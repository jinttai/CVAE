"""
Dataset Generation: Waypoint -> Orientation Change

Generates (waypoint, orientation_change) pairs by:
1. Sampling random waypoints within joint limits
2. Running forward physics simulation (PhysicsLayer)
3. Recording the resulting final orientation as Euler angles (orientation change from identity)

Since q0_init is always identity [0,0,0,1] and start/goal joints are zeros,
the final quaternion directly represents the orientation change caused by the waypoints.

Output fields:
  - waypoints: [N, 18]  (3 waypoints x 6 DoF)
  - q_final:   [N, 4]   (final quaternion [x,y,z,w])
  - euler_final: [N, 3] (final Euler angles [yaw, pitch, roll] in rad)
"""

import torch
import os
import sys
import time
import math
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.func import vmap

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


# ---------------------------------------------------------------------------
# Quaternion / Euler helpers
# ---------------------------------------------------------------------------
def quat_to_rot_batch(q):
    """Quaternion [B, 4] (x,y,z,w) -> rotation matrix [B, 3, 3]."""
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        torch.stack([1 - 2*(yy+zz), 2*(xy-wz), 2*(xz+wy)], dim=-1),
        torch.stack([2*(xy+wz), 1 - 2*(xx+zz), 2*(yz-wx)], dim=-1),
        torch.stack([2*(xz-wy), 2*(yz+wx), 1 - 2*(xx+yy)], dim=-1),
    ], dim=-2)
    return R


def rot_to_euler_batch(R):
    """
    Rotation matrix [B, 3, 3] -> Euler angles [B, 3] (yaw, pitch, roll).
    ZYX convention.
    """
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
    singular = sy < 1e-6

    yaw = torch.where(singular,
                       torch.atan2(-R[:, 0, 1], R[:, 1, 1]),
                       torch.atan2(R[:, 1, 0], R[:, 0, 0]))
    pitch = torch.atan2(-R[:, 2, 0], sy)
    roll = torch.where(singular,
                        torch.zeros_like(yaw),
                        torch.atan2(R[:, 2, 1], R[:, 2, 2]))

    return torch.stack([yaw, pitch, roll], dim=-1)


def quat_to_euler_batch(q):
    """Quaternion [B, 4] -> Euler [B, 3] (yaw, pitch, roll) in rad."""
    R = quat_to_rot_batch(q)
    return rot_to_euler_batch(R)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data_batch(physics, batch_size, n_q, num_waypoints,
                        joint_min, joint_max, device):
    """
    Generate one batch of (waypoints, q_final, euler_final).
    start/goal joints = 0, q0_init = identity.
    """
    start_joint = torch.zeros(batch_size, n_q, device=device)
    goal_joint = torch.zeros(batch_size, n_q, device=device)

    # Random waypoints within joint limits
    waypoints = (torch.rand(batch_size, num_waypoints * n_q, device=device)
                 * (joint_max - joint_min) + joint_min)

    # Identity initial orientation
    q0_init = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(batch_size, 4)
    # Dummy goal (not used for dataset, only needed by simulate_single signature)
    q0_goal_dummy = q0_init.clone()

    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(
            waypoints, q_start=start_joint, q_end=goal_joint
        )
        batch_sim = vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        _, q_final = batch_sim(q_traj, q_dot_traj, q0_init, q0_goal_dummy)

        euler_final = quat_to_euler_batch(q_final)

    return waypoints, q_final, euler_final


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_distributions(waypoints_np, euler_np, q_final_np, n_q, num_waypoints, save_dir, name):
    """Save distribution summary plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Waypoint distributions (all joints overlaid)
    wp_deg = np.rad2deg(waypoints_np)
    ax = axes[0, 0]
    for wp_idx in range(num_waypoints):
        s = wp_idx * n_q
        ax.hist(wp_deg[:, s:s+n_q].flatten(), bins=60, alpha=0.4,
                label=f"WP{wp_idx+1}")
    ax.set_title("Waypoints (all joints)")
    ax.set_xlabel("Angle (deg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Euler angle distributions
    euler_deg = np.rad2deg(euler_np)
    euler_labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    for i in range(3):
        ax = axes[0, 1] if i == 0 else axes[0, 2] if i == 1 else axes[1, 0]
        ax.hist(euler_deg[:, i], bins=60, edgecolor='black', alpha=0.7)
        ax.set_title(f"{euler_labels[i]} distribution")
        ax.set_xlabel("Angle (deg)")
        ax.grid(True, alpha=0.3)

    # Quaternion components
    ax = axes[1, 1]
    for i, lbl in enumerate(["qx", "qy", "qz", "qw"]):
        ax.hist(q_final_np[:, i], bins=60, alpha=0.5, label=lbl)
    ax.set_title("Quaternion components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Euler angle scatter: yaw vs pitch
    ax = axes[1, 2]
    ax.scatter(euler_deg[:, 0], euler_deg[:, 1], s=1, alpha=0.3)
    ax.set_xlabel("Yaw (deg)")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Yaw vs Pitch coverage")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Dataset: {name} ({len(waypoints_np)} samples)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name}_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate waypoint -> orientation change dataset")
    parser.add_argument("--num-samples", type=int, default=100000,
                        help="Total samples (default: 100000)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size (default: 1024)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-name", type=str,
                        default="waypoint_orientation_dataset")
    parser.add_argument("--save-format", type=str, default="pt",
                        choices=["pt", "npy", "csv"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Waypoint -> Orientation Change Dataset Generation ({device}) ===")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)

    n_q = robot["n_q"]
    NUM_WAYPOINTS = 3
    TOTAL_TIME = 10.0
    JOINT_MIN_RAD = math.radians(-140.0)
    JOINT_MAX_RAD = math.radians(140.0)

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    output_dir = args.output_dir or os.path.join(ROOT_DIR, "outputs/data")
    os.makedirs(output_dir, exist_ok=True)

    all_waypoints = []
    all_q_finals = []
    all_eulers = []

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    total_generated = 0

    print(f"\nSamples: {args.num_samples}  |  Batch: {args.batch_size}")
    print(f"Joint limits: [-140, 140] deg  |  Waypoints: {NUM_WAYPOINTS}")
    print(f"Total time: {TOTAL_TIME}s  |  Start/Goal joints: zeros\n")

    t0 = time.time()

    with tqdm(total=args.num_samples, desc="Generating", unit="samp") as pbar:
        for _ in range(num_batches):
            bs = min(args.batch_size, args.num_samples - total_generated)
            wp, qf, ef = generate_data_batch(
                physics, bs, n_q, NUM_WAYPOINTS,
                JOINT_MIN_RAD, JOINT_MAX_RAD, device
            )
            all_waypoints.append(wp.cpu())
            all_q_finals.append(qf.cpu())
            all_eulers.append(ef.cpu())
            total_generated += bs
            pbar.update(bs)

    elapsed = time.time() - t0

    waypoints_t = torch.cat(all_waypoints, dim=0)   # [N, 18]
    q_finals_t = torch.cat(all_q_finals, dim=0)     # [N, 4]
    eulers_t = torch.cat(all_eulers, dim=0)          # [N, 3]

    print(f"\nGenerated {total_generated} samples in {elapsed:.1f}s "
          f"({total_generated/elapsed:.0f} samp/s)")
    print(f"  waypoints:    {waypoints_t.shape}")
    print(f"  q_final:      {q_finals_t.shape}")
    print(f"  euler_final:  {eulers_t.shape}")

    # ---- Statistics ----
    euler_deg = eulers_t.numpy() * 180.0 / math.pi
    print(f"\nEuler angle range (deg):")
    for i, lbl in enumerate(["Yaw", "Pitch", "Roll"]):
        print(f"  {lbl:>5}: [{euler_deg[:, i].min():+7.1f}, {euler_deg[:, i].max():+7.1f}]  "
              f"mean={euler_deg[:, i].mean():+6.1f}  std={euler_deg[:, i].std():.1f}")

    q_norms = torch.norm(q_finals_t, dim=1)
    print(f"\nQuaternion norm: mean={q_norms.mean():.6f}, std={q_norms.std():.2e}")

    # ---- Save ----
    name = args.output_name
    if args.save_format == "pt":
        save_path = os.path.join(output_dir, f"{name}.pt")
        torch.save({
            'waypoints': waypoints_t,
            'q_final': q_finals_t,
            'euler_final': eulers_t,
            'metadata': {
                'n_q': n_q,
                'num_waypoints': NUM_WAYPOINTS,
                'total_time': TOTAL_TIME,
                'joint_min_rad': JOINT_MIN_RAD,
                'joint_max_rad': JOINT_MAX_RAD,
                'num_samples': total_generated,
                'start_joint': 'zeros',
                'goal_joint': 'zeros',
                'q0_init': [0.0, 0.0, 0.0, 1.0],
                'euler_convention': 'ZYX (yaw, pitch, roll)',
            }
        }, save_path)
        print(f"\nSaved: {save_path}")

    elif args.save_format == "npy":
        np.save(os.path.join(output_dir, f"{name}_waypoints.npy"),
                waypoints_t.numpy())
        np.save(os.path.join(output_dir, f"{name}_q_final.npy"),
                q_finals_t.numpy())
        np.save(os.path.join(output_dir, f"{name}_euler_final.npy"),
                eulers_t.numpy())
        print(f"\nSaved: {output_dir}/{name}_*.npy")

    elif args.save_format == "csv":
        header = []
        header.extend([f"wp_{i+1}" for i in range(NUM_WAYPOINTS * n_q)])
        header.extend(["qx", "qy", "qz", "qw"])
        header.extend(["yaw", "pitch", "roll"])
        all_data = np.concatenate([
            waypoints_t.numpy(), q_finals_t.numpy(), eulers_t.numpy()
        ], axis=1)
        save_path = os.path.join(output_dir, f"{name}.csv")
        np.savetxt(save_path, all_data, delimiter=",",
                   header=",".join(header), comments="")
        print(f"\nSaved: {save_path}")

    # ---- Distribution plots ----
    plot_distributions(
        waypoints_t.numpy(), eulers_t.numpy(), q_finals_t.numpy(),
        n_q, NUM_WAYPOINTS, output_dir, name
    )


if __name__ == "__main__":
    main()
