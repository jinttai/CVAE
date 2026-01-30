"""
Latent space meaningfulness check: start/final joint fixed at 0.

- Condition: same posture control (q0_start, q0_goal).
- Trajectory: q_start_joint=0, q_end_joint=0 (start and final joint fixed at 0).
- CVAE (v5_joint_change.pth): 1024 batch sampling → loss distribution.
- MLP (v4.pth): single output loss → vertical line on the plot.
- Random-weight CVAE (same architecture): 1024 sampling → loss distribution.
"""

import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import math
import argparse

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.models.cvae import CVAE, MLP
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    """Euler (ZYX) to quaternion (x, y, z, w)."""
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return torch.stack([qx, qy, qz, qw], dim=-1)


def load_cvae(weights_path, input_dim, output_dim, latent_dim, device, joint_limits):
    model = CVAE(input_dim, output_dim, latent_dim, joint_limits=joint_limits).to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[Warning] strict load failed, trying strict=False. {e}")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_mlp(weights_path, input_dim, output_dim, device, joint_limits):
    model = MLP(input_dim, output_dim, joint_limits=joint_limits).to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[Warning] strict load failed, trying strict=False. {e}")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Latent loss distribution (start/final joint = 0)")
    parser.add_argument("--num-samples", type=int, default=1024, help="Batch size for CVAE sampling")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: outputs/results/latent_loss_dist)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = args.num_samples

    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 3
    TOTAL_TIME = 10.0
    JOINT_SQUARED_WEIGHT = 0.01
    JOINT_CHANGE_WEIGHT = 0.01
    MAX_JOINT_WEIGHT = 0.1

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    n_q = robot["n_q"]

    # Same condition as optimize_cvae: posture control
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    roll_rad = math.radians(15.0)
    pitch_rad = math.radians(15.0)
    yaw_rad = math.radians(-15.0)
    q0_goal = euler_to_quaternion(
        torch.tensor([roll_rad], device=device),
        torch.tensor([pitch_rad], device=device),
        torch.tensor([yaw_rad], device=device),
    )
    if q0_goal.dim() == 1:
        q0_goal = q0_goal.unsqueeze(0)  # [1, 4]
    condition = torch.cat([q0_start, q0_goal], dim=1)  # [1, 8]

    # Start/final joint fixed at 0
    q_start_joint = torch.zeros(num_samples, n_q, device=device, dtype=torch.float32)
    q_end_joint = torch.zeros(num_samples, n_q, device=device, dtype=torch.float32)

    # ----- 1) CVAE trained (v5_joint_change.pth) -----
    cvae_path = os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v5_joint_change.pth")
    cvae_trained = load_cvae(cvae_path, COND_DIM, OUTPUT_DIM, LATENT_DIM, device, robot["joint_limits"])
    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1)
        candidates = cvae_trained.decode(cond_batch, z)
        q0_start_batch = q0_start.repeat(num_samples, 1)
        q0_goal_batch = q0_goal.repeat(num_samples, 1)
        total_loss, _ = physics.calculate_total_loss(
            candidates, q0_start_batch, q0_goal_batch,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False,
            q_start_joint=q_start_joint,
            q_end_joint=q_end_joint,
        )
    losses_cvae_trained = torch.where(
        torch.isfinite(total_loss), total_loss, torch.full_like(total_loss, float("inf"))
    )
    losses_cvae_trained_np = losses_cvae_trained.cpu().numpy()
    print(f"[CVAE trained] min={losses_cvae_trained_np.min():.6f} max={losses_cvae_trained_np.max():.6f} mean={losses_cvae_trained_np.mean():.6f}")

    # ----- 2) MLP (v4.pth) single output → one loss -----
    mlp_path = os.path.join(ROOT_DIR, "outputs/weights/mlp_debug/v4.pth")
    mlp_model = load_mlp(mlp_path, COND_DIM, OUTPUT_DIM, device, robot["joint_limits"])
    with torch.no_grad():
        mlp_waypoints = mlp_model(condition)  # [1, OUTPUT_DIM]
        mlp_loss, _ = physics.calculate_total_loss(
            mlp_waypoints, q0_start, q0_goal,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=True,
            q_start_joint=torch.zeros(1, n_q, device=device),
            q_end_joint=torch.zeros(1, n_q, device=device),
        )
    mlp_loss_val = mlp_loss.item()
    print(f"[MLP v4] loss = {mlp_loss_val:.6f}")

    # ----- 3) CVAE random weights (same architecture) -----
    cvae_random = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM, joint_limits=robot["joint_limits"]).to(device)
    cvae_random.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1)
        candidates_rand = cvae_random.decode(cond_batch, z)
        q0_start_batch = q0_start.repeat(num_samples, 1)
        q0_goal_batch = q0_goal.repeat(num_samples, 1)
        total_loss_rand, _ = physics.calculate_total_loss(
            candidates_rand, q0_start_batch, q0_goal_batch,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False,
            q_start_joint=q_start_joint,
            q_end_joint=q_end_joint,
        )
    losses_cvae_random = torch.where(
        torch.isfinite(total_loss_rand), total_loss_rand, torch.full_like(total_loss_rand, float("inf"))
    )
    losses_cvae_random_np = losses_cvae_random.cpu().numpy()
    print(f"[CVAE random] min={losses_cvae_random_np.min():.6f} max={losses_cvae_random_np.max():.6f} mean={losses_cvae_random_np.mean():.6f}")

    # ----- Plot: loss distributions + MLP as vertical line -----
    out_dir = args.out_dir or os.path.join(ROOT_DIR, "outputs/results/latent_loss_dist")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Filter inf for histogram
    finite_trained = losses_cvae_trained_np[np.isfinite(losses_cvae_trained_np)]
    finite_random = losses_cvae_random_np[np.isfinite(losses_cvae_random_np)]
    all_vals = np.concatenate([finite_trained, finite_random, [mlp_loss_val]])
    lo = min(np.percentile(all_vals, 0.5), mlp_loss_val) - 1e-6
    hi = max(np.percentile(all_vals, 99.5), mlp_loss_val) + 1e-6
    bins = np.linspace(lo, hi, 50)

    ax.hist(finite_trained, bins=bins, alpha=0.6, label=f"CVAE trained (v5_joint_change) n={len(finite_trained)}", color="C0", density=True)
    ax.hist(finite_random, bins=bins, alpha=0.6, label=f"CVAE random weights n={len(finite_random)}", color="C1", density=True)
    ax.axvline(mlp_loss_val, color="C2", linestyle="-", linewidth=2, label=f"MLP (v4.pth) = {mlp_loss_val:.4f}")

    ax.set_xlabel("Total loss")
    ax.set_ylabel("Density")
    ax.set_title("Latent loss distribution (start/final joint = 0, posture condition, 1024 samples)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "latent_loss_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
    return 0


if __name__ == "__main__":
    main()
