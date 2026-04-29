"""
1024 random goal orientations에 대해 random waypoints를 Adam으로 최적화.
각 goal별 수렴 loss/angle error 분포를 확인.
"""

import sys
import time
import os
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.utils.runtime_env import configure_windows_runtime

configure_windows_runtime()

import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2); sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2); sy = torch.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return torch.stack([qx, qy, qz, qw], dim=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    JOINT_SQUARED_WEIGHT = 0.01
    JOINT_CHANGE_WEIGHT = 0.01
    MAX_JOINT_WEIGHT = 0.1

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    # 1024 random goals (full range uniform)
    N_GOALS = 1

    torch.manual_seed(42)
    yaw   = 2 * math.pi * torch.rand(N_GOALS, device=device) - math.pi    # [-pi, pi]
    pitch = math.pi * torch.rand(N_GOALS, device=device) - math.pi / 2    # [-pi/2, pi/2]
    roll  = 2 * math.pi * torch.rand(N_GOALS, device=device) - math.pi    # [-pi, pi]

    q0_goals = euler_to_quaternion(roll, pitch, yaw)  # [1024, 4]
    q0_start = torch.zeros(N_GOALS, 4, device=device)
    q0_start[:, 3] = 1.0  # [1024, 4] identity
    euler_goals = torch.stack([yaw, pitch, roll], dim=1)  # [1024, 3]

    # Random initial waypoints (small values)
    waypoints_param = (torch.randn(N_GOALS, OUTPUT_DIM, device=device) * 0.1).requires_grad_(True)

    # Adam optimization
    NUM_ITERS = 500
    optimizer = torch.optim.Adam([waypoints_param], lr=1e-3)
    loss_history = []

    print(f"\n--- Adam optimization: {N_GOALS} goals x {NUM_ITERS} iters ---")
    t0 = time.time()

    for it in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()

        total_loss, _ = physics.calculate_total_loss(
            waypoints_param, q0_start, q0_goals,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False
        )

        valid = torch.isfinite(total_loss)
        safe_loss = torch.where(valid, total_loss, torch.zeros_like(total_loss))
        safe_loss.sum().backward()
        optimizer.step()

        mean_loss = safe_loss[valid].mean().item() if valid.any() else float("nan")
        loss_history.append(mean_loss)

        if it <= 10 or it % 50 == 0 or it == NUM_ITERS:
            median_loss = float(np.median(safe_loss[valid].detach().cpu().numpy())) if valid.any() else float("nan")
            print(f"  Iter [{it:3d}/{NUM_ITERS}] mean={mean_loss:.6f}  median={median_loss:.6f}  valid={valid.sum()}/{N_GOALS}")

    t_opt = time.time() - t0
    print(f"Done in {t_opt:.1f}s")

    # Final evaluation
    with torch.no_grad():
        final_loss, final_dict = physics.calculate_total_loss(
            waypoints_param, q0_start, q0_goals,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False
        )
        final_total = final_loss.cpu().numpy()
        final_physics = final_dict['physics_loss'].cpu().numpy()

        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        batch_sim = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        _, q_final_all = batch_sim(q_traj, q_dot_traj, q0_start, q0_goals)

        dot = (q_final_all * q0_goals).sum(dim=1).abs().clamp(-1.0, 1.0)
        angle_errors = (2.0 * torch.acos(dot) * 180.0 / math.pi).cpu().numpy()

    euler_np = euler_goals.cpu().numpy()

    print(f"\n=== Converged Loss Distribution (N={N_GOALS}) ===")
    print(f"Total loss  : mean={final_total.mean():.4f}, std={final_total.std():.4f}, "
          f"median={np.median(final_total):.4f}, min={final_total.min():.4f}, max={final_total.max():.4f}")
    print(f"Physics loss: mean={final_physics.mean():.4f}, std={final_physics.std():.4f}, "
          f"median={np.median(final_physics):.4f}, min={final_physics.min():.4f}, max={final_physics.max():.4f}")
    print(f"Angle error : mean={angle_errors.mean():.2f} deg, std={angle_errors.std():.2f}, "
          f"median={np.median(angle_errors):.2f}, min={angle_errors.min():.2f}, max={angle_errors.max():.2f}")

    # Plots
    save_dir = os.path.join(ROOT_DIR, "outputs/plots/eval_random_goals")
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(final_total, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(final_total), color="red", linestyle="--", label=f"median={np.median(final_total):.4f}")
    ax.set_xlabel("Converged Total Loss")
    ax.set_ylabel("Count")
    ax.set_title("Total Loss Distribution")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(final_physics, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(final_physics), color="red", linestyle="--", label=f"median={np.median(final_physics):.4f}")
    ax.set_xlabel("Converged Physics Loss")
    ax.set_ylabel("Count")
    ax.set_title("Physics Loss Distribution")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(angle_errors, bins=50, color="seagreen", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(angle_errors), color="red", linestyle="--", label=f"median={np.median(angle_errors):.2f} deg")
    ax.set_xlabel("Angle Error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Converged Angle Error Distribution")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(loss_history, color="steelblue", linewidth=1)
    ax.set_xlabel("Adam Iteration")
    ax.set_ylabel("Mean Total Loss")
    ax.set_title("Loss Convergence (mean over 1024 goals)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Random Init + Adam: {N_GOALS} Random Goals, {NUM_ITERS} iters", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "loss_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {save_path}")

    np.savez(
        os.path.join(save_dir, "eval_results.npz"),
        euler_goals=euler_np,
        final_total_losses=final_total,
        final_physics_losses=final_physics,
        angle_errors=angle_errors,
        loss_history=np.array(loss_history),
    )
    print(f"Saved results to {save_dir}/eval_results.npz")


if __name__ == "__main__":
    main()
