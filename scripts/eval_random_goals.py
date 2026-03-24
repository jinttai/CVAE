"""
Evaluate CVAE + physics loss for 1024 random goal orientations (euler angles).
For each goal: sample N CVAE candidates, pick best, record converged loss.
Output: loss distribution histogram + statistics.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import sys
import time
import numpy as np
import math
import matplotlib.pyplot as plt

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.models.cvae import CVAE
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

    # Setup
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

    # Load CVAE
    cvae_path = os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v5_joint_change.pth")
    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM, joint_limits=robot['joint_limits']).to(device)
    state_dict = torch.load(cvae_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("CVAE loaded")

    # Generate 1024 random goal euler angles (uniform in [-30deg, 30deg])
    N_GOALS = 1024
    NUM_SAMPLES_PER_GOAL = 256  # CVAE samples per goal
    max_rad = math.radians(30.0)

    torch.manual_seed(42)
    yaw = (2 * max_rad) * torch.rand(N_GOALS, device=device) - max_rad
    pitch = (2 * max_rad) * torch.rand(N_GOALS, device=device) - max_rad
    roll = (2 * max_rad) * torch.rand(N_GOALS, device=device) - max_rad

    q0_goals = euler_to_quaternion(roll, pitch, yaw)  # [1024, 4]
    q0_start = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0)  # [1, 4]

    euler_goals = torch.stack([yaw, pitch, roll], dim=1)  # [1024, 3]

    print(f"Evaluating {N_GOALS} random goals, {NUM_SAMPLES_PER_GOAL} CVAE samples each...")

    best_losses = []
    best_physics_losses = []
    angle_errors = []

    t0 = time.time()

    # Process in mini-batches to avoid OOM
    # Each goal needs NUM_SAMPLES_PER_GOAL candidates
    # Total batch = goals_per_batch * NUM_SAMPLES_PER_GOAL
    GOALS_PER_BATCH = 32  # 32 * 256 = 8192 total samples per batch

    with torch.no_grad():
        for batch_start in range(0, N_GOALS, GOALS_PER_BATCH):
            batch_end = min(batch_start + GOALS_PER_BATCH, N_GOALS)
            n_goals_batch = batch_end - batch_start

            # Goals for this batch: [n_goals_batch, 4]
            goals_batch = q0_goals[batch_start:batch_end]

            # Expand each goal to NUM_SAMPLES_PER_GOAL copies
            # [n_goals_batch, 4] -> [n_goals_batch * NUM_SAMPLES_PER_GOAL, 4]
            q0_goal_expanded = goals_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_GOAL, -1).reshape(-1, 4)
            q0_start_expanded = q0_start.expand(n_goals_batch * NUM_SAMPLES_PER_GOAL, -1)

            # Condition: [q0_start, q0_goal]
            cond = torch.cat([q0_start_expanded, q0_goal_expanded], dim=1)  # [n*S, 8]

            # CVAE decode
            z = torch.randn(n_goals_batch * NUM_SAMPLES_PER_GOAL, LATENT_DIM, device=device)
            candidates = model.decode(cond, z)  # [n*S, output_dim]

            # Evaluate losses
            total_loss_all, loss_dict = physics.calculate_total_loss(
                candidates, q0_start_expanded, q0_goal_expanded,
                joint_squared_weight=JOINT_SQUARED_WEIGHT,
                joint_change_weight=JOINT_CHANGE_WEIGHT,
                max_joint_weight=MAX_JOINT_WEIGHT,
                return_mean=False
            )
            physics_loss_all = loss_dict['physics_loss']  # [n*S]

            # Reshape to [n_goals_batch, NUM_SAMPLES_PER_GOAL]
            total_loss_grid = total_loss_all.view(n_goals_batch, NUM_SAMPLES_PER_GOAL)
            physics_loss_grid = physics_loss_all.view(n_goals_batch, NUM_SAMPLES_PER_GOAL)

            # Handle non-finite
            total_loss_grid = torch.where(
                torch.isfinite(total_loss_grid), total_loss_grid,
                torch.full_like(total_loss_grid, float("inf"))
            )

            # Best per goal
            best_idx = total_loss_grid.argmin(dim=1)  # [n_goals_batch]
            best_total = total_loss_grid[torch.arange(n_goals_batch), best_idx]  # [n_goals_batch]
            best_phys = physics_loss_grid[torch.arange(n_goals_batch), best_idx]  # [n_goals_batch]
            best_losses.append(best_total.cpu().numpy())
            best_physics_losses.append(best_phys.cpu().numpy())

            # Angle error for best candidates (batched)
            all_candidates = candidates.view(n_goals_batch, NUM_SAMPLES_PER_GOAL, -1)
            best_wp = all_candidates[torch.arange(n_goals_batch), best_idx]  # [n_goals_batch, output_dim]
            q0_start_bg = q0_start.expand(n_goals_batch, -1)  # [n_goals_batch, 4]

            q_traj, q_dot_traj = physics.generate_trajectory(best_wp)  # [n_goals_batch, T, n_q]
            batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
            _, q_final_batch = batch_sim_fn(q_traj, q_dot_traj, q0_start_bg, goals_batch)  # [n_goals_batch, 4]

            dot = (q_final_batch * goals_batch).sum(dim=1).abs().clamp(-1.0, 1.0)  # [n_goals_batch]
            err_deg = 2.0 * torch.acos(dot) * 180.0 / math.pi  # [n_goals_batch]
            angle_errors.append(err_deg.cpu().numpy())

            elapsed = time.time() - t0
            print(f"  [{batch_end}/{N_GOALS}] elapsed={elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")

    best_losses = np.concatenate(best_losses)
    best_physics_losses = np.concatenate(best_physics_losses)
    angle_errors = np.concatenate(angle_errors)
    euler_np = euler_goals.cpu().numpy()  # [1024, 3] in rad

    # Statistics
    print(f"\n=== Loss Distribution (N={N_GOALS}) ===")
    print(f"Total loss  : mean={best_losses.mean():.4f}, std={best_losses.std():.4f}, "
          f"median={np.median(best_losses):.4f}, min={best_losses.min():.4f}, max={best_losses.max():.4f}")
    print(f"Physics loss: mean={best_physics_losses.mean():.4f}, std={best_physics_losses.std():.4f}, "
          f"median={np.median(best_physics_losses):.4f}, min={best_physics_losses.min():.4f}, max={best_physics_losses.max():.4f}")
    print(f"Angle error : mean={angle_errors.mean():.2f} deg, std={angle_errors.std():.2f}, "
          f"median={np.median(angle_errors):.2f}, min={angle_errors.min():.2f}, max={angle_errors.max():.2f}")

    # Plots
    save_dir = os.path.join(ROOT_DIR, "outputs/plots/eval_random_goals")
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Total loss histogram
    ax = axes[0, 0]
    ax.hist(best_losses, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(best_losses), color="red", linestyle="--", label=f"median={np.median(best_losses):.4f}")
    ax.set_xlabel("Best Total Loss (per goal)")
    ax.set_ylabel("Count")
    ax.set_title(f"Total Loss Distribution (N={N_GOALS})")
    ax.legend()

    # 2) Physics loss histogram
    ax = axes[0, 1]
    ax.hist(best_physics_losses, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(best_physics_losses), color="red", linestyle="--", label=f"median={np.median(best_physics_losses):.4f}")
    ax.set_xlabel("Best Physics Loss (per goal)")
    ax.set_ylabel("Count")
    ax.set_title("Physics Loss Distribution")
    ax.legend()

    # 3) Angle error histogram
    ax = axes[1, 0]
    ax.hist(angle_errors, bins=50, color="seagreen", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(angle_errors), color="red", linestyle="--", label=f"median={np.median(angle_errors):.2f} deg")
    ax.set_xlabel("Quaternion Angle Error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Angle Error Distribution")
    ax.legend()

    # 4) Angle error vs euler angle magnitude
    ax = axes[1, 1]
    euler_mag = np.linalg.norm(euler_np, axis=1)  # L2 norm of (yaw, pitch, roll)
    sc = ax.scatter(np.degrees(euler_mag), angle_errors, s=3, alpha=0.5, c=best_physics_losses, cmap="viridis")
    ax.set_xlabel("Goal Euler Angle Magnitude (deg)")
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title("Error vs Goal Magnitude")
    plt.colorbar(sc, ax=ax, label="Physics Loss")

    fig.suptitle(f"CVAE Sampling Evaluation: {N_GOALS} Random Goals, {NUM_SAMPLES_PER_GOAL} samples/goal", fontsize=13)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "loss_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {save_path}")

    # Save raw data
    np.savez(
        os.path.join(save_dir, "eval_results.npz"),
        euler_goals=euler_np,
        best_losses=best_losses,
        best_physics_losses=best_physics_losses,
        angle_errors=angle_errors,
    )
    print(f"Saved results to {save_dir}/eval_results.npz")


if __name__ == "__main__":
    main()
