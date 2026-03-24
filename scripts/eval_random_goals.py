"""
Evaluate CVAE + Adam optimization for 1024 random goal orientations.
1. CVAE sampling: pick best of 256 candidates per goal (warm start)
2. Adam optimization: optimize all 1024 waypoints in parallel until convergence
3. Plot converged loss / angle error distributions
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

    # =====================================================================
    # 1. Generate 1024 random goals
    # =====================================================================
    N_GOALS = 1024
    NUM_SAMPLES_PER_GOAL = 256
    max_rad = math.radians(30.0)

    torch.manual_seed(42)
    yaw = (2 * max_rad) * torch.rand(N_GOALS, device=device) - max_rad
    pitch = (2 * max_rad) * torch.rand(N_GOALS, device=device) - max_rad
    roll = (2 * max_rad) * torch.rand(N_GOALS, device=device) - max_rad

    q0_goals = euler_to_quaternion(roll, pitch, yaw)  # [1024, 4]
    q0_start_single = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # [4]
    q0_start_batch = q0_start_single.unsqueeze(0).expand(N_GOALS, -1)  # [1024, 4]
    euler_goals = torch.stack([yaw, pitch, roll], dim=1)  # [1024, 3]

    # =====================================================================
    # 2. CVAE warm start: pick best of 256 samples per goal
    # =====================================================================
    print(f"\n--- Phase 1: CVAE sampling ({NUM_SAMPLES_PER_GOAL} samples/goal) ---")
    t0 = time.time()

    GOALS_PER_BATCH = 32
    best_waypoints_list = []

    with torch.no_grad():
        for batch_start in range(0, N_GOALS, GOALS_PER_BATCH):
            batch_end = min(batch_start + GOALS_PER_BATCH, N_GOALS)
            n = batch_end - batch_start

            goals_b = q0_goals[batch_start:batch_end]  # [n, 4]
            q0_goal_exp = goals_b.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_GOAL, -1).reshape(-1, 4)
            q0_start_exp = q0_start_single.unsqueeze(0).expand(n * NUM_SAMPLES_PER_GOAL, -1)

            cond = torch.cat([q0_start_exp, q0_goal_exp], dim=1)
            z = torch.randn(n * NUM_SAMPLES_PER_GOAL, LATENT_DIM, device=device)
            candidates = model.decode(cond, z)

            total_loss_all, _ = physics.calculate_total_loss(
                candidates, q0_start_exp, q0_goal_exp,
                joint_squared_weight=JOINT_SQUARED_WEIGHT,
                joint_change_weight=JOINT_CHANGE_WEIGHT,
                max_joint_weight=MAX_JOINT_WEIGHT,
                return_mean=False
            )

            loss_grid = total_loss_all.view(n, NUM_SAMPLES_PER_GOAL)
            loss_grid = torch.where(torch.isfinite(loss_grid), loss_grid, torch.full_like(loss_grid, float("inf")))
            best_idx = loss_grid.argmin(dim=1)

            all_cand = candidates.view(n, NUM_SAMPLES_PER_GOAL, -1)
            best_wp = all_cand[torch.arange(n, device=device), best_idx]  # [n, output_dim]
            best_waypoints_list.append(best_wp)

            print(f"  [{batch_end}/{N_GOALS}] best loss range: [{loss_grid[torch.arange(n), best_idx].min():.4f}, {loss_grid[torch.arange(n), best_idx].max():.4f}]")

    # [1024, output_dim]
    init_waypoints = torch.cat(best_waypoints_list, dim=0)
    t_cvae = time.time() - t0
    print(f"CVAE sampling done in {t_cvae:.1f}s")

    # Pre-optimization loss
    with torch.no_grad():
        pre_loss, pre_dict = physics.calculate_total_loss(
            init_waypoints, q0_start_batch, q0_goals,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False
        )
    pre_total = pre_loss.cpu().numpy()
    pre_physics = pre_dict['physics_loss'].cpu().numpy()
    print(f"Pre-opt loss: mean={pre_total.mean():.4f}, median={np.median(pre_total):.4f}")

    # =====================================================================
    # 3. Adam optimization: all 1024 goals in parallel
    # =====================================================================
    print(f"\n--- Phase 2: Adam optimization (1024 goals in parallel) ---")
    t1 = time.time()

    waypoints_param = init_waypoints.clone().detach().requires_grad_(True)  # [1024, output_dim]
    optimizer = torch.optim.Adam([waypoints_param], lr=1e-3)

    NUM_ITERS = 300
    loss_history = []

    for it in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()

        total_loss, loss_dict = physics.calculate_total_loss(
            waypoints_param, q0_start_batch, q0_goals,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False  # [1024]
        )

        # Replace NaN/Inf with 0 gradient (don't corrupt other goals)
        valid = torch.isfinite(total_loss)
        safe_loss = torch.where(valid, total_loss, torch.zeros_like(total_loss))
        loss_sum = safe_loss.sum()
        loss_sum.backward()
        optimizer.step()

        mean_loss = safe_loss[valid].mean().item() if valid.any() else float("nan")
        loss_history.append(mean_loss)

        if it <= 10 or it % 50 == 0 or it == NUM_ITERS:
            median_loss = np.median(safe_loss[valid].detach().cpu().numpy()) if valid.any() else float("nan")
            print(f"  Iter [{it:3d}/{NUM_ITERS}] mean={mean_loss:.6f}  median={median_loss:.6f}  valid={valid.sum()}/{N_GOALS}")

    t_opt = time.time() - t1
    print(f"Optimization done in {t_opt:.1f}s")

    # =====================================================================
    # 4. Final evaluation (batched)
    # =====================================================================
    print(f"\n--- Phase 3: Final evaluation ---")
    with torch.no_grad():
        final_loss, final_dict = physics.calculate_total_loss(
            waypoints_param, q0_start_batch, q0_goals,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False
        )
        final_total = final_loss.cpu().numpy()
        final_physics = final_dict['physics_loss'].cpu().numpy()

        # Angle errors (batched)
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        _, q_final_all = batch_sim_fn(q_traj, q_dot_traj, q0_start_batch, q0_goals)

        dot = (q_final_all * q0_goals).sum(dim=1).abs().clamp(-1.0, 1.0)
        angle_errors = (2.0 * torch.acos(dot) * 180.0 / math.pi).cpu().numpy()

    euler_np = euler_goals.cpu().numpy()

    # Statistics
    print(f"\n=== Converged Loss Distribution (N={N_GOALS}, {NUM_ITERS} Adam iters) ===")
    print(f"Total loss  : mean={final_total.mean():.4f}, std={final_total.std():.4f}, "
          f"median={np.median(final_total):.4f}, min={final_total.min():.4f}, max={final_total.max():.4f}")
    print(f"Physics loss: mean={final_physics.mean():.4f}, std={final_physics.std():.4f}, "
          f"median={np.median(final_physics):.4f}, min={final_physics.min():.4f}, max={final_physics.max():.4f}")
    print(f"Angle error : mean={angle_errors.mean():.2f} deg, std={angle_errors.std():.2f}, "
          f"median={np.median(angle_errors):.2f}, min={angle_errors.min():.2f}, max={angle_errors.max():.2f}")

    # =====================================================================
    # 5. Plots
    # =====================================================================
    save_dir = os.path.join(ROOT_DIR, "outputs/plots/eval_random_goals")
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1) Total loss histogram (before vs after)
    ax = axes[0, 0]
    ax.hist(pre_total, bins=50, color="lightcoral", edgecolor="white", alpha=0.6, label="Before opt")
    ax.hist(final_total, bins=50, color="steelblue", edgecolor="white", alpha=0.8, label="After opt")
    ax.axvline(np.median(final_total), color="red", linestyle="--", label=f"median={np.median(final_total):.4f}")
    ax.set_xlabel("Total Loss")
    ax.set_ylabel("Count")
    ax.set_title("Total Loss: Before vs After Optimization")
    ax.legend()

    # 2) Physics loss histogram
    ax = axes[0, 1]
    ax.hist(pre_physics, bins=50, color="lightsalmon", edgecolor="white", alpha=0.6, label="Before opt")
    ax.hist(final_physics, bins=50, color="darkorange", edgecolor="white", alpha=0.8, label="After opt")
    ax.axvline(np.median(final_physics), color="red", linestyle="--", label=f"median={np.median(final_physics):.4f}")
    ax.set_xlabel("Physics Loss")
    ax.set_ylabel("Count")
    ax.set_title("Physics Loss: Before vs After")
    ax.legend()

    # 3) Angle error histogram
    ax = axes[0, 2]
    ax.hist(angle_errors, bins=50, color="seagreen", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(angle_errors), color="red", linestyle="--", label=f"median={np.median(angle_errors):.2f} deg")
    ax.set_xlabel("Quaternion Angle Error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Converged Angle Error Distribution")
    ax.legend()

    # 4) Loss convergence curve
    ax = axes[1, 0]
    ax.plot(loss_history, color="steelblue", linewidth=1)
    ax.set_xlabel("Adam Iteration")
    ax.set_ylabel("Mean Total Loss")
    ax.set_title("Loss Convergence (mean over 1024 goals)")
    ax.grid(True, alpha=0.3)

    # 5) Angle error vs euler angle magnitude
    ax = axes[1, 1]
    euler_mag = np.linalg.norm(euler_np, axis=1)
    sc = ax.scatter(np.degrees(euler_mag), angle_errors, s=3, alpha=0.5, c=final_physics, cmap="viridis")
    ax.set_xlabel("Goal Euler Angle Magnitude (deg)")
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title("Error vs Goal Magnitude")
    plt.colorbar(sc, ax=ax, label="Physics Loss")

    # 6) Per-goal loss improvement
    ax = axes[1, 2]
    improvement = pre_total - final_total
    ax.hist(improvement, bins=50, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(improvement), color="red", linestyle="--", label=f"median={np.median(improvement):.4f}")
    ax.set_xlabel("Loss Improvement (pre - post)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Goal Loss Improvement")
    ax.legend()

    fig.suptitle(f"CVAE + Adam Optimization: {N_GOALS} Random Goals, {NUM_ITERS} iters", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "loss_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {save_path}")

    # Save raw data
    np.savez(
        os.path.join(save_dir, "eval_results.npz"),
        euler_goals=euler_np,
        pre_total_losses=pre_total,
        pre_physics_losses=pre_physics,
        final_total_losses=final_total,
        final_physics_losses=final_physics,
        angle_errors=angle_errors,
        loss_history=np.array(loss_history),
    )
    print(f"Saved results to {save_dir}/eval_results.npz")


if __name__ == "__main__":
    main()
