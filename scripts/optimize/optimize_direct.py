import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import math
from torch.func import vmap

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.models.cvae import CVAE, MLP
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart


def euler_to_quaternion(roll, pitch, yaw):
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2); sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2); sy = torch.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return torch.stack([qx, qy, qz, qw], dim=-1)


def quat_to_rot(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)]),
        torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)]),
        torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)])
    ])
    return R


def skew(v):
    vx, vy, vz = v
    zero = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])


def rot_from_omega(wb, dt):
    device, dtype = wb.device, wb.dtype
    theta = torch.linalg.norm(wb) * dt
    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew(axis)
    I = torch.eye(3, device=device, dtype=dtype)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def rot_to_euler(R):
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        yaw = torch.atan2(R[1, 0], R[0, 0])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.atan2(R[2, 1], R[2, 2])
    else:
        yaw = torch.atan2(-R[0, 1], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.zeros_like(yaw)
    return torch.stack([yaw, pitch, roll])


def compute_orientation_traj(physics, q_traj, q_dot_traj, q0_init):
    device = physics.device
    R_curr = quat_to_rot(q0_init)

    def compute_wb_single_step(qm, qd):
        R0 = torch.eye(3, device=device)
        r0 = torch.zeros(3, device=device)
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device)
        return torch.linalg.solve(H0_damped, rhs)[:3]

    wb_all = vmap(compute_wb_single_step, in_dims=(0, 0))(q_traj, q_dot_traj)
    R_delta_all = vmap(rot_from_omega, in_dims=(0, None))(wb_all, physics.dt)

    eulers = []
    for t in range(physics.num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))
    return torch.stack(eulers, dim=0)


# =====================================================================
# Multi-Seed Optimization with Initial Guess Analysis
# =====================================================================
def evaluate_batch(physics, waypoints, q0_start_batch, q0_goal_batch, num_waypoints, n_q, max_joint_weight):
    """Evaluate per-seed loss, angle error, max joint for a batch of waypoints."""
    num_seeds = waypoints.shape[0]
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    loss_batch, q_final_batch = physics._batch_sim_fn(
        q_traj, q_dot_traj, q0_start_batch, q0_goal_batch
    )
    dots = torch.sum(q_final_batch * q0_goal_batch, dim=-1).abs().clamp(-1.0, 1.0)
    angle_errors = 2.0 * torch.acos(dots) * 180.0 / math.pi

    wp = waypoints.view(num_seeds, num_waypoints, n_q)
    max_joints = wp.abs().view(num_seeds, -1).max(dim=1)[0]
    total_losses = loss_batch + max_joint_weight * max_joints

    return {
        "physics_loss": loss_batch,
        "total_loss": total_losses,
        "angle_error_deg": angle_errors,
        "max_joint_rad": max_joints,
    }


def run_multiseed_optimization(physics, num_seeds, output_dim, q0_start, q0_goal,
                                robot, num_waypoints, max_joint_weight,
                                init_scale=0.5, max_lbfgs_iter=50, verbose=True):
    device = q0_start.device
    n_q = robot["n_q"]

    q0_start_batch = q0_start.expand(num_seeds, -1)
    q0_goal_batch = q0_goal.expand(num_seeds, -1)

    torch.manual_seed(42)
    waypoints_param = (torch.randn(num_seeds, output_dim, device=device) * init_scale)

    # ======== Record initial state BEFORE optimization ========
    with torch.no_grad():
        init_waypoints = waypoints_param.clone()
        init_eval = evaluate_batch(
            physics, init_waypoints, q0_start_batch, q0_goal_batch,
            num_waypoints, n_q, max_joint_weight,
        )
        init_info = {
            "waypoints": init_waypoints.cpu(),
            "physics_loss": init_eval["physics_loss"].cpu(),
            "total_loss": init_eval["total_loss"].cpu(),
            "angle_error_deg": init_eval["angle_error_deg"].cpu(),
            "max_joint_rad": init_eval["max_joint_rad"].cpu(),
            # Per-seed statistics
            "l2_norm": init_waypoints.norm(dim=1).cpu(),
            "mean_abs": init_waypoints.abs().mean(dim=1).cpu(),
            "std": init_waypoints.std(dim=1).cpu(),
            # Per-waypoint-per-joint breakdown
            "per_wp": init_waypoints.view(num_seeds, num_waypoints, n_q).cpu(),
        }

    # ======== Optimization ========
    waypoints_param = waypoints_param.requires_grad_(True)
    optimizer = optim.LBFGS(
        [waypoints_param], lr=1.0, max_iter=max_lbfgs_iter,
        history_size=100, line_search_fn="strong_wolfe",
    )

    loss_history = []
    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        loss_batch, _ = physics._batch_sim_fn(q_traj, q_dot_traj, q0_start_batch, q0_goal_batch)
        wp = waypoints_param.view(num_seeds, num_waypoints, n_q)
        max_joint_per_seed = wp.abs().view(num_seeds, -1).max(dim=1)[0]
        total_per_seed = loss_batch + max_joint_weight * max_joint_per_seed
        loss = total_per_seed.mean()
        loss.backward()
        loss_history.append(loss.item())
        iteration_count[0] += 1
        if verbose and (iteration_count[0] <= 5 or iteration_count[0] % 10 == 0):
            print(f"  Iter {iteration_count[0]:3d}  Mean: {loss.item():.6f}  "
                  f"Best: {total_per_seed.min().item():.6f}  Worst: {total_per_seed.max().item():.6f}")
        return loss

    t0 = time.time()
    optimizer.step(closure)
    elapsed = time.time() - t0

    # ======== Record final state AFTER optimization ========
    with torch.no_grad():
        final_eval = evaluate_batch(
            physics, waypoints_param, q0_start_batch, q0_goal_batch,
            num_waypoints, n_q, max_joint_weight,
        )
        final_info = {
            "waypoints": waypoints_param.detach().cpu(),
            "physics_loss": final_eval["physics_loss"].cpu(),
            "total_loss": final_eval["total_loss"].cpu(),
            "angle_error_deg": final_eval["angle_error_deg"].cpu(),
            "max_joint_rad": final_eval["max_joint_rad"].cpu(),
        }

    return {
        "init": init_info,
        "final": final_info,
        "loss_history": loss_history,
        "iterations": len(loss_history),
        "time": elapsed,
        "num_seeds": num_seeds,
    }


# =====================================================================
# Analysis & Plots
# =====================================================================
def analyze_initial_guesses(result, save_dir, threshold_deg=1.0):
    """수렴/미수렴 그룹의 초기값 특성을 비교 분석 + 시각화."""
    init = result["init"]
    final = result["final"]
    num_seeds = result["num_seeds"]

    final_errors = final["angle_error_deg"].numpy()
    converged_mask = final_errors < threshold_deg
    n_conv = converged_mask.sum()
    n_fail = num_seeds - n_conv

    print(f"\n{'='*70}")
    print(f"INITIAL GUESS ANALYSIS: Converged vs Failed")
    print(f"{'='*70}")
    print(f"Converged: {n_conv}/{num_seeds} ({n_conv/num_seeds*100:.1f}%)")
    print(f"Failed   : {n_fail}/{num_seeds} ({n_fail/num_seeds*100:.1f}%)")

    # ---- 1) Initial value statistics comparison ----
    metrics = {
        "Init Physics Loss": init["physics_loss"].numpy(),
        "Init Total Loss":   init["total_loss"].numpy(),
        "Init Angle Err(deg)": init["angle_error_deg"].numpy(),
        "Init L2 Norm":      init["l2_norm"].numpy(),
        "Init Mean |w|":     init["mean_abs"].numpy(),
        "Init Std(w)":       init["std"].numpy(),
        "Init Max Joint":    init["max_joint_rad"].numpy(),
    }

    print(f"\n{'Metric':<22} {'Converged (mean±std)':>24} {'Failed (mean±std)':>24} {'p-value':>10}")
    print(f"{'-'*82}")

    from scipy import stats as sp_stats
    p_values = {}
    for name, vals in metrics.items():
        conv_vals = vals[converged_mask]
        fail_vals = vals[~converged_mask]
        if len(conv_vals) > 1 and len(fail_vals) > 1:
            _, pval = sp_stats.mannwhitneyu(conv_vals, fail_vals, alternative='two-sided')
        else:
            pval = float('nan')
        p_values[name] = pval
        conv_str = f"{conv_vals.mean():.4f} ± {conv_vals.std():.4f}" if len(conv_vals) > 0 else "N/A"
        fail_str = f"{fail_vals.mean():.4f} ± {fail_vals.std():.4f}" if len(fail_vals) > 0 else "N/A"
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"{name:<22} {conv_str:>24} {fail_str:>24} {pval:>8.4f} {sig}")

    # ---- 2) Per-joint initial value comparison ----
    per_wp = init["per_wp"].numpy()  # [N, num_wp, n_q]
    num_waypoints = per_wp.shape[1]
    n_q = per_wp.shape[2]

    print(f"\n{'='*70}")
    print(f"PER-JOINT INITIAL VALUE (mean)")
    print(f"{'='*70}")
    for w in range(num_waypoints):
        print(f"\n  Waypoint {w+1}:")
        for j in range(n_q):
            c_mean = per_wp[converged_mask, w, j].mean()
            f_mean = per_wp[~converged_mask, w, j].mean()
            c_std = per_wp[converged_mask, w, j].std()
            f_std = per_wp[~converged_mask, w, j].std()
            print(f"    J{j+1}: Conv {c_mean:+.4f}±{c_std:.4f}  |  Fail {f_mean:+.4f}±{f_std:.4f}")

    # ---- 3) Plots ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Scatter: Init physics loss vs Final angle error
    ax = axes[0, 0]
    ax.scatter(init["physics_loss"].numpy()[converged_mask],
               final_errors[converged_mask],
               c='tab:green', alpha=0.7, label=f'Converged ({n_conv})', s=20)
    ax.scatter(init["physics_loss"].numpy()[~converged_mask],
               final_errors[~converged_mask],
               c='tab:red', alpha=0.5, label=f'Failed ({n_fail})', s=20)
    ax.axhline(threshold_deg, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Initial Physics Loss")
    ax.set_ylabel("Final Angle Error (deg)")
    ax.set_title("Init Physics Loss vs Final Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Scatter: Init L2 norm vs Final angle error
    ax = axes[0, 1]
    ax.scatter(init["l2_norm"].numpy()[converged_mask],
               final_errors[converged_mask],
               c='tab:green', alpha=0.7, label='Converged', s=20)
    ax.scatter(init["l2_norm"].numpy()[~converged_mask],
               final_errors[~converged_mask],
               c='tab:red', alpha=0.5, label='Failed', s=20)
    ax.axhline(threshold_deg, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Initial Waypoints L2 Norm")
    ax.set_ylabel("Final Angle Error (deg)")
    ax.set_title("Init L2 Norm vs Final Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) Scatter: Init angle error vs Final angle error
    ax = axes[0, 2]
    ax.scatter(init["angle_error_deg"].numpy()[converged_mask],
               final_errors[converged_mask],
               c='tab:green', alpha=0.7, label='Converged', s=20)
    ax.scatter(init["angle_error_deg"].numpy()[~converged_mask],
               final_errors[~converged_mask],
               c='tab:red', alpha=0.5, label='Failed', s=20)
    ax.axhline(threshold_deg, color='k', linestyle='--', linewidth=0.8)
    ax.plot([0, 180], [0, 180], 'k:', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Initial Angle Error (deg)")
    ax.set_ylabel("Final Angle Error (deg)")
    ax.set_title("Init Angle Error vs Final Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Histogram: Init physics loss by group
    ax = axes[1, 0]
    bins = 30
    ax.hist(init["physics_loss"].numpy()[converged_mask], bins=bins, alpha=0.7,
            color='tab:green', label='Converged', density=True)
    ax.hist(init["physics_loss"].numpy()[~converged_mask], bins=bins, alpha=0.5,
            color='tab:red', label='Failed', density=True)
    ax.set_xlabel("Initial Physics Loss")
    ax.set_ylabel("Density")
    ax.set_title("Init Physics Loss Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Histogram: Init L2 norm by group
    ax = axes[1, 1]
    ax.hist(init["l2_norm"].numpy()[converged_mask], bins=bins, alpha=0.7,
            color='tab:green', label='Converged', density=True)
    ax.hist(init["l2_norm"].numpy()[~converged_mask], bins=bins, alpha=0.5,
            color='tab:red', label='Failed', density=True)
    ax.set_xlabel("Initial Waypoints L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("Init L2 Norm Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Box plot: per-joint initial values by group
    ax = axes[1, 2]
    conv_flat = per_wp[converged_mask].reshape(-1, n_q)   # [N_conv * num_wp, n_q]
    fail_flat = per_wp[~converged_mask].reshape(-1, n_q)  # [N_fail * num_wp, n_q]
    positions_c = np.arange(n_q) * 3
    positions_f = np.arange(n_q) * 3 + 1
    if conv_flat.shape[0] > 0:
        bp1 = ax.boxplot([conv_flat[:, j] for j in range(n_q)], positions=positions_c,
                         widths=0.8, patch_artist=True, showfliers=False)
        for patch in bp1['boxes']:
            patch.set_facecolor('tab:green')
            patch.set_alpha(0.6)
    if fail_flat.shape[0] > 0:
        bp2 = ax.boxplot([fail_flat[:, j] for j in range(n_q)], positions=positions_f,
                         widths=0.8, patch_artist=True, showfliers=False)
        for patch in bp2['boxes']:
            patch.set_facecolor('tab:red')
            patch.set_alpha(0.6)
    ax.set_xticks(np.arange(n_q) * 3 + 0.5)
    ax.set_xticklabels([f"J{j+1}" for j in range(n_q)])
    ax.set_ylabel("Initial Joint Value (rad)")
    ax.set_title("Per-Joint Init Values (green=conv, red=fail)")
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f"Initial Guess Analysis: {n_conv} converged / {n_fail} failed "
                 f"(threshold={threshold_deg}°)", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "initial_guess_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved analysis plot to {plot_path}")

    # ---- 4) Loss improvement analysis ----
    init_loss = init["total_loss"].numpy()
    final_loss = final["total_loss"].numpy()
    improvement = init_loss - final_loss

    print(f"\n{'='*70}")
    print(f"LOSS IMPROVEMENT")
    print(f"{'='*70}")
    print(f"{'Group':<12} {'Init Loss (mean)':>16} {'Final Loss (mean)':>18} {'Improvement':>14}")
    print(f"{'-'*62}")
    print(f"{'Converged':<12} {init_loss[converged_mask].mean():>16.4f} "
          f"{final_loss[converged_mask].mean():>18.4f} "
          f"{improvement[converged_mask].mean():>14.4f}")
    print(f"{'Failed':<12} {init_loss[~converged_mask].mean():>16.4f} "
          f"{final_loss[~converged_mask].mean():>18.4f} "
          f"{improvement[~converged_mask].mean():>14.4f}")

    return converged_mask


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Multi-Seed Optimization + Initial Guess Analysis on {device} ===\n")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    MAX_JOINT_WEIGHT = 0.01
    NUM_SEEDS = 200

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/opt_multiseed")
    os.makedirs(save_dir, exist_ok=True)

    # Target orientation
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    roll_deg, pitch_deg, yaw_deg = 15, 15, -15
    q0_goal = euler_to_quaternion(
        torch.tensor([math.radians(roll_deg)], device=device),
        torch.tensor([math.radians(pitch_deg)], device=device),
        torch.tensor([math.radians(yaw_deg)], device=device),
    )
    print(f"Target: roll={roll_deg}, pitch={pitch_deg}, yaw={yaw_deg}")
    print(f"Seeds : {NUM_SEEDS}  |  init_scale=0.5\n")

    # ---- Optimization ----
    result = run_multiseed_optimization(
        physics=physics,
        num_seeds=NUM_SEEDS,
        output_dim=OUTPUT_DIM,
        q0_start=q0_start,
        q0_goal=q0_goal,
        robot=robot,
        num_waypoints=NUM_WAYPOINTS,
        max_joint_weight=MAX_JOINT_WEIGHT,
        init_scale=0.5,
        max_lbfgs_iter=50,
        verbose=True,
    )

    print(f"\nOptimization: {result['iterations']} iters in {result['time']:.1f}s")

    # ---- Initial Guess Analysis ----
    converged_mask = analyze_initial_guesses(result, save_dir, threshold_deg=1.0)

    # ---- Summary Table (top 10 best + bottom 10 worst) ----
    final = result["final"]
    init = result["init"]
    errors = final["angle_error_deg"].numpy()
    sorted_idx = np.argsort(errors)

    print(f"\n{'='*100}")
    print(f"TOP 10 BEST + BOTTOM 10 WORST")
    print(f"{'='*100}")
    print(f"{'Seed':>6} {'Init Loss':>11} {'Init Err':>10} {'Init L2':>9} "
          f"{'Final Loss':>12} {'Final Err':>10} {'Status':>10}")
    print(f"{'-'*100}")

    show_idx = list(sorted_idx[:10]) + list(sorted_idx[-10:])
    for i in show_idx:
        status = "CONV" if converged_mask[i] else "FAIL"
        print(f"{i:>6d} {init['total_loss'][i].item():>11.4f} "
              f"{init['angle_error_deg'][i].item():>10.4f} "
              f"{init['l2_norm'][i].item():>9.4f} "
              f"{final['total_loss'][i].item():>12.4f} "
              f"{errors[i]:>10.4f} "
              f"{status:>10}")


if __name__ == "__main__":
    main()
