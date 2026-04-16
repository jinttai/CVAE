import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import math
from torch.func import vmap

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.models.cvae import CVAE, MLP
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w). ZYX convention."""
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


# === Orientation & Trajectory Helpers ===
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
    M = torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])
    return M


def rot_from_omega(wb, dt):
    device = wb.device
    dtype = wb.dtype
    theta = torch.linalg.norm(wb) * dt
    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew(axis)
    I = torch.eye(3, device=device, dtype=dtype)
    R_delta = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R_delta


def rot_to_euler(R):
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
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
    num_steps = physics.num_steps
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
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        return u0_sol[:3]

    batch_compute_wb = vmap(compute_wb_single_step, in_dims=(0, 0))
    wb_all = batch_compute_wb(q_traj, q_dot_traj)
    batch_rot_from_omega = vmap(rot_from_omega, in_dims=(0, None))
    R_delta_all = batch_rot_from_omega(wb_all, physics.dt)

    eulers = []
    for t in range(num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))

    return torch.stack(eulers, dim=0)


def plot_trajectory(q_traj, q_dot_traj, euler_traj, title, save_path, total_time, target_euler=None):
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    euler_traj = euler_traj.detach().cpu().numpy()

    target_deg = None
    if target_euler is not None:
        target_deg = np.rad2deg(target_euler.detach().cpu().numpy())

    num_steps = q_traj.shape[0]
    t = np.linspace(0.0, total_time, num_steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    for i in range(q_traj.shape[1]):
        axes[0].plot(t, q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"{title} - Joint Angles")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", fontsize=8)

    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(t, q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    euler_deg = np.rad2deg(euler_traj)
    labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    for i in range(3):
        axes[2].plot(t, euler_deg[:, i], label=labels[i])
        if target_deg is not None:
            axes[2].axhline(target_deg[i], linestyle="--", linewidth=1.5, label=f"Target {labels[i]}")
    axes[2].set_title("Body Orientation (Euler)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Angle [deg]")
    axes[2].grid(True)
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


# =====================================================================
# Multi-Seed Batch Optimization
# =====================================================================
def run_multiseed_optimization(physics, num_seeds, output_dim, q0_start, q0_goal,
                                robot, num_waypoints, max_joint_weight,
                                init_scale=0.1, max_lbfgs_iter=50, verbose=True):
    """
    Run batched LBFGS optimization over multiple random seeds simultaneously.

    All seeds share a single LBFGS optimizer. Since per-seed physics losses are
    independent, optimizing the mean loss is equivalent to optimizing each seed
    individually (up to a constant gradient scale that the line search absorbs).

    Args:
        physics: PhysicsLayer instance
        num_seeds: number of random initial guesses
        output_dim: NUM_WAYPOINTS * n_q
        q0_start: [1, 4] initial quaternion (will be expanded)
        q0_goal: [1, 4] goal quaternion (will be expanded)
        robot: robot dict
        num_waypoints: number of intermediate waypoints
        max_joint_weight: penalty weight for max joint angle
        init_scale: std of random initialization
        max_lbfgs_iter: max inner iterations for LBFGS
        verbose: print progress

    Returns:
        dict with best_idx, all per-seed results, timing, loss history
    """
    device = q0_start.device
    n_q = robot["n_q"]

    # Expand targets to batch dimension
    q0_start_batch = q0_start.expand(num_seeds, -1)  # [N, 4]
    q0_goal_batch = q0_goal.expand(num_seeds, -1)     # [N, 4]

    # Initialize all seeds: [num_seeds, output_dim]
    torch.manual_seed(42)
    waypoints_param = (torch.randn(num_seeds, output_dim, device=device) * init_scale)
    waypoints_param = waypoints_param.requires_grad_(True)

    optimizer = optim.LBFGS(
        [waypoints_param],
        lr=1.0,
        max_iter=max_lbfgs_iter,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_history = []
    per_seed_loss_history = []
    iteration_count = [0]

    def closure():
        optimizer.zero_grad()

        # Physics loss (batched) — returns mean over seeds
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        loss_batch, _ = physics._batch_sim_fn(q_traj, q_dot_traj, q0_start_batch, q0_goal_batch)

        # Max joint penalty per seed, then mean
        wp = waypoints_param.view(num_seeds, num_waypoints, n_q)
        max_joint_per_seed = wp.abs().view(num_seeds, -1).max(dim=1)[0]

        total_per_seed = loss_batch + max_joint_weight * max_joint_per_seed
        loss = total_per_seed.mean()

        loss.backward()

        loss_history.append(loss.item())
        per_seed_loss_history.append(total_per_seed.detach().cpu().tolist())
        iteration_count[0] += 1

        if verbose and (iteration_count[0] <= 10 or iteration_count[0] % 10 == 0):
            best_loss = total_per_seed.min().item()
            worst_loss = total_per_seed.max().item()
            print(f"  Iter {iteration_count[0]:3d}  "
                  f"Mean: {loss.item():.6f}  Best: {best_loss:.6f}  Worst: {worst_loss:.6f}")

        return loss

    t0 = time.time()
    optimizer.step(closure)
    elapsed = time.time() - t0

    # === Evaluate all seeds ===
    with torch.no_grad():
        q_traj_all, q_dot_traj_all = physics.generate_trajectory(waypoints_param)
        loss_batch, q_final_batch = physics._batch_sim_fn(
            q_traj_all, q_dot_traj_all, q0_start_batch, q0_goal_batch
        )

        # Angle error per seed
        dots = torch.sum(q_final_batch * q0_goal_batch, dim=-1).abs().clamp(-1.0, 1.0)
        angle_errors = 2.0 * torch.acos(dots) * 180.0 / math.pi  # [num_seeds]

        # Max joint per seed
        wp = waypoints_param.view(num_seeds, num_waypoints, n_q)
        max_joints = wp.abs().view(num_seeds, -1).max(dim=1)[0]

        total_losses = loss_batch + max_joint_weight * max_joints

        best_idx = angle_errors.argmin().item()

    # Build per-seed result list
    seed_results = []
    for i in range(num_seeds):
        seed_results.append({
            "seed": i,
            "physics_loss": loss_batch[i].item(),
            "total_loss": total_losses[i].item(),
            "angle_error_deg": angle_errors[i].item(),
            "max_joint_rad": max_joints[i].item(),
            "waypoints": waypoints_param[i].detach(),
        })

    return {
        "best_idx": best_idx,
        "seed_results": seed_results,
        "loss_history": loss_history,
        "per_seed_loss_history": per_seed_loss_history,
        "iterations": len(loss_history),
        "time": elapsed,
        "waypoints_all": waypoints_param.detach(),
    }


def plot_multiseed_summary(result, save_path, target_info=""):
    """Plot convergence and per-seed comparison."""
    seed_results = result["seed_results"]
    best_idx = result["best_idx"]
    num_seeds = len(seed_results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Mean loss convergence
    ax = axes[0]
    ax.plot(result["loss_history"], "k-", linewidth=1.5, label="Mean loss")
    # Per-seed loss curves
    per_seed = np.array(result["per_seed_loss_history"])  # [iters, num_seeds]
    for i in range(num_seeds):
        style = "-" if i == best_idx else "--"
        alpha = 1.0 if i == best_idx else 0.3
        label = f"Seed {i} (best)" if i == best_idx else (f"Seed {i}" if i < 5 else None)
        ax.plot(per_seed[:, i], style, alpha=alpha, linewidth=1.0, label=label)
    ax.set_xlabel("LBFGS Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Loss Convergence (all seeds)")
    ax.legend(fontsize=7)
    ax.grid(True)

    # 2) Angle error bar chart
    ax = axes[1]
    errors = [r["angle_error_deg"] for r in seed_results]
    colors = ["tab:green" if i == best_idx else "tab:blue" for i in range(num_seeds)]
    ax.bar(range(num_seeds), errors, color=colors)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title("Final Angle Error per Seed")
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1, label="1° threshold")
    ax.legend()
    ax.grid(True, axis="y")

    # 3) Total loss bar chart
    ax = axes[2]
    losses = [r["total_loss"] for r in seed_results]
    ax.bar(range(num_seeds), losses, color=colors)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Total Loss")
    ax.set_title("Final Total Loss per Seed")
    ax.grid(True, axis="y")

    fig.suptitle(f"Multi-Seed Optimization ({num_seeds} seeds)  {target_info}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved summary plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Multi-Seed Batch Optimization (LBFGS) on {device} ===")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    MAX_JOINT_WEIGHT = 0.01
    NUM_SEEDS = 16

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
    print(f"Target: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°")
    print(f"Seeds : {NUM_SEEDS}")

    # ---- Multi-Seed Optimization ----
    result = run_multiseed_optimization(
        physics=physics,
        num_seeds=NUM_SEEDS,
        output_dim=OUTPUT_DIM,
        q0_start=q0_start,
        q0_goal=q0_goal,
        robot=robot,
        num_waypoints=NUM_WAYPOINTS,
        max_joint_weight=MAX_JOINT_WEIGHT,
        init_scale=0.1,
        max_lbfgs_iter=50,
        verbose=True,
    )

    best_idx = result["best_idx"]
    best = result["seed_results"][best_idx]

    # ---- Summary Table ----
    print(f"\n{'='*80}")
    print(f"  Completed in {result['time']:.2f}s  |  {result['iterations']} LBFGS iterations")
    print(f"{'='*80}")
    print(f"{'Seed':>6} {'Total Loss':>12} {'Physics Loss':>14} {'Angle Err(°)':>14} {'Max Joint':>10}")
    print(f"{'-'*80}")
    for r in result["seed_results"]:
        marker = " <-- BEST" if r["seed"] == best_idx else ""
        print(f"{r['seed']:>6d} {r['total_loss']:>12.6f} {r['physics_loss']:>14.6f} "
              f"{r['angle_error_deg']:>14.4f} {r['max_joint_rad']:>10.4f}{marker}")
    print(f"{'='*80}")

    converged = [r for r in result["seed_results"] if r["angle_error_deg"] < 1.0]
    print(f"\nConverged (< 1°): {len(converged)} / {NUM_SEEDS}")
    print(f"Best seed: {best_idx}  |  Angle error: {best['angle_error_deg']:.4f}°  |  Loss: {best['total_loss']:.6f}")

    # ---- Summary Plot ----
    target_info = f"(target: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°)"
    plot_multiseed_summary(result, os.path.join(save_dir, "multiseed_summary.png"), target_info)

    # ---- Best Trajectory Plot ----
    R_goal = quat_to_rot(q0_goal[0])
    target_euler = rot_to_euler(R_goal)

    with torch.no_grad():
        best_wp = best["waypoints"].unsqueeze(0)  # [1, output_dim]
        q_traj, q_dot_traj = physics.generate_trajectory(best_wp)
        euler_traj = compute_orientation_traj(physics, q_traj[0], q_dot_traj[0], q0_start[0])
        plot_trajectory(
            q_traj[0], q_dot_traj[0], euler_traj,
            f"Best Seed {best_idx} (Err: {best['angle_error_deg']:.4f}°)",
            os.path.join(save_dir, f"traj_best_seed{best_idx}.png"),
            TOTAL_TIME,
            target_euler=target_euler,
        )

    # ---- Converged Waypoints Comparison ----
    n_q = robot["n_q"]
    if len(converged) >= 2:
        print(f"\n{'='*80}")
        print("CONVERGED WAYPOINTS COMPARISON")
        print(f"{'='*80}")

        for r in converged:
            wp = r["waypoints"].cpu().numpy().reshape(NUM_WAYPOINTS, n_q)
            print(f"\nSeed {r['seed']}  (loss={r['total_loss']:.4f}, err={r['angle_error_deg']:.4f}°)")
            for w in range(NUM_WAYPOINTS):
                joint_str = "  ".join([f"{v:+7.3f}" for v in wp[w]])
                print(f"  WP{w+1}: [{joint_str}]")

        wp_tensors = torch.stack([r["waypoints"] for r in converged])
        N = wp_tensors.shape[0]
        print(f"\nPairwise L2 distance between converged waypoints:")
        labels_c = [f"Seed {r['seed']}" for r in converged]
        header = f"{'':>12}" + "".join([f"{l:>12}" for l in labels_c])
        print(header)
        for i in range(N):
            row = f"{labels_c[i]:>12}"
            for j in range(N):
                dist = torch.norm(wp_tensors[i] - wp_tensors[j]).item()
                row += f"{dist:>12.4f}"
            print(row)

        # Overlay converged joint trajectories
        fig, axes = plt.subplots(n_q, 1, figsize=(10, 2.5 * n_q), sharex=True)
        if n_q == 1:
            axes = [axes]
        with torch.no_grad():
            for r in converged:
                wp = r["waypoints"].unsqueeze(0)
                q_traj, q_dot_traj = physics.generate_trajectory(wp)
                q_np = q_traj[0].cpu().numpy()
                t_arr = np.linspace(0.0, TOTAL_TIME, q_np.shape[0])
                for j in range(n_q):
                    lbl = f"Seed {r['seed']}" if j == 0 else None
                    axes[j].plot(t_arr, q_np[:, j], linewidth=1.2, label=lbl)
            for j in range(n_q):
                axes[j].set_ylabel(f"J{j+1} (rad)")
                axes[j].grid(True)
                if j == 0:
                    axes[j].legend(fontsize=7, loc="upper right")
            axes[-1].set_xlabel("Time [s]")
        fig.suptitle("Joint Trajectories - Converged Seeds", fontsize=12)
        plt.tight_layout()
        traj_path = os.path.join(save_dir, "trajectories_converged.png")
        plt.savefig(traj_path, dpi=150)
        plt.close()
        print(f"Saved trajectories plot to {traj_path}")


if __name__ == "__main__":
    main()
