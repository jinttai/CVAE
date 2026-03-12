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

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.models.cvae import CVAE, MLP
from src.training.physics_layer import PhysicsLayer   # default
from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    Using ZYX convention (yaw around Z, pitch around Y, roll around X)
    """
    # Half angles
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    
    # Quaternion components
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    return torch.stack([qx, qy, qz, qw], dim=-1)


def generate_random_quaternion_from_euler(batch_size, max_angle_deg=30.0, device='cpu'):
    """
    Generate random quaternions from Euler angles within specified range
    Args:
        batch_size: Number of quaternions to generate
        max_angle_deg: Maximum angle in degrees for each Euler angle (default: 10 degrees)
        device: Device to create tensors on
    Returns:
        quaternions: [batch_size, 4] tensor of quaternions (x, y, z, w)
    """
    max_angle_rad = math.radians(max_angle_deg)
    
    # Generate random Euler angles in [-max_angle_deg, max_angle_deg]
    # Using torch.rand to generate uniform distribution in [0, 1], then scale to [-max, max]
    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    
    # Convert to quaternion
    quaternions = euler_to_quaternion(roll, pitch, yaw)
    
    return quaternions


# === Orientation & Trajectory Helpers ===
def quat_to_rot(q):
    """
    쿼터니언 q = [x, y, z, w] 를 회전행렬 R (3x3) 로 변환.
    """
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
    """
    회전 행렬 R (3x3)을 Euler angle (ZYX 순서, yaw-pitch-roll)로 변환.
    """
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
    """
    PhysicsLayer에서 사용하는 동역학과 동일하게 body orientation 궤적을 적분하여
    각 스텝의 Euler angle (yaw, pitch, roll)을 반환.
    Vectorized version: wb 계산은 vmap으로 병렬 처리 (PhysicsLayer와 동일)

    Args:
        physics: PhysicsLayer 인스턴스
        q_traj: [num_steps, n_q]
        q_dot_traj: [num_steps, n_q]
        q0_init: [4]
    Returns:
        euler_traj: [num_steps, 3] (rad)
    """
    device = physics.device
    num_steps = physics.num_steps

    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)

    R_curr = quat_to_rot(q0_init)

    # Vectorized: 모든 step의 wb를 한번에 계산 (PhysicsLayer.simulate_single과 동일)
    def compute_wb_single_step(qm, qd):
        """Single step의 wb 계산"""
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)

        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]  # Angular Velocity part
        return wb
    
    # vmap으로 모든 step을 병렬 처리
    batch_compute_wb = vmap(compute_wb_single_step, in_dims=(0, 0))
    wb_all = batch_compute_wb(q_traj, q_dot_traj)  # [num_steps, 3]
    
    # 모든 step의 R_delta를 한번에 계산
    batch_rot_from_omega = vmap(rot_from_omega, in_dims=(0, None))
    R_delta_all = batch_rot_from_omega(wb_all, physics.dt)  # [num_steps, 3, 3]
    
    # 순차적으로 R_curr 업데이트 및 euler 계산 (각 스텝마다 euler가 필요하므로)
    eulers = []
    for t in range(num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))

    euler_traj = torch.stack(eulers, dim=0)
    return euler_traj


def plot_trajectory(q_traj, q_dot_traj, euler_traj, title, save_path, total_time, target_euler=None):
    """
    joint trajectory는 PhysicsLayer.generate_trajectory의 5차 다항식(quintic) 결과를 그대로 사용하고,
    body orientation 궤적은 Euler angle 로 함께 plot.
    """
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    euler_traj = euler_traj.detach().cpu().numpy()  # [T, 3], rad

    # Optional target Euler angle (single 3-vector, rad)
    target_deg = None
    if target_euler is not None:
        target_deg = np.rad2deg(target_euler.detach().cpu().numpy())  # [3]

    num_steps = q_traj.shape[0]
    t = np.linspace(0.0, total_time, num_steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # 1) Joint Angles
    for i in range(q_traj.shape[1]):
        axes[0].plot(t, q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"{title} - Joint Angles (Half-cosine)")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", fontsize=8)

    # 2) Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(t, q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    # 3) Body Orientation (Euler, deg)
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


def run_single_optimization(physics, waypoints_init, q0_start, q0_goal, robot,
                             NUM_WAYPOINTS, MAX_JOINT_WEIGHT, label, verbose=False):
    """
    Run LBFGS optimization from a given initial waypoint guess.
    Returns dict with loss_history, final_loss, angle_error, time, waypoints.
    """
    device = waypoints_init.device
    waypoints_param = waypoints_init.clone().detach().requires_grad_(True)

    optimizer = optim.LBFGS(
        [waypoints_param],
        lr=1.0,
        max_iter=50,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_history = []
    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        physics_loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        waypoints_reshaped = waypoints_param.view(1, NUM_WAYPOINTS, robot["n_q"])
        max_joint_angle = waypoints_reshaped.abs().view(1, -1).max(dim=1)[0]
        max_joint_penalty = max_joint_angle.mean()
        loss = physics_loss + MAX_JOINT_WEIGHT * max_joint_penalty
        loss.backward()
        loss_history.append(loss.item())
        iteration_count[0] += 1
        if verbose and (iteration_count[0] <= 10 or iteration_count[0] % 10 == 0):
            print(f"  [{label}] Iter {iteration_count[0]:3d}  Loss: {loss.item():.6f}")
        return loss

    t0 = time.time()
    optimizer.step(closure)
    elapsed = time.time() - t0

    # Final metrics
    with torch.no_grad():
        physics_loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal).item()
        waypoints_reshaped = waypoints_param.view(1, NUM_WAYPOINTS, robot["n_q"])
        max_joint_angle = waypoints_reshaped.abs().view(1, -1).max(dim=1)[0].item()
        total_loss = physics_loss + max_joint_angle * MAX_JOINT_WEIGHT

        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        sim_out = physics.simulate_single(q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])
        q_final = sim_out[1]
        dot = torch.sum(q_final * q0_goal[0]).abs().clamp(-1.0, 1.0)
        angle_deg = (2.0 * torch.acos(dot) * 180.0 / math.pi).item()

    return {
        "label": label,
        "loss_history": loss_history,
        "final_loss": total_loss,
        "physics_loss": physics_loss,
        "angle_error_deg": angle_deg,
        "time": elapsed,
        "iterations": len(loss_history),
        "waypoints": waypoints_param.detach(),
    }


def plot_comparison(results, save_path, target_info=""):
    """
    Plot loss convergence curves and a summary bar chart for all initial guesses.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Loss convergence curves
    ax = axes[0]
    for r in results:
        ax.plot(r["loss_history"], label=r["label"], linewidth=1.5)
    ax.set_xlabel("LBFGS Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Loss Convergence")
    ax.legend(fontsize=7)
    ax.grid(True)

    # 2) Final loss bar chart
    ax = axes[1]
    labels = [r["label"] for r in results]
    final_losses = [r["final_loss"] for r in results]
    colors = ["tab:blue" if "Zero" in l else "tab:orange" for l in labels]
    ax.bar(range(len(labels)), final_losses, color=colors, tick_label=labels)
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss by Init")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y")

    # 3) Angle error bar chart
    ax = axes[2]
    angle_errors = [r["angle_error_deg"] for r in results]
    ax.bar(range(len(labels)), angle_errors, color=colors, tick_label=labels)
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title("Final Angle Error by Init")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y")

    fig.suptitle(f"Initial Guess Comparison  {target_info}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Initial Guess Comparison (LBFGS) on {device} ===")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    MAX_JOINT_WEIGHT = 0.01
    NUM_RANDOM_SEEDS = 5  # number of random initial guesses to try

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/opt_init_comparison")
    os.makedirs(save_dir, exist_ok=True)

    # Target orientation (fixed for fair comparison)
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    roll_deg, pitch_deg, yaw_deg = 20, 25, 15
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    q0_goal = euler_to_quaternion(
        torch.tensor([roll_rad], device=device),
        torch.tensor([pitch_rad], device=device),
        torch.tensor([yaw_rad], device=device),
    )
    print(f"Target orientation: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°")

    # ---- Build list of initial guesses ----
    init_guesses = []

    # Zero init
    init_guesses.append(("Zero", torch.zeros(1, OUTPUT_DIM, device=device)))

    # Random inits with different seeds
    for seed in range(NUM_RANDOM_SEEDS):
        torch.manual_seed(seed)
        init_guesses.append((f"Rand(seed={seed})", torch.randn(1, OUTPUT_DIM, device=device)))

    # ---- Run optimization for each init ----
    results = []
    for label, init in init_guesses:
        print(f"\n--- {label} ---")
        r = run_single_optimization(
            physics, init, q0_start, q0_goal, robot,
            NUM_WAYPOINTS, MAX_JOINT_WEIGHT, label, verbose=True,
        )
        print(f"  Final Loss: {r['final_loss']:.6f}  |  Angle Error: {r['angle_error_deg']:.4f}°  |  Iters: {r['iterations']}  |  Time: {r['time']:.2f}s")
        results.append(r)

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print(f"{'Init':<18} {'Final Loss':>12} {'Physics Loss':>14} {'Angle Err(°)':>14} {'Iters':>6} {'Time(s)':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<18} {r['final_loss']:>12.6f} {r['physics_loss']:>14.6f} {r['angle_error_deg']:>14.4f} {r['iterations']:>6d} {r['time']:>8.2f}")
    print("=" * 80)

    # ---- Comparison plot ----
    target_info = f"(target: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°)"
    plot_comparison(results, os.path.join(save_dir, "init_comparison.png"), target_info)

    # ---- Compare converged waypoints ----
    # Only compare runs that actually converged (exclude zero if it got stuck)
    converged = [r for r in results if r["angle_error_deg"] < 1.0]
    n_q = robot["n_q"]

    if len(converged) >= 2:
        print("\n" + "=" * 80)
        print("CONVERGED WAYPOINTS COMPARISON")
        print("=" * 80)

        # Print waypoints reshaped as (NUM_WAYPOINTS x n_q) for each run
        for r in converged:
            wp = r["waypoints"].cpu().numpy().reshape(NUM_WAYPOINTS, n_q)
            print(f"\n{r['label']}  (loss={r['final_loss']:.4f}, angle_err={r['angle_error_deg']:.4f}°)")
            for w in range(NUM_WAYPOINTS):
                joint_str = "  ".join([f"{v:+7.3f}" for v in wp[w]])
                print(f"  WP{w+1}: [{joint_str}]")

        # Pairwise L2 distance between converged waypoints
        wp_tensors = torch.stack([r["waypoints"].squeeze(0) for r in converged])  # [N, OUTPUT_DIM]
        N = wp_tensors.shape[0]
        print(f"\nPairwise L2 distance between converged waypoints:")
        labels_c = [r["label"] for r in converged]
        header = f"{'':>18}" + "".join([f"{l:>18}" for l in labels_c])
        print(header)
        for i in range(N):
            row = f"{labels_c[i]:>18}"
            for j in range(N):
                dist = torch.norm(wp_tensors[i] - wp_tensors[j]).item()
                row += f"{dist:>18.4f}"
            print(row)

        # Plot: overlay all converged joint trajectories
        fig, axes = plt.subplots(NUM_WAYPOINTS, 1, figsize=(10, 3 * NUM_WAYPOINTS))
        if NUM_WAYPOINTS == 1:
            axes = [axes]
        for w in range(NUM_WAYPOINTS):
            ax = axes[w]
            for r in converged:
                wp = r["waypoints"].cpu().numpy().reshape(NUM_WAYPOINTS, n_q)
                ax.bar(
                    np.arange(n_q) + converged.index(r) * 0.15,
                    wp[w],
                    width=0.15,
                    label=r["label"],
                )
            ax.set_title(f"Waypoint {w+1} joint values")
            ax.set_xlabel("Joint index")
            ax.set_ylabel("Angle (rad)")
            ax.set_xticks(np.arange(n_q))
            ax.set_xticklabels([f"J{i+1}" for i in range(n_q)])
            ax.legend(fontsize=7)
            ax.grid(True, axis="y")
        plt.tight_layout()
        wp_plot_path = os.path.join(save_dir, "waypoints_comparison.png")
        plt.savefig(wp_plot_path, dpi=150)
        plt.close()
        print(f"\nSaved waypoints comparison plot to {wp_plot_path}")

        # Plot: overlay all converged joint angle trajectories
        fig, axes = plt.subplots(n_q, 1, figsize=(10, 2.5 * n_q), sharex=True)
        if n_q == 1:
            axes = [axes]
        with torch.no_grad():
            for r in converged:
                q_traj, q_dot_traj = physics.generate_trajectory(r["waypoints"])
                q_np = q_traj[0].cpu().numpy()  # [T, n_q]
                t_arr = np.linspace(0.0, TOTAL_TIME, q_np.shape[0])
                for j in range(n_q):
                    axes[j].plot(t_arr, q_np[:, j], label=r["label"], linewidth=1.2)
            for j in range(n_q):
                axes[j].set_ylabel(f"J{j+1} (rad)")
                axes[j].grid(True)
                if j == 0:
                    axes[j].legend(fontsize=7, loc="upper right")
            axes[-1].set_xlabel("Time [s]")
        fig.suptitle("Joint Trajectories - All Converged Runs", fontsize=12)
        plt.tight_layout()
        traj_plot_path = os.path.join(save_dir, "trajectories_comparison.png")
        plt.savefig(traj_plot_path, dpi=150)
        plt.close()
        print(f"Saved trajectories comparison plot to {traj_plot_path}")

    # ---- Per-run trajectory plots for best and worst ----
    best = min(results, key=lambda r: r["angle_error_deg"])
    worst = max(results, key=lambda r: r["angle_error_deg"])

    R_goal = quat_to_rot(q0_goal[0])
    target_euler = rot_to_euler(R_goal)

    for tag, r in [("best", best), ("worst", worst)]:
        with torch.no_grad():
            q_traj, q_dot_traj = physics.generate_trajectory(r["waypoints"].unsqueeze(0) if r["waypoints"].dim() == 1 else r["waypoints"])
            euler_traj = compute_orientation_traj(physics, q_traj[0], q_dot_traj[0], q0_start[0])
            plot_trajectory(
                q_traj[0], q_dot_traj[0], euler_traj,
                f"{tag.upper()} ({r['label']}, Err: {r['angle_error_deg']:.4f}°)",
                os.path.join(save_dir, f"traj_{tag}_{r['label'].replace('=','').replace('(','').replace(')','')}.png"),
                TOTAL_TIME,
                target_euler=target_euler,
            )

    print(f"\nBest  init: {best['label']}  (angle error: {best['angle_error_deg']:.4f}°)")
    print(f"Worst init: {worst['label']}  (angle error: {worst['angle_error_deg']:.4f}°)")


if __name__ == "__main__":
    main()


