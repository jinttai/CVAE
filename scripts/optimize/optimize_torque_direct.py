"""
Direct optimization: torque cost minimization with orientation constraint.

Augmented Lagrangian method:
  minimize   torque_cost(w)
  subject to orient_loss(w) ≤ ε

AL subproblem (inner LBFGS):
  L(w; λ, ρ) = torque_cost + (ρ/2) · max(0, c(w) + λ/ρ)²

Outer loop updates:
  λ ← max(0, λ + ρ · c(w*))
  ρ ← γ · ρ   (if constraint not sufficiently reduced)
"""

import os
import sys
import time
import math

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from src.utils.runtime_env import configure_windows_runtime

configure_windows_runtime()

import torch
import torch.optim as optim
import numpy as np
from torch.func import vmap

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from src.models.cvae import CVAE
import src.dynamics.spart_functions_torch as spart


# ── Config ──────────────────────────────────────────────────────────────
NUM_WAYPOINTS = 3
TOTAL_TIME = 10.0
N_SAMPLES = 128       # CVAE candidates for initial guess selection

# Augmented Lagrangian parameters
ORIENT_TOL_DEG = 0.5  # orientation feasibility threshold (degrees)
RHO_INIT = 10.0       # initial penalty parameter
RHO_MAX = 1e6         # maximum penalty
GAMMA = 5.0           # penalty growth factor
MAX_OUTER = 30        # max AL outer iterations
INNER_MAX_ITER = 30   # LBFGS iterations per inner solve

CVAE_WEIGHT_PATH = os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v5_joint_change.pth")


# ── Helpers ─────────────────────────────────────────────────────────────
def euler_to_quaternion(roll, pitch, yaw):
    """Batched Euler (rad) -> quaternion [x,y,z,w]."""
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2); sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2); sy = torch.sin(yaw / 2)
    return torch.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dim=-1)


def angle_error_deg(q_final, q_goal):
    """Non-differentiable angle error (uses abs)."""
    dot = (q_final * q_goal).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot) * 180.0 / math.pi


def angle_error_deg_smooth(q_final, q_goal):
    """Differentiable angle error (squared dot avoids abs gradient issue)."""
    dot = (q_final * q_goal).sum(dim=-1)
    dot_sq = dot * dot  # ∈ [0, 1], avoids non-differentiable abs
    dot_sq = dot_sq.clamp(0.0, 1.0 - 1e-7)
    return 2.0 * torch.acos(torch.sqrt(dot_sq)) * 180.0 / math.pi


def quat_to_rot(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)]),
        torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)]),
        torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)])
    ])


def skew(v):
    vx, vy, vz = v
    zero = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])


def rot_from_omega(wb, dt):
    theta = torch.linalg.norm(wb) * dt
    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew(axis)
    I = torch.eye(3, device=wb.device, dtype=wb.dtype)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


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
    """Compute Euler angle trajectory via vmap wb + sequential R accumulation."""
    R_curr = quat_to_rot(q0_init)

    wb_all = vmap(physics._compute_wb)(q_traj, q_dot_traj)
    R_delta_all = vmap(rot_from_omega, in_dims=(0, None))(wb_all, physics.dt)

    eulers = []
    for t in range(physics.num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))
    return torch.stack(eulers, dim=0)


# ── Augmented Lagrangian optimization ──────────────────────────────────
def run_optimization(physics, waypoints_init, q0_start, q0_goal, robot,
                     orient_tol_deg=ORIENT_TOL_DEG, rho_init=RHO_INIT,
                     rho_max=RHO_MAX, gamma=GAMMA, max_outer=MAX_OUTER,
                     inner_max_iter=INNER_MAX_ITER,
                     label="", verbose=False):
    """
    Augmented Lagrangian optimization:
      minimize   torque_cost(w)
      subject to angle_error(w) ≤ orient_tol_deg  (degrees)

    Inner solve (LBFGS):
      L(w; λ, ρ) = torque_cost + (ρ/2) · max(0, c(w) + λ/ρ)²
      where c(w) = angle_error_deg(w) - orient_tol_deg

    Outer updates:
      λ ← max(0, λ + ρ · c(w*))
      ρ ← γ · ρ   if infeasibility not sufficiently reduced
    """
    device = waypoints_init.device
    waypoints_param = waypoints_init.clone().detach().requires_grad_(True)

    lam = 0.0        # Lagrange multiplier
    rho = rho_init   # penalty parameter

    loss_history = []
    orient_history = []
    torque_history = []
    lam_history = []
    rho_history = []
    prev_violation = float('inf')

    t0 = time.time()

    for outer in range(max_outer):
        # ── Inner LBFGS solve ──
        optimizer = optim.LBFGS(
            [waypoints_param],
            lr=1.0,
            max_iter=inner_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        lam_k = lam
        rho_k = rho
        inner_count = [0]

        def closure():
            optimizer.zero_grad()

            # Objective: torque cost
            q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
            tc = physics.compute_torque_cost(q_traj[0], q_dot_traj[0])

            # Constraint: c(w) = angle_error - tol  (feasible when c ≤ 0)
            _, q_final = physics.simulate_single(
                q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])
            ae = angle_error_deg_smooth(q_final.unsqueeze(0), q0_goal)
            c = ae - orient_tol_deg

            # AL penalty for inequality: (ρ/2) · max(0, c + λ/ρ)²
            shifted = c + lam_k / rho_k
            penalty = (rho_k / 2.0) * torch.clamp(shifted, min=0.0) ** 2

            loss = tc + penalty
            loss.backward()

            inner_count[0] += 1
            return loss

        optimizer.step(closure)

        # ── Evaluate after inner solve ──
        with torch.no_grad():
            q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
            tc = physics.compute_torque_cost(q_traj[0], q_dot_traj[0]).item()
            _, q_final = physics.simulate_single(
                q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])
            ae = angle_error_deg(q_final.unsqueeze(0), q0_goal).item()

        c_val = ae - orient_tol_deg  # constraint violation (deg)

        loss_history.append(tc + max(0.0, lam * c_val + 0.5 * rho * c_val**2))
        orient_history.append(ae)
        torque_history.append(tc)
        lam_history.append(lam)
        rho_history.append(rho)

        if verbose:
            status = "FEAS" if c_val <= 0 else "INFEAS"
            print(f"  [{label}] Outer {outer+1:2d}  "
                  f"torque={tc:.1f}  angle={ae:.2f}deg  "
                  f"λ={lam:.2f}  ρ={rho:.1f}  "
                  f"[{status}]  (inner={inner_count[0]})")

        # ── Outer updates ──
        # Update multiplier
        lam = max(0.0, lam + rho * c_val)

        # Update penalty if constraint violation not improving enough
        if c_val > 0 and c_val > 0.25 * prev_violation:
            rho = min(rho * gamma, rho_max)

        prev_violation = max(c_val, 0.0)

        # ── Convergence check ──
        if c_val <= 0 and outer >= 2:
            # Feasible — check if torque cost stabilized
            if len(torque_history) >= 3:
                tc_change = abs(torque_history[-1] - torque_history[-2])
                if tc_change < 1.0:
                    if verbose:
                        print(f"  Converged: feasible & torque stable "
                              f"(Δtc={tc_change:.2f})")
                    break

    elapsed = time.time() - t0

    # Final metrics
    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        tc = physics.compute_torque_cost(q_traj[0], q_dot_traj[0]).item()
        torques = physics.compute_torques(q_traj[0], q_dot_traj[0])

        _, q_final = physics.simulate_single(
            q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])
        ae = angle_error_deg(q_final.unsqueeze(0), q0_goal).item()

    return {
        "label": label,
        "loss_history": loss_history,
        "orient_history": orient_history,
        "torque_history": torque_history,
        "lam_history": lam_history,
        "rho_history": rho_history,
        "torque_cost": tc,
        "angle_error_deg": ae,
        "time": elapsed,
        "iterations": len(loss_history),
        "waypoints": waypoints_param.detach(),
        "q_traj": q_traj[0].detach(),
        "q_dot_traj": q_dot_traj[0].detach(),
        "torques": torques.detach(),
    }


# ── Plotting ────────────────────────────────────────────────────────────
def plot_results(result, q0_start, q0_goal, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_q = result["q_traj"].shape[1]
    t_axis = np.linspace(0, TOTAL_TIME, result["q_traj"].shape[0])

    q_np = result["q_traj"].cpu().numpy()
    qd_np = result["q_dot_traj"].cpu().numpy()
    tau_np = result["torques"].cpu().numpy()

    # ── Figure 1: Trajectory + Torque (3 rows x n_q cols) ──
    fig, axes = plt.subplots(3, n_q, figsize=(4 * n_q, 10), sharex=True)
    row_data = [
        ("Joint Angle (rad)", q_np),
        ("Joint Velocity (rad/s)", qd_np),
        ("Torque (Nm)", tau_np),
    ]
    for row, (ylabel, data) in enumerate(row_data):
        for j in range(n_q):
            ax = axes[row, j]
            ax.plot(t_axis, data[:, j], linewidth=1.2)
            if row == 0:
                ax.set_title(f"J{j+1}", fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == 2:
                ax.set_xlabel("Time (s)", fontsize=9)
            ax.grid(True, alpha=0.2)

    fig.suptitle(f"Augmented Lagrangian Result\n"
                 f"angle_err={result['angle_error_deg']:.2f} deg, "
                 f"torque_cost={result['torque_cost']:.1f}", fontsize=13, y=1.02)
    fig.tight_layout()
    path1 = os.path.join(save_dir, "trajectory.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: AL convergence (4 subplots) ──
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    outer_iters = range(1, len(result["torque_history"]) + 1)

    ax = axes2[0, 0]
    ax.plot(outer_iters, result["torque_history"], 'o-', linewidth=1.5,
            markersize=4, color="#59a14f")
    ax.set_xlabel("AL Outer Iteration")
    ax.set_ylabel("Torque Cost")
    ax.set_title("Objective: Torque Cost")
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    ax.plot(outer_iters, result["orient_history"], 'o-', linewidth=1.5,
            markersize=4, color="#f28e2b")
    ax.axhline(y=ORIENT_TOL_DEG, color='red', linestyle='--', alpha=0.7,
               label=f'tol = {ORIENT_TOL_DEG:.1f} deg')
    ax.set_xlabel("AL Outer Iteration")
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title("Constraint: Angle Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    ax.plot(outer_iters, result["lam_history"], 'o-', linewidth=1.5,
            markersize=4, color="#e15759")
    ax.set_xlabel("AL Outer Iteration")
    ax.set_ylabel("λ (Multiplier)")
    ax.set_title("Lagrange Multiplier λ")
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    ax.plot(outer_iters, result["rho_history"], 'o-', linewidth=1.5,
            markersize=4, color="#4e79a7")
    ax.set_xlabel("AL Outer Iteration")
    ax.set_ylabel("ρ (Penalty)")
    ax.set_title("Penalty Parameter ρ")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig2.suptitle("Augmented Lagrangian Convergence", fontsize=14)
    fig2.tight_layout()
    path2 = os.path.join(save_dir, "convergence.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    # ── Figure 3: Torque norm ──
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    tau_norm = np.linalg.norm(tau_np, axis=1)
    ax3.plot(t_axis, tau_norm, linewidth=1.5, color="#59a14f")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("||torque|| (Nm)")
    ax3.set_title(f"Torque Norm (total cost={result['torque_cost']:.1f})")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    path3 = os.path.join(save_dir, "torque_norm.png")
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)

    return path1, path2, path3


# ── Main ────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)
    n_q = robot["n_q"]
    output_dim = NUM_WAYPOINTS * n_q
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    # ── Load CVAE ──
    COND_DIM = 8  # start_quat(4) + goal_quat(4)
    LATENT_DIM = 3
    cvae = CVAE(COND_DIM, output_dim, LATENT_DIM,
                joint_limits=robot["joint_limits"]).to(device)

    if os.path.exists(CVAE_WEIGHT_PATH):
        cvae.load_state_dict(torch.load(CVAE_WEIGHT_PATH, map_location=device,
                                        weights_only=True))
        cvae.eval()
        print(f"Loaded CVAE from {CVAE_WEIGHT_PATH}")
        use_cvae = True
    else:
        print(f"CVAE weights not found, using random init")
        use_cvae = False

    # ── Goal ──
    torch.manual_seed(42)
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    q0_goal = euler_to_quaternion(
        torch.tensor([math.radians(roll_deg)], device=device),
        torch.tensor([math.radians(pitch_deg)], device=device),
        torch.tensor([math.radians(yaw_deg)], device=device),
    )
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    print(f"Goal: roll={roll_deg}, pitch={pitch_deg}, yaw={yaw_deg} deg")

    # ── Select best CVAE sample ──
    with torch.no_grad():
        q0_start_batch = q0_start.expand(N_SAMPLES, -1)
        q0_goal_batch = q0_goal.expand(N_SAMPLES, -1)
        condition = torch.cat([q0_start_batch, q0_goal_batch], dim=1)

        if use_cvae:
            waypoints_all = cvae.inference(condition)
        else:
            waypoints_all = torch.randn(N_SAMPLES, output_dim, device=device) * 0.1

        q_traj_all, qd_traj_all = physics.generate_trajectory(waypoints_all)
        batch_sim = vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        losses_all, qf_all = batch_sim(q_traj_all, qd_traj_all,
                                        q0_start_batch, q0_goal_batch)
        ae_all = angle_error_deg(qf_all, q0_goal_batch)

    best_idx = losses_all.argmin().item()
    print(f"\nCVAE selection: {N_SAMPLES} samples")
    print(f"  loss: min={losses_all.min():.4f}, mean={losses_all.mean():.4f}")
    print(f"  angle_err: min={ae_all.min():.2f}, mean={ae_all.mean():.2f} deg")
    print(f"  best: #{best_idx} (loss={losses_all[best_idx]:.4f}, "
          f"angle_err={ae_all[best_idx]:.2f} deg)")

    w_best = waypoints_all[best_idx].unsqueeze(0)  # [1, output_dim]

    # ── Run optimization ──
    save_dir = os.path.join(ROOT_DIR, "outputs/plots/torque_direct")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Augmented Lagrangian: orient_tol={ORIENT_TOL_DEG:.1f} deg")
    print(f"{'='*60}")

    result = run_optimization(
        physics, w_best, q0_start, q0_goal, robot,
        label="AL", verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"Torque cost : {result['torque_cost']:.1f}")
    print(f"Angle error : {result['angle_error_deg']:.2f} deg (tol={ORIENT_TOL_DEG:.1f})")
    print(f"Time        : {result['time']:.1f}s ({result['iterations']} outer iters)")
    print(f"Final λ     : {result['lam_history'][-1]:.2f}")
    print(f"Final ρ     : {result['rho_history'][-1]:.1f}")
    print(f"{'='*60}")

    try:
        p1, p2, p3 = plot_results(result, q0_start, q0_goal, save_dir)
        print(f"  -> {p1}")
        print(f"  -> {p2}")
        print(f"  -> {p3}")
    except Exception as e:
        print(f"  Plotting failed: {e}")

    np.savez(
        os.path.join(save_dir, "results.npz"),
        orient_tol_deg=ORIENT_TOL_DEG,
        torque_cost=result["torque_cost"],
        angle_error=result["angle_error_deg"],
        best_cvae_idx=best_idx,
        waypoints_optimized=result["waypoints"].cpu().numpy(),
        lam_history=result["lam_history"],
        rho_history=result["rho_history"],
    )
    print(f"Results saved to {save_dir}/results.npz")


if __name__ == "__main__":
    main()
