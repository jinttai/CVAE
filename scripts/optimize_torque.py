"""
Torque-optimal trajectory optimization.

Flow:
  1. CVAE initial guess (waypoints)
  2. Newton correction → orientation manifold 위에 올림
     - J = d(error_vec) / d(waypoints),  minimum-norm: Δw = -α * J^T (JJ^T + λI)^{-1} e
     - Damped step size + error monitoring for stability
  3. Null-space torque descent
     - torque cost gradient g = ∇_w (Σ ||τ||²)
     - null-space projector N = I - J^T (JJ^T)^{-1} J
     - Δw = -lr * N @ g
     - 중간중간 Newton correction 삽입
"""

import os
import sys
import time
import math

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.utils.runtime_env import configure_windows_runtime

configure_windows_runtime()

import torch
import numpy as np

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from src.models.cvae import CVAE


# ── Config ──────────────────────────────────────────────────────────────
NUM_WAYPOINTS = 3
TOTAL_TIME = 10.0
N_SAMPLES = 4

# Newton correction
NEWTON_ITERS = 10
NEWTON_DAMPING = 1e-3     # Levenberg-Marquardt damping (larger = more stable)
NEWTON_STEP_SIZE = 0.5    # Damped step size (< 1 for stability)

# Null-space torque optimization
NULLSPACE_ITERS = 30
NULLSPACE_LR = 0.1
CORRECTION_INTERVAL = 5   # 매 N step 마다 Newton correction
CORRECTION_ITERS = 3      # correction 당 Newton iteration 수

# CVAE weights
CVAE_WEIGHT_PATH = os.path.join(ROOT_DIR, "outputs/weights/cvae/v5.pth")


# ── Helpers ─────────────────────────────────────────────────────────────
def euler_to_quaternion(roll, pitch, yaw):
    """Batched Euler (rad) → quaternion [x,y,z,w]."""
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2); sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2); sy = torch.sin(yaw / 2)
    return torch.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dim=-1)


def random_goal_quaternions(n, device):
    """Full-range uniform random orientations."""
    yaw   = 2 * math.pi * torch.rand(n, device=device) - math.pi
    pitch = math.pi * torch.rand(n, device=device) - math.pi / 2
    roll  = 2 * math.pi * torch.rand(n, device=device) - math.pi
    return euler_to_quaternion(roll, pitch, yaw)


def angle_error_deg(q_final, q_goal):
    """Quaternion angle error in degrees."""
    dot = (q_final * q_goal).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot) * 180.0 / math.pi


# ── Core: per-sample Newton + null-space optimization ───────────────────
def optimize_single_sample(physics, w_init, q0_init, q0_goal, device, verbose=False):
    """
    Single sample torque optimization.

    Args:
        w_init: [n_params] initial waypoints from CVAE
        q0_init: [4] initial quaternion
        q0_goal: [4] goal quaternion
    Returns:
        w_opt: [n_params] optimized waypoints
        info: dict with metrics at each stage
    """
    n_params = w_init.shape[0]
    eye_n = torch.eye(n_params, device=device)
    eye3 = torch.eye(3, device=device)

    # ── Helper closures ──
    def _error_fn(w):
        """w [n_params] → orientation error vec [3]"""
        q_traj, q_dot_traj = physics.generate_trajectory(w.unsqueeze(0))
        return physics.simulate_single_error_vec(
            q_traj[0], q_dot_traj[0], q0_init, q0_goal
        )

    def _torque_cost_fn(w):
        """w [n_params] → scalar torque cost"""
        q_traj, q_dot_traj = physics.generate_trajectory(w.unsqueeze(0))
        return physics.compute_torque_cost(q_traj[0], q_dot_traj[0])

    def _get_final_quat(w):
        """w [n_params] → final quaternion [4]"""
        q_traj, q_dot_traj = physics.generate_trajectory(w.unsqueeze(0))
        _, q_final = physics.simulate_single(
            q_traj[0], q_dot_traj[0], q0_init, q0_goal
        )
        return q_final

    def _newton_step(w, step_size=NEWTON_STEP_SIZE, damping=NEWTON_DAMPING):
        """
        Damped Newton correction step (Levenberg-Marquardt style).
        Δw = -step_size * J^T (JJ^T + λI)^{-1} e
        """
        e = _error_fn(w)
        e_norm = e.norm().item()

        # Skip if already converged
        if e_norm < 1e-6:
            return w, e_norm

        J = torch.autograd.functional.jacobian(_error_fn, w)  # [3, n_params]
        JJT = J @ J.T + damping * eye3
        dw = -step_size * J.T @ torch.linalg.solve(JJT, e)

        # Clamp step size to prevent divergence
        dw_norm = dw.norm().item()
        max_step = 1.0  # max waypoint change per step (rad)
        if dw_norm > max_step:
            dw = dw * (max_step / dw_norm)

        w_new = (w + dw).detach().requires_grad_(True)

        # Check if error actually decreased; if not, halve step
        with torch.no_grad():
            e_new = _error_fn(w_new)
            e_new_norm = e_new.norm().item()

        if e_new_norm > e_norm * 1.5:  # allow slight increase
            # Backtrack: try smaller step
            dw = dw * 0.25
            w_new = (w.detach() + dw).requires_grad_(True)
            with torch.no_grad():
                e_new = _error_fn(w_new)
                e_new_norm = e_new.norm().item()

        if verbose:
            print(f"    Newton: ||e|| {e_norm:.4f} → {e_new_norm:.4f}")

        return w_new, e_new_norm

    info = {}
    w = w_init.clone().detach().requires_grad_(True)

    # ── Phase 0: Metrics before optimization ──
    with torch.no_grad():
        tc_init = _torque_cost_fn(w.detach())
        qf_init = _get_final_quat(w.detach())
        ae_init = angle_error_deg(qf_init.unsqueeze(0), q0_goal.unsqueeze(0)).item()
    info['torque_init'] = tc_init.item()
    info['angle_err_init'] = ae_init
    info['w_cvae'] = w.detach().clone()

    # ── Phase 1: Newton correction (orientation manifold projection) ──
    if verbose:
        print(f"  Phase 1: Newton correction ({NEWTON_ITERS} iters)")

    for k in range(NEWTON_ITERS):
        w, e_norm = _newton_step(w)
        if e_norm < 1e-5:
            if verbose:
                print(f"    Converged at iter {k+1}")
            break

    with torch.no_grad():
        tc_after_newton = _torque_cost_fn(w.detach())
        qf_newton = _get_final_quat(w.detach())
        ae_newton = angle_error_deg(qf_newton.unsqueeze(0), q0_goal.unsqueeze(0)).item()
        e_after = _error_fn(w.detach())
    info['torque_after_newton'] = tc_after_newton.item()
    info['angle_err_after_newton'] = ae_newton
    info['error_norm_after_newton'] = e_after.norm().item()
    info['w_newton'] = w.detach().clone()

    # ── Phase 2: Null-space torque descent ──
    if verbose:
        print(f"  Phase 2: Null-space optimization ({NULLSPACE_ITERS} iters)")

    torque_history = []
    error_history = []

    for k in range(NULLSPACE_ITERS):
        # Torque cost gradient
        w_grad = w.detach().requires_grad_(True)
        tc = _torque_cost_fn(w_grad)
        g = torch.autograd.grad(tc, w_grad)[0]  # [n_params]
        torque_history.append(tc.item())

        # Check for NaN
        if torch.isnan(g).any() or torch.isnan(tc):
            if verbose:
                print(f"    NaN detected at iter {k}, stopping")
            break

        # Orientation Jacobian (at current point)
        J = torch.autograd.functional.jacobian(_error_fn, w.detach())  # [3, n_params]

        # Null-space projection: N = I - J^T (JJ^T + λI)^{-1} J
        JJT = J @ J.T + NEWTON_DAMPING * eye3
        JJT_inv_J = torch.linalg.solve(JJT, J)  # [3, n_params]
        N = eye_n - J.T @ JJT_inv_J              # [n_params, n_params]

        # Normalize gradient for stable step size
        g_norm = g.norm()
        if g_norm > 1e-10:
            g_normalized = g / g_norm
        else:
            break  # gradient vanished

        # Null-space descent step
        dw = -NULLSPACE_LR * (N @ g_normalized)
        w = (w.detach() + dw).requires_grad_(True)

        # Monitor error drift
        with torch.no_grad():
            e_cur = _error_fn(w.detach())
            error_history.append(e_cur.norm().item())

        if verbose and (k + 1) % 10 == 0:
            print(f"    NS iter {k+1}: torque={tc.item():.1f}, ||e||={e_cur.norm().item():.6f}")

        # Intermittent Newton correction to stay on manifold
        if (k + 1) % CORRECTION_INTERVAL == 0:
            for _ in range(CORRECTION_ITERS):
                w, _ = _newton_step(w)

    # ── Final metrics ──
    with torch.no_grad():
        tc_final = _torque_cost_fn(w.detach())
        qf_final = _get_final_quat(w.detach())
        ae_final = angle_error_deg(qf_final.unsqueeze(0), q0_goal.unsqueeze(0)).item()
        e_final = _error_fn(w.detach())
    info['torque_final'] = tc_final.item()
    info['angle_err_final'] = ae_final
    info['error_norm_final'] = e_final.norm().item()
    info['torque_history'] = torque_history
    info['error_history'] = error_history
    info['w_nullspace'] = w.detach().clone()

    return w.detach(), info


def plot_sample_trajectories(physics, info, q0_init, q0_goal, sample_idx, save_dir):
    """
    Plot trajectory (joint angles, velocities) and torque profiles
    for each optimization stage of a single sample.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = {
        'CVAE init': info['w_cvae'],
        'After Newton': info['w_newton'],
        'After Null-space': info['w_nullspace'],
    }
    stage_colors = {
        'CVAE init': '#4e79a7',
        'After Newton': '#f28e2b',
        'After Null-space': '#59a14f',
    }

    n_q = physics.n_q
    t_axis = np.linspace(0, physics.total_time, physics.num_steps)
    joint_labels = [f'J{j+1}' for j in range(n_q)]

    # Collect data for each stage
    stage_data = {}
    for name, w in stages.items():
        with torch.no_grad():
            q_traj, q_dot_traj = physics.generate_trajectory(w.unsqueeze(0))
            torques = physics.compute_torques(q_traj[0], q_dot_traj[0])
        stage_data[name] = {
            'q': q_traj[0].cpu().numpy(),
            'qd': q_dot_traj[0].cpu().numpy(),
            'tau': torques.cpu().numpy(),
        }

    # ── Figure 1: Overlay all stages per joint (3 rows x n_q cols) ──
    fig, axes = plt.subplots(3, n_q, figsize=(4 * n_q, 10), sharex=True)

    row_labels = ['Joint Angle (rad)', 'Joint Velocity (rad/s)', 'Torque (Nm)']
    data_keys = ['q', 'qd', 'tau']

    for row, (label, key) in enumerate(zip(row_labels, data_keys)):
        for j in range(n_q):
            ax = axes[row, j]
            for name in stages:
                ax.plot(t_axis, stage_data[name][key][:, j],
                        color=stage_colors[name], label=name,
                        linewidth=1.2, alpha=0.85)
            if row == 0:
                ax.set_title(joint_labels[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 2:
                ax.set_xlabel('Time (s)', fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'Sample {sample_idx}: Trajectory & Torque Comparison', fontsize=13, y=1.03)
    fig.tight_layout()
    path1 = os.path.join(save_dir, f'sample_{sample_idx}_trajectories.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: Torque norm over time ──
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for name in stages:
        tau = stage_data[name]['tau']
        tau_norm = np.linalg.norm(tau, axis=1)
        ax2.plot(t_axis, tau_norm, color=stage_colors[name], label=name,
                 linewidth=1.5, alpha=0.85)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('||torque|| (Nm)')
    ax2.set_title(f'Sample {sample_idx}: Torque Norm Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path2 = os.path.join(save_dir, f'sample_{sample_idx}_torque_norm.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    return path1, path2


# ── Main ────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Robot & Physics
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    n_q = robot["n_q"]
    output_dim = NUM_WAYPOINTS * n_q

    # ── Load CVAE ──
    COND_DIM = n_q + n_q + 4  # 16
    LATENT_DIM = 3
    cvae = CVAE(COND_DIM, output_dim, LATENT_DIM,
                joint_limits=robot['joint_limits']).to(device)

    if os.path.exists(CVAE_WEIGHT_PATH):
        cvae.load_state_dict(torch.load(CVAE_WEIGHT_PATH, map_location=device,
                                        weights_only=True))
        cvae.eval()
        print(f"Loaded CVAE from {CVAE_WEIGHT_PATH}")
        use_cvae = True
    else:
        print(f"CVAE weights not found at {CVAE_WEIGHT_PATH}, using random init")
        use_cvae = False

    # ── Fixed Euler goal: [15°, 15°, -15°] (roll, pitch, yaw) for all samples ──
    torch.manual_seed(42)
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll = torch.tensor(math.radians(roll_deg), device=device).expand(N_SAMPLES)
    pitch = torch.tensor(math.radians(pitch_deg), device=device).expand(N_SAMPLES)
    yaw = torch.tensor(math.radians(yaw_deg), device=device).expand(N_SAMPLES)
    q0_goals = euler_to_quaternion(roll, pitch, yaw)
    q0_start = torch.zeros(N_SAMPLES, 4, device=device)
    q0_start[:, 3] = 1.0

    # ── CVAE initial guess ──
    with torch.no_grad():
        if use_cvae:
            start_joints = torch.zeros(N_SAMPLES, n_q, device=device)
            goal_joints = torch.zeros(N_SAMPLES, n_q, device=device)
            condition = torch.cat([start_joints, goal_joints, q0_goals], dim=1)
            waypoints_init = cvae.inference(condition)
        else:
            waypoints_init = torch.randn(N_SAMPLES, output_dim, device=device) * 0.1

    print(f"\n{'='*70}")
    print(f"Torque Optimization: {N_SAMPLES} samples")
    print(f"  Newton: {NEWTON_ITERS} iters, damping={NEWTON_DAMPING}, step={NEWTON_STEP_SIZE}")
    print(f"  Null-space: {NULLSPACE_ITERS} iters, lr={NULLSPACE_LR}")
    print(f"  Correction: every {CORRECTION_INTERVAL} steps, {CORRECTION_ITERS} iters each")
    print(f"{'='*70}\n")

    # ── Optimize each sample ──
    save_dir = os.path.join(ROOT_DIR, "outputs/plots/torque_opt")
    os.makedirs(save_dir, exist_ok=True)

    all_info = []
    w_optimized = torch.zeros_like(waypoints_init)
    t_start = time.time()

    for i in range(N_SAMPLES):
        t_i = time.time()
        w_opt, info = optimize_single_sample(
            physics,
            waypoints_init[i],
            q0_start[i],
            q0_goals[i],
            device,
            verbose=True,
        )
        w_optimized[i] = w_opt
        all_info.append(info)
        dt_i = time.time() - t_i

        print(f"[{i+1:3d}/{N_SAMPLES}] {dt_i:.1f}s | "
              f"torque: {info['torque_init']:.1f} -> {info['torque_after_newton']:.1f} -> {info['torque_final']:.1f} | "
              f"angle err: {info['angle_err_init']:.2f} -> {info['angle_err_after_newton']:.2f} -> {info['angle_err_final']:.2f} deg")

        # Per-sample trajectory & torque plots
        try:
            p1, p2 = plot_sample_trajectories(
                physics, info, q0_start[i], q0_goals[i], i, save_dir)
            print(f"  -> {p1}")
            print(f"  -> {p2}")
        except Exception as e:
            print(f"  Plotting failed: {e}")
        print()

    total_time_elapsed = time.time() - t_start
    print(f"Total time: {total_time_elapsed:.1f}s ({total_time_elapsed/N_SAMPLES:.1f}s/sample)")

    # ── Summary statistics ──
    torque_init = np.array([info['torque_init'] for info in all_info])
    torque_newton = np.array([info['torque_after_newton'] for info in all_info])
    torque_final = np.array([info['torque_final'] for info in all_info])
    ae_init = np.array([info['angle_err_init'] for info in all_info])
    ae_newton = np.array([info['angle_err_after_newton'] for info in all_info])
    ae_final = np.array([info['angle_err_final'] for info in all_info])

    print(f"\n{'='*70}")
    print(f"{'Stage':<20s} | {'Torque Cost (mean +/- std)':>28s} | {'Angle Error (mean +/- std)':>28s}")
    print(f"{'─'*70}")
    print(f"{'CVAE init':<20s} | {torque_init.mean():12.2f} +/- {torque_init.std():8.2f} | {ae_init.mean():8.2f} +/- {ae_init.std():6.2f} deg")
    print(f"{'After Newton':<20s} | {torque_newton.mean():12.2f} +/- {torque_newton.std():8.2f} | {ae_newton.mean():8.2f} +/- {ae_newton.std():6.2f} deg")
    print(f"{'After null-space':<20s} | {torque_final.mean():12.2f} +/- {torque_final.std():8.2f} | {ae_final.mean():8.2f} +/- {ae_final.std():6.2f} deg")
    print(f"{'─'*70}")

    if not np.isnan(torque_final.mean()) and torque_newton.mean() > 0:
        reduction = (1 - torque_final.mean() / torque_newton.mean()) * 100
        print(f"Torque reduction (null-space): {reduction:.1f}%")
    print(f"{'='*70}")

    # ── Save results ──
    np.savez(
        os.path.join(save_dir, "results.npz"),
        torque_init=torque_init,
        torque_newton=torque_newton,
        torque_final=torque_final,
        angle_err_init=ae_init,
        angle_err_newton=ae_newton,
        angle_err_final=ae_final,
        waypoints_init=waypoints_init.cpu().numpy(),
        waypoints_optimized=w_optimized.cpu().numpy(),
        goals=q0_goals.cpu().numpy(),
    )

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Torque cost comparison (bar)
        ax = axes[0, 0]
        stages = ['CVAE init', 'Newton', 'Null-space']
        means = [torque_init.mean(), torque_newton.mean(), torque_final.mean()]
        stds = [torque_init.std(), torque_newton.std(), torque_final.std()]
        bars = ax.bar(stages, means, yerr=stds, capsize=5,
                       color=['#4e79a7', '#f28e2b', '#59a14f'], edgecolor='white')
        ax.set_ylabel("Torque Cost")
        ax.set_title("Torque Cost by Stage")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{m:.1f}', ha='center', va='bottom', fontsize=9)

        # Angle error comparison (bar)
        ax = axes[0, 1]
        means_ae = [ae_init.mean(), ae_newton.mean(), ae_final.mean()]
        stds_ae = [ae_init.std(), ae_newton.std(), ae_final.std()]
        bars = ax.bar(stages, means_ae, yerr=stds_ae, capsize=5,
                       color=['#4e79a7', '#f28e2b', '#59a14f'], edgecolor='white')
        ax.set_ylabel("Angle Error (deg)")
        ax.set_title("Orientation Error by Stage")
        for bar, m in zip(bars, means_ae):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{m:.2f}', ha='center', va='bottom', fontsize=9)

        # Per-sample torque comparison
        ax = axes[1, 0]
        x = np.arange(N_SAMPLES)
        w_bar = 0.25
        ax.bar(x - w_bar, torque_init, w_bar, label='CVAE init', color='#4e79a7', alpha=0.8)
        ax.bar(x, torque_newton, w_bar, label='After Newton', color='#f28e2b', alpha=0.8)
        ax.bar(x + w_bar, torque_final, w_bar, label='After Null-space', color='#59a14f', alpha=0.8)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Torque Cost")
        ax.set_title("Per-sample Torque Cost")
        ax.legend()

        # Torque convergence (first sample)
        ax = axes[1, 1]
        if all_info[0]['torque_history']:
            ax.plot(all_info[0]['torque_history'], color='#59a14f', linewidth=1.5)
            ax.set_xlabel("Null-space Iteration")
            ax.set_ylabel("Torque Cost")
            ax.set_title("Torque Convergence (sample 0)")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Torque Optimization: CVAE + Newton + Null-space ({N_SAMPLES} samples)",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "torque_optimization.png"), dpi=150)
        plt.close(fig)
        print(f"\nPlot saved to {save_dir}/torque_optimization.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"Results saved to {save_dir}/results.npz")


if __name__ == "__main__":
    main()
