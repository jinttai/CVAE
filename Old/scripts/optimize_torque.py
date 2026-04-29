"""
Torque-optimal trajectory optimization.

Flow:
  1. CVAE initial guess (waypoints)
  2. Newton correction → orientation manifold 위에 올림
     - J = d(error_vec) / d(waypoints),  minimum-norm: Δw = -α * J^T (JJ^T + λI)^{-1} e
  3. Null-space L-BFGS torque descent
     - Projected gradient: g_proj = N @ ∇torque  (N = I - J^T (JJ^T)^{-1} J)
     - L-BFGS with strong_wolfe line search
     - Lazy Jacobian recomputation (threshold-based)
     - Adaptive Newton correction (error-triggered)
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
import torch.optim as optim
import numpy as np

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from src.models.cvae import CVAE


# ── Config ──────────────────────────────────────────────────────────────
NUM_WAYPOINTS = 3
TOTAL_TIME = 10.0
N_SAMPLES = 128

# Newton correction
NEWTON_ITERS = 10
NEWTON_DAMPING = 1e-3     # Levenberg-Marquardt damping (larger = more stable)
NEWTON_STEP_SIZE = 0.5    # Damped step size (< 1 for stability)

# Null-space gradient descent (with Jacobian caching)
NULLSPACE_ITERS = 1000
NULLSPACE_LR = 0.1
J_RECOMPUTE_THRESHOLD = 0.05       # recompute J when ||Δw|| > this (rad)
J_RECOMPUTE_MAX_INTERVAL = 20      # force J recompute every N iters
CORRECTION_ERROR_THRESHOLD = 0.01  # Newton correct when ||e|| > this (rad, ~0.57 deg)
CORRECTION_MIN_INTERVAL = 20       # min iters between corrections
CORRECTION_NEWTON_ITERS = 3        # Newton iters per correction block
CONVERGENCE_WINDOW = 50            # early termination window
CONVERGENCE_REL_TOL = 1e-3         # stop if <0.1% improvement

# CVAE weights
CVAE_WEIGHT_PATH = os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v5_joint_change.pth")


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


# ── Core: per-sample Newton + null-space L-BFGS optimization ────────────
def optimize_single_sample(physics, w_init, q0_init, q0_goal, device, verbose=False):
    """
    Single sample torque optimization.

    Phase 1: Newton correction → project onto orientation manifold
    Phase 2: Null-space L-BFGS with lazy Jacobian + adaptive corrections

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
        Returns: (w_new, e_new_norm, J)  — also returns Jacobian for reuse.
        """
        e = _error_fn(w)
        e_norm = e.norm().item()

        if e_norm < 1e-6:
            return w, e_norm, None

        J = torch.autograd.functional.jacobian(_error_fn, w)  # [3, n_params]
        JJT = J @ J.T + damping * eye3
        dw = -step_size * J.T @ torch.linalg.solve(JJT, e)

        dw_norm = dw.norm().item()
        if dw_norm > 1.0:
            dw = dw * (1.0 / dw_norm)

        w_new = (w + dw).detach().requires_grad_(True)

        with torch.no_grad():
            e_new = _error_fn(w_new)
            e_new_norm = e_new.norm().item()

        if e_new_norm > e_norm * 1.5:
            dw = dw * 0.25
            w_new = (w.detach() + dw).requires_grad_(True)
            with torch.no_grad():
                e_new = _error_fn(w_new)
                e_new_norm = e_new.norm().item()

        if verbose:
            print(f"    Newton: ||e|| {e_norm:.4f} → {e_new_norm:.4f}")

        return w_new, e_new_norm, J

    def _compute_nullspace(J):
        """Compute null-space projector N = I - J^T (JJ^T + λI)^{-1} J."""
        JJT = J @ J.T + NEWTON_DAMPING * eye3
        JJT_inv_J = torch.linalg.solve(JJT, J)
        return eye_n - J.T @ JJT_inv_J

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

    J_last = None
    for k in range(NEWTON_ITERS):
        w, e_norm, J_step = _newton_step(w)
        if J_step is not None:
            J_last = J_step
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

    # ── Phase 2: Null-space gradient descent (with Jacobian caching) ──
    if verbose:
        print(f"  Phase 2: Null-space descent ({NULLSPACE_ITERS} iters, cached J)")

    torque_history = []
    error_history = []
    j_compute_count = [0]

    # Initialize Jacobian & null-space projector (reuse from Phase 1)
    if J_last is None:
        J_last = torch.autograd.functional.jacobian(_error_fn, w.detach())
        j_compute_count[0] += 1
    N_cached = _compute_nullspace(J_last)
    w_at_J = w.detach().clone()
    last_J_iter = 0
    last_correction_iter = -CORRECTION_MIN_INTERVAL

    for k in range(NULLSPACE_ITERS):
        # Torque cost gradient
        w_grad = w.detach().requires_grad_(True)
        tc = _torque_cost_fn(w_grad)
        g = torch.autograd.grad(tc, w_grad)[0]
        torque_history.append(tc.item())

        if torch.isnan(g).any() or torch.isnan(tc):
            if verbose:
                print(f"    NaN at iter {k}, stopping")
            break

        # Normalize gradient for stable step size
        g_norm = g.norm()
        if g_norm < 1e-10:
            break

        # Null-space descent step (using cached N)
        dw = -NULLSPACE_LR * (N_cached @ (g / g_norm))
        w = (w.detach() + dw).requires_grad_(True)

        # ── Lazy Jacobian recomputation ──
        w_moved = (w.detach() - w_at_J).norm().item()
        if w_moved > J_RECOMPUTE_THRESHOLD or (k - last_J_iter) >= J_RECOMPUTE_MAX_INTERVAL:
            J_last = torch.autograd.functional.jacobian(_error_fn, w.detach())
            N_cached = _compute_nullspace(J_last)
            w_at_J = w.detach().clone()
            last_J_iter = k
            j_compute_count[0] += 1

        # ── Error monitoring & adaptive Newton correction ──
        with torch.no_grad():
            e_cur = _error_fn(w.detach())
            e_norm = e_cur.norm().item()
        error_history.append(e_norm)

        if e_norm > CORRECTION_ERROR_THRESHOLD and (k - last_correction_iter) >= CORRECTION_MIN_INTERVAL:
            for _ in range(CORRECTION_NEWTON_ITERS):
                w, e_after, J_step = _newton_step(w)
                if J_step is not None:
                    J_last = J_step
                    N_cached = _compute_nullspace(J_last)
                    w_at_J = w.detach().clone()
                    last_J_iter = k
                    j_compute_count[0] += 1
                if e_after < 1e-5:
                    break
            last_correction_iter = k

        if verbose and (k + 1) % 50 == 0:
            print(f"    Iter {k+1}: torque={tc.item():.1f}, "
                  f"||e||={e_norm:.6f}, J_calls={j_compute_count[0]}")

        # ── Early termination ──
        if len(torque_history) > CONVERGENCE_WINDOW:
            old_tc = torque_history[-CONVERGENCE_WINDOW]
            new_tc = torque_history[-1]
            if old_tc > 0 and (old_tc - new_tc) / old_tc < CONVERGENCE_REL_TOL:
                if verbose:
                    print(f"    Converged at iter {k+1}: "
                          f"Δtc/tc < {CONVERGENCE_REL_TOL} over {CONVERGENCE_WINDOW} iters")
                break

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
    info['j_compute_count'] = j_compute_count[0]

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
    COND_DIM = 8  # start_quat(4) + goal_quat(4), matching train_cvae.py
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
    # condition = [start_quat, goal_quat] (COND_DIM=8)
    with torch.no_grad():
        condition = torch.cat([q0_start, q0_goals], dim=1)
        waypoints_init = cvae.inference(condition)

    # ── Phase 0: Batch evaluate all CVAE samples, pick best ──
    print(f"\n{'='*70}")
    print(f"Evaluating {N_SAMPLES} CVAE samples to select best initial guess...")
    with torch.no_grad():
        q_traj_all, qd_traj_all = physics.generate_trajectory(waypoints_init)
        batch_sim = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        losses_all, qf_all = batch_sim(q_traj_all, qd_traj_all, q0_start, q0_goals)
        ae_all = angle_error_deg(qf_all, q0_goals)

    print(f"  Physics loss : min={losses_all.min():.4f}, mean={losses_all.mean():.4f}, max={losses_all.max():.4f}")
    print(f"  Angle error  : min={ae_all.min():.2f}, mean={ae_all.mean():.2f}, max={ae_all.max():.2f} deg")

    best_idx = losses_all.argmin().item()
    torque_cost_best = physics.compute_torque_cost(q_traj_all[best_idx], qd_traj_all[best_idx])
    print(f"  Best sample: #{best_idx} (loss={losses_all[best_idx]:.4f}, angle_err={ae_all[best_idx]:.2f} deg, torque_cost={torque_cost_best:.4f})")

    # ── Phase 1-2: Optimize only the best sample ──
    print(f"\nOptimizing best sample #{best_idx}...")
    print(f"  Newton: {NEWTON_ITERS} iters, damping={NEWTON_DAMPING}, step={NEWTON_STEP_SIZE}")
    print(f"  Null-space: {NULLSPACE_ITERS} iters, lr={NULLSPACE_LR}, J_cache_interval={J_RECOMPUTE_MAX_INTERVAL}")
    print(f"{'='*70}\n")

    save_dir = os.path.join(ROOT_DIR, "outputs/plots/torque_opt")
    os.makedirs(save_dir, exist_ok=True)

    t_start = time.time()
    w_opt, info = optimize_single_sample(
        physics,
        waypoints_init[best_idx],
        q0_start[best_idx],
        q0_goals[best_idx],
        device,
        verbose=True,
    )
    total_time_elapsed = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"{'Stage':<20s} | {'Torque Cost':>14s} | {'Angle Error':>14s}")
    print(f"{'─'*55}")
    print(f"{'CVAE init':<20s} | {info['torque_init']:14.2f} | {info['angle_err_init']:10.2f} deg")
    print(f"{'After Newton':<20s} | {info['torque_after_newton']:14.2f} | {info['angle_err_after_newton']:10.2f} deg")
    print(f"{'After null-space':<20s} | {info['torque_final']:14.2f} | {info['angle_err_final']:10.2f} deg")
    print(f"{'─'*55}")

    if info['torque_after_newton'] > 0:
        reduction = (1 - info['torque_final'] / info['torque_after_newton']) * 100
        print(f"Torque reduction (null-space): {reduction:.1f}%")
    print(f"Total time: {total_time_elapsed:.1f}s")
    print(f"L-BFGS iters: {len(info['torque_history'])}, Jacobian calls: {info.get('j_compute_count', 'N/A')}")
    print(f"{'='*70}")

    # ── Per-sample trajectory & torque plots ──
    try:
        p1, p2 = plot_sample_trajectories(
            physics, info, q0_start[best_idx], q0_goals[best_idx], best_idx, save_dir)
        print(f"  -> {p1}")
        print(f"  -> {p2}")
    except Exception as e:
        print(f"  Plotting failed: {e}")

    # ── Save results ──
    np.savez(
        os.path.join(save_dir, "results.npz"),
        best_idx=best_idx,
        all_losses=losses_all.cpu().numpy(),
        all_angle_errors=ae_all.cpu().numpy(),
        torque_init=info['torque_init'],
        torque_newton=info['torque_after_newton'],
        torque_final=info['torque_final'],
        angle_err_init=info['angle_err_init'],
        angle_err_newton=info['angle_err_after_newton'],
        angle_err_final=info['angle_err_final'],
        waypoints_init=waypoints_init[best_idx].cpu().numpy(),
        waypoints_optimized=w_opt.cpu().numpy(),
        goal=q0_goals[best_idx].cpu().numpy(),
    )

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # All CVAE samples: loss distribution
        ax = axes[0, 0]
        losses_np = losses_all.cpu().numpy()
        ax.hist(losses_np, bins=30, color='#4e79a7', edgecolor='white', alpha=0.8)
        ax.axvline(losses_np[best_idx], color='red', linestyle='--',
                   label=f'best #{best_idx} = {losses_np[best_idx]:.4f}')
        ax.set_xlabel("Physics Loss")
        ax.set_ylabel("Count")
        ax.set_title(f"CVAE Sample Selection ({N_SAMPLES} candidates)")
        ax.legend()

        # Optimization stages (bar)
        ax = axes[0, 1]
        stages = ['CVAE init', 'Newton', 'Null-space']
        vals = [info['torque_init'], info['torque_after_newton'], info['torque_final']]
        bars = ax.bar(stages, vals,
                       color=['#4e79a7', '#f28e2b', '#59a14f'], edgecolor='white')
        ax.set_ylabel("Torque Cost")
        ax.set_title("Torque Cost by Stage (best sample)")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9)

        # Angle error stages (bar)
        ax = axes[1, 0]
        ae_vals = [info['angle_err_init'], info['angle_err_after_newton'], info['angle_err_final']]
        bars = ax.bar(stages, ae_vals,
                       color=['#4e79a7', '#f28e2b', '#59a14f'], edgecolor='white')
        ax.set_ylabel("Angle Error (deg)")
        ax.set_title("Orientation Error by Stage (best sample)")
        for bar, v in zip(bars, ae_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)

        # Torque convergence
        ax = axes[1, 1]
        if info['torque_history']:
            ax.plot(info['torque_history'], color='#59a14f', linewidth=1.5)
            ax.set_xlabel("Null-space Iteration")
            ax.set_ylabel("Torque Cost")
            ax.set_title("Torque Convergence (best sample)")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Torque Optimization: best of {N_SAMPLES} CVAE samples",
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
