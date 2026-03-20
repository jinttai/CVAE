"""
Simulate DDP trajectory in MuJoCo and compare with DDP result.

Loads the DDP-optimized states/controls, runs a MuJoCo simulation with
Computed Torque Control (CTC) to track the trajectory, then generates
comparison plots for joint angles, joint velocities, and base orientation.
"""

import os
import sys
import numpy as np
import mujoco
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from scenario import SCENARIO, get_goal_quaternion


def quat_to_euler_wxyz(q):
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in rad."""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def quat_to_euler_xyzw(q):
    """Convert quaternion [x, y, z, w] to Euler angles [roll, pitch, yaw] in rad."""
    x, y, z, w = q
    return quat_to_euler_wxyz([w, x, y, z])


def main():
    # ── Load DDP results ──
    results_dir = os.path.join(ROOT_DIR, 'ddp', 'results')
    X_ddp = np.load(os.path.join(results_dir, 'trajectory_casadi_ddp_states.npy'))
    U_ddp = np.load(os.path.join(results_dir, 'trajectory_casadi_ddp_controls.npy'))

    n_q = SCENARIO['n_q']  # 6
    dt_ddp = SCENARIO['dt']  # 0.1
    T = U_ddp.shape[0]  # 100
    total_time = T * dt_ddp

    # Extract DDP trajectories
    # State: [q(6), qd(6), quat_xyzw(4)]
    q_ddp = X_ddp[:, :n_q]
    qd_ddp = X_ddp[:, n_q:2*n_q]
    quat_ddp_xyzw = X_ddp[:, 2*n_q:]  # [x, y, z, w]
    # Convert to MuJoCo convention [w, x, y, z]
    quat_ddp_wxyz = quat_ddp_xyzw[:, [3, 0, 1, 2]]
    times_ddp = np.arange(T + 1) * dt_ddp

    print(f"Loaded DDP trajectory: {T} steps, dt={dt_ddp}s, total={total_time}s")
    print(f"  States shape: {X_ddp.shape}, Controls shape: {U_ddp.shape}")

    # ── Compute base angular velocity from DDP quaternions ──
    w_ddp = np.zeros((T + 1, 3))
    for i in range(T):
        q_cur = quat_ddp_wxyz[i]
        q_nxt = quat_ddp_wxyz[i + 1]
        # Body-frame angular velocity: dq_body = q_cur^{-1} * q_next
        q_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
        # quaternion multiply q_inv * q_nxt
        aw, av = q_inv[0], q_inv[1:]
        bw, bv = q_nxt[0], q_nxt[1:]
        rw = aw * bw - np.dot(av, bv)
        rv = aw * bv + bw * av + np.cross(av, bv)
        # log map
        rw = np.clip(rw, -1.0, 1.0)
        if rw > 0.999999:
            w_ddp[i] = (2.0 / dt_ddp) * rv
        else:
            theta = 2.0 * np.arccos(rw)
            sin_half = np.sin(theta / 2.0)
            if sin_half < 1e-6:
                w_ddp[i] = np.zeros(3)
            else:
                w_ddp[i] = (theta / dt_ddp) * (rv / sin_half)
    w_ddp[-1] = w_ddp[-2]

    # ── Compute joint acceleration from DDP ──
    qdd_ddp = np.zeros((T + 1, n_q))
    qdd_ddp[:T] = U_ddp
    qdd_ddp[T] = U_ddp[-1]

    # ── MuJoCo simulation ──
    xml_path = os.path.join(ROOT_DIR, 'assets', 'spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)

    dt_sim = model.opt.timestep  # 0.01s
    n_sim_steps = int(total_time / dt_sim)

    # CTC gains
    Kp = np.array([500.0] * 3 + [250.0] * 3)
    Kd = np.array([50.0] * 3 + [25.0] * 3)

    # Set initial state
    data.qpos[0:3] = np.zeros(3)        # base position
    data.qpos[3:7] = quat_ddp_wxyz[0]   # base quaternion
    data.qpos[7:] = q_ddp[0]            # joint angles
    data.qvel[0:3] = np.zeros(3)        # base linear velocity
    data.qvel[3:6] = w_ddp[0]           # base angular velocity (body frame)
    data.qvel[6:] = qd_ddp[0]           # joint velocities
    mujoco.mj_forward(model, data)

    # Logging
    sim_q_log = []
    sim_qd_log = []
    sim_quat_log = []
    sim_t_log = []

    print(f"Running MuJoCo simulation: {n_sim_steps} steps at dt={dt_sim}s ...")

    for step in range(n_sim_steps):
        sim_time = step * dt_sim

        # Find reference index by interpolation
        idx = int(sim_time / dt_ddp)
        if idx >= T:
            idx = T - 1
        alpha = (sim_time - idx * dt_ddp) / dt_ddp
        alpha = np.clip(alpha, 0.0, 1.0)

        # Interpolate reference
        q_ref = (1 - alpha) * q_ddp[idx] + alpha * q_ddp[idx + 1]
        qd_ref = (1 - alpha) * qd_ddp[idx] + alpha * qd_ddp[idx + 1]
        qdd_ref = (1 - alpha) * qdd_ddp[idx] + alpha * qdd_ddp[idx + 1]

        # CTC: desired acceleration = feedforward + PD feedback
        q_err = q_ref - data.qpos[7:]
        qd_err = qd_ref - data.qvel[6:]
        qacc_des = qdd_ref + Kp * q_err + Kd * qd_err

        # Inverse dynamics
        data_ref.qpos[:] = data.qpos[:]
        data_ref.qvel[:] = data.qvel[:]
        data_ref.qacc[0:6] = np.zeros(6)  # unactuated base
        data_ref.qacc[6:] = qacc_des
        mujoco.mj_inverse(model, data_ref)

        data.ctrl[:] = data_ref.qfrc_inverse[6:]

        # Log
        sim_q_log.append(data.qpos[7:].copy())
        sim_qd_log.append(data.qvel[6:].copy())
        sim_quat_log.append(data.qpos[3:7].copy())
        sim_t_log.append(sim_time)

        # Step
        mujoco.mj_step(model, data)

    sim_q = np.array(sim_q_log)
    sim_qd = np.array(sim_qd_log)
    sim_quat = np.array(sim_quat_log)
    sim_t = np.array(sim_t_log)

    print("Simulation complete. Generating comparison plots...")

    # ── Compute Euler angles for both ──
    euler_ddp = np.array([quat_to_euler_xyzw(quat_ddp_xyzw[i] / (np.linalg.norm(quat_ddp_xyzw[i]) + 1e-8))
                          for i in range(T + 1)])
    euler_sim = np.array([quat_to_euler_wxyz(sim_quat[i] / (np.linalg.norm(sim_quat[i]) + 1e-8))
                          for i in range(len(sim_t))])

    euler_ddp_deg = np.rad2deg(euler_ddp)
    euler_sim_deg = np.rad2deg(euler_sim)

    # Goal orientation
    q_goal = get_goal_quaternion()  # [x,y,z,w]
    goal_euler_deg = np.rad2deg(quat_to_euler_xyzw(q_goal))

    # ── Save output dir ──
    out_dir = os.path.join(results_dir, 'mujoco_comparison')
    os.makedirs(out_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 1: Joint Angles Comparison (2x3 grid)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    for i in range(n_q):
        ax = axes[i]
        ax.plot(times_ddp, np.rad2deg(q_ddp[:, i]), 'b-', linewidth=2, label='DDP')
        ax.plot(sim_t, np.rad2deg(sim_q[:, i]), 'r--', linewidth=1.5, label='MuJoCo')
        ax.set_ylabel('Angle (deg)')
        ax.set_title(f'Joint {i+1}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize='small')
    for i in range(3, 6):
        axes[i].set_xlabel('Time (s)')
    plt.suptitle('Joint Angles: DDP vs MuJoCo Simulation', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'joint_angles_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 2: Joint Velocities Comparison (2x3 grid)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    for i in range(n_q):
        ax = axes[i]
        ax.plot(times_ddp, qd_ddp[:, i], 'b-', linewidth=2, label='DDP')
        ax.plot(sim_t, sim_qd[:, i], 'r--', linewidth=1.5, label='MuJoCo')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title(f'Joint {i+1}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize='small')
    for i in range(3, 6):
        axes[i].set_xlabel('Time (s)')
    plt.suptitle('Joint Velocities: DDP vs MuJoCo Simulation', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'joint_velocities_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 3: Base Orientation Comparison (Euler angles)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax = axes[i]
        ax.plot(times_ddp, euler_ddp_deg[:, i], 'b-', linewidth=2, label='DDP')
        ax.plot(sim_t, euler_sim_deg[:, i], 'r--', linewidth=1.5, label='MuJoCo')
        ax.axhline(y=goal_euler_deg[i], color='g', linestyle=':', linewidth=1.5, label='Goal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
    plt.suptitle('Base Orientation (Euler): DDP vs MuJoCo Simulation', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'base_orientation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 4: Tracking Error Summary (2x2)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [0,0] Joint angle tracking error (RMS per joint over time)
    # Resample sim to DDP time steps for fair comparison
    sim_q_at_ddp = np.zeros_like(q_ddp)
    sim_qd_at_ddp = np.zeros_like(qd_ddp)
    sim_euler_at_ddp = np.zeros_like(euler_ddp_deg)
    for k in range(T + 1):
        t_target = k * dt_ddp
        idx_sim = int(t_target / dt_sim)
        if idx_sim >= len(sim_t):
            idx_sim = len(sim_t) - 1
        sim_q_at_ddp[k] = sim_q[idx_sim]
        sim_qd_at_ddp[k] = sim_qd[idx_sim]
        sim_euler_at_ddp[k] = euler_sim_deg[idx_sim]

    q_err_deg = np.rad2deg(sim_q_at_ddp - q_ddp)
    for i in range(n_q):
        axes[0, 0].plot(times_ddp, q_err_deg[:, i], label=f'J{i+1}')
    axes[0, 0].set_ylabel('Error (deg)')
    axes[0, 0].set_title('Joint Angle Tracking Error (MuJoCo - DDP)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize='small', ncol=2)

    # [0,1] Joint velocity tracking error
    qd_err = sim_qd_at_ddp - qd_ddp
    for i in range(n_q):
        axes[0, 1].plot(times_ddp, qd_err[:, i], label=f'J{i+1}')
    axes[0, 1].set_ylabel('Error (rad/s)')
    axes[0, 1].set_title('Joint Velocity Tracking Error (MuJoCo - DDP)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize='small', ncol=2)

    # [1,0] Orientation tracking error (Euler)
    orient_err = sim_euler_at_ddp - euler_ddp_deg
    for i, lbl in enumerate(labels):
        axes[1, 0].plot(times_ddp, orient_err[:, i], label=lbl)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Error (deg)')
    axes[1, 0].set_title('Base Orientation Tracking Error (MuJoCo - DDP)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize='small')

    # [1,1] RMS tracking error summary
    rms_q = np.sqrt(np.mean(q_err_deg ** 2, axis=0))
    rms_orient = np.sqrt(np.mean(orient_err ** 2, axis=0))
    x_pos = np.arange(n_q)
    bars = axes[1, 1].bar(x_pos, rms_q, color='steelblue', alpha=0.8)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'J{i+1}' for i in range(n_q)])
    axes[1, 1].set_ylabel('RMS Error (deg)')
    axes[1, 1].set_title('RMS Joint Tracking Error')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    # Add orientation RMS as text
    orient_txt = ', '.join([f'{labels[i]}: {rms_orient[i]:.3f}°' for i in range(3)])
    axes[1, 1].text(0.5, 0.95, f'Orientation RMS: {orient_txt}',
                     transform=axes[1, 1].transAxes, fontsize=9,
                     va='top', ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Tracking Error Analysis: DDP vs MuJoCo', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tracking_error_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Print summary ──
    print(f"\nTracking Error Summary:")
    print(f"  Joint angle RMS errors (deg): {np.round(rms_q, 4)}")
    print(f"  Total joint RMS: {np.sqrt(np.mean(rms_q**2)):.4f} deg")
    print(f"  Orientation RMS errors (deg): Roll={rms_orient[0]:.4f}, Pitch={rms_orient[1]:.4f}, Yaw={rms_orient[2]:.4f}")

    # Final state comparison
    final_sim_euler = euler_sim_deg[-1]
    final_ddp_euler = euler_ddp_deg[-1]
    print(f"\nFinal Base Orientation (deg):")
    print(f"  DDP:    Roll={final_ddp_euler[0]:.2f}, Pitch={final_ddp_euler[1]:.2f}, Yaw={final_ddp_euler[2]:.2f}")
    print(f"  MuJoCo: Roll={final_sim_euler[0]:.2f}, Pitch={final_sim_euler[1]:.2f}, Yaw={final_sim_euler[2]:.2f}")
    print(f"  Goal:   Roll={goal_euler_deg[0]:.2f}, Pitch={goal_euler_deg[1]:.2f}, Yaw={goal_euler_deg[2]:.2f}")

    print(f"\nPlots saved to: {out_dir}")


if __name__ == '__main__':
    main()
