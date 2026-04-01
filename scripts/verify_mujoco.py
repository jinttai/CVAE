"""
MuJoCo 검증: SPART dynamics와 MuJoCo dynamics가 동일한지 비교.

3단계 검증:
  1. Mass matrix 비교 (zero config)
  2. Step-by-step acceleration 비교 (모든 timestep)
  3. Computed-torque control로 trajectory 추종 → base orientation 비교
"""

import os
import sys
import math

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.utils.runtime_env import configure_windows_runtime

configure_windows_runtime()

import torch
import numpy as np
import mujoco

import src.dynamics.spart_functions_torch as spart_fn
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from torch.func import vmap


# ── Config ──────────────────────────────────────────────────────────────
NUM_WAYPOINTS = 3
TOTAL_TIME = 10.0
MJCF_PATH = os.path.join(ROOT_DIR, "assets/spacerobot_urdf_match.xml")
RESULTS_PATH = os.path.join(ROOT_DIR, "outputs/plots/torque_direct/results.npz")


def euler_to_quaternion_np(roll, pitch, yaw):
    """Euler (rad) -> quaternion [x,y,z,w]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quat_angle_error(q1, q2):
    """Angle error in degrees between two quaternions [x,y,z,w]."""
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot) * 180.0 / np.pi


def mujoco_quat_to_xyzw(q_wxyz):
    """MuJoCo [w,x,y,z] -> our convention [x,y,z,w]."""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def rot_to_euler(R):
    """Rotation matrix -> Euler angles [yaw, pitch, roll] (ZYX)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.array([yaw, pitch, roll])


def quat_to_rotmat(q_xyzw):
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Mass matrix comparison
# ═══════════════════════════════════════════════════════════════════════

def compare_mass_matrices(robot, mjcf_path):
    """Compare SPART vs MuJoCo generalized mass matrix at zero config."""
    n_q = robot['n_q']

    # SPART mass matrix
    R0 = torch.eye(3)
    r0 = torch.zeros(3)
    qm = torch.zeros(n_q)
    RJ, RL, rJ, rL, e, g = spart_fn.kinematics(R0, r0, qm, robot)
    Bij, Bi0, P0, pm = spart_fn.diff_kinematics(R0, r0, rL, e, g, robot)
    I0, Im = spart_fn.inertia_projection(R0, RL, robot)
    M0_t, Mm_t = spart_fn.mass_composite_body(I0, Im, Bij, Bi0, robot)
    H0, H0m, Hm = spart_fn.generalized_inertia_matrix(
        M0_t, Mm_t, Bij, Bi0, P0, pm, robot)

    H_spart = torch.zeros(6 + n_q, 6 + n_q)
    H_spart[:6, :6] = H0
    H_spart[:6, 6:] = H0m
    H_spart[6:, :6] = H0m.T
    H_spart[6:, 6:] = Hm

    # Reorder SPART [ang, lin, joints] -> MuJoCo [lin, ang, joints]
    reorder = [3, 4, 5, 0, 1, 2] + list(range(6, 6 + n_q))
    H_spart_reord = H_spart.numpy()[np.ix_(reorder, reorder)]

    # MuJoCo mass matrix
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    nv = model.nv
    M_mj = np.zeros((nv, nv), dtype=np.float64)
    mujoco.mj_fullM(model, M_mj, data.qM)

    diff = H_spart_reord - M_mj
    max_diff = np.abs(diff).max()
    frob_norm = np.linalg.norm(diff)

    return max_diff, frob_norm


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Step-by-step acceleration comparison
# ═══════════════════════════════════════════════════════════════════════

def compare_accelerations(physics, robot, mjcf_path, q_np, qd_np, tau_np):
    """
    At each SPART timestep, set MuJoCo to the same state and apply the same
    torque, then compare joint accelerations.
    """
    runtime = physics._get_runtime_constants(torch.float32)
    R0 = runtime['R0']
    r0 = runtime['r0']
    damping = runtime['damping_term']

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # SPART q_ddot
    q_ddot = np.zeros_like(qd_np)
    q_ddot[:-1] = (qd_np[1:] - qd_np[:-1]) / physics.dt
    q_ddot[-1] = q_ddot[-2]

    qacc_diffs = []
    for t in range(len(q_np)):
        qm_t = torch.tensor(q_np[t], dtype=torch.float32)
        qd_t = torch.tensor(qd_np[t], dtype=torch.float32)

        # Compute SPART base velocity (momentum conservation)
        RJ, RL, rJ, rL, e, g = spart_fn.kinematics(R0, r0, qm_t, robot)
        Bij, Bi0, P0, pm = spart_fn.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart_fn.inertia_projection(R0, RL, robot)
        M0_t, Mm_t = spart_fn.mass_composite_body(I0, Im, Bij, Bi0, robot)
        H0, H0m, Hm = spart_fn.generalized_inertia_matrix(
            M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
        u0 = -(torch.linalg.inv(H0 + damping) @ H0m @ qd_t).numpy()

        # Set MuJoCo state
        mujoco.mj_resetData(model, data)
        data.qpos[3] = 1.0
        data.qpos[7:13] = q_np[t]
        data.qvel[0:3] = u0[3:6]  # linear vel
        data.qvel[3:6] = u0[0:3]  # angular vel
        data.qvel[6:12] = qd_np[t]
        mujoco.mj_forward(model, data)

        data.ctrl[:6] = tau_np[t]
        mujoco.mj_step(model, data)

        qacc_diff = np.linalg.norm(q_ddot[t] - data.qacc[6:12])
        qacc_diffs.append(qacc_diff)

    return np.array(qacc_diffs)


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Computed-torque trajectory tracking
# ═══════════════════════════════════════════════════════════════════════

def run_spart_sim(physics, waypoints, q0_start_quat):
    """Run SPART simulation: trajectory + base orientation."""
    with torch.no_grad():
        q_traj, qd_traj = physics.generate_trajectory(waypoints.unsqueeze(0))
        q_t = q_traj[0]
        qd_t = qd_traj[0]
        torques = physics.compute_torques(q_t, qd_t)

        all_wb = vmap(physics._compute_wb)(q_t, qd_t)
        R_curr = physics._quat_to_rot(q0_start_quat)
        dt = physics.dt

        base_quats = [physics._rot_to_quat(R_curr).cpu().numpy()]
        for t in range(physics.num_steps):
            R_delta = physics._rot_from_omega(all_wb[t], dt)
            R_curr = R_curr @ R_delta
            base_quats.append(physics._rot_to_quat(R_curr).cpu().numpy())

    return (np.stack(base_quats, axis=0),
            q_t.cpu().numpy(), qd_t.cpu().numpy(), torques.cpu().numpy())


def run_mujoco_computed_torque(physics, robot, mjcf_path, q_np, qd_np):
    """
    MuJoCo simulation with computed-torque control.

    At each MuJoCo substep, recomputes SPART inverse dynamics using the
    actual MuJoCo joint state + desired acceleration from interpolated reference.
    This eliminates integration-scheme drift.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    dt_spart = physics.dt
    dt_mj = model.opt.timestep
    substeps = int(round(dt_spart / dt_mj))
    num_steps = len(q_np)
    n_joints = 6

    # Precompute q_ddot
    q_ddot = np.zeros_like(qd_np)
    q_ddot[:-1] = (qd_np[1:] - qd_np[:-1]) / dt_spart
    q_ddot[-1] = q_ddot[-2]

    # Initialize
    mujoco.mj_resetData(model, data)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    mj_joint_pos = np.zeros((num_steps + 1, n_joints))
    mj_joint_vel = np.zeros((num_steps + 1, n_joints))
    mj_base_quat = np.zeros((num_steps + 1, 4))
    mj_base_pos = np.zeros((num_steps + 1, 3))

    mj_joint_pos[0] = data.qpos[7:13]
    mj_joint_vel[0] = data.qvel[6:12]
    mj_base_quat[0] = mujoco_quat_to_xyzw(data.qpos[3:7])
    mj_base_pos[0] = data.qpos[0:3]

    runtime = physics._get_runtime_constants(torch.float32)

    for t in range(num_steps):
        for sub in range(substeps):
            # Interpolate reference
            alpha = sub / substeps
            if t + 1 < num_steps:
                qm_ref = (1 - alpha) * q_np[t] + alpha * q_np[t + 1]
                qd_ref = (1 - alpha) * qd_np[t] + alpha * qd_np[t + 1]
                qdd_ref = (1 - alpha) * q_ddot[t] + alpha * q_ddot[t + 1]
            else:
                qm_ref = q_np[t]
                qd_ref = qd_np[t]
                qdd_ref = q_ddot[t]

            # Actual MuJoCo state
            qm_actual = data.qpos[7:13].copy()
            qd_actual = data.qvel[6:12].copy()

            # Desired acceleration with PD correction on joint errors
            kp_ct, kd_ct = 1000.0, 100.0
            qdd_desired = qdd_ref + kp_ct * (qm_ref - qm_actual) + kd_ct * (qd_ref - qd_actual)

            # Compute torque using SPART inverse dynamics at actual state
            qm_t = torch.tensor(qm_actual, dtype=torch.float32)
            qd_torch = torch.tensor(qd_actual, dtype=torch.float32)
            qdd_torch = torch.tensor(qdd_desired, dtype=torch.float32)
            tau = physics._compute_tau_single_step(qm_t, qd_torch, qdd_torch, runtime)

            data.ctrl[:n_joints] = tau.numpy()
            mujoco.mj_step(model, data)

        mj_joint_pos[t + 1] = data.qpos[7:13]
        mj_joint_vel[t + 1] = data.qvel[6:12]
        mj_base_quat[t + 1] = mujoco_quat_to_xyzw(data.qpos[3:7])
        mj_base_pos[t + 1] = data.qpos[0:3]

    return mj_joint_pos, mj_joint_vel, mj_base_quat, mj_base_pos


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison(spart_quat, mj_quat, spart_joints, mj_joints,
                    spart_torques, mj_joint_vel, spart_qd, save_dir,
                    mass_diff, qacc_diffs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_steps = spart_joints.shape[0]
    n_q = spart_joints.shape[1]
    t_full = np.linspace(0, TOTAL_TIME, num_steps + 1)
    t_steps = np.linspace(0, TOTAL_TIME, num_steps)

    # ── Figure 1: Validation summary ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.text(0.5, 0.7, f"Mass Matrix\nMax Diff: {mass_diff[0]:.2e}\nFrob Norm: {mass_diff[1]:.2e}",
            transform=ax.transAxes, ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8))
    ax.text(0.5, 0.25, "PASS" if mass_diff[0] < 1e-3 else "FAIL",
            transform=ax.transAxes, ha='center', va='center', fontsize=20, fontweight='bold',
            color='green' if mass_diff[0] < 1e-3 else 'red')
    ax.set_title("Phase 1: Mass Matrix")
    ax.axis('off')

    ax = axes[1]
    ax.plot(t_steps, qacc_diffs, linewidth=1.5, color="#4e79a7")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||qacc_SPART - qacc_MJ||")
    ax.set_title(f"Phase 2: Acceleration (max={qacc_diffs.max():.2e})")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    angle_errors = np.array([quat_angle_error(spart_quat[i], mj_quat[i])
                             for i in range(num_steps + 1)])
    ax = axes[2]
    ax.plot(t_full, angle_errors, linewidth=1.5, color="#e15759")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title(f"Phase 3: Base Orient (final={angle_errors[-1]:.2f} deg)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("SPART vs MuJoCo Dynamics Verification", fontsize=14)
    fig.tight_layout()
    p0 = os.path.join(save_dir, "validation_summary.png")
    fig.savefig(p0, dpi=150)
    plt.close(fig)

    # ── Figure 2: Base orientation comparison ──
    spart_euler = np.array([rot_to_euler(quat_to_rotmat(q)) for q in spart_quat])
    mj_euler = np.array([rot_to_euler(quat_to_rotmat(q)) for q in mj_quat])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    for i in range(3):
        ax = axes[i]
        ax.plot(t_full, np.degrees(spart_euler[:, i]), label="SPART", linewidth=1.5)
        ax.plot(t_full, np.degrees(mj_euler[:, i]), label="MuJoCo", linewidth=1.5, linestyle="--")
        ax.set_ylabel(f"{labels[i]} (deg)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Base Orientation: SPART vs MuJoCo (Computed-Torque)", fontsize=13)
    fig.tight_layout()
    p1 = os.path.join(save_dir, "base_orientation_comparison.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ── Figure 3: Joint position comparison ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j in range(n_q):
        ax = axes[j // 3, j % 3]
        ax.plot(t_steps, spart_joints[:, j], label="SPART (ref)", linewidth=1.5)
        ax.plot(t_full, mj_joints[:, j], label="MuJoCo", linewidth=1.5, linestyle="--")
        ax.set_title(f"J{j+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if j >= 3:
            ax.set_xlabel("Time (s)")
    fig.suptitle("Joint Positions: SPART reference vs MuJoCo actual", fontsize=13)
    fig.tight_layout()
    p2 = os.path.join(save_dir, "joint_pos_comparison.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ── Figure 4: Joint velocity comparison ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j in range(n_q):
        ax = axes[j // 3, j % 3]
        ax.plot(t_steps, spart_qd[:, j], label="SPART (ref)", linewidth=1.5)
        ax.plot(t_full, mj_joint_vel[:, j], label="MuJoCo", linewidth=1.5, linestyle="--")
        ax.set_title(f"J{j+1} vel")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if j >= 3:
            ax.set_xlabel("Time (s)")
    fig.suptitle("Joint Velocities: SPART reference vs MuJoCo actual", fontsize=13)
    fig.tight_layout()
    p3 = os.path.join(save_dir, "joint_vel_comparison.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ── Figure 5: Error analysis ──
    joint_errors = np.array([np.linalg.norm(spart_joints[t] - mj_joints[t + 1])
                             for t in range(num_steps)])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    tau_norm = np.linalg.norm(spart_torques, axis=1)
    ax.plot(t_steps, tau_norm, linewidth=1.5, color="#59a14f")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||torque|| (Nm)")
    ax.set_title("SPART Torque Norm")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_full, angle_errors, linewidth=1.5, color="#e15759")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title("Base Orientation Error")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_steps, joint_errors, linewidth=1.5, color="#4e79a7")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Error (rad)")
    ax.set_title("Joint Position Error (L2)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p4 = os.path.join(save_dir, "error_analysis.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    return p0, p1, p2, p3, p4


def main():
    device = "cpu"
    print(f"Device: {device}")

    # ── Load robot & physics ──
    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    # ── Load optimized waypoints ──
    if os.path.exists(RESULTS_PATH):
        data = np.load(RESULTS_PATH)
        waypoints_np = data["waypoints_optimized"]
        print(f"Loaded waypoints from {RESULTS_PATH}")
    else:
        print(f"Results not found at {RESULTS_PATH}, using random waypoints")
        waypoints_np = np.random.randn(NUM_WAYPOINTS * robot["n_q"]) * 0.1

    waypoints = torch.tensor(waypoints_np, dtype=torch.float32, device=device)

    # ── Goal orientation ──
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    q0_goal = euler_to_quaternion_np(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
    q0_start = np.array([0.0, 0.0, 0.0, 1.0])
    q0_start_torch = torch.tensor(q0_start, dtype=torch.float32, device=device)

    print(f"Goal: roll={roll_deg}, pitch={pitch_deg}, yaw={yaw_deg} deg")

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Mass matrix comparison
    # ════════════════════════════════════════════════════════════════
    print("\n[Phase 1] Mass matrix comparison...")
    max_diff, frob_norm = compare_mass_matrices(robot, MJCF_PATH)
    print(f"  Max element diff: {max_diff:.2e}")
    print(f"  Frobenius norm:   {frob_norm:.2e}")
    print(f"  Result: {'PASS' if max_diff < 1e-3 else 'FAIL'}")

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Step-by-step acceleration comparison
    # ════════════════════════════════════════════════════════════════
    print("\n[Phase 2] Running SPART simulation...")
    spart_quat, q_traj_np, qd_traj_np, torques_np = run_spart_sim(
        physics, waypoints, q0_start_torch)

    spart_final_quat = spart_quat[-1]
    spart_angle_err = quat_angle_error(spart_final_quat, q0_goal)
    print(f"  SPART final angle error to goal: {spart_angle_err:.2f} deg")
    print(f"  Torque cost: {np.sum(torques_np**2):.1f}")

    print("\n[Phase 2] Comparing accelerations at each timestep...")
    with torch.no_grad():
        qacc_diffs = compare_accelerations(
            physics, robot, MJCF_PATH, q_traj_np, qd_traj_np, torques_np)
    print(f"  Max qacc diff:  {qacc_diffs.max():.2e}")
    print(f"  Mean qacc diff: {qacc_diffs.mean():.2e}")
    print(f"  Result: {'PASS' if qacc_diffs.max() < 1e-3 else 'FAIL'}")

    # ════════════════════════════════════════════════════════════════
    # Phase 3: MuJoCo computed-torque tracking
    # ════════════════════════════════════════════════════════════════
    print("\n[Phase 3] Running MuJoCo with computed-torque control...")
    with torch.no_grad():
        mj_joints, mj_joint_vel, mj_quat, mj_pos = run_mujoco_computed_torque(
            physics, robot, MJCF_PATH, q_traj_np, qd_traj_np)

    mj_final_quat = mj_quat[-1]
    mj_angle_err = quat_angle_error(mj_final_quat, q0_goal)
    cross_err = quat_angle_error(spart_final_quat, mj_final_quat)
    joint_err = np.linalg.norm(q_traj_np[-1] - mj_joints[-1])

    print(f"  MuJoCo final angle error to goal: {mj_angle_err:.2f} deg")
    print(f"\n{'='*60}")
    print(f"SPART vs MuJoCo comparison:")
    print(f"  Base orientation diff:  {cross_err:.4f} deg")
    print(f"  Final joint pos diff:   {joint_err:.6f} rad")
    print(f"  SPART -> goal:          {spart_angle_err:.2f} deg")
    print(f"  MuJoCo -> goal:         {mj_angle_err:.2f} deg")
    print(f"{'='*60}")

    # ── Plots ──
    save_dir = os.path.join(ROOT_DIR, "outputs/plots/mujoco_verify")
    os.makedirs(save_dir, exist_ok=True)

    try:
        paths = plot_comparison(
            spart_quat, mj_quat, q_traj_np, mj_joints,
            torques_np, mj_joint_vel, qd_traj_np, save_dir,
            (max_diff, frob_norm), qacc_diffs)
        for p in paths:
            print(f"  -> {p}")
    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
