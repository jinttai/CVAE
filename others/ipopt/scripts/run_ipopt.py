"""
Run trajectory optimization using CasADi IPOPT.

Same optimization problem as ddp_acc but solved via direct transcription (NLP) + IPOPT.
State: [q(6), qd(6), q_base_quat(4)]
Control: [qdd(6)] (joint acceleration)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from others.ipopt.src.nlp_ipopt import (
    load_robot_from_urdf,
    CasadiSpaceRobotDynamics,
    solve_ipopt,
)
from ddp.src.trajectory_utils import save_trajectory_csv


def euler_to_quaternion(roll, pitch, yaw):
    """Convert ZYX Euler angles to quaternion [x, y, z, w]."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q) + 1e-8
    return q


def quat_to_euler(q):
    """Convert quaternion [x, y, z, w] to ZYX Euler angles [roll, pitch, yaw] (rad)."""
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def quat_to_rot(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


def geodesic_distance_so3(R1, R2):
    """Geodesic distance between rotation matrices (radians)."""
    trace = np.trace(R1.T @ R2)
    val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(val)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    urdf_path = os.path.join(root_dir, "assets", "SC_ur10e.urdf")

    robot = load_robot_from_urdf(urdf_path)
    n_q = robot["n_q"]
    n_x = 2 * n_q + 4

    T = 100
    dt = 0.1
    total_time = T * dt
    print(f"Time horizon: {total_time}s ({T} steps, dt={dt}s)")

    dyn = CasadiSpaceRobotDynamics(robot)

    q0 = np.zeros(n_q)
    qd0 = np.zeros(n_q)
    q_base0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    x0 = np.concatenate([q0, qd0, q_base0])

    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    q_goal = euler_to_quaternion(roll, pitch, yaw)
    goal_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")

    joint_limits = None
    if "joints" in robot:
        moving_joints = [j for j in robot["joints"] if j["q_id"] != -1]
        moving_joints.sort(key=lambda x: x["q_id"])
        if len(moving_joints) == n_q:
            jl_lower = np.array([j["limit"]["lower"] for j in moving_joints])
            jl_upper = np.array([j["limit"]["upper"] for j in moving_joints])
            joint_limits = (jl_lower, jl_upper)
            print("Joint limits loaded from URDF (rad)")
            print(f"  Lower: {jl_lower}")
            print(f"  Upper: {jl_upper}")

    print("\nStarting CasADi IPOPT optimization (Acceleration Control)...")
    print("-" * 50)
    start_time = time.time()

    X_opt, U_opt = solve_ipopt(
        dyn,
        x0,
        q_goal,
        goal_joints,
        T=T,
        dt=dt,
        R_weight=0.01,
        orientation_weight=20.0,
        joint_weight=1.0,
        joint_vel_weight=1.0,
        joint_limits=joint_limits,
    )

    elapsed_time = time.time() - start_time

    # Cost (for comparison)
    R = 0.01 * np.eye(n_q)
    run_cost = sum(U_opt[t] @ R @ U_opt[t] for t in range(T))
    goal_R = quat_to_rot(q_goal)
    final_q = X_opt[-1, 2*n_q:]
    final_q /= np.linalg.norm(final_q) + 1e-8
    final_R = quat_to_rot(final_q)
    orient_term = 20.0 * (3.0 - np.trace(final_R.T @ goal_R))
    joint_term = np.sum((X_opt[-1, :n_q] - goal_joints) ** 2)
    vel_term = np.sum(X_opt[-1, n_q:2*n_q] ** 2)
    total_cost = run_cost + orient_term + joint_term + vel_term

    print("-" * 50)
    print("Optimization completed!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Final cost: {total_cost:.6f}")

    orient_err_rad = geodesic_distance_so3(final_R, goal_R)
    orient_err_deg = np.rad2deg(orient_err_rad)
    joint_err_norm = np.linalg.norm(X_opt[-1, :n_q] - goal_joints)
    print(f"\nFinal orientation error: {orient_err_deg:.2f}°")
    print(f"Final joint error: {joint_err_norm:.4f} rad")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "trajectory_casadi_ipopt_states.npy"), X_opt)
    np.save(os.path.join(results_dir, "trajectory_casadi_ipopt_controls.npy"), U_opt)

    csv_path = os.path.join(results_dir, "trajectory_casadi_ipopt.csv")
    save_trajectory_csv(X_opt, U_opt, dt, csv_path, method_name="casadi_ipopt")

    print(f"\nSaved trajectories to: {results_dir}")

    # Plotting
    time_steps = np.arange(T + 1) * dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Base orientation as Euler angles (deg) with target
    euler_traj = np.array([quat_to_euler(X_opt[t, 2*n_q:] / (np.linalg.norm(X_opt[t, 2*n_q:]) + 1e-8))
                          for t in range(T + 1)])
    euler_deg = np.rad2deg(euler_traj)
    axes[0, 0].plot(time_steps, euler_deg[:, 0], label="Roll")
    axes[0, 0].plot(time_steps, euler_deg[:, 1], label="Pitch")
    axes[0, 0].plot(time_steps, euler_deg[:, 2], label="Yaw")
    # Target orientation (constant)
    target_euler_deg = np.rad2deg(quat_to_euler(q_goal))
    axes[0, 0].axhline(y=target_euler_deg[0], color="C0", linestyle="--", alpha=0.6, linewidth=1)
    axes[0, 0].axhline(y=target_euler_deg[1], color="C1", linestyle="--", alpha=0.6, linewidth=1)
    axes[0, 0].axhline(y=target_euler_deg[2], color="C2", linestyle="--", alpha=0.6, linewidth=1)
    axes[0, 0].set_ylabel("Euler Angle (deg)")
    axes[0, 0].set_title("Base Orientation (Euler: Roll, Pitch, Yaw) — dashed = target")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize="small")

    errors = []
    for t in range(T + 1):
        qt = X_opt[t, 2*n_q:]
        qt /= np.linalg.norm(qt) + 1e-8
        Rt = quat_to_rot(qt)
        err = geodesic_distance_so3(Rt, goal_R)
        errors.append(np.rad2deg(err))
    axes[0, 1].plot(time_steps, errors, "r-", linewidth=2)
    axes[0, 1].set_ylabel("Error (deg)")
    axes[0, 1].set_title("Orientation Error")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    axes[1, 0].plot(time_steps, np.rad2deg(X_opt[:, :n_q]))
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Joint Angle (deg)")
    axes[1, 0].set_title("Joint Angles")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend([f"J{i+1}" for i in range(n_q)], loc="upper right", fontsize="small", ncol=2)

    axes[1, 1].plot(time_steps, X_opt[:, n_q:2*n_q])
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Joint Velocity (rad/s)")
    axes[1, 1].set_title("Joint Velocities")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "trajectory_casadi_ipopt_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plots to results directory.")
    print("\nTo compute torques from this trajectory, run:")
    print("  python others/ipopt/scripts/result_to_torque.py")


if __name__ == "__main__":
    main()
