"""
Run DDP / iLQR using CasADi + SPART dynamics.

This is a CasADi counterpart of `run_ddp.py` that uses the analytical
Jacobians from CasADi instead of PyTorch autograd.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path based on current file location
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ddp.src.ddp_casadi import (
    load_robot_from_urdf,
    CasadiSpaceRobotDynamics,
    CasadiRunningCost,
    CasadiTerminalCost,
    CasadiDDP,
)
from ddp.src.trajectory_utils import save_trajectory_csv


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to quaternion [x, y, z, w].
    roll, pitch, yaw are in radians (floats).
    """
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q) + 1e-8
    return q


def quat_to_euler(q):
    """
    Convert quaternion [x, y, z, w] to ZYX Euler angles [roll, pitch, yaw].
    Returns angles in radians.
    """
    x, y, z, w = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def quat_to_rot(q):
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix (NumPy).
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])


def geodesic_distance_so3(R1, R2):
    """
    Compute geodesic distance between two rotation matrices.
    theta = arccos((trace(R1^T @ R2) - 1) / 2)
    """
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    val = (trace - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    urdf_path = os.path.join(root_dir, "assets","SC_ur10e.urdf")

    robot = load_robot_from_urdf(urdf_path)
    n_q = robot["n_q"]
    n_x = n_q + 4

    # Time horizon
    T = 100
    dt = 0.1
    total_time = T * dt

    print(f"Time horizon: {total_time}s ({T} steps, dt={dt}s)")

    # Dynamics
    dyn = CasadiSpaceRobotDynamics(robot)

    # Initial state: joints zero, base identity quaternion
    q0 = np.zeros(n_q)
    q_base0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    x0 = np.concatenate([q0, q_base0])

    # Goal orientation & joints
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    q_goal = euler_to_quaternion(roll, pitch, yaw)
    goal_joints = np.zeros(n_q)

    print(f"Initial orientation: Identity")
    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")

    # Costs
    running_cost = CasadiRunningCost(R_weight=0.01, n_u=n_q)
    terminal_cost = CasadiTerminalCost(
        goal_quaternion=q_goal,
        goal_joints=goal_joints,
        orientation_weight=20.0,
        joint_weight=1.0,
        velocity_weight=0.5,
        n_u=n_q,
    )

    # Initial controls (zero)
    U0 = np.zeros((T, n_q))

    # ============================================================================
    # SWITCH: Choose between iLQR and DDP
    # ============================================================================
    USE_ILQR = True  # Set to True for iLQR (faster), False for full DDP (slower, more accurate)
    # ============================================================================

    solver = CasadiDDP(
        dynamics_model=dyn,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        max_iter=500,
        tol=1e-4,
        use_full_ddp=not USE_ILQR,
    )

    method_name = "iLQR" if USE_ILQR else "DDP"
    print(f"\nStarting CasADi {method_name} optimization...")
    if USE_ILQR:
        print("Mode: iLQR (dynamics curvature terms disabled - faster)")
    else:
        print("Mode: Full DDP (dynamics curvature terms enabled - more accurate)")
    print("-" * 50)
    start_time = time.time()
    
    X_opt, U_opt, cost_history = solver.solve(x0, U0, dt)
    
    elapsed_time = time.time() - start_time

    print("-" * 50)
    print(f"Optimization completed!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Final   cost: {cost_history[-1]:.6f}")
    print(f"Cost reduction: {cost_history[0] - cost_history[-1]:.6f}")
    print(f"Iterations  : {len(cost_history) - 1}")

    # --- Compute final errors ---
    final_q_base = X_opt[-1, n_q:]
    final_q_base /= np.linalg.norm(final_q_base) + 1e-8
    final_R = quat_to_rot(final_q_base)
    goal_R = quat_to_rot(q_goal)
    
    orient_err_rad = geodesic_distance_so3(final_R, goal_R)
    orient_err_deg = np.rad2deg(orient_err_rad)
    
    final_joints = X_opt[-1, :n_q]
    joint_err_norm = np.linalg.norm(final_joints - goal_joints)

    print(f"\nFinal orientation error: {orient_err_deg:.2f}°")
    print(f"Final joint error: {joint_err_norm:.4f} rad")

    # --- Save Results ---
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save NPY
    np.save(os.path.join(results_dir, "trajectory_casadi_ddp_states.npy"), X_opt)
    np.save(os.path.join(results_dir, "trajectory_casadi_ddp_controls.npy"), U_opt)
    np.save(os.path.join(results_dir, "cost_history_casadi_ddp.npy"), np.array(cost_history))

    # Save CSV
    csv_path = os.path.join(results_dir, f"trajectory_casadi_{method_name.lower()}.csv")
    save_trajectory_csv(X_opt, U_opt, dt, csv_path, method_name=f"casadi_{method_name.lower()}")

    print(f"\nSaved trajectories and cost history to: {results_dir}")

    # --- Plotting ---
    # 1. Cost History
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.title(f'CasADi {method_name} Cost History')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(results_dir, f"cost_history_casadi_{method_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Trajectory Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_steps = np.arange(T + 1) * dt

    # Joint angles
    axes[0, 0].plot(time_steps, X_opt[:, :n_q])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Joint Angle (rad)')
    axes[0, 0].set_title(f'Joint Angles ({method_name})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend([f'J{i+1}' for i in range(n_q)])

    # Controls
    axes[0, 1].plot(time_steps[:-1], U_opt)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Joint Velocity (rad/s)')
    axes[0, 1].set_title('Controls (Joint Velocities)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend([f'J{i+1}' for i in range(n_q)])

    # Quaternion to Euler
    euler_angles = []
    for t in range(T + 1):
        qt = X_opt[t, n_q:]
        qt /= np.linalg.norm(qt) + 1e-8
        euler_angles.append(np.rad2deg(quat_to_euler(qt)))
    euler_angles = np.array(euler_angles)

    axes[1, 0].plot(time_steps, euler_angles)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Euler Angles (deg)')
    axes[1, 0].set_title('Base Orientation (Euler)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(['Roll', 'Pitch', 'Yaw'])
    # Goal lines
    goals = [roll_deg, pitch_deg, yaw_deg]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['Goal Roll', 'Goal Pitch', 'Goal Yaw']
    for i, val in enumerate(goals):
        axes[1, 0].axhline(y=val, color=colors[i], linestyle='--', alpha=0.5, label=labels[i])


    # Orientation Error
    errors = []
    for t in range(T + 1):
        qt = X_opt[t, n_q:]
        qt /= np.linalg.norm(qt) + 1e-8
        Rt = quat_to_rot(qt)
        err = geodesic_distance_so3(Rt, goal_R)
        errors.append(np.rad2deg(err))
    
    axes[1, 1].plot(time_steps, errors, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (deg)')
    axes[1, 1].set_title('Orientation Error')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"trajectory_casadi_{method_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved plots to results directory.")


if __name__ == "__main__":
    main()
