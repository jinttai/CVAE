"""
Run DDP / iLQR using CasADi + SPART dynamics.

This is a CasADi counterpart of `run_ddp.py` that uses the analytical
Jacobians from CasADi.
Modified for Acceleration control and Velocity in State.
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

from ilqr.src.ddp_casadi import (
    load_robot_from_urdf,
    CasadiSpaceRobotDynamics,
    CasadiRunningCost,
    CasadiTerminalCost,
    CasadiDDP,
)
from ilqr.src.trajectory_utils import save_trajectory_csv
from physics.utils import euler_to_quaternion_np as euler_to_quaternion, quat_to_euler_np as quat_to_euler, quat_to_rot_np as quat_to_rot, geodesic_distance_so3_np as geodesic_distance_so3


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    urdf_path = os.path.join(root_dir, "assets","SC_ur10e.urdf")

    robot = load_robot_from_urdf(urdf_path)
    n_q = robot["n_q"]
    # New state dim: q(n_q) + qd(n_q) + q_base(4)
    n_x = 2 * n_q + 4

    # Time horizon
    T = 100
    dt = 0.1
    total_time = T * dt

    print(f"Time horizon: {total_time}s ({T} steps, dt={dt}s)")

    # Dynamics
    dyn = CasadiSpaceRobotDynamics(robot)

    # Initial state: joints zero, joint vels zero, base identity quaternion
    q0 = np.zeros(n_q)
    qd0 = np.zeros(n_q)
    q_base0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    x0 = np.concatenate([q0, qd0, q_base0])

    # Goal orientation & joints
    roll_deg, pitch_deg, yaw_deg = 150.0, 150.0, -15.0
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    q_goal = euler_to_quaternion(roll, pitch, yaw)
    goal_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    print(f"Initial orientation: Identity")
    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")

    # Costs
    # R_weight applies to control input u (acceleration)
    # Extract joint limits from robot model (if available)
    # The urdf2robot parser provides limits in radians (default URDF standard)
    joint_limits = None
    if 'joints' in robot:
        # Filter only moving joints (q_id != -1) and sort by q_id
        moving_joints = [j for j in robot['joints'] if j['q_id'] != -1]
        moving_joints.sort(key=lambda x: x['q_id'])
        
        if len(moving_joints) == n_q:
            jl_lower = np.array([j['limit']['lower'] for j in moving_joints])
            jl_upper = np.array([j['limit']['upper'] for j in moving_joints])
            joint_limits = (jl_lower, jl_upper)
            print("Joint limits loaded from URDF (rad):")
            print(f"  Lower: {jl_lower}")
            print(f"  Upper: {jl_upper}")
            
    running_cost = CasadiRunningCost(
        R_weight=0.01, 
        n_u=n_q, 
        joint_limits=joint_limits,
        mu_init=1.0,        # Initial ALM penalty parameter
        lambda_init=0.0,    # Initial Lagrange multipliers
    )
    
    # Terminal cost
    terminal_cost = CasadiTerminalCost(
        goal_quaternion=q_goal,
        goal_joints=goal_joints,
        orientation_weight=20.0,
        joint_weight=1.0,
        joint_vel_weight=1.0, # Penalizes terminal joint velocity (qd -> 0)
        n_u=n_q,
    )

    # Initial controls (zero acceleration)
    U0 = np.zeros((T, n_q))

    # ============================================================================
    # SWITCH: Choose between iLQR and DDP
    # ============================================================================
    USE_ILQR = True  # Set to True for iLQR (faster), False for full DDP (slower, more accurate)
    USE_ALM = True   # Set to True to use Augmented Lagrangian Method for joint limits
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
    constraint_method = "ALM" if USE_ALM else "Barrier"
    print(f"\nStarting CasADi {method_name} optimization (Acceleration Control)...")
    if USE_ILQR:
        print("Mode: iLQR (dynamics curvature terms disabled - faster)")
    else:
        print("Mode: Full DDP (dynamics curvature terms enabled - more accurate)")
    print(f"Constraint handling: {constraint_method}")
    print("-" * 50)
    start_time = time.time()
    
    if USE_ALM:
        # Use Augmented Lagrangian outer loop for constraint handling
        X_opt, U_opt, cost_history = solver.solve_alm(
            x0, U0, dt,
            alm_max_iter=10,          # Max ALM outer iterations
            constraint_tol=1e-4,       # Constraint satisfaction tolerance
            mu_increase_factor=10.0,   # Penalty increase factor
        )
    else:
        X_opt, U_opt, cost_history = solver.solve(x0, U0, dt)
    
    elapsed_time = time.time() - start_time

    print("-" * 50)
    print(f"Optimization completed!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Final   cost: {cost_history[-1]:.6f}")
    print(f"Cost reduction: {cost_history[0] - cost_history[-1]:.6f}")
    print(f"Iterations  : {len(cost_history) - 1}")
    print(f"time per iteration: {elapsed_time / (len(cost_history) - 1):.3f} seconds")

    # --- Compute final errors ---
    # State structure: [q(n_q), qd(n_q), q_base(4)]
    final_q_base = X_opt[-1, 2*n_q:]
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
    plt.title(f'CasADi {method_name} Cost History ({constraint_method})')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(results_dir, f"cost_history_casadi_{method_name.lower()}_{constraint_method.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Trajectory Overview
    time_steps = np.arange(T + 1) * dt
    
    # 3. Combined Orientation and Joint Plot (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # [0,0] Orientation (Quaternion)
    # quaternion components (x, y, z, w)
    quaternions = X_opt[:, 2*n_q:]
    
    axes[0, 0].plot(time_steps, quaternions)
    axes[0, 0].set_ylabel('Quaternion')
    axes[0, 0].set_title('Base Orientation (Quaternion)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(['qx', 'qy', 'qz', 'qw'])
    
    # Goal lines for quaternion
    # q_goal is [x, y, z, w]
    # Check if final quaternion is flipped relative to goal.
    # If dot(q_final, q_goal) < 0, they are in opposite hemispheres.
    # For visualization, we flip q_goal to match q_final's hemisphere.
    final_q = quaternions[-1]
    if np.dot(final_q, q_goal) < 0:
        q_goal_plot = -q_goal
        print("Note: Goal quaternion flipped for visualization to match final state hemisphere.")
    else:
        q_goal_plot = q_goal

    q_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(4):
        axes[0, 0].axhline(y=q_goal_plot[i], color=q_colors[i], linestyle='--', alpha=0.5)

    # [0,1] Orientation Error
    errors = []
    for t in range(T + 1):
        qt = X_opt[t, 2*n_q:]
        qt /= np.linalg.norm(qt) + 1e-8
        Rt = quat_to_rot(qt)
        err = geodesic_distance_so3(Rt, goal_R)
        errors.append(np.rad2deg(err))
    
    axes[0, 1].plot(time_steps, errors, 'r-', linewidth=2)
    axes[0, 1].set_ylabel('Error (deg)')
    axes[0, 1].set_title('Orientation Error')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # [1,0] Joint Angles
    axes[1, 0].plot(time_steps, np.rad2deg(X_opt[:, :n_q]))
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Joint Angle (deg)')
    axes[1, 0].set_title(f'Joint Angles')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend([f'J{i+1}' for i in range(n_q)], loc='upper right', fontsize='small', ncol=2)

    # [1,1] Joint Velocities
    axes[1, 1].plot(time_steps, X_opt[:, n_q:2*n_q])
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Joint Velocity (rad/s)')
    axes[1, 1].set_title('Joint Velocities')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"trajectory_casadi_{method_name.lower()}_{constraint_method.lower()}_combined.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved plots to results directory.")


if __name__ == "__main__":
    main()
