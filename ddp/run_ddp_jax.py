"""
Main execution script for DDP/iLQR trajectory optimization (JAX version).

Switch between DDP and iLQR:
- Set USE_ILQR = True for iLQR (faster, less accurate)
- Set USE_ILQR = False for full DDP (slower, more accurate)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time as time_module

# ============================================================================
# SWITCH: Choose between iLQR and DDP
# ============================================================================
USE_ILQR = True  # Set to True for iLQR, False for full DDP
# ============================================================================

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dynamics.urdf2robot_jax import urdf2robot
import ddp.dynamics_jax as dynamics
import ddp.cost_jax as cost
import ddp.solver_jax as solver
from ddp.solver_jax_fn import ilqr_solve_jit


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (ZYX) to quaternion [x, y, z, w]."""
    cr = jnp.cos(roll / 2)
    sr = jnp.sin(roll / 2)
    cp = jnp.cos(pitch / 2)
    sp = jnp.sin(pitch / 2)
    cy = jnp.cos(yaw / 2)
    sy = jnp.sin(yaw / 2)
    
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    return jnp.array([qx, qy, qz, qw])


def main():
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Load robot model
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False)
    
    # Parameters
    T = 100  # Number of time steps
    dt = 0.1  # Time step (seconds)
    total_time = T * dt  # 10 seconds
    
    print(f"Time horizon: {total_time}s ({T} steps, dt={dt}s)")
    
    # Initialize dynamics
    dynamics_model = dynamics.SpaceRobotDynamics(robot)
    
    # Initial state: [joint_angles (6), base_quaternion (4)]
    q_joints_init = jnp.zeros(6)  # All joints at zero
    q_base_init = jnp.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    initial_state = jnp.concatenate([q_joints_init, q_base_init], axis=0)  # [10]
    
    # Goal orientation (example: 15 degrees roll, pitch, yaw)
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)
    goal_quat = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
    
    # Goal joint angles (example: small angles in radians)
    goal_joints = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [6] in radians
    
    print(f"Initial orientation: Identity")
    print(f"Initial joints: {np.array(q_joints_init)}")
    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")
    print(f"Goal joints: {np.array(goal_joints)}")
    
    # Cost functions
    # NOTE: Use a slightly larger control weight than Torch version to keep
    # JAX iLQR updates numerically stable (prevents huge control jumps)
    R_weight = 0.1  # Control cost weight
    joint_limits = jnp.array(robot['joint_limits'])  # [6, 2]
    running_cost = cost.RunningCost(
        R_weight=R_weight,
        joint_limits=joint_limits,
        barrier_weight=0.1
    )
    
    terminal_cost = cost.TerminalCost(
        goal_quaternion=goal_quat,
        goal_joints=goal_joints,
        orientation_weight=20.0,  # High weight for orientation
        joint_weight=5.0  # Weight for joint goal
    )
    
    # Initial control sequence (zero initialization)
    initial_controls = jnp.zeros((T, 6))
    # Alternative: random initialization
    # key = jax.random.PRNGKey(0)
    # initial_controls = 0.1 * jax.random.normal(key, (T, 6))
    
    # Initialize DDP/iLQR solver
    method_name = "iLQR" if USE_ILQR else "DDP"
    use_full_ddp = not USE_ILQR
    
    print(f"\nStarting {method_name} optimization (JAX)...")
    print("-" * 50)
    if USE_ILQR:
        print("Mode: iLQR (dynamics curvature terms disabled - faster, JIT)")
    else:
        print("Mode: Full DDP (dynamics curvature terms enabled - more accurate, no JIT)")
    print("-" * 50)
    
    # Solve with timing
    start_time = time_module.time()

    if USE_ILQR:
        # Functional JAX iLQR solver (fully JIT-able)
        states, controls, cost_history = ilqr_solve_jit(
            initial_state=initial_state,
            initial_controls=initial_controls,
            dt=dt,
            dynamics_model=dynamics_model,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
            max_iter=50,
        )
    else:
        # Class-based DDP solver (full second-order dynamics, no JIT)
        ddp_solver = solver.DDP(
            dynamics_model=dynamics_model,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
            max_iter=50,
            tol=1e-4,
            reg_init=1.0,
            use_full_ddp=use_full_ddp,
        )
        states, controls, cost_history = ddp_solver.solve(
            initial_state=initial_state,
            initial_controls=initial_controls,
            dt=dt
        )
    elapsed_time = time_module.time() - start_time
    
    print(f"\nOptimization completed! ({method_name}, JAX)")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Initial cost: {cost_history[0]:.6f}")
    print(f"Cost reduction: {cost_history[0] - cost_history[-1]:.6f}")
    print(f"Iterations: {len(cost_history) - 1}")
    if len(cost_history) > 1:
        avg_time_per_iter = elapsed_time / (len(cost_history) - 1)
        print(f"Average time per iteration: {avg_time_per_iter:.3f} seconds")
    
    # Compute final errors
    final_quat = states[-1, 6:]
    final_quat = dynamics.normalize_quat(final_quat)
    final_R = dynamics.quat_to_rot(final_quat)
    goal_R = dynamics.quat_to_rot(goal_quat)
    orientation_error = cost.geodesic_distance_so3(final_R, goal_R)
    print(f"\nFinal orientation error: {np.rad2deg(float(orientation_error)):.2f}°")
    
    # Compute final joint error
    final_joints = states[-1, :6]
    joint_error = final_joints - goal_joints
    joint_error_norm = jnp.linalg.norm(joint_error)
    print(f"Final joint error: {float(joint_error_norm):.4f} rad (L2 norm)")
    print(f"Final joint angles: {np.array(final_joints)}")
    print(f"Goal joint angles: {np.array(goal_joints)}")
    
    # Save results (convert to numpy for saving)
    save_dir = os.path.dirname(__file__)
    os.makedirs(save_dir, exist_ok=True)
    
    results_filename = f"results_jax_{method_name.lower()}.npz"
    results_path = os.path.join(save_dir, results_filename)
    np.savez(
        results_path,
        states=np.array(states),
        controls=np.array(controls),
        cost_history=np.array(cost_history),
        initial_state=np.array(initial_state),
        goal_quaternion=np.array(goal_quat),
        dt=dt,
        T=T,
        method=method_name,
        use_full_ddp=use_full_ddp
    )
    print(f"\nResults saved to: {results_path}")
    
    # Plot cost history
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.title(f'{method_name} Cost History (JAX)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    cost_history_filename = f"cost_history_jax_{method_name.lower()}.png"
    cost_history_path = os.path.join(save_dir, cost_history_filename)
    plt.savefig(cost_history_path, dpi=150, bbox_inches='tight')
    print(f"Cost history plot saved to: {cost_history_path}")
    
    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time = np.arange(T + 1) * dt
    
    # Convert to numpy for visualization
    states_np = np.array(states)
    controls_np = np.array(controls)
    goal_quat_np = np.array(goal_quat)
    
    # Joint angles
    axes[0, 0].plot(time, states_np[:, :6])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Joint Angle (rad)')
    axes[0, 0].set_title('Joint Angles')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend([f'Joint {i+1}' for i in range(6)])
    
    # Joint velocities (controls)
    axes[0, 1].plot(time[:-1], controls_np)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Joint Velocity (rad/s)')
    axes[0, 1].set_title('Joint Velocities (Controls)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend([f'Joint {i+1}' for i in range(6)])
    
    # Base quaternion
    axes[1, 0].plot(time, states_np[:, 6:])
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Quaternion Component')
    axes[1, 0].set_title('Base Orientation (Quaternion)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(['qx', 'qy', 'qz', 'qw'])
    axes[1, 0].axhline(y=goal_quat_np[0], color='r', linestyle='--', alpha=0.5, label='Goal qx')
    axes[1, 0].axhline(y=goal_quat_np[1], color='g', linestyle='--', alpha=0.5, label='Goal qy')
    axes[1, 0].axhline(y=goal_quat_np[2], color='b', linestyle='--', alpha=0.5, label='Goal qz')
    axes[1, 0].axhline(y=goal_quat_np[3], color='m', linestyle='--', alpha=0.5, label='Goal qw')
    
    # Orientation error over time
    orientation_errors = []
    for t in range(T + 1):
        q_t = dynamics.normalize_quat(states[t, 6:])
        R_t = dynamics.quat_to_rot(q_t)
        err = cost.geodesic_distance_so3(R_t, goal_R)
        orientation_errors.append(np.rad2deg(float(err)))
    
    axes[1, 1].plot(time, orientation_errors, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Orientation Error (deg)')
    axes[1, 1].set_title('Orientation Error vs Goal')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    trajectory_filename = f"trajectory_jax_{method_name.lower()}.png"
    trajectory_path = os.path.join(save_dir, trajectory_filename)
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory plot saved to: {trajectory_path}")
    
    print("\n" + "=" * 50)
    print(f"{method_name} Optimization Complete! (JAX)")
    print("=" * 50)


if __name__ == "__main__":
    main()

