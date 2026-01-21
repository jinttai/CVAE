"""
Main execution script for DDP/iLQR trajectory optimization.

Switch between DDP and iLQR:
- Set USE_ILQR = True for iLQR (faster, less accurate)
- Set USE_ILQR = False for full DDP (slower, more accurate)
"""

import torch
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

from src.dynamics.urdf2robot_torch import urdf2robot
import ddp.dynamics_torch as dynamics
import ddp.cost as cost
import ddp.solver as solver


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (ZYX) to quaternion [x, y, z, w]."""
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    return torch.stack([qx, qy, qz, qw])


def main():
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load robot model
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    
    # Parameters
    T = 100  # Number of time steps
    dt = 0.1  # Time step (seconds)
    total_time = T * dt  # 10 seconds
    
    print(f"Time horizon: {total_time}s ({T} steps, dt={dt}s)")
    
    # Initialize dynamics
    dynamics_model = dynamics.SpaceRobotDynamics(robot, device=device)
    
    # Initial state: [joint_angles (6), base_quaternion (4)]
    q_joints_init = torch.zeros(6, device=device)  # All joints at zero
    q_base_init = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # Identity quaternion
    initial_state = torch.cat([q_joints_init, q_base_init], dim=0)  # [10]
    
    # Goal orientation (example: 15 degrees roll, pitch, yaw)
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll_rad = torch.tensor(np.deg2rad(roll_deg), device=device)
    pitch_rad = torch.tensor(np.deg2rad(pitch_deg), device=device)
    yaw_rad = torch.tensor(np.deg2rad(yaw_deg), device=device)
    goal_quat = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
    
    # Goal joint angles (example: small angles in radians)
    # You can modify these to set desired joint positions
    goal_joints = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)  # [6] in radians
    
    print(f"Initial orientation: Identity")
    print(f"Initial joints: {q_joints_init.cpu().numpy()}")
    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")
    print(f"Goal joints: {goal_joints.cpu().numpy()}")
    
    # Cost functions
    R_weight = 0.01  # Control cost weight
    joint_limits = robot['joint_limits'].to(device)  # [6, 2]
    running_cost = cost.RunningCost(
        R_weight=R_weight,
        joint_limits=joint_limits,
        barrier_weight=0.1,
        device=device
    )
    
    terminal_cost = cost.TerminalCost(
        goal_quaternion=goal_quat,
        goal_joints=goal_joints,
        orientation_weight=20.0,  # High weight for orientation
        joint_weight=5.0,  # Weight for joint goal
        device=device
    )
    
    # Initial control sequence (zero initialization)
    initial_controls = torch.zeros(T, 6, device=device)
    # Alternative: random initialization
    # initial_controls = 0.1 * torch.randn(T, 6, device=device)
    
    # Initialize DDP/iLQR solver
    method_name = "iLQR" if USE_ILQR else "DDP"
    use_full_ddp = not USE_ILQR
    
    ddp_solver = solver.DDP(
        dynamics_model=dynamics_model,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        max_iter=50,
        tol=1e-4,
        reg_init=1.0,
        use_full_ddp=use_full_ddp,
        terminal_control_weight=1.0,  # Final joint velocity penalty weight
        device=device
    )
    
    print(f"\nStarting {method_name} optimization...")
    print("-" * 50)
    if USE_ILQR:
        print("Mode: iLQR (dynamics curvature terms disabled - faster)")
    else:
        print("Mode: Full DDP (dynamics curvature terms enabled - more accurate)")
    print("-" * 50)
    
    # Solve with timing
    start_time = time_module.time()
    states, controls, cost_history = ddp_solver.solve(
        initial_state=initial_state,
        initial_controls=initial_controls,
        dt=dt
    )
    elapsed_time = time_module.time() - start_time
    
    print(f"\nOptimization completed!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Initial cost: {cost_history[0]:.6f}")
    print(f"Cost reduction: {cost_history[0] - cost_history[-1]:.6f}")
    print(f"Iterations: {len(cost_history) - 1}")
    if len(cost_history) > 1:
        avg_time_per_iter = elapsed_time / (len(cost_history) - 1)
        print(f"Average time per iteration: {avg_time_per_iter:.3f} seconds")
    
    # Compute final errors
    final_quat = states[-1, 6:].detach()
    final_quat = dynamics.normalize_quat(final_quat)
    final_R = dynamics.quat_to_rot(final_quat)
    goal_R = dynamics.quat_to_rot(goal_quat.detach())
    orientation_error = cost.geodesic_distance_so3(final_R, goal_R)
    print(f"\nFinal orientation error: {torch.rad2deg(orientation_error.detach()):.2f}°")
    
    # Compute final joint error
    final_joints = states[-1, :6].detach()
    joint_error = final_joints - goal_joints
    joint_error_norm = torch.linalg.norm(joint_error)
    print(f"Final joint error: {joint_error_norm.item():.4f} rad (L2 norm)")
    print(f"Final joint angles: {final_joints.cpu().numpy()}")
    print(f"Goal joint angles: {goal_joints.cpu().numpy()}")
    
    # Save results
    save_dir = os.path.dirname(__file__)
    os.makedirs(save_dir, exist_ok=True)
    
    results_filename = f"results_{method_name.lower()}.pt"
    results_path = os.path.join(save_dir, results_filename)
    torch.save({
        'states': states.cpu(),
        'controls': controls.cpu(),
        'cost_history': cost_history,
        'initial_state': initial_state.cpu(),
        'goal_quaternion': goal_quat.cpu(),
        'dt': dt,
        'T': T,
        'method': method_name,
        'use_full_ddp': use_full_ddp
    }, results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Plot cost history
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.title(f'{method_name} Cost History', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    cost_history_filename = f"cost_history_{method_name.lower()}.png"
    cost_history_path = os.path.join(save_dir, cost_history_filename)
    plt.savefig(cost_history_path, dpi=150, bbox_inches='tight')
    print(f"Cost history plot saved to: {cost_history_path}")
    
    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time = np.arange(T + 1) * dt
    
    # Detach tensors from computation graph for visualization
    states_np = states.detach().cpu().numpy()
    controls_np = controls.detach().cpu().numpy()
    goal_quat_np = goal_quat.detach().cpu().numpy()
    
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
    goal_R_detached = goal_R.detach()
    for t in range(T + 1):
        q_t = dynamics.normalize_quat(states[t, 6:].detach())
        R_t = dynamics.quat_to_rot(q_t)
        err = cost.geodesic_distance_so3(R_t, goal_R_detached)
        orientation_errors.append(torch.rad2deg(err.detach()).item())
    
    axes[1, 1].plot(time, orientation_errors, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Orientation Error (deg)')
    axes[1, 1].set_title('Orientation Error vs Goal')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    trajectory_filename = f"trajectory_{method_name.lower()}.png"
    trajectory_path = os.path.join(save_dir, trajectory_filename)
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory plot saved to: {trajectory_path}")
    
    print("\n" + "=" * 50)
    print(f"{method_name} Optimization Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

