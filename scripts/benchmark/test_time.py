import torch
import time
import os
import sys
from typing import Tuple

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    Using ZYX convention (yaw around Z, pitch around Y, roll around X)
    """
    # Half angles
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    
    # Quaternion components
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    return torch.stack([qx, qy, qz, qw], dim=-1)


def generate_random_goal(max_angle_deg: float = 30.0, device: str = "cpu") -> torch.Tensor:
    """
    Generate random quaternion goal from Euler angles within specified range
    """
    import math
    max_angle_rad = math.radians(max_angle_deg)
    
    # Generate random Euler angles in [-max_angle_deg, max_angle_deg]
    roll = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    
    # Convert to quaternion
    roll_t = torch.tensor([roll], device=device)
    pitch_t = torch.tensor([pitch], device=device)
    yaw_t = torch.tensor([yaw], device=device)
    
    q0_goal = euler_to_quaternion(roll_t, pitch_t, yaw_t)
    return q0_goal  # [1, 4]


def test_simulate_single_timing(physics: PhysicsLayer, device: str = "cpu", num_runs: int = 1):
    """
    Test timing for simulate_single function with detailed per-step and per-process timing
    
    Args:
        physics: PhysicsLayer instance
        device: Device to run on
        num_runs: Number of runs to average over
    """
    print("\n" + "="*70)
    print("Testing simulate_single Calculation Time")
    print("="*70)
    
    # Generate test data
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = generate_random_goal(30.0, device)
    
    # Generate a simple trajectory
    OUTPUT_DIM = physics.num_waypoints * physics.n_q
    waypoints_param = torch.randn(1, OUTPUT_DIM, device=device, dtype=torch.float32)
    
    traj_start = time.time()
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
    traj_time = time.time() - traj_start
    print(f"Trajectory generation time: {traj_time*1000:.4f} ms")
    
    # Extract single trajectory (remove batch dimension)
    q_traj_single = q_traj[0]  # [num_steps, n_q]
    q_dot_traj_single = q_dot_traj[0]  # [num_steps, n_q]
    q0_start_single = q0_start[0]  # [4]
    q0_goal_single = q0_goal[0]  # [4]
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Number of steps: {physics.num_steps}")
    print(f"  Number of joints: {physics.n_q}")
    print(f"  Number of waypoints: {physics.num_waypoints}")
    print(f"  Total time: {physics.total_time} s")
    print(f"  Time step (dt): {physics.dt} s")
    print(f"  Number of runs: {num_runs}")
    print()
    
    # Storage for timing results
    all_total_times = []
    all_init_times = []
    all_spart_times = []
    all_constraint_times = []
    all_rotation_times = []
    all_final_times = []
    
    # Detailed per-step timing (only for first run)
    step_spart_times = []
    step_constraint_times = []
    step_rotation_times = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}:")
        
        # Total function timing
        total_start = time.time()
        
        # Initial quaternion to rotation matrix conversion
        init_start = time.time()
        R0 = physics.R0
        r0 = physics.r0
        R_curr = physics._quat_to_rot(q0_start_single)
        R_goal = physics._quat_to_rot(q0_goal_single)
        init_time = time.time() - init_start
        all_init_times.append(init_time)
        
        if run == 0:  # Only print details for first run
            print(f"  Initial quat-to-rot conversion: {init_time*1000:.4f} ms")
        
        # Timing accumulators for each process
        total_spart_time = 0.0
        total_constraint_time = 0.0
        total_rotation_time = 0.0
        
        # Import spart functions
        import src.dynamics.spart_functions_torch as spart
        
        for t in range(physics.num_steps):
            qm = q_traj_single[t]
            qd = q_dot_traj_single[t]
            
            # --- 1. SPART Dynamics Calculations ---
            spart_start = time.time()
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
            I0, Im = spart.inertia_projection(R0, RL, physics.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
            spart_time = time.time() - spart_start
            total_spart_time += spart_time
            
            if run == 0:  # Store per-step timing only for first run
                step_spart_times.append(spart_time)
            
            # --- 2. Non-holonomic Constraint Solver ---
            constraint_start = time.time()
            rhs = -H0m @ qd
            H0_damped = H0 + 1e-6 * physics.eye6
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]  # Angular Velocity part
            constraint_time = time.time() - constraint_start
            total_constraint_time += constraint_time
            
            if run == 0:  # Store per-step timing only for first run
                step_constraint_times.append(constraint_time)
            
            # --- 3. Rotation Matrix Integration ---
            rotation_start = time.time()
            R_delta = physics._rot_from_omega(wb, physics.dt)
            R_curr = R_curr @ R_delta
            rotation_time = time.time() - rotation_start
            total_rotation_time += rotation_time
            
            if run == 0:  # Store per-step timing only for first run
                step_rotation_times.append(rotation_time)
            
            # Print timing for first few and last steps (only first run)
            if run == 0 and (t < 3 or t == physics.num_steps - 1):
                print(f"  Step {t:3d}: SPART={spart_time*1000:7.4f} ms, "
                      f"Constraint={constraint_time*1000:7.4f} ms, "
                      f"Rotation={rotation_time*1000:7.4f} ms")
        
        # --- 4. Final Orientation Error ---
        final_start = time.time()
        R_err = R_goal.T @ R_curr
        trace = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = torch.acos(trace)
        q_final = physics._rot_to_quat(R_curr)
        final_time = time.time() - final_start
        all_final_times.append(final_time)
        
        if run == 0:
            print(f"  Final error calculation: {final_time*1000:.4f} ms")
        
        total_time = time.time() - total_start
        all_total_times.append(total_time)
        all_spart_times.append(total_spart_time)
        all_constraint_times.append(total_constraint_time)
        all_rotation_times.append(total_rotation_time)
        
        if run == 0:
            loss = angle_error ** 2
            print(f"  Total time: {total_time*1000:.4f} ms")
            print(f"  Final loss: {loss.item():.6f}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Timing Summary Statistics")
    print("="*70)
    
    import numpy as np
    
    def print_stat(name, times):
        if len(times) > 0:
            times_ms = [t * 1000 for t in times]
            mean_time = np.mean(times_ms)
            std_time = np.std(times_ms)
            min_time = np.min(times_ms)
            max_time = np.max(times_ms)
            print(f"{name:35s}: Mean={mean_time:8.4f} ms, Std={std_time:8.4f} ms, "
                  f"Min={min_time:8.4f} ms, Max={max_time:8.4f} ms")
    
    print_stat("Total simulate_single time", all_total_times)
    print_stat("Initial quat-to-rot conversion", all_init_times)
    print_stat("SPART dynamics (total)", all_spart_times)
    print_stat("Constraint solver (total)", all_constraint_times)
    print_stat("Rotation integration (total)", all_rotation_times)
    print_stat("Final error calculation", all_final_times)
    
    # Per-step averages
    if len(step_spart_times) > 0:
        avg_spart_per_step = np.mean(step_spart_times) * 1000
        avg_constraint_per_step = np.mean(step_constraint_times) * 1000
        avg_rotation_per_step = np.mean(step_rotation_times) * 1000
        
        print(f"\nPer-step averages (from first run):")
        print(f"  SPART dynamics per step:     {avg_spart_per_step:8.4f} ms")
        print(f"  Constraint solver per step:  {avg_constraint_per_step:8.4f} ms")
        print(f"  Rotation integration per step: {avg_rotation_per_step:8.4f} ms")
        print(f"  Total per step:              {(avg_spart_per_step + avg_constraint_per_step + avg_rotation_per_step):8.4f} ms")
    
    print("="*70 + "\n")


def main():
    """
    Main function to run timing tests
    """
    # Force CUDA if available, otherwise use CPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"=== Calculation Time Test (CUDA) ===")
        print(f"Device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"=== Calculation Time Test (CPU) ===")
        print(f"Device: {device}")
        print("Warning: CUDA not available, using CPU")
    
    # Setup physics layer
    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), 
                          verbose_flag=False, device=device)
    
    NUM_WAYPOINTS = 3
    TOTAL_TIME = 10.0
    
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # Run timing tests
    # Single run for detailed output
    test_simulate_single_timing(physics, device, num_runs=1)
    
    # Multiple runs for statistics
    print("\nRunning multiple iterations for better statistics...")
    test_simulate_single_timing(physics, device, num_runs=10)


if __name__ == "__main__":
    main()

