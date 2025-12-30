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
    Test timing for simulate_single function with vectorized approach
    (Same as physics_layer.py implementation - vectorized SPART calculations)
    
    Args:
        physics: PhysicsLayer instance
        device: Device to run on
        num_runs: Number of runs to average over
    """
    print("\n" + "="*70)
    print("Testing simulate_single Calculation Time (Vectorized)")
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
    all_spart_times = []  # Vectorized SPART + Constraint time
    all_rotation_times = []  # Vectorized R_delta calculation time
    all_constraint_times = []  # Cumulative product time (reused variable name)
    all_final_times = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}:")
        
        # Total function timing
        total_start = time.time()
        
        # Initial quaternion to rotation matrix conversion
        init_start = time.time()
        R0 = physics.R0
        r0 = physics.r0
        R_init = physics._quat_to_rot(q0_start_single)
        R_goal = physics._quat_to_rot(q0_goal_single)
        init_time = time.time() - init_start
        all_init_times.append(init_time)
        
        if run == 0:  # Only print details for first run
            print(f"  Initial quat-to-rot conversion: {init_time*1000:.4f} ms")
        
        # Vectorized approach: same as physics_layer.py
        # Import spart functions and vmap
        import src.dynamics.spart_functions_torch as spart
        from torch.func import vmap
        
        # --- 1. Vectorized SPART Dynamics Calculations ---
        spart_start = time.time()
        
        def compute_wb_single_step(qm, qd):
            """Single step의 wb 계산 (벡터화용)"""
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
            I0, Im = spart.inertia_projection(R0, RL, physics.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
            
            # Constraint solver
            rhs = -H0m @ qd
            H0_damped = H0 + physics._damping_term
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]
            return wb
        
        # vmap으로 모든 step을 병렬 처리
        batch_compute_wb = vmap(compute_wb_single_step, in_dims=(0, 0))
        wb_all = batch_compute_wb(q_traj_single, q_dot_traj_single)  # [num_steps, 3]
        
        spart_time = time.time() - spart_start
        all_spart_times.append(spart_time)
        
        if run == 0:
            print(f"  Vectorized SPART + Constraint (all steps): {spart_time*1000:.4f} ms")
        
        # --- 2. Vectorized Rotation Matrix Calculation ---
        rotation_start = time.time()
        
        # 모든 step의 R_delta를 한번에 계산
        batch_rot_from_omega = vmap(physics._rot_from_omega, in_dims=(0, None))
        R_delta_all = batch_rot_from_omega(wb_all, physics.dt)  # [num_steps, 3, 3]
        
        rotation_time = time.time() - rotation_start
        all_rotation_times.append(rotation_time)
        
        if run == 0:
            print(f"  Vectorized R_delta calculation (all steps): {rotation_time*1000:.4f} ms")
        
        # --- 3. Cumulative Product (sequential but optimized) ---
        cumprod_start = time.time()
        
        # Cumulative product: R_final = R_init @ R_delta[0] @ R_delta[1] @ ... @ R_delta[n-1]
        # 순서가 중요: 왼쪽부터 곱해야 함 (순차적 버전과 동일)
        R_curr = R_init
        for t in range(physics.num_steps):
            R_curr = R_curr @ R_delta_all[t]
        
        cumprod_time = time.time() - cumprod_start
        all_constraint_times.append(cumprod_time)  # Reuse for cumulative product time
        
        if run == 0:
            print(f"  Cumulative product (sequential): {cumprod_time*1000:.4f} ms")
        
        # --- 4. Final Orientation Error ---
        final_start = time.time()
        R_err = R_goal.T @ R_curr
        trace_val = torch.trace(R_err)
        chordal_dist_sq = 3.0 - trace_val
        epsilon = 1e-8
        loss = torch.log(chordal_dist_sq + epsilon)
        q_final = physics._rot_to_quat(R_curr)
        final_time = time.time() - final_start
        all_final_times.append(final_time)
        
        if run == 0:
            print(f"  Final error calculation: {final_time*1000:.4f} ms")
        
        total_time = time.time() - total_start
        all_total_times.append(total_time)
        
        if run == 0:
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
    print_stat("Vectorized SPART+Constraint (all steps)", all_spart_times)
    print_stat("Vectorized R_delta calculation (all steps)", all_rotation_times)
    print_stat("Cumulative product (sequential)", all_constraint_times)
    print_stat("Final error calculation", all_final_times)
    
    # Calculate per-step averages (vectorized approach)
    if len(all_spart_times) > 0 and physics.num_steps > 0:
        avg_spart_per_step = (np.mean(all_spart_times) / physics.num_steps) * 1000
        avg_rotation_per_step = (np.mean(all_rotation_times) / physics.num_steps) * 1000
        avg_cumprod_per_step = (np.mean(all_constraint_times) / physics.num_steps) * 1000
        
        print(f"\nPer-step averages (vectorized approach):")
        print(f"  SPART+Constraint per step:   {avg_spart_per_step:8.4f} ms")
        print(f"  R_delta calculation per step: {avg_rotation_per_step:8.4f} ms")
        print(f"  Cumulative product per step:  {avg_cumprod_per_step:8.4f} ms")
        print(f"  Total per step (avg):         {(avg_spart_per_step + avg_rotation_per_step + avg_cumprod_per_step):8.4f} ms")
    
    print("="*70 + "\n")


def verify_vectorized_equivalence(physics: PhysicsLayer, device: str = "cpu"):
    """
    Verify that vectorized version produces same results as sequential version
    """
    print("\n" + "="*70)
    print("Verifying Vectorized vs Sequential Equivalence")
    print("="*70)
    
    # Generate test data
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = generate_random_goal(30.0, device)
    
    OUTPUT_DIM = physics.num_waypoints * physics.n_q
    waypoints_param = torch.randn(1, OUTPUT_DIM, device=device, dtype=torch.float32)
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
    
    q_traj_single = q_traj[0]
    q_dot_traj_single = q_dot_traj[0]
    q0_start_single = q0_start[0]
    q0_goal_single = q0_goal[0]
    
    # Vectorized version (current implementation)
    loss_vec, q_final_vec = physics.simulate_single(q_traj_single, q_dot_traj_single, 
                                                      q0_start_single, q0_goal_single)
    
    # Sequential version (old implementation for comparison)
    import src.dynamics.spart_functions_torch as spart
    R0 = physics.R0
    r0 = physics.r0
    R_init = physics._quat_to_rot(q0_start_single)
    R_goal = physics._quat_to_rot(q0_goal_single)
    R_curr = R_init.clone()
    
    # Sequential computation
    for t in range(physics.num_steps):
        qm = q_traj_single[t]
        qd = q_dot_traj_single[t]
        
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
        
        rhs = -H0m @ qd
        H0_damped = H0 + physics._damping_term
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]
        
        R_delta = physics._rot_from_omega(wb, physics.dt)
        R_curr = R_curr @ R_delta
    
    # Final error (sequential)
    R_err = R_goal.T @ R_curr
    trace_val = torch.trace(R_err)
    chordal_dist_sq = 3.0 - trace_val
    epsilon = 1e-8
    loss_seq = torch.log(chordal_dist_sq + epsilon)
    q_final_seq = physics._rot_to_quat(R_curr)
    
    # Compare results
    loss_diff = torch.abs(loss_vec - loss_seq).item()
    q_final_diff = torch.norm(q_final_vec - q_final_seq).item()
    
    print(f"\nComparison Results:")
    print(f"  Loss difference: {loss_diff:.2e}")
    print(f"  Quaternion difference (L2 norm): {q_final_diff:.2e}")
    print(f"  Vectorized loss: {loss_vec.item():.6f}")
    print(f"  Sequential loss: {loss_seq.item():.6f}")
    
    # Tolerance check
    loss_tol = 1e-5
    quat_tol = 1e-4
    
    if loss_diff < loss_tol and q_final_diff < quat_tol:
        print(f"\n✓ PASS: Results match within tolerance")
        print(f"  Loss tolerance: {loss_tol:.2e}")
        print(f"  Quaternion tolerance: {quat_tol:.2e}")
        return True
    else:
        print(f"\n✗ FAIL: Results differ beyond tolerance")
        print(f"  Loss tolerance: {loss_tol:.2e}")
        print(f"  Quaternion tolerance: {quat_tol:.2e}")
        return False


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
    
    # First verify equivalence
    verify_vectorized_equivalence(physics, device)
    
    # Run timing tests
    # Single run for detailed output
    test_simulate_single_timing(physics, device, num_runs=1)
    
    # Multiple runs for statistics
    print("\nRunning multiple iterations for better statistics...")
    test_simulate_single_timing(physics, device, num_runs=10)


if __name__ == "__main__":
    main()

