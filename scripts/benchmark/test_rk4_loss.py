"""
RK4 vs Rotation Matrix Simulation Comparison

이 스크립트는 physics_layer의 두 가지 시뮬레이션 방법을 비교합니다:
1. simulate_single: Rotation Matrix 기반 (dt=0.1s, vectorized)
2. simulate_single_rk4: RK4 기반 (dt=0.01s, high-fidelity)

비교 항목:
- Loss 값
- 최종 quaternion
- 여러 케이스 (opt_nn_lbfgs 결과, 랜덤 waypoints)
"""

import torch
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def load_csv_tensor(path, device):
    """Load CSV file as tensor"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    tensor = torch.tensor(data, device=device, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def load_meta(path):
    """Load meta information from CSV"""
    if not os.path.exists(path):
        print(f"Meta file not found: {path}. Using default total_time=10.0")
        return 10.0
    
    df = pd.read_csv(path)
    if 'total_time' in df.columns:
        return float(df['total_time'].iloc[0])
    return 10.0


def quaternion_distance(q1, q2):
    """Compute quaternion distance (angle between quaternions in radians)"""
    # Quaternion dot product (clamped to [-1, 1])
    dot = torch.clamp(torch.abs(torch.dot(q1, q2)), -1.0, 1.0)
    # Angle between quaternions: 2 * arccos(|dot|)
    angle = 2 * torch.acos(dot)
    return angle.item()


def compare_simulations(physics, waypoints, q0_start, q0_goal, case_name=""):
    """
    Compare simulate_single and simulate_single_rk4 for given waypoints.
    
    Returns:
        dict with comparison results
    """
    # Generate trajectory from waypoints
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    q_traj_single = q_traj[0]  # [num_steps, n_q]
    q_dot_traj_single = q_dot_traj[0]  # [num_steps, n_q]
    
    # Run simulate_single (Rotation Matrix)
    loss_rmat, q_final_rmat = physics.simulate_single(
        q_traj_single, q_dot_traj_single, q0_start[0], q0_goal[0]
    )
    
    # Run simulate_single_rk4
    loss_rk4, q_final_rk4 = physics.simulate_single_rk4(
        q_traj_single, q_dot_traj_single, q0_start[0], q0_goal[0]
    )
    
    # Compute differences
    loss_diff = abs(loss_rmat.item() - loss_rk4.item())
    loss_rel_diff = loss_diff / (loss_rk4.item() + 1e-10) * 100  # relative difference (%)
    
    quat_dist = quaternion_distance(q_final_rmat, q_final_rk4)
    quat_dist_deg = np.rad2deg(quat_dist)
    
    quat_l2_diff = torch.norm(q_final_rmat - q_final_rk4).item()
    
    return {
        'case_name': case_name,
        'loss_rmat': loss_rmat.item(),
        'loss_rk4': loss_rk4.item(),
        'loss_diff': loss_diff,
        'loss_rel_diff': loss_rel_diff,
        'q_final_rmat': q_final_rmat.cpu().numpy(),
        'q_final_rk4': q_final_rk4.cpu().numpy(),
        'quat_dist_rad': quat_dist,
        'quat_dist_deg': quat_dist_deg,
        'quat_l2_diff': quat_l2_diff
    }


def test_opt_nn_lbfgs(physics, device):
    """Test with optimized waypoints from opt_nn_lbfgs"""
    print("\n" + "="*60)
    print("Test Case 1: opt_nn_lbfgs Results")
    print("="*60)
    
    opt_dir = os.path.join(ROOT_DIR, "outputs/results/opt_nn_lbfgs")
    
    try:
        waypoints = load_csv_tensor(os.path.join(opt_dir, "waypoints.csv"), device)
        q0_start = load_csv_tensor(os.path.join(opt_dir, "q0_start.csv"), device)
        q0_goal = load_csv_tensor(os.path.join(opt_dir, "q0_goal.csv"), device)
        
        print(f"Loaded waypoints: shape {waypoints.shape}")
        print(f"q0_start: {q0_start[0].cpu().numpy()}")
        print(f"q0_goal: {q0_goal[0].cpu().numpy()}")
        
        result = compare_simulations(physics, waypoints, q0_start, q0_goal, "opt_nn_lbfgs")
        return result
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Skipping opt_nn_lbfgs test")
        return None


def test_random_waypoints(physics, device, num_tests=5):
    """Test with random waypoints"""
    print("\n" + "="*60)
    print(f"Test Case 2: Random Waypoints ({num_tests} tests)")
    print("="*60)
    
    results = []
    n_q = physics.n_q
    num_waypoints = physics.num_waypoints
    output_dim = num_waypoints * n_q
    
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    for i in range(num_tests):
        # Generate random waypoints
        waypoints = torch.randn(1, output_dim, device=device, dtype=torch.float32) * 0.1
        
        # Generate random goal orientation
        # Random Euler angles in [-30, 30] degrees
        roll_deg = np.random.uniform(-30, 30)
        pitch_deg = np.random.uniform(-30, 30)
        yaw_deg = np.random.uniform(-30, 30)
        
        roll_rad = np.deg2rad(roll_deg)
        pitch_rad = np.deg2rad(pitch_deg)
        yaw_rad = np.deg2rad(yaw_deg)
        
        # Convert to quaternion (ZYX convention)
        cr = np.cos(roll_rad / 2)
        sr = np.sin(roll_rad / 2)
        cp = np.cos(pitch_rad / 2)
        sp = np.sin(pitch_rad / 2)
        cy = np.cos(yaw_rad / 2)
        sy = np.sin(yaw_rad / 2)
        
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        
        q0_goal = torch.tensor([[qx, qy, qz, qw]], device=device, dtype=torch.float32)
        
        print(f"\n  Test {i+1}/{num_tests}:")
        print(f"    Goal Euler (deg): roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}")
        
        result = compare_simulations(physics, waypoints, q0_start, q0_goal, f"random_{i+1}")
        results.append(result)
    
    return results


def test_zero_waypoints(physics, device):
    """Test with zero waypoints (straight trajectory)"""
    print("\n" + "="*60)
    print("Test Case 3: Zero Waypoints")
    print("="*60)
    
    n_q = physics.n_q
    num_waypoints = physics.num_waypoints
    output_dim = num_waypoints * n_q
    
    waypoints = torch.zeros(1, output_dim, device=device, dtype=torch.float32)
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    # Small goal rotation
    roll_rad = np.deg2rad(15.0)
    pitch_rad = np.deg2rad(15.0)
    yaw_rad = np.deg2rad(-15.0)
    
    cr = np.cos(roll_rad / 2)
    sr = np.sin(roll_rad / 2)
    cp = np.cos(pitch_rad / 2)
    sp = np.sin(pitch_rad / 2)
    cy = np.cos(yaw_rad / 2)
    sy = np.sin(yaw_rad / 2)
    
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    q0_goal = torch.tensor([[qx, qy, qz, qw]], device=device, dtype=torch.float32)
    
    print(f"q0_start: {q0_start[0].cpu().numpy()}")
    print(f"q0_goal: {q0_goal[0].cpu().numpy()}")
    
    result = compare_simulations(physics, waypoints, q0_start, q0_goal, "zero_waypoints")
    return result


def print_comparison(result):
    """Print comparison results"""
    print(f"\n  Case: {result['case_name']}")
    print(f"  Rotation Matrix Loss: {result['loss_rmat']:.8f}")
    print(f"  RK4 Loss:             {result['loss_rk4']:.8f}")
    print(f"  Loss Difference:      {result['loss_diff']:.8f} ({result['loss_rel_diff']:.2f}%)")
    print(f"  Quaternion Distance:  {result['quat_dist_rad']:.6f} rad ({result['quat_dist_deg']:.4f} deg)")
    print(f"  Quaternion L2 Diff:  {result['quat_l2_diff']:.6f}")
    print(f"  Final Quat (RMat):    {result['q_final_rmat']}")
    print(f"  Final Quat (RK4):     {result['q_final_rk4']}")


def print_summary(all_results):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("No valid results to summarize")
        return
    
    # Extract metrics
    loss_diffs = [r['loss_diff'] for r in valid_results]
    loss_rel_diffs = [r['loss_rel_diff'] for r in valid_results]
    quat_dists_deg = [r['quat_dist_deg'] for r in valid_results]
    quat_l2_diffs = [r['quat_l2_diff'] for r in valid_results]
    
    print(f"\nNumber of tests: {len(valid_results)}")
    print(f"\nLoss Differences:")
    print(f"  Mean:   {np.mean(loss_diffs):.8f}")
    print(f"  Std:    {np.std(loss_diffs):.8f}")
    print(f"  Min:    {np.min(loss_diffs):.8f}")
    print(f"  Max:    {np.max(loss_diffs):.8f}")
    
    print(f"\nRelative Loss Differences (%):")
    print(f"  Mean:   {np.mean(loss_rel_diffs):.4f}%")
    print(f"  Std:    {np.std(loss_rel_diffs):.4f}%")
    print(f"  Min:    {np.min(loss_rel_diffs):.4f}%")
    print(f"  Max:    {np.max(loss_rel_diffs):.4f}%")
    
    print(f"\nQuaternion Distances (deg):")
    print(f"  Mean:   {np.mean(quat_dists_deg):.4f}°")
    print(f"  Std:    {np.std(quat_dists_deg):.4f}°")
    print(f"  Min:    {np.min(quat_dists_deg):.4f}°")
    print(f"  Max:    {np.max(quat_dists_deg):.4f}°")
    
    print(f"\nQuaternion L2 Differences:")
    print(f"  Mean:   {np.mean(quat_l2_diffs):.6f}")
    print(f"  Std:    {np.std(quat_l2_diffs):.6f}")
    print(f"  Min:    {np.min(quat_l2_diffs):.6f}")
    print(f"  Max:    {np.max(quat_l2_diffs):.6f}")
    
    # Warning if differences are large
    if np.max(quat_dists_deg) > 10.0:
        print(f"\n⚠ WARNING: Large quaternion distance detected ({np.max(quat_dists_deg):.4f}°)")
        print("  This may indicate significant differences between the two methods.")
    elif np.max(quat_dists_deg) > 5.0:
        print(f"\n⚠ Note: Moderate quaternion distance detected ({np.max(quat_dists_deg):.4f}°)")
    else:
        print(f"\n✓ Quaternion distances are within reasonable range (< 5°)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RK4 vs Rotation Matrix Simulation Comparison ({device}) ===")
    
    # Setup Robot & Physics
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    n_q = robot["n_q"]
    
    NUM_WAYPOINTS = 3
    TOTAL_TIME = 10.0
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Waypoints: {NUM_WAYPOINTS}")
    print(f"  Total Time: {TOTAL_TIME}s")
    print(f"  Joints (n_q): {n_q}")
    
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    print(f"  Physics Steps: {physics.num_steps}")
    print(f"  Physics dt: {physics.dt}s")
    print(f"  RK4 dt: 0.01s")
    
    all_results = []
    
    # Test 1: opt_nn_lbfgs results
    result1 = test_opt_nn_lbfgs(physics, device)
    if result1:
        print_comparison(result1)
        all_results.append(result1)
    
    # Test 2: Random waypoints
    random_results = test_random_waypoints(physics, device, num_tests=5)
    for result in random_results:
        print_comparison(result)
        all_results.append(result)
    
    # Test 3: Zero waypoints
    result3 = test_zero_waypoints(physics, device)
    if result3:
        print_comparison(result3)
        all_results.append(result3)
    
    # Print summary
    print_summary(all_results)
    
    print("\n" + "="*60)
    print("Done")
    print("="*60)


if __name__ == "__main__":
    main()

