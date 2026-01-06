"""
RK4 Comparison Script

이 스크립트는 다음 세 가지 결과를 비교합니다:
1. sim_result.csv: MATLAB 시뮬레이션 결과
2. opt_nn_lbfgs 결과: 최적화된 waypoint로 생성된 궤적
3. opt_nn_lbfgs waypoint + RK4: 최적화된 waypoint를 RK4로 시뮬레이션한 결과
"""

import torch
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from torch.func import vmap
import src.dynamics.spart_functions_torch as spart


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


# Note: quat_to_rot, skew, rot_from_omega are now using physics_layer methods
# These helper functions are kept for compatibility but should use physics methods


def rot_to_euler(R):
    """Convert rotation matrix R (3x3) to Euler angles (ZYX order: yaw, pitch, roll)"""
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        yaw = torch.atan2(R[1, 0], R[0, 0])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.atan2(R[2, 1], R[2, 2])
    else:
        yaw = torch.atan2(-R[0, 1], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.zeros_like(yaw)

    return torch.stack([yaw, pitch, roll])


def quat_to_euler(q, physics):
    """Convert quaternion [x, y, z, w] to Euler angles [yaw, pitch, roll]"""
    R_mat = physics._quat_to_rot(q)
    return rot_to_euler(R_mat)


def compute_orientation_traj(physics, q_traj, q_dot_traj, q0_init):
    """
    Compute body orientation trajectory (Euler angles) from joint trajectory.
    Uses the same dynamics as PhysicsLayer.simulate_single.
    """
    device = physics.device
    num_steps = physics.num_steps

    R0 = physics.R0
    r0 = physics.r0

    R_curr = physics._quat_to_rot(q0_init)

    # Vectorized: compute wb for all steps (same as PhysicsLayer.simulate_single)
    def compute_wb_single_step(qm, qd):
        """Single step wb calculation"""
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)

        rhs = -H0m @ qd
        H0_damped = H0 + physics._damping_term
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]
        return wb
    
    batch_compute_wb = vmap(compute_wb_single_step, in_dims=(0, 0))
    wb_all = batch_compute_wb(q_traj, q_dot_traj)  # [num_steps, 3]
    
    # Use physics_layer's _rot_from_omega method
    batch_rot_from_omega = vmap(physics._rot_from_omega, in_dims=(0, None))
    R_delta_all = batch_rot_from_omega(wb_all, physics.dt)  # [num_steps, 3, 3]
    
    # Sequentially update R_curr and compute euler
    eulers = []
    for t in range(num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))

    euler_traj = torch.stack(eulers, dim=0)
    return euler_traj


def load_sim_result(sim_result_path):
    """
    Load sim_result.csv from MATLAB simulation.
    MATLAB format: [time, quat_w, quat_x, quat_y, quat_z, euler_yaw, euler_pitch, euler_roll]
    Convert to: [x, y, z, w] format for consistency with codebase
    """
    if not os.path.exists(sim_result_path):
        raise FileNotFoundError(f"sim_result.csv not found: {sim_result_path}")
    
    data = np.loadtxt(sim_result_path, delimiter=",")
    
    # MATLAB format: time, quat[w, x, y, z], euler[3]
    time = data[:, 0]
    quat_matlab = data[:, 1:5]  # [w, x, y, z] from MATLAB
    euler = data[:, 5:8]  # [yaw, pitch, roll] (assuming ZYX order)
    
    # Convert MATLAB [w, x, y, z] to [x, y, z, w] format
    quat = np.zeros_like(quat_matlab)
    quat[:, 0] = quat_matlab[:, 1]  # x
    quat[:, 1] = quat_matlab[:, 2]  # y
    quat[:, 2] = quat_matlab[:, 3]  # z
    quat[:, 3] = quat_matlab[:, 0]  # w
    
    return {
        'time': time,
        'quat': quat,  # [x, y, z, w]
        'euler': euler
    }


def compute_rk4_trajectory(physics, waypoints, q0_start, q0_goal):
    """
    Compute trajectory using RK4 simulation.
    Uses the exact same logic as physics_layer.simulate_single_rk4 but stores trajectory.
    Returns quaternion and euler angle trajectories.
    """
    # Generate trajectory from waypoints
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    q_traj_single = q_traj[0]  # [num_steps, n_q]
    q_dot_traj_single = q_dot_traj[0]  # [num_steps, n_q]
    
    # Use exact same logic as physics_layer.simulate_single_rk4
    dt_eval = 0.01
    num_steps_eval = int(physics.total_time / dt_eval)
    actual_dt = physics.total_time / num_steps_eval if num_steps_eval > 0 else dt_eval
    
    # 초기화 (same as physics_layer)
    R0 = physics.R0
    r0 = physics.r0
    q_curr = q0_start[0].clone()
    q_goal = q0_goal[0].clone()
    
    def normalize_quat(q):
        return q / (torch.linalg.norm(q) + 1e-8)
    
    # --- Helper: 특정 시간 t에서의 입력(qm, qd) 보간 함수 (same as physics_layer) ---
    def get_interpolated_input(t):
        t_clamped = max(0.0, min(float(t), physics.total_time - 1e-10))
        if physics.num_steps > 1:
            idx_float = t_clamped * (physics.num_steps - 1) / physics.total_time
        else:
            idx_float = 0.0
        idx_floor = int(idx_float)
        idx_ceil = min(idx_floor + 1, physics.num_steps - 1)
        alpha = idx_float - idx_floor
        qm_interp = (1 - alpha) * q_traj_single[idx_floor] + alpha * q_traj_single[idx_ceil]
        qd_interp = (1 - alpha) * q_dot_traj_single[idx_floor] + alpha * q_dot_traj_single[idx_ceil]
        return qm_interp, qd_interp
    
    # --- Helper: 현재 쿼터니언(q)과 시간(t)에서 각속도(wb) 계산 (same as physics_layer) ---
    def compute_omega(current_q, current_t):
        # 입력 보간값 가져오기
        qm_sub, qd_sub = get_interpolated_input(current_t)
        
        # SPART Dynamics 재계산 (same as physics_layer)
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm_sub, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
        
        # Constraint Solver (use same damping as physics_layer)
        rhs = -H0m @ qd_sub
        H0_damped = H0 + 1e-6 * physics.eye6  # Use physics_layer's eye6
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        return u0_sol[:3]  # wb
    
    quat_traj_rk4 = []
    euler_traj_rk4 = []
    current_time = 0.0
    
    # Sample trajectory at regular intervals (every 10 steps = 0.1s)
    sample_interval = max(1, num_steps_eval // physics.num_steps)
    
    # --- Main Loop (same as physics_layer) ---
    for step in range(num_steps_eval):
        dt_step = actual_dt
        if step == num_steps_eval - 1:
            remaining_time = physics.total_time - current_time
            if remaining_time > 1e-10:
                dt_step = remaining_time
        
        # RK4 Integration (exact same as physics_layer)
        w1 = compute_omega(q_curr, current_time)
        k1 = spart.quat_dot(q_curr, w1)
        
        q_k2 = normalize_quat(q_curr + 0.5 * dt_step * k1)
        w2 = compute_omega(q_k2, current_time + 0.5 * dt_step)
        k2 = spart.quat_dot(q_k2, w2)
        
        q_k3 = normalize_quat(q_curr + 0.5 * dt_step * k2)
        w3 = compute_omega(q_k3, current_time + 0.5 * dt_step)
        k3 = spart.quat_dot(q_k3, w3)
        
        q_k4 = normalize_quat(q_curr + dt_step * k3)
        w4 = compute_omega(q_k4, current_time + dt_step)
        k4 = spart.quat_dot(q_k4, w4)
        
        q_curr = normalize_quat(q_curr + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        current_time += dt_step
        
        # Store at sample intervals
        if step % sample_interval == 0 or step == num_steps_eval - 1:
            quat_traj_rk4.append(q_curr.cpu().numpy())
            R_curr = physics._quat_to_rot(q_curr)
            euler_curr = rot_to_euler(R_curr)
            euler_traj_rk4.append(euler_curr.cpu().numpy())
        
        if current_time >= physics.total_time - 1e-10:
            break
    
    quat_traj_rk4 = np.array(quat_traj_rk4)  # [num_samples, 4]
    euler_traj_rk4 = np.array(euler_traj_rk4)  # [num_samples, 3]
    time_rk4 = np.linspace(0.0, physics.total_time, len(quat_traj_rk4))
    
    # Compute final loss (same as physics_layer)
    R_curr = physics._quat_to_rot(q_curr)
    R_goal = physics._quat_to_rot(q_goal)
    R_diff = R_curr - R_goal
    R_diff_T = R_diff.T
    R_diff_sq = R_diff_T @ R_diff
    trace_val = 0.5 * torch.trace(R_diff_sq)
    epsilon = 1e-8
    loss = torch.log(epsilon + trace_val)
    
    return {
        'time': time_rk4,
        'quat': quat_traj_rk4,
        'euler': euler_traj_rk4,
        'loss': loss.item(),
        'q_final': q_curr.cpu().numpy()
    }


def plot_comparison(sim_result, opt_result, rk4_result, save_path):
    """Plot comparison of three results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Quaternion comparison
    quat_labels = ['qx', 'qy', 'qz', 'qw']
    for i in range(4):
        ax = axes[0, i] if i < 3 else axes[1, 0]
        ax.plot(sim_result['time'], sim_result['quat'][:, i], 'b-', label='MATLAB Sim', linewidth=2)
        ax.plot(opt_result['time'], opt_result['quat'][:, i], 'r--', label='opt_nn_lbfgs', linewidth=2)
        ax.plot(rk4_result['time'], rk4_result['quat'][:, i], 'g:', label='RK4', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(quat_labels[i])
        ax.set_title(f'Quaternion {quat_labels[i]}')
        ax.legend()
        ax.grid(True)
    
    # Euler angle comparison
    euler_labels = ['Yaw', 'Pitch', 'Roll']
    for i in range(3):
        ax = axes[1, i+1] if i < 2 else axes[0, 2]
        ax.plot(sim_result['time'], np.rad2deg(sim_result['euler'][:, i]), 'b-', label='MATLAB Sim', linewidth=2)
        ax.plot(opt_result['time'], np.rad2deg(opt_result['euler'][:, i]), 'r--', label='opt_nn_lbfgs', linewidth=2)
        ax.plot(rk4_result['time'], np.rad2deg(rk4_result['euler'][:, i]), 'g:', label='RK4', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{euler_labels[i]} (deg)')
        ax.set_title(f'Euler Angle: {euler_labels[i]}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RK4 Comparison Script ({device}) ===")
    
    # Paths
    result_dir = os.path.join(ROOT_DIR, "outputs/results")
    opt_dir = os.path.join(result_dir, "opt_nn_lbfgs")
    sim_result_path = os.path.join(result_dir, "sim_result.csv")
    
    # 1. Load sim_result.csv (MATLAB simulation)
    print("\n[1] Loading sim_result.csv...")
    try:
        sim_result = load_sim_result(sim_result_path)
        print(f"  Loaded {len(sim_result['time'])} time steps")
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Skipping MATLAB simulation comparison")
        sim_result = None
    
    # 2. Load opt_nn_lbfgs results
    print("\n[2] Loading opt_nn_lbfgs results...")
    try:
        waypoints = load_csv_tensor(os.path.join(opt_dir, "waypoints.csv"), device)
        q0_start = load_csv_tensor(os.path.join(opt_dir, "q0_start.csv"), device)
        q0_goal = load_csv_tensor(os.path.join(opt_dir, "q0_goal.csv"), device)
        total_time = load_meta(os.path.join(opt_dir, "meta.csv"))
        body_orientation = pd.read_csv(os.path.join(opt_dir, "body_orientation.csv"))
        
        print(f"  Loaded waypoints: shape {waypoints.shape}")
        print(f"  Total time: {total_time}s")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return
    
    # 3. Setup Robot & Physics
    print("\n[3] Setting up robot and physics...")
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    n_q = robot["n_q"]
    
    num_elements = waypoints.numel()
    if num_elements % n_q != 0:
        print(f"Error: Waypoints size {num_elements} is not divisible by n_q={n_q}")
        return
    num_waypoints = num_elements // n_q
    
    print(f"  Waypoints: {num_waypoints}, Joints: {n_q}")
    
    physics = PhysicsLayer(robot, num_waypoints, total_time, device)
    
    # 4. Generate trajectory from opt_nn_lbfgs waypoints
    print("\n[4] Generating trajectory from opt_nn_lbfgs waypoints...")
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    q_traj_single = q_traj[0]
    q_dot_traj_single = q_dot_traj[0]
    
    # Compute orientation trajectory
    euler_traj_opt = compute_orientation_traj(physics, q_traj_single, q_dot_traj_single, q0_start[0])
    
    # Convert euler trajectory to quaternion trajectory
    num_steps = q_traj_single.shape[0]
    time_opt = np.linspace(0.0, total_time, num_steps)
    
    # Compute quaternion trajectory from euler trajectory
    # Use the same approach as compute_orientation_traj but track quaternions
    quat_traj_opt = []
    R_curr = physics._quat_to_rot(q0_start[0])
    
    # Use the same vectorized approach as compute_orientation_traj
    R0 = physics.R0
    r0 = physics.r0
    
    def compute_wb_single_step(qm, qd):
        """Single step wb calculation"""
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart.inertia_projection(R0, RL, robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
        rhs = -H0m @ qd
        H0_damped = H0 + physics._damping_term
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]
        return wb
    
    batch_compute_wb = vmap(compute_wb_single_step, in_dims=(0, 0))
    wb_all = batch_compute_wb(q_traj_single, q_dot_traj_single)  # [num_steps, 3]
    
    batch_rot_from_omega = vmap(physics._rot_from_omega, in_dims=(0, None))
    R_delta_all = batch_rot_from_omega(wb_all, physics.dt)  # [num_steps, 3, 3]
    
    # Sequentially update R_curr and convert to quaternion
    for t in range(num_steps):
        R_curr = R_curr @ R_delta_all[t]
        q_curr = physics._rot_to_quat(R_curr)
        quat_traj_opt.append(q_curr.cpu().numpy())
    
    quat_traj_opt = np.array(quat_traj_opt)
    euler_traj_opt_np = euler_traj_opt.cpu().numpy()
    
    opt_result = {
        'time': time_opt,
        'quat': quat_traj_opt,
        'euler': euler_traj_opt_np
    }
    
    # 5. Run RK4 simulation
    print("\n[5] Running RK4 simulation...")
    
    # First, verify using physics_layer's simulate_single_rk4 directly
    q_traj_check, q_dot_traj_check = physics.generate_trajectory(waypoints)
    loss_rk4_direct, q_final_rk4_direct = physics.simulate_single_rk4(
        q_traj_check[0], q_dot_traj_check[0], q0_start[0], q0_goal[0]
    )
    print(f"  Direct RK4 (physics_layer):")
    print(f"    Loss: {loss_rk4_direct.item():.8f}")
    print(f"    Final quaternion: {q_final_rk4_direct.cpu().numpy()}")
    
    # Then compute full trajectory
    rk4_result = compute_rk4_trajectory(physics, waypoints, q0_start, q0_goal)
    print(f"  Trajectory RK4:")
    print(f"    Loss: {rk4_result['loss']:.8f}")
    print(f"    Final quaternion: {rk4_result['q_final']}")
    
    # Compare
    q_diff = np.linalg.norm(q_final_rk4_direct.cpu().numpy() - rk4_result['q_final'])
    print(f"  Difference between direct and trajectory: {q_diff:.8f}")
    if q_diff > 1e-4:
        print(f"  WARNING: Large difference detected! Trajectory computation may be incorrect.")
    
    # 6. Compare results
    print("\n[6] Comparing results...")
    
    if sim_result is not None:
        print("\n=== Comparison Summary ===")
        print(f"MATLAB Sim:     {len(sim_result['time'])} steps")
        print(f"opt_nn_lbfgs:   {len(opt_result['time'])} steps")
        print(f"RK4:            {len(rk4_result['time'])} steps")
        
        # Final orientation comparison
        print("\n=== Final Orientation Comparison ===")
        print("MATLAB Sim (final):")
        print(f"  Quaternion: {sim_result['quat'][-1]}")
        print(f"  Euler (deg): {(sim_result['euler'][-1])}")
        
        print("\nopt_nn_lbfgs (final):")
        print(f"  Quaternion: {opt_result['quat'][-1]}")
        print(f"  Euler (deg): {np.rad2deg(opt_result['euler'][-1])}")
        
        print("\nRK4 (final):")
        print(f"  Quaternion: {rk4_result['q_final']}")
        print(f"  Euler (deg): {np.rad2deg(rk4_result['euler'][-1])}")
        
        # Plot comparison
        save_path = os.path.join(result_dir, "rk4_comparison.png")
        plot_comparison(sim_result, opt_result, rk4_result, save_path)
    else:
        print("\n=== opt_nn_lbfgs vs RK4 Comparison ===")
        print(f"opt_nn_lbfgs (final):")
        print(f"  Quaternion: {opt_result['quat'][-1]}")
        print(f"  Euler (deg): {np.rad2deg(opt_result['euler'][-1])}")
        
        print("\nRK4 (final):")
        print(f"  Quaternion: {rk4_result['q_final']}")
        print(f"  Euler (deg): {np.rad2deg(rk4_result['euler'][-1])}")
        
        # Compute differences
        quat_diff = np.linalg.norm(opt_result['quat'][-1] - rk4_result['q_final'])
        euler_diff = np.linalg.norm(np.rad2deg(opt_result['euler'][-1] - rk4_result['euler'][-1]))
        print(f"\nDifference:")
        print(f"  Quaternion L2 norm: {quat_diff:.6f}")
        print(f"  Euler L2 norm (deg): {euler_diff:.4f}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
