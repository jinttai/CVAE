import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys
from torch.func import vmap

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.training.physics_layer import PhysicsLayer   # default
from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart

# --- Helper Functions (기존과 동일) ---
def quat_to_rot(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    R = torch.stack([
        torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)]),
        torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)]),
        torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)])
    ])
    return R

def skew(v):
    vx, vy, vz = v
    zero = torch.zeros_like(vx)
    M = torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])
    return M

def rot_from_omega(wb, dt):
    device = wb.device
    dtype = wb.dtype
    theta = torch.linalg.norm(wb) * dt
    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew(axis)
    I = torch.eye(3, device=device, dtype=dtype)
    R_delta = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R_delta

# --- [수정됨] 인자를 분해해서 받도록 변경 ---
def calculate_axis_rotation(robot, dt, num_steps, q_traj, q_dot_traj, q0_init):
    """
    단일 샘플에 대해, 최종 회전이 각 축(X, Y, Z)에 대해 얼마나 투영되는지 계산
    인자가 physics 객체가 아니라 분해된 변수들임에 주의!
    """
    device = q0_init.device
    R0 = torch.eye(3, device=device, dtype=q0_init.dtype)
    r0 = torch.zeros(3, device=device, dtype=q0_init.dtype)
    
    R_init = quat_to_rot(q0_init)
    R_curr = R_init.clone()
    
    # physics.num_steps 대신 인자로 받은 num_steps 사용
    for t in range(num_steps):
        qm = q_traj[t]
        qd = q_dot_traj[t]
        
        # physics.robot 대신 인자로 받은 robot 사용
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart.inertia_projection(R0, RL, robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
        
        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device, dtype=q0_init.dtype)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]
        
        # physics.dt 대신 인자로 받은 dt 사용
        R_delta = rot_from_omega(wb, dt)
        R_curr = R_curr @ R_delta
    
    R_rel = R_init.T @ R_curr
    
    tr = torch.trace(R_rel)
    theta = torch.acos(torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7))
    


    # vmap 호환: torch.stack() 사용 (torch.tensor() 대신)
    v = torch.stack([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1]
    ])
    
    # theta가 0에 가까울 때 division by zero 방지
    sin_theta = torch.sin(theta)
    sin_theta = torch.clamp(sin_theta, min=1e-8)  # 최소값 설정
    axis = v / (2 * sin_theta)
        
    proj_angles = torch.abs(axis * theta)
    return proj_angles

def calculate_axis_rotation_batch(physics, q_traj, q_dot_traj, q0_init):
    # vmap 설정: robot, dt, num_steps는 배치 차원이 없음(None)
    batch_sim_fn = vmap(
        calculate_axis_rotation, 
        in_dims=(None, None, None, 0, 0, 0)
    )
    # 인자를 6개로 펼쳐서 전달
    angles_vec = batch_sim_fn(
        physics.robot,
        physics.dt,
        physics.num_steps,
        q_traj,
        q_dot_traj,
        q0_init
    )
    return angles_vec

def monte_carlo_axis_analysis(physics, num_samples=1000, waypoint_range=(-3.14, 3.14)):
    device = physics.device
    n_q = physics.n_q
    num_waypoints = physics.num_waypoints
    
    print(f"--- Monte Carlo Axis Analysis (N={num_samples}) ---")
    
    waypoints_flat = torch.rand(num_samples, num_waypoints * n_q, device=device)
    waypoints_flat = waypoints_flat * (waypoint_range[1] - waypoint_range[0]) + waypoint_range[0]
    
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints_flat)
    
    q0_init = torch.tensor([[0., 0., 0., 1.]], device=device).repeat(num_samples, 1)
    
    start_time = time.time()
    axis_angles = calculate_axis_rotation_batch(physics, q_traj, q_dot_traj, q0_init)
    elapsed = time.time() - start_time
    
    max_x = torch.max(axis_angles[:, 0]).item()
    max_y = torch.max(axis_angles[:, 1]).item()
    max_z = torch.max(axis_angles[:, 2]).item()
    
    total_angles = torch.linalg.norm(axis_angles, dim=1)
    max_total = torch.max(total_angles).item()
    
    print(f"Simulation Time: {elapsed:.2f}s")
    print(f"[Result] Max Rotation per Axis (Independent):")
    print(f"  X-Axis Max: {max_x:.4f} rad ({np.rad2deg(max_x):.2f}°)")
    print(f"  Y-Axis Max: {max_y:.4f} rad ({np.rad2deg(max_y):.2f}°)")
    print(f"  Z-Axis Max: {max_z:.4f} rad ({np.rad2deg(max_z):.2f}°)")
    print(f"  Any-Axis Max: {max_total:.4f} rad ({np.rad2deg(max_total):.2f}°)")
    
    min_max_angle = min(max_x, max_y, max_z)
    print(f"\n>>> Conservative Limit (Min of Maxes): {min_max_angle:.4f} rad ({np.rad2deg(min_max_angle):.2f}°)")
    print("이 값을 학습 시 Curriculum Learning의 목표치로 사용하세요.")
    
    return axis_angles

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Axis-Specific Monte Carlo Analysis on {device} ===")
    
    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=device)
    
    # 3.14 (180도)로 범위를 넓혀서 최대한의 가능성을 탐색
    physics = PhysicsLayer(robot, num_waypoints=3, total_time=10.0, device=device)
    
    NUM_SAMPLES = 100000 
    RANGE = (-140.0 * np.pi / 180.0, 140.0 * np.pi / 180.0)
    
    axis_angles = monte_carlo_axis_analysis(physics, NUM_SAMPLES, RANGE)
    
    angles_np = axis_angles.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['X-Axis Rotation', 'Y-Axis Rotation', 'Z-Axis Rotation']
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        axes[i].hist(np.rad2deg(angles_np[:, i]), bins=50, color=colors[i], alpha=0.7)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Angle (Deg)')
        axes[i].set_ylabel('Count')
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    save_path = os.path.join(ROOT_DIR, "outputs/results/axis_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nDistribution plot saved to {save_path}")

if __name__ == "__main__":
    main()