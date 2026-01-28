"""
Data Generation Script for Reachable Set Construction

목표: (start_joint, goal_joint, waypoint, q_final) 데이터 수집
- start_joint: 시작 관절 각도 [6]
- goal_joint: 목표 관절 각도 [6]
- waypoint: 중간 waypoint [18] (3 waypoints × 6 DoF)
- q_final: 시뮬레이션 후 최종 orientation quaternion [4]

모든 값은 joint limit (-140° ~ 140°) 범위 내에서 랜덤 생성
batch_size=10으로 처리하여 데이터 축적
"""

import torch
import os
import sys
import time
import math
import argparse
import numpy as np
from tqdm import tqdm

# Add root directory to sys.path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def generate_random_joints(batch_size, n_joints, joint_min, joint_max, device):
    """
    joint limit 범위 내에서 랜덤 관절 각도 생성
    
    Args:
        batch_size: 배치 크기
        n_joints: 관절 수 (6)
        joint_min: 최소 관절 각도 (rad)
        joint_max: 최대 관절 각도 (rad)
        device: torch device
    
    Returns:
        joints: [batch_size, n_joints]
    """
    return torch.rand(batch_size, n_joints, device=device) * (joint_max - joint_min) + joint_min


def generate_data_batch(physics, batch_size, n_q, num_waypoints, joint_min, joint_max, device):
    """
    한 배치의 데이터 생성
    
    Args:
        physics: PhysicsLayer 인스턴스
        batch_size: 배치 크기
        n_q: 관절 수 (6)
        num_waypoints: waypoint 수 (3)
        joint_min: 최소 관절 각도 (rad)
        joint_max: 최대 관절 각도 (rad)
        device: torch device
    
    Returns:
        start_joint: [batch_size, n_q]
        goal_joint: [batch_size, n_q]
        waypoints: [batch_size, num_waypoints * n_q]
        q_final: [batch_size, 4]
    """
    # 1. 랜덤하게 start_joint, goal_joint, waypoints 생성
    start_joint = generate_random_joints(batch_size, n_q, joint_min, joint_max, device)
    goal_joint = generate_random_joints(batch_size, n_q, joint_min, joint_max, device)
    waypoints = generate_random_joints(batch_size, num_waypoints * n_q, joint_min, joint_max, device)
    
    # 2. 초기 orientation (identity quaternion)
    q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(batch_size, 1)
    
    # 3. 시뮬레이션을 위한 dummy goal quaternion (physics loss 계산에만 사용, q_final 계산에는 영향 없음)
    q0_goal_dummy = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(batch_size, 1)
    
    # 4. trajectory 생성 및 시뮬레이션
    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(
            waypoints, 
            q_start=start_joint, 
            q_end=goal_joint
        )
        
        # vmap으로 배치 시뮬레이션
        from torch.func import vmap
        batch_sim_fn = vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        _, q_final = batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal_dummy)
    
    return start_joint, goal_joint, waypoints, q_final


def main():
    parser = argparse.ArgumentParser(description="Generate reachable set data")
    parser.add_argument("--num-samples", type=int, default=100000000,
                        help="Total number of samples to generate (default: 1000000)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for data generation (default: 1024)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for saving data (default: outputs/data)")
    parser.add_argument("--output-name", type=str, default="reachable_set",
                        help="Output file name prefix (default: reachable_set)")
    parser.add_argument("--save-format", type=str, default="pt", choices=["pt", "npy", "csv"],
                        help="Save format: pt (PyTorch), npy (NumPy), csv (default: pt)")
    args = parser.parse_args()
    
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Data Generation for Reachable Set on {device} ===")
    
    # 로봇 로드
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    
    # 파라미터 설정
    n_q = robot["n_q"]  # 6
    NUM_WAYPOINTS = 3
    TOTAL_TIME = 10.0
    
    # Joint limit: -140° ~ 140°
    JOINT_MIN_DEG = -140.0
    JOINT_MAX_DEG = 140.0
    JOINT_MIN_RAD = math.radians(JOINT_MIN_DEG)
    JOINT_MAX_RAD = math.radians(JOINT_MAX_DEG)
    
    # Physics Layer 초기화
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "outputs/data")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 저장용 리스트
    all_start_joints = []
    all_goal_joints = []
    all_waypoints = []
    all_q_finals = []
    
    # 배치 수 계산
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    total_generated = 0
    
    print(f"\nGenerating {args.num_samples} samples with batch_size={args.batch_size}")
    print(f"Joint limits: [{JOINT_MIN_DEG}°, {JOINT_MAX_DEG}°]")
    print(f"Number of waypoints: {NUM_WAYPOINTS}")
    print(f"Total simulation time: {TOTAL_TIME}s")
    print()
    
    start_time = time.time()
    
    # 진행 표시줄과 함께 데이터 생성
    with tqdm(total=args.num_samples, desc="Generating data", unit="samples") as pbar:
        for batch_idx in range(num_batches):
            # 마지막 배치는 남은 샘플 수만큼만 생성
            current_batch_size = min(args.batch_size, args.num_samples - total_generated)
            
            # 배치 데이터 생성
            start_joint, goal_joint, waypoints, q_final = generate_data_batch(
                physics, current_batch_size, n_q, NUM_WAYPOINTS,
                JOINT_MIN_RAD, JOINT_MAX_RAD, device
            )
            
            # CPU로 이동하여 저장
            all_start_joints.append(start_joint.cpu())
            all_goal_joints.append(goal_joint.cpu())
            all_waypoints.append(waypoints.cpu())
            all_q_finals.append(q_final.cpu())
            
            total_generated += current_batch_size
            pbar.update(current_batch_size)
    
    elapsed_time = time.time() - start_time
    
    # 데이터 합치기
    start_joints_tensor = torch.cat(all_start_joints, dim=0)  # [N, 6]
    goal_joints_tensor = torch.cat(all_goal_joints, dim=0)    # [N, 6]
    waypoints_tensor = torch.cat(all_waypoints, dim=0)        # [N, 18]
    q_finals_tensor = torch.cat(all_q_finals, dim=0)          # [N, 4]
    
    print(f"\n=== Data Generation Complete ===")
    print(f"Total samples: {total_generated}")
    print(f"Time elapsed: {elapsed_time:.2f}s ({total_generated/elapsed_time:.1f} samples/sec)")
    print(f"\nData shapes:")
    print(f"  start_joint: {start_joints_tensor.shape}")
    print(f"  goal_joint:  {goal_joints_tensor.shape}")
    print(f"  waypoints:   {waypoints_tensor.shape}")
    print(f"  q_final:     {q_finals_tensor.shape}")
    
    # 데이터 저장
    if args.save_format == "pt":
        # PyTorch format
        save_path = os.path.join(output_dir, f"{args.output_name}.pt")
        torch.save({
            'start_joint': start_joints_tensor,
            'goal_joint': goal_joints_tensor,
            'waypoints': waypoints_tensor,
            'q_final': q_finals_tensor,
            'metadata': {
                'n_q': n_q,
                'num_waypoints': NUM_WAYPOINTS,
                'total_time': TOTAL_TIME,
                'joint_min_rad': JOINT_MIN_RAD,
                'joint_max_rad': JOINT_MAX_RAD,
                'num_samples': total_generated
            }
        }, save_path)
        print(f"\nSaved to: {save_path}")
        
    elif args.save_format == "npy":
        # NumPy format (각각 별도 파일)
        np.save(os.path.join(output_dir, f"{args.output_name}_start_joint.npy"), 
                start_joints_tensor.numpy())
        np.save(os.path.join(output_dir, f"{args.output_name}_goal_joint.npy"), 
                goal_joints_tensor.numpy())
        np.save(os.path.join(output_dir, f"{args.output_name}_waypoints.npy"), 
                waypoints_tensor.numpy())
        np.save(os.path.join(output_dir, f"{args.output_name}_q_final.npy"), 
                q_finals_tensor.numpy())
        print(f"\nSaved to: {output_dir}/{args.output_name}_*.npy")
        
    elif args.save_format == "csv":
        # CSV format (하나의 파일에 모든 데이터)
        # 컬럼: start_J1~J6, goal_J1~J6, wp_1~18, qx, qy, qz, qw
        header = []
        header.extend([f"start_J{i+1}" for i in range(n_q)])
        header.extend([f"goal_J{i+1}" for i in range(n_q)])
        header.extend([f"wp_{i+1}" for i in range(NUM_WAYPOINTS * n_q)])
        header.extend(["qx", "qy", "qz", "qw"])
        
        # 데이터 합치기
        all_data = np.concatenate([
            start_joints_tensor.numpy(),
            goal_joints_tensor.numpy(),
            waypoints_tensor.numpy(),
            q_finals_tensor.numpy()
        ], axis=1)
        
        save_path = os.path.join(output_dir, f"{args.output_name}.csv")
        np.savetxt(save_path, all_data, delimiter=",", header=",".join(header), comments="")
        print(f"\nSaved to: {save_path}")
    
    # 데이터 통계 출력
    print(f"\n=== Data Statistics ===")
    print(f"start_joint - min: {start_joints_tensor.min().item():.4f}, max: {start_joints_tensor.max().item():.4f}")
    print(f"goal_joint  - min: {goal_joints_tensor.min().item():.4f}, max: {goal_joints_tensor.max().item():.4f}")
    print(f"waypoints   - min: {waypoints_tensor.min().item():.4f}, max: {waypoints_tensor.max().item():.4f}")
    print(f"q_final     - min: {q_finals_tensor.min().item():.4f}, max: {q_finals_tensor.max().item():.4f}")
    
    # Quaternion norm 확인 (정규화 확인)
    q_norms = torch.norm(q_finals_tensor, dim=1)
    print(f"q_final norm - mean: {q_norms.mean().item():.6f}, std: {q_norms.std().item():.6f}")


if __name__ == "__main__":
    main()
