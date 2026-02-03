"""
Data Generation Script for Orientation Reachable Set Construction

목표: (waypoint, q_final) 데이터 수집
- waypoint: 중간 waypoint [18] (3 waypoints × 6 DoF)
- q_final: 시뮬레이션 후 최종 orientation quaternion [4]

start_joint와 goal_joint는 모두 0으로 고정
waypoint만 joint limit (-140° ~ 140°) 범위 내에서 랜덤 생성
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
import matplotlib.pyplot as plt

# Add root directory to sys.path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)

from NN_opt.training.physics_layer import PhysicsLayer
from physics.dynamics.urdf2robot_torch import urdf2robot
from physics.utils import (
    euler_to_quaternion_torch,
    quat_to_rot_torch,
    generate_random_quaternion_from_euler_torch,
)


def plot_distribution(data, labels, title, save_path, xlabel="Value (rad)", bins=50):
    """
    데이터 분포 히스토그램을 그리고 저장
    
    Args:
        data: [N, num_features] 형태의 데이터
        labels: 각 feature의 라벨 리스트
        title: 그래프 제목
        save_path: 저장 경로
        xlabel: x축 라벨
        bins: 히스토그램 bin 수
    """
    num_features = data.shape[1]
    
    # 그래프 레이아웃 결정 (2~3열)
    if num_features <= 4:
        ncols = 2
    else:
        ncols = 3
    nrows = (num_features + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if num_features > 1 else [axes]
    
    for i in range(num_features):
        ax = axes[i]
        ax.hist(data[:, i], bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(labels[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    
    # 빈 subplot 숨기기
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_quaternion_distribution(q_data, save_path, bins=50):
    """
    Quaternion 분포를 그리고 저장 (각 성분 + norm)
    
    Args:
        q_data: [N, 4] quaternion 데이터 (x, y, z, w)
        save_path: 저장 경로
        bins: 히스토그램 bin 수
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    labels = ['qx', 'qy', 'qz', 'qw']
    
    # 각 quaternion 성분
    for i in range(4):
        ax = axes[i]
        ax.hist(q_data[:, i], bins=bins, edgecolor='black', alpha=0.7, color=f'C{i}')
        ax.set_title(f'{labels[i]} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    # Quaternion norm 분포
    q_norms = np.linalg.norm(q_data, axis=1)
    axes[4].hist(q_norms, bins=bins, edgecolor='black', alpha=0.7, color='purple')
    axes[4].set_title('Quaternion Norm')
    axes[4].set_xlabel('Norm')
    axes[4].set_ylabel('Count')
    axes[4].grid(True, alpha=0.3)
    
    # 빈 subplot 숨기기
    axes[5].set_visible(False)
    
    plt.suptitle('q_final (Final Orientation) Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_distributions(waypoints, q_finals, n_q, num_waypoints, output_dir, output_name):
    """
    모든 데이터의 분포 그래프를 생성하고 저장
    
    Args:
        waypoints: [N, num_waypoints * n_q] 중간 waypoint
        q_finals: [N, 4] 최종 orientation quaternion
        n_q: 관절 수 (6)
        num_waypoints: waypoint 수 (3)
        output_dir: 출력 디렉토리
        output_name: 출력 파일 이름 prefix
    """
    print("\n=== Saving Distribution Plots ===")
    
    # numpy로 변환
    wp_np = waypoints.numpy() if torch.is_tensor(waypoints) else waypoints
    q_np = q_finals.numpy() if torch.is_tensor(q_finals) else q_finals
    
    # degree로 변환 (시각화 편의)
    wp_deg = np.rad2deg(wp_np)
    
    # 1. Waypoint 분포 (각 waypoint별로)
    for wp_idx in range(num_waypoints):
        start_col = wp_idx * n_q
        end_col = start_col + n_q
        wp_labels = [f'WP{wp_idx+1}_J{i+1}' for i in range(n_q)]
        plot_distribution(
            wp_deg[:, start_col:end_col], wp_labels,
            f'Waypoint {wp_idx+1} Distribution',
            os.path.join(output_dir, f"{output_name}_dist_waypoint{wp_idx+1}.png"),
            xlabel="Angle (deg)"
        )
    
    # 2. q_final (Quaternion) 분포
    plot_quaternion_distribution(
        q_np,
        os.path.join(output_dir, f"{output_name}_dist_q_final.png")
    )
    
    # 3. 전체 요약 그래프
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # All waypoints - overlapped
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, num_waypoints))
    for wp_idx in range(num_waypoints):
        start_col = wp_idx * n_q
        end_col = start_col + n_q
        wp_all = wp_deg[:, start_col:end_col].flatten()
        ax.hist(wp_all, bins=50, alpha=0.4, label=f'WP{wp_idx+1}', color=colors[wp_idx])
    ax.set_title('Waypoints (All Values)')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Quaternion components
    ax = axes[1]
    q_labels = ['qx', 'qy', 'qz', 'qw']
    for i in range(4):
        ax.hist(q_np[:, i], bins=50, alpha=0.5, label=q_labels[i])
    ax.set_title('q_final Components')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Data Distribution Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{output_name}_dist_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


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
        waypoints: [batch_size, num_waypoints * n_q]
        q_final: [batch_size, 4]
    """
    # 1. start_joint, goal_joint는 0으로 고정
    start_joint = torch.zeros(batch_size, n_q, device=device)
    goal_joint = torch.zeros(batch_size, n_q, device=device)
    
    # 2. waypoints만 랜덤 생성
    waypoints = generate_random_joints(batch_size, num_waypoints * n_q, joint_min, joint_max, device)
    
    # 3. 초기 orientation (identity quaternion)
    q0_init = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(batch_size, 1)
    
    # 4. 시뮬레이션을 위한 dummy goal quaternion
    q0_goal_dummy = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(batch_size, 1)
    
    # 5. trajectory 생성 및 시뮬레이션
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
    
    return waypoints, q_final


def main():
    parser = argparse.ArgumentParser(description="Generate reachable set data")
    parser.add_argument("--num-samples", type=int, default=100000000,
                        help="Total number of samples to generate (default: 100000000)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for data generation (default: 1024)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for saving data (default: outputs/data)")
    parser.add_argument("--output-name", type=str, default="orientation_reachable_set",
                        help="Output file name prefix (default: orientation_reachable_set)")
    parser.add_argument("--save-format", type=str, default="pt", choices=["pt", "npy", "csv"],
                        help="Save format: pt (PyTorch), npy (NumPy), csv (default: pt)")
    args = parser.parse_args()
    
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Data Generation for Orientation Reachable Set on {device} ===")
    
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
    all_waypoints = []
    all_q_finals = []
    
    # 배치 수 계산
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    total_generated = 0
    
    print(f"\nGenerating {args.num_samples} samples with batch_size={args.batch_size}")
    print(f"Joint limits: [{JOINT_MIN_DEG}°, {JOINT_MAX_DEG}°]")
    print(f"Number of waypoints: {NUM_WAYPOINTS}")
    print(f"Total simulation time: {TOTAL_TIME}s")
    print(f"Start/Goal joints: Fixed at 0")
    print()
    
    start_time = time.time()
    
    # 진행 표시줄과 함께 데이터 생성
    with tqdm(total=args.num_samples, desc="Generating data", unit="samples") as pbar:
        for batch_idx in range(num_batches):
            # 마지막 배치는 남은 샘플 수만큼만 생성
            current_batch_size = min(args.batch_size, args.num_samples - total_generated)
            
            # 배치 데이터 생성
            waypoints, q_final = generate_data_batch(
                physics, current_batch_size, n_q, NUM_WAYPOINTS,
                JOINT_MIN_RAD, JOINT_MAX_RAD, device
            )
            
            # CPU로 이동하여 저장
            all_waypoints.append(waypoints.cpu())
            all_q_finals.append(q_final.cpu())
            
            total_generated += current_batch_size
            pbar.update(current_batch_size)
    
    elapsed_time = time.time() - start_time
    
    # 데이터 합치기
    waypoints_tensor = torch.cat(all_waypoints, dim=0)        # [N, 18]
    q_finals_tensor = torch.cat(all_q_finals, dim=0)          # [N, 4]
    
    print(f"\n=== Data Generation Complete ===")
    print(f"Total samples: {total_generated}")
    print(f"Time elapsed: {elapsed_time:.2f}s ({total_generated/elapsed_time:.1f} samples/sec)")
    print(f"\nData shapes:")
    print(f"  waypoints:   {waypoints_tensor.shape}")
    print(f"  q_final:     {q_finals_tensor.shape}")
    
    # 데이터 저장
    if args.save_format == "pt":
        # PyTorch format
        save_path = os.path.join(output_dir, f"{args.output_name}.pt")
        torch.save({
            'waypoints': waypoints_tensor,
            'q_final': q_finals_tensor,
            'metadata': {
                'n_q': n_q,
                'num_waypoints': NUM_WAYPOINTS,
                'total_time': TOTAL_TIME,
                'joint_min_rad': JOINT_MIN_RAD,
                'joint_max_rad': JOINT_MAX_RAD,
                'num_samples': total_generated,
                'start_joint': 'zeros',
                'goal_joint': 'zeros'
            }
        }, save_path)
        print(f"\nSaved to: {save_path}")
        
    elif args.save_format == "npy":
        # NumPy format (각각 별도 파일)
        np.save(os.path.join(output_dir, f"{args.output_name}_waypoints.npy"), 
                waypoints_tensor.numpy())
        np.save(os.path.join(output_dir, f"{args.output_name}_q_final.npy"), 
                q_finals_tensor.numpy())
        print(f"\nSaved to: {output_dir}/{args.output_name}_*.npy")
        
    elif args.save_format == "csv":
        # CSV format (하나의 파일에 모든 데이터)
        # 컬럼: wp_1~18, qx, qy, qz, qw
        header = []
        header.extend([f"wp_{i+1}" for i in range(NUM_WAYPOINTS * n_q)])
        header.extend(["qx", "qy", "qz", "qw"])
        
        # 데이터 합치기
        all_data = np.concatenate([
            waypoints_tensor.numpy(),
            q_finals_tensor.numpy()
        ], axis=1)
        
        save_path = os.path.join(output_dir, f"{args.output_name}.csv")
        np.savetxt(save_path, all_data, delimiter=",", header=",".join(header), comments="")
        print(f"\nSaved to: {save_path}")
    
    # 데이터 통계 출력
    print(f"\n=== Data Statistics ===")
    print(f"waypoints   - min: {waypoints_tensor.min().item():.4f}, max: {waypoints_tensor.max().item():.4f}")
    print(f"q_final     - min: {q_finals_tensor.min().item():.4f}, max: {q_finals_tensor.max().item():.4f}")
    
    # Quaternion norm 확인 (정규화 확인)
    q_norms = torch.norm(q_finals_tensor, dim=1)
    print(f"q_final norm - mean: {q_norms.mean().item():.6f}, std: {q_norms.std().item():.6f}")
    
    # 분포 그래프 저장
    plot_all_distributions(
        waypoints_tensor, q_finals_tensor,
        n_q, NUM_WAYPOINTS, output_dir, args.output_name
    )


if __name__ == "__main__":
    main()
