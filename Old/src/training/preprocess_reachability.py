"""
Reachability Predictor 전처리 스크립트

(start_joint, goal_joint)를 3등분 그리드로 binning하여
각 bin별로 도달 가능한 q_final들을 그룹화

- 12차원 (6 start + 6 goal) × 3등분 = 3^12 = 531,441 bins
- 각 bin에 해당하는 q_final들을 저장
"""

import torch
import os
import sys
import time
import math
import argparse
from tqdm import tqdm
from collections import defaultdict

# Add root directory to sys.path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)


def compute_bin_index(start_joint, goal_joint, joint_limits, num_divisions=3):
    """
    (start_joint, goal_joint) → 단일 bin index
    
    Args:
        start_joint: [6] tensor
        goal_joint: [6] tensor
        joint_limits: [6, 2] tensor (lower, upper)
        num_divisions: 등분 수 (default=3)
    
    Returns:
        bin_index: int (0 ~ 3^12-1)
    """
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    
    def joint_to_bin(joint):
        # [lower, upper] → [0, 1] → [0, num_divisions-1]
        normalized = (joint - lower) / (upper - lower + 1e-8)
        bin_idx = (normalized * num_divisions).long().clamp(0, num_divisions - 1)
        return bin_idx
    
    start_bins = joint_to_bin(start_joint)  # [6]
    goal_bins = joint_to_bin(goal_joint)    # [6]
    all_bins = torch.cat([start_bins, goal_bins])  # [12]
    
    # 12D → 1D index (혼합 기수 변환)
    index = 0
    for i, b in enumerate(all_bins):
        index += b.item() * (num_divisions ** i)
    
    return index


def compute_bin_indices_batch(start_joints, goal_joints, joint_limits, num_divisions=3):
    """
    배치로 bin indices 계산 (빠른 버전)
    
    Args:
        start_joints: [N, 6] tensor
        goal_joints: [N, 6] tensor
        joint_limits: [6, 2] tensor
        num_divisions: 등분 수
    
    Returns:
        bin_indices: [N] tensor (int64)
    """
    lower = joint_limits[:, 0].unsqueeze(0)  # [1, 6]
    upper = joint_limits[:, 1].unsqueeze(0)  # [1, 6]
    
    def joints_to_bins(joints):
        normalized = (joints - lower) / (upper - lower + 1e-8)
        bin_idx = (normalized * num_divisions).long().clamp(0, num_divisions - 1)
        return bin_idx
    
    start_bins = joints_to_bins(start_joints)  # [N, 6]
    goal_bins = joints_to_bins(goal_joints)    # [N, 6]
    all_bins = torch.cat([start_bins, goal_bins], dim=1)  # [N, 12]
    
    # 12D → 1D index (배치 버전)
    multipliers = torch.tensor([num_divisions ** i for i in range(12)], 
                               dtype=torch.int64, device=all_bins.device)
    bin_indices = (all_bins * multipliers.unsqueeze(0)).sum(dim=1)
    
    return bin_indices


def main():
    parser = argparse.ArgumentParser(description="Preprocess reachability data with binning")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to reachable_set.pt (default: outputs/data/reachable_set.pt)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path (default: outputs/data/reachable_set_binned.pt)")
    parser.add_argument("--num-divisions", type=int, default=3,
                        help="Number of divisions per joint (default: 3)")
    parser.add_argument("--batch-size", type=int, default=100000,
                        help="Batch size for processing (default: 100000)")
    args = parser.parse_args()
    
    # 경로 설정
    if args.data_path is None:
        data_path = os.path.join(ROOT_DIR, "outputs/data/reachable_set.pt")
    else:
        data_path = args.data_path
    
    if args.output_path is None:
        output_path = os.path.join(ROOT_DIR, "outputs/data/reachable_set_binned.pt")
    else:
        output_path = args.output_path
    
    print(f"=== Reachability Data Preprocessing ===")
    print(f"Input: {data_path}")
    print(f"Output: {output_path}")
    print(f"Num divisions: {args.num_divisions}")
    print(f"Total bins: {args.num_divisions ** 12:,}")
    
    # 데이터 로드
    print(f"\nLoading data...")
    start_time = time.time()
    data = torch.load(data_path, map_location='cpu')
    
    start_joint = data['start_joint']  # [N, 6]
    goal_joint = data['goal_joint']    # [N, 6]
    q_final = data['q_final']          # [N, 4]
    metadata = data.get('metadata', {})
    
    N = start_joint.shape[0]
    print(f"Loaded {N:,} samples in {time.time() - start_time:.1f}s")
    print(f"  start_joint: {start_joint.shape}")
    print(f"  goal_joint: {goal_joint.shape}")
    print(f"  q_final: {q_final.shape}")
    
    # Joint limits 설정
    if 'joint_min_rad' in metadata and 'joint_max_rad' in metadata:
        joint_min = metadata['joint_min_rad']
        joint_max = metadata['joint_max_rad']
    else:
        # 기본값: -140도 ~ 140도 (약 -2.44 ~ 2.44 rad)
        joint_min = -2.44
        joint_max = 2.44
    
    joint_limits = torch.tensor([[joint_min, joint_max]] * 6, dtype=torch.float32)  # [6, 2]
    print(f"\nJoint limits: [{joint_min:.4f}, {joint_max:.4f}] rad")
    
    # Bin indices 계산 (배치 처리)
    print(f"\nComputing bin indices...")
    bin_indices = torch.zeros(N, dtype=torch.int64)
    
    num_batches = (N + args.batch_size - 1) // args.batch_size
    with tqdm(total=N, desc="Computing bins", unit="samples") as pbar:
        for i in range(num_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, N)
            
            batch_indices = compute_bin_indices_batch(
                start_joint[start_idx:end_idx],
                goal_joint[start_idx:end_idx],
                joint_limits,
                args.num_divisions
            )
            bin_indices[start_idx:end_idx] = batch_indices
            pbar.update(end_idx - start_idx)
    
    # Bin별로 q_final 그룹화
    print(f"\nGrouping q_finals by bin...")
    bin_to_indices = defaultdict(list)
    
    for i in tqdm(range(N), desc="Grouping", unit="samples"):
        bin_idx = bin_indices[i].item()
        bin_to_indices[bin_idx].append(i)
    
    # bin_to_qfinals 생성 (dict of tensors)
    print(f"\nCreating bin_to_qfinals...")
    bin_to_qfinals = {}
    for bin_idx, sample_indices in tqdm(bin_to_indices.items(), desc="Creating tensors"):
        indices_tensor = torch.tensor(sample_indices, dtype=torch.long)
        bin_to_qfinals[bin_idx] = q_final[indices_tensor]  # [K, 4]
    
    # 통계 출력
    non_empty_bins = len(bin_to_qfinals)
    total_bins = args.num_divisions ** 12
    
    samples_per_bin = [len(indices) for indices in bin_to_indices.values()]
    min_samples = min(samples_per_bin) if samples_per_bin else 0
    max_samples = max(samples_per_bin) if samples_per_bin else 0
    avg_samples = sum(samples_per_bin) / len(samples_per_bin) if samples_per_bin else 0
    
    print(f"\n=== Binning Statistics ===")
    print(f"Total bins: {total_bins:,}")
    print(f"Non-empty bins: {non_empty_bins:,} ({100*non_empty_bins/total_bins:.2f}%)")
    print(f"Empty bins: {total_bins - non_empty_bins:,}")
    print(f"Samples per bin:")
    print(f"  Min: {min_samples}")
    print(f"  Max: {max_samples}")
    print(f"  Avg: {avg_samples:.1f}")
    
    # 저장
    print(f"\nSaving preprocessed data...")
    preprocessed = {
        'start_joint': start_joint,
        'goal_joint': goal_joint,
        'q_final': q_final,
        'bin_indices': bin_indices,
        'bin_to_qfinals': bin_to_qfinals,
        'joint_limits': joint_limits,
        'num_divisions': args.num_divisions,
        'metadata': metadata,
        'stats': {
            'total_samples': N,
            'total_bins': total_bins,
            'non_empty_bins': non_empty_bins,
            'min_samples_per_bin': min_samples,
            'max_samples_per_bin': max_samples,
            'avg_samples_per_bin': avg_samples
        }
    }
    
    torch.save(preprocessed, output_path)
    
    total_time = time.time() - start_time
    print(f"\n=== Preprocessing Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
