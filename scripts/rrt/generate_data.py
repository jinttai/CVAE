"""RRT로 궤적 데이터 생성 → CVAE 학습용 저장.

Usage:
    python scripts/rrt/generate_data.py --num_trajectories 10000 --output outputs/data/rrt
"""

import argparse
import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.RRT import RRTPlanner, RRTConfig, State
from src.RRT.postprocess import path_to_training_sample, smooth_path
from src.dynamics.urdf2robot_torch import urdf2robot
from scenario import scenario


def random_goal_rotation(max_angle_deg: float = 60.0) -> torch.Tensor:
    """랜덤 목표 자세 생성 (axis-angle → rotation matrix)."""
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)
    angle = torch.empty(1).uniform_(0, max_angle_deg * torch.pi / 180).item()

    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    return R


def make_start_state(device: str) -> State:
    """초기 상태: 관절 0, 베이스 identity."""
    return State(
        q_joints=torch.zeros(6, device=device),
        R_base=torch.eye(3, device=device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajectories", type=int, default=10000)
    parser.add_argument("--output", type=str, default="outputs/data/rrt")
    parser.add_argument("--max_angle", type=float, default=60.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 로봇 모델 로드
    robot = urdf2robot(scenario["urdf_path"])
    config = RRTConfig(max_trajectories=args.num_trajectories)
    planner = RRTPlanner(robot, config)

    # 데이터 수집
    dataset = []
    success_count = 0

    pbar = tqdm(total=args.num_trajectories, desc="RRT data generation")
    while success_count < args.num_trajectories:
        R_goal = random_goal_rotation(args.max_angle).to(config.device)
        start = make_start_state(config.device)

        path = planner.plan(start, R_goal)
        if path is None:
            continue

        # 후처리
        if config.smoothing:
            path = smooth_path(path, planner.steerer)

        sample = path_to_training_sample(path, R_goal)
        dataset.append(sample)
        success_count += 1
        pbar.update(1)

    pbar.close()

    # 저장
    save_path = os.path.join(args.output, "rrt_dataset.pt")
    torch.save(dataset, save_path)
    print(f"Saved {len(dataset)} trajectories to {save_path}")


if __name__ == "__main__":
    main()
