"""RRT 하이퍼파라미터 설정."""

from dataclasses import dataclass
import torch


@dataclass
class RRTConfig:
    # 트리 탐색
    max_iter: int = 50000           # 최대 반복 수
    goal_tolerance: float = 0.1     # 목표 orientation 허용 오차 (chordal)
    goal_bias: float = 0.1          # goal 방향 샘플링 확률

    # steering
    step_size: float = 0.3          # 관절 공간 최대 스텝 (rad)
    n_substeps: int = 10            # steer 내 물리 시뮬레이션 서브스텝
    dt: float = 0.1                 # 시뮬레이션 타임스텝 (s)

    # 관절 제한
    joint_min: float = -2.44        # rad (~-140°)
    joint_max: float = 2.44         # rad (~+140°)
    n_joints: int = 6

    # 데이터 수집
    max_trajectories: int = 10000   # 수집할 총 궤적 수
    smoothing: bool = True          # 후처리 스무딩 적용 여부

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
