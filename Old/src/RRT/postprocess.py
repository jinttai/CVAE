"""RRT 경로 후처리 → CVAE 학습 데이터 형식으로 변환.

RRT 경로 (가변 길이 State 리스트) 를:
1) 스무딩 (shortcut)
2) 리샘플링 (고정 타임스텝)
3) 웨이포인트 추출 (CVAE 입력 형식)
4) 저장
"""

import torch
from typing import List
from .state import State
from .config import RRTConfig


def smooth_path(path: List[State], steerer, max_attempts: int = 200) -> list[State]:
    """Shortcut smoothing: 랜덤 두 점을 직접 연결 시도."""
    # TODO: 구현
    return path


def resample_path(path: List[State], num_steps: int = 100) -> list[State]:
    """등간격 리샘플링 (관절 공간 선형보간 + 동역학 재전파)."""
    # TODO: 구현
    return path


def extract_waypoints(path: List[State], num_waypoints: int = 3) -> torch.Tensor:
    """경로에서 CVAE 형식 웨이포인트 추출.

    Returns:
        [num_waypoints, n_joints] 텐서 (= CVAE decoder 출력과 동일한 형식)
    """
    n = len(path)
    indices = torch.linspace(0, n - 1, num_waypoints + 2).long()[1:-1]  # 양 끝 제외
    waypoints = torch.stack([path[i].q_joints for i in indices])
    return waypoints


def path_to_training_sample(
    path: List[State],
    R_goal: torch.Tensor,
    num_waypoints: int = 3,
) -> dict:
    """RRT 경로 하나를 CVAE 학습 샘플로 변환.

    Returns:
        {
            "condition": [8] (q0_start + q0_goal quaternion),
            "waypoints": [num_waypoints * n_joints],
            "q_trajectory": [num_steps, n_joints],
        }
    """
    # TODO: rotation matrix → quaternion 변환 등
    waypoints = extract_waypoints(path, num_waypoints)
    return {
        "condition": None,       # 구현 필요
        "waypoints": waypoints.flatten(),
        "q_trajectory": torch.stack([s.q_joints for s in path]),
    }
