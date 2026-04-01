"""RRT 트리 구조 및 핵심 알고리즘.

1) 랜덤 관절 샘플링 (goal bias 포함)
2) nearest neighbor 탐색
3) steer → 새 노드 추가
4) goal 도달 확인
5) 경로 추출 → 궤적 데이터로 변환
"""

from __future__ import annotations
from typing import Optional
import torch

from .state import State
from .steer import Steerer
from .distance import combined_distance
from .config import RRTConfig


class Node:
    """트리 노드."""

    def __init__(self, state: State, parent: Optional[Node] = None):
        self.state = state
        self.parent = parent


class RRTPlanner:
    """RRT 기반 궤적 플래너."""

    def __init__(self, robot: dict, config: RRTConfig):
        self.config = config
        self.steerer = Steerer(robot, config)
        self.nodes: list[Node] = []

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def plan(self, start: State, R_goal: torch.Tensor) -> Optional[list[State]]:
        """start에서 R_goal까지 RRT 탐색.

        Returns:
            성공 시 State 리스트 (start → goal), 실패 시 None
        """
        self.nodes = [Node(start)]

        for _ in range(self.config.max_iter):
            # 1) 샘플
            q_sample = self._sample(R_goal)

            # 2) nearest
            nearest_node = self._nearest(q_sample)

            # 3) steer
            new_state = self.steerer.steer(nearest_node.state, q_sample)
            new_node = Node(new_state, parent=nearest_node)
            self.nodes.append(new_node)

            # 4) goal 체크
            if self._is_goal_reached(new_state, R_goal):
                return self._extract_path(new_node)

        return None  # 실패

    # ------------------------------------------------------------------
    # 서브루틴
    # ------------------------------------------------------------------
    def _sample(self, R_goal: torch.Tensor) -> torch.Tensor:
        """관절 공간 랜덤 샘플 (goal bias 포함)."""
        cfg = self.config
        if torch.rand(1).item() < cfg.goal_bias:
            # goal bias: 현재 트리에서 goal에 가장 가까운 노드의 관절 근처 샘플
            # (단순 구현: 랜덤 관절 반환 — goal은 관절 공간에 직접 대응 없음)
            pass
        return torch.empty(cfg.n_joints).uniform_(cfg.joint_min, cfg.joint_max)

    def _nearest(self, q_target: torch.Tensor) -> Node:
        """가장 가까운 노드 탐색 (brute-force)."""
        # 임시 State (R_base=I) 로 관절 거리만 사용
        target_state = State(q_joints=q_target, R_base=torch.eye(3))
        best_node = self.nodes[0]
        best_dist = float("inf")
        for node in self.nodes:
            d = combined_distance(node.state, target_state)
            if d < best_dist:
                best_dist = d
                best_node = node
        return best_node

    def _is_goal_reached(self, state: State, R_goal: torch.Tensor) -> bool:
        """chordal distance로 goal 도달 판정."""
        from .distance import orientation_distance
        return orientation_distance(state.R_base, R_goal) < self.config.goal_tolerance

    def _extract_path(self, goal_node: Node) -> list[State]:
        """goal 노드에서 root까지 역추적."""
        path = []
        node = goal_node
        while node is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()
        return path
