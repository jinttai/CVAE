"""RRT 노드 상태 정의.

비홀로노믹 시스템이므로 상태 = (관절 위치, 베이스 자세).
같은 관절 위치라도 경로에 따라 베이스 자세가 달라진다.
"""

from dataclasses import dataclass
import torch


@dataclass
class State:
    q_joints: torch.Tensor    # [n_joints] 관절 위치 (rad)
    R_base: torch.Tensor      # [3, 3]    베이스 rotation matrix (SO(3))

    def clone(self) -> "State":
        return State(
            q_joints=self.q_joints.clone(),
            R_base=self.R_base.clone(),
        )
