"""RRT 거리 메트릭.

- 관절 거리: L2 norm
- 자세 거리: chordal distance (||R1 - R2||_F)
- 결합 거리: 가중합
"""

import torch


def joint_distance(q1: torch.Tensor, q2: torch.Tensor) -> float:
    """관절 공간 L2 거리."""
    return torch.norm(q1 - q2).item()


def orientation_distance(R1: torch.Tensor, R2: torch.Tensor) -> float:
    """Chordal distance between rotation matrices."""
    diff = R1 - R2
    return torch.sqrt(0.5 * torch.sum(diff * diff)).item()


def combined_distance(state1, state2, w_joint: float = 0.3, w_orient: float = 0.7) -> float:
    """관절 + 자세 가중 거리. RRT nearest neighbor에 사용."""
    d_joint = joint_distance(state1.q_joints, state2.q_joints)
    d_orient = orientation_distance(state1.R_base, state2.R_base)
    return w_joint * d_joint + w_orient * d_orient
