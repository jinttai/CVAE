"""Steering 함수: 현재 상태에서 목표 방향으로 한 스텝 전진.

관절 변화량을 정하고 SPART 동역학으로 베이스 자세를 전파한다.
핵심: u0 = -H0^{-1} H0m q_dot (운동량 보존)
"""

import torch
from .state import State
from .config import RRTConfig


class Steerer:
    """SPART 동역학 기반 local planner."""

    def __init__(self, robot: dict, config: RRTConfig):
        self.robot = robot
        self.config = config

    def steer(self, from_state: State, toward_q: torch.Tensor) -> State:
        """from_state에서 toward_q 방향으로 step_size만큼 이동.

        Args:
            from_state: 현재 노드 상태
            toward_q: 목표 관절 위치 (샘플 or goal)

        Returns:
            새로운 State (관절 + 전파된 베이스 자세)
        """
        # 1) 관절 변화 방향 & 크기 제한
        delta = toward_q - from_state.q_joints
        dist = torch.norm(delta)
        if dist > self.config.step_size:
            delta = delta / dist * self.config.step_size

        q_new = from_state.q_joints + delta
        q_new = torch.clamp(q_new, self.config.joint_min, self.config.joint_max)

        # 2) 서브스텝으로 나눠서 동역학 전파
        R_curr = from_state.R_base.clone()
        q_curr = from_state.q_joints.clone()
        q_step = (q_new - q_curr) / self.config.n_substeps

        for _ in range(self.config.n_substeps):
            q_dot = q_step / self.config.dt
            R_curr = self._propagate_rotation(q_curr, q_dot, R_curr)
            q_curr = q_curr + q_step

        return State(q_joints=q_new, R_base=R_curr)

    def _propagate_rotation(
        self, q: torch.Tensor, q_dot: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """SPART 동역학으로 베이스 각속도 계산 → rotation matrix 적분.

        TODO: spart_functions_torch에서 실제 함수 연결
            1) kinematics(robot, q, R) → RJ, RL, ...
            2) diff_kinematics(robot, ...) → Bij, Bi0, P0, pm
            3) generalized_inertia_matrix(...) → H0, H0m, Hm
            4) u0 = -H0^{-1} @ H0m @ q_dot
            5) R_new = R @ rodrigues(u0 * dt)
        """
        raise NotImplementedError("SPART 동역학 연결 필요")

    @staticmethod
    def _rodrigues(omega: torch.Tensor, dt: float) -> torch.Tensor:
        """각속도 → 회전 행렬 (Rodrigues' formula)."""
        theta = torch.norm(omega) * dt
        if theta < 1e-8:
            return torch.eye(3, device=omega.device)
        k = omega / torch.norm(omega)
        K = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ], device=omega.device)
        return torch.eye(3, device=omega.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
