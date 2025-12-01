import torch
import sys
import os

# -------------------------------------------------------------------------
# [경로 설정] 프로젝트 루트를 sys.path에 추가하여 모듈 import 에러 방지
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/training
src_dir = os.path.dirname(current_dir)                    # src
root_dir = os.path.dirname(src_dir)                       # space_robot_planning_Rmat

if src_dir not in sys.path:
    sys.path.append(src_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import dynamics.spart_functions_torch as spart
from torch.func import vmap  # Auto-Batching


class PhysicsLayer:
    """
    Rotation-Matrix 기반 물리 레이어.

    - 기존 버전은 쿼터니언 q 를 미분/적분(Euler, RK4)하여 자세를 추적
    - 이 버전은 각속도 wb 로부터 회전행렬 R 을 타임스텝마다 곱해 나가는 방식으로 진행
      (R_{k+1} = R_k @ R_delta(wb, dt))
    - 외부 인터페이스는 기존 PhysicsLayer 와 동일하게 유지:
        - generate_trajectory
        - simulate_single
        - calculate_loss
      단, 내부적으로만 회전행렬을 사용.
    """

    def __init__(self, robot, num_waypoints, total_time, device):
        self.robot = robot
        self.n_q = robot["n_q"]
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        self.dt = 0.1  # Simulation step size (0.1s)
        self.num_steps = int(total_time / self.dt)
        self.device = device

        # Spline Basis 미리 생성 (GPU 메모리에 상주)
        self.basis = self._get_spline_basis(self.num_waypoints, self.num_steps).to(device)

    # ------------------------------------------------------------------
    # Trajectory Generation (원본과 동일)
    # ------------------------------------------------------------------
    def _get_spline_basis(self, num_points, num_steps):
        """
        Linear Spline Basis Matrix 생성
        """
        basis = torch.zeros((num_steps, num_points + 2))
        t = torch.linspace(0, 1, num_steps)
        segment_len = 1.0 / (num_points + 1)

        for i in range(num_points + 2):
            center = i * segment_len
            dist = torch.abs(t - center)
            val = torch.clamp(1 - dist / segment_len, 0, 1)
            basis[:, i] = val
        return basis

    def generate_trajectory(self, waypoints_flat):
        """
        [Batch, Waypoints*Joints] -> [Batch, Steps, Joints] (Pos, Vel)
        """
        batch_size = waypoints_flat.size(0)
        w_mid = waypoints_flat.view(batch_size, self.num_waypoints, self.n_q)

        # 시작점(0)과 끝점(0)은 고정 (속도 0 제약 조건을 간접적으로 부여)
        zeros = torch.zeros(batch_size, 1, self.n_q, device=self.device)
        w_full = torch.cat([zeros, w_mid, zeros], dim=1)

        # Matrix Multiplication으로 보간 (고속)
        q_traj = torch.einsum("st,btj->bsj", self.basis, w_full)

        # 수치 미분으로 속도 계산
        q_dot_traj = torch.zeros_like(q_traj)
        q_dot_traj[:, 1:, :] = (q_traj[:, 1:, :] - q_traj[:, :-1, :]) / self.dt

        return q_traj, q_dot_traj

    # ------------------------------------------------------------------
    # 회전행렬 유틸리티
    # ------------------------------------------------------------------
    def _quat_to_rot(self, q):
        """
        쿼터니언 q = [x, y, z, w] 를 회전행렬 R (3x3) 로 변환.
        입력: 1D 텐서 (4,)
        """
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = torch.empty((3, 3), device=self.device, dtype=q.dtype)
        R[0, 0] = 1.0 - 2.0 * (yy + zz)
        R[0, 1] = 2.0 * (xy - wz)
        R[0, 2] = 2.0 * (xz + wy)

        R[1, 0] = 2.0 * (xy + wz)
        R[1, 1] = 1.0 - 2.0 * (xx + zz)
        R[1, 2] = 2.0 * (yz - wx)

        R[2, 0] = 2.0 * (xz - wy)
        R[2, 1] = 2.0 * (yz + wx)
        R[2, 2] = 1.0 - 2.0 * (xx + yy)
        return R

    def _skew(self, v):
        """
        3D 벡터 v에 대한 skew-symmetric 행렬 [v]_x
        """
        vx, vy, vz = v
        M = torch.tensor(
            [
                [0.0, -vz, vy],
                [vz, 0.0, -vx],
                [-vy, vx, 0.0],
            ],
            device=self.device,
            dtype=v.dtype,
        )
        return M

    def _rot_from_omega(self, wb, dt):
        """
        각속도 wb 에 대해 dt 동안의 회전을 나타내는 회전행렬 R_delta 계산.
        Rodrigues 공식을 사용.
        """
        theta = torch.linalg.norm(wb) * dt
        if theta < 1e-8:
            # 매우 작은 회전: 1차 근사 (I + [w*dt]_x)
            K = self._skew(wb * dt)
            return torch.eye(3, device=self.device, dtype=wb.dtype) + K

        axis = wb / (torch.linalg.norm(wb) + 1e-12)
        K = self._skew(axis)
        I = torch.eye(3, device=self.device, dtype=wb.dtype)
        R_delta = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
        return R_delta

    # ------------------------------------------------------------------
    # 핵심 시뮬레이션 (회전행렬 기반)
    # ------------------------------------------------------------------
    def simulate_single(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Core Physics Engine - Rotation Matrix Version]

        - 각 스텝에서 SPART 동역학을 통해 각속도 wb 를 구함
        - 회전행렬 R 를 R_{k+1} = R_k @ R_delta(wb, dt) 로 업데이트
        - 최종 R 와 목표 쿼터니언 q0_goal 을 회전행렬로 변환한 R_goal 간의
          각도 오차를 반환 (angle_error^2)
        """
        # 기준 좌표계
        R0 = torch.eye(3, device=self.device)
        r0 = torch.zeros(3, device=self.device)

        # 초기/목표 자세를 회전행렬로 변환
        R_curr = self._quat_to_rot(q0_init)
        R_goal = self._quat_to_rot(q0_goal)

        for t in range(self.num_steps):
            qm = q_traj[t]
            qd = q_dot_traj[t]

            # --- 1. SPART Dynamics Calculations (기존과 동일) ---
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
            I0, Im = spart.inertia_projection(R0, RL, self.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)

            # --- 2. Non-holonomic Constraint Solver --- 
            rhs = -H0m @ qd
            H0_damped = H0 + 1e-6 * torch.eye(6, device=self.device)
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]  # Angular Velocity part

            # --- 3. Rotation Matrix Integration ---
            R_delta = self._rot_from_omega(wb, self.dt)
            R_curr = R_curr @ R_delta

        # --- 4. Final Orientation Error ---
        # 상대 회전 행렬 R_err = R_goal^T * R_curr
        R_err = R_goal.T @ R_curr
        trace = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = torch.acos(trace) * 2.0  # 동일한 convention 유지
        return angle_error ** 2

    def calculate_loss(self, waypoints_flat, q0_init, q0_goal):
        """
        Batched Physics Simulation using vmap (Rotation Matrix Version)
        """
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat)

        # simulate_single 을 배치 차원에 대해 병렬화
        batch_sim_fn = vmap(self.simulate_single, in_dims=(0, 0, 0, 0))
        loss_batch = batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)
        return loss_batch.mean()


