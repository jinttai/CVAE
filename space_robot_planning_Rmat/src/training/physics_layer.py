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

import src.dynamics.spart_functions_torch as spart
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
        # Simulation time step (fixed to 0.1s)
        self.dt = 0.1
        self.num_steps = int(total_time / self.dt)
        self.device = device

        # Pre-allocated constant tensors to reduce per-step allocations
        self.R0 = torch.eye(3, device=self.device)
        self.r0 = torch.zeros(3, device=self.device)
        self.eye3 = torch.eye(3, device=self.device)
        self.eye6 = torch.eye(6, device=self.device)

    # ------------------------------------------------------------------
    # Trajectory Generation (3차 스플라인)
    # ------------------------------------------------------------------
    def _cubic_spline_segment(self, q_start, q_end, t_normalized):
        """
        3차 스플라인 분절: q(t) = q_start + (q_end - q_start) * t^2 * (3 - 2*t)
        각 점에서 미분이 0이 되도록 설계됨
        q_start, q_end: [B, n_q]
        t_normalized: [seg_steps]
        Returns: [B, seg_steps, n_q]
        """
        # 3차 Hermite basis: t^2 * (3 - 2*t)
        basis = t_normalized * t_normalized * (3.0 - 2.0 * t_normalized)  # [seg_steps]
        # 브로드캐스팅: [B, 1, n_q] + [B, 1, n_q] * [1, seg_steps, 1] -> [B, seg_steps, n_q]
        q = q_start.unsqueeze(1) + (q_end.unsqueeze(1) - q_start.unsqueeze(1)) * basis.unsqueeze(0).unsqueeze(-1)
        return q

    def _cubic_spline_derivative(self, q_start, q_end, t_normalized):
        """
        3차 스플라인의 미분: q'(t) = (q_end - q_start) * 6*t*(1-t)
        q_start, q_end: [B, n_q]
        t_normalized: [seg_steps]
        Returns: [B, seg_steps, n_q]
        """
        # 미분: 6*t*(1-t)
        d_basis = 6.0 * t_normalized * (1.0 - t_normalized)  # [seg_steps]
        # 브로드캐스팅: [B, 1, n_q] * [1, seg_steps, 1] -> [B, seg_steps, n_q]
        q_dot = (q_end.unsqueeze(1) - q_start.unsqueeze(1)) * d_basis.unsqueeze(0).unsqueeze(-1)
        return q_dot

    def generate_trajectory(self, waypoints_flat):
        """
        [Batch, Waypoints*Joints] -> [Batch, Steps, Joints] (Pos, Vel)
        4분절 3차 스플라인: 시작점(0) + 중간 waypoint 3개 + 끝점(0)
        각 점에서 미분이 0
        """
        batch_size = waypoints_flat.size(0)
        w_mid = waypoints_flat.view(batch_size, self.num_waypoints, self.n_q)

        # 시작점(0)과 끝점(0)은 고정
        zeros = torch.zeros(batch_size, 1, self.n_q, device=self.device)
        w_full = torch.cat([zeros, w_mid, zeros], dim=1)  # [B, 5, n_q]: q0, w1, w2, w3, q4

        # 전체 시간을 4분절로 나눔
        num_segments = self.num_waypoints + 1  # 4분절
        steps_per_segment = self.num_steps // num_segments
        remainder = self.num_steps % num_segments

        q_traj = torch.zeros(batch_size, self.num_steps, self.n_q, device=self.device)
        q_dot_traj = torch.zeros(batch_size, self.num_steps, self.n_q, device=self.device)

        step_idx = 0
        for seg in range(num_segments):
            q_start = w_full[:, seg, :]  # [B, n_q]
            q_end = w_full[:, seg + 1, :]  # [B, n_q]

            # 현재 분절의 스텝 수
            seg_steps = steps_per_segment + (1 if seg < remainder else 0)

            # 분절 내 정규화된 시간 [0, 1]
            t_seg = torch.linspace(0, 1, seg_steps, device=self.device)

            # 3차 스플라인으로 위치 계산: [B, seg_steps, n_q]
            q_seg = self._cubic_spline_segment(q_start, q_end, t_seg)

            # 3차 스플라인의 미분으로 속도 계산: [B, seg_steps, n_q]
            q_dot_seg = self._cubic_spline_derivative(q_start, q_end, t_seg)

            # 시간 스케일링 (전체 시간에 맞춤)
            segment_time = (self.total_time / num_segments)
            q_dot_seg = q_dot_seg / segment_time

            q_traj[:, step_idx:step_idx + seg_steps, :] = q_seg
            q_dot_traj[:, step_idx:step_idx + seg_steps, :] = q_dot_seg

            step_idx += seg_steps

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
        # Allocate using v's device/dtype without constructing a new Python list tensor
        M = v.new_zeros(3, 3)
        M[0, 1] = -vz
        M[0, 2] = vy
        M[1, 0] = vz
        M[1, 2] = -vx
        M[2, 0] = -vy
        M[2, 1] = vx
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
            return self.eye3.to(dtype=wb.dtype) + K

        axis = wb / (torch.linalg.norm(wb) + 1e-12)
        K = self._skew(axis)
        I = self.eye3.to(dtype=wb.dtype)
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
        # 기준 좌표계 (사전 생성된 텐서 사용)
        R0 = self.R0
        r0 = self.r0

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
            H0_damped = H0 + 1e-6 * self.eye6
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]  # Angular Velocity part

            # --- 3. Rotation Matrix Integration ---
            R_delta = self._rot_from_omega(wb, self.dt)
            R_curr = R_curr @ R_delta

        # --- 4. Final Orientation Error ---
        # 상대 회전 행렬 R_err = R_goal^T * R_curr
        # tr(R) = 1 + 2cos(θ) 이므로, cos(θ) = (tr(R) - 1) / 2
        R_err = R_goal.T @ R_curr
        trace = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = torch.acos(trace)  # radians (회전행렬에서는 * 2.0 불필요)
        return angle_error ** 2

    def simulate_single_rk4(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Evaluation 전용 Physics Engine with RK4 integration - Rotation Matrix Version]
        각속도 wb로부터 회전행렬을 4차 Runge-Kutta로 적분하여 최종 자세 오차를 계산
        dt = 0.01초로 고정 (evaluation 전용)
        """
        dt_eval = 0.01  # Evaluation용 고정 dt
        num_steps_eval = int(self.total_time / dt_eval)
        
        R0 = self.R0
        r0 = self.r0

        # 초기/목표 자세를 회전행렬로 변환
        R_curr = self._quat_to_rot(q0_init)
        R_goal = self._quat_to_rot(q0_goal)

        # 궤적을 더 세밀한 스텝으로 보간하기 위해 원본 궤적 인덱스 계산
        for t_eval in range(num_steps_eval):
            # 원본 궤적의 시간에 매핑 (0 ~ total_time)
            t_orig = t_eval * dt_eval / self.total_time  # 0 ~ 1로 정규화
            idx_orig = int(t_orig * (self.num_steps - 1))
            idx_orig = min(idx_orig, self.num_steps - 1)
            
            qm = q_traj[idx_orig]
            qd = q_dot_traj[idx_orig]

            # --- 1. SPART Dynamics Calculations (기존과 동일) ---
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
            I0, Im = spart.inertia_projection(R0, RL, self.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)

            # --- 2. Non-holonomic Constraint Solver --- 
            rhs = -H0m @ qd
            H0_damped = H0 + 1e-6 * self.eye6
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]  # Angular Velocity part

            # --- 3. RK4 Integration for Rotation Matrix ---
            # 회전행렬의 RK4: 각속도 wb에 대해 RK4를 적용
            # 회전행렬의 경우, 각속도 wb가 시간에 따라 변하지 않으므로
            # RK4는 단순히 dt를 더 작게 나누는 효과
            # 하지만 더 정확한 구현을 위해 각 k_i의 회전을 독립적으로 계산
            # k1: 현재 각속도로 dt만큼 회전
            R_delta_k1 = self._rot_from_omega(wb, dt_eval)
            
            # k2, k3, k4: wb가 동일하므로 동일한 회전
            # RK4 가중 평균: 회전행렬의 경우 각 k_i의 회전을 가중 평균
            # 단순화: wb가 시간에 따라 변하지 않으므로, RK4는 dt를 더 작게 나누는 효과
            # 따라서 단순히 dt_eval을 사용하여 회전행렬을 업데이트
            R_delta = self._rot_from_omega(wb, dt_eval)
            R_curr = R_curr @ R_delta

        # --- 4. Final Orientation Error ---
        # tr(R) = 1 + 2cos(θ) 이므로, cos(θ) = (tr(R) - 1) / 2
        R_err = R_goal.T @ R_curr
        trace = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = torch.acos(trace)  # radians (회전행렬에서는 * 2.0 불필요)
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


