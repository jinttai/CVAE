import torch
import sys
import os

from torch.func import vmap

# -------------------------------------------------------------------------
# [경로 설정] 프로젝트 / 워크스페이스 루트를 sys.path에 추가하여 import 에러 방지
#   - Windows PowerShell에서 cwd가 space_robot_planning_rk4일 때도
#     상위 워크스페이스(`<workspace_root>/CVAE/src`)의 `src.dynamics`를 찾을 수 있게 함.
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../space_robot_planning_rk4/src/training
src_dir = os.path.dirname(current_dir)                     # .../space_robot_planning_rk4/src
root_dir = os.path.dirname(src_dir)                        # .../space_robot_planning_rk4
workspace_root = os.path.dirname(root_dir)                 # .../CVAE

for p in (src_dir, root_dir, workspace_root):
    if p not in sys.path:
        sys.path.append(p)

# dynamics는 원본 프로젝트 패키지인 space_robot_planning 아래에서 가져온다.
from space_robot_planning.src.dynamics import spart_functions_torch as spart


class PhysicsLayer:
    """
    Quaternion RK4 적분을 **학습부터** 사용하는 rk4 전용 PhysicsLayer.

    - generate_trajectory: 4분절 3차 스플라인 기반 궤적/속도 생성
    - simulate_single: (Euler, 호환용)
    - simulate_single_rk4: dt=0.01, 평가 전용 RK4 (evaluate에서 사용)
    - _simulate_single_rk4_train: dt=self.dt, 학습/최적화용 RK4
    - calculate_loss: 학습/최적화용 RK4 기반 loss (vmap + _simulate_single_rk4_train)
    """

    def __init__(self, robot, num_waypoints, total_time, device):
        self.robot = robot
        self.n_q = robot['n_q']
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        self.dt = 0.1  # Simulation step size (0.1s)
        self.num_steps = int(total_time / self.dt)
        self.device = device

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

        # waypoints_flat: [B, NUM_WAYPOINTS * n_q] (중간 3개만 포함)
        waypoints = waypoints_flat.view(batch_size, self.num_waypoints, self.n_q)  # [B, 3, n_q]

        # 시작점/끝점은 정해진 상태라고 가정 (여기서는 모두 0으로 두고, 쿼터니언 w=1)
        q_start = torch.zeros(batch_size, self.n_q, device=self.device)
        q_start[..., 3] = 1.0  # 단위 쿼터니언

        q_end = torch.zeros(batch_size, self.n_q, device=self.device)
        q_end[..., 3] = 1.0

        # 5개 점: [start] + 3개 waypoint + [end]
        control_points = torch.cat([q_start.unsqueeze(1), waypoints, q_end.unsqueeze(1)], dim=1)  # [B, 5, n_q]

        # 분절 수: 4 (0-1, 1-2, 2-3, 3-4)
        num_segments = control_points.size(1) - 1

        # 각 분절당 스텝 수 (정수 나눗셈)
        seg_steps = self.num_steps // num_segments
        t_local = torch.linspace(0.0, 1.0, seg_steps, device=self.device)

        qs = []
        qdots = []

        for i in range(num_segments):
            q_s = control_points[:, i, :]      # [B, n_q]
            q_e = control_points[:, i + 1, :]  # [B, n_q]

            q_seg = self._cubic_spline_segment(q_s, q_e, t_local)          # [B, seg_steps, n_q]
            qd_seg = self._cubic_spline_derivative(q_s, q_e, t_local)      # [B, seg_steps, n_q]

            qs.append(q_seg)
            qdots.append(qd_seg)

        q_traj = torch.cat(qs, dim=1)      # [B, num_segments*seg_steps, n_q]
        q_dot_traj = torch.cat(qdots, dim=1)

        # self.num_steps 와 맞추기 위해 잘라내기/패딩
        if q_traj.size(1) > self.num_steps:
            q_traj = q_traj[:, :self.num_steps, :]
            q_dot_traj = q_dot_traj[:, :self.num_steps, :]
        elif q_traj.size(1) < self.num_steps:
            pad_steps = self.num_steps - q_traj.size(1)
            pad_q = q_traj[:, -1:, :].repeat(1, pad_steps, 1)
            pad_qd = torch.zeros_like(pad_q)
            q_traj = torch.cat([q_traj, pad_q], dim=1)
            q_dot_traj = torch.cat([q_dot_traj, pad_qd], dim=1)

        return q_traj, q_dot_traj

    # ------------------------------------------------------------------
    # Quaternion 도함수 (wb 기반)
    # ------------------------------------------------------------------
    def _quat_deriv(self, q, wb):
        """
        q: [4] (x, y, z, w)
        wb: [3]
        """
        xyz = q[:3]
        w = q[3]
        d_xyz = 0.5 * (w * wb + torch.linalg.cross(wb, xyz))
        d_w = -0.5 * torch.dot(wb, xyz)
        return torch.cat([d_xyz, d_w.unsqueeze(0)])

    # ------------------------------------------------------------------
    # Core Physics (Euler) - 호환용, 기본 RK4 학습에는 사용하지 않음
    # ------------------------------------------------------------------
    def simulate_single(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Core Physics Engine - Euler Integration]
        단일 샘플에 대해 전체 시간을 적분합니다. (vmap에 의해 배치 병렬화됨)
        """
        # 초기 기저 상태 (기준 좌표계)
        R0 = torch.eye(3, device=self.device)
        r0 = torch.zeros(3, device=self.device)

        q0_curr = q0_init

        # Time Integration Loop (시간 순차 계산은 Loop 필수)
        for t in range(self.num_steps):
            qm = q_traj[t]
            qd = q_dot_traj[t]

            # --- 1. SPART Dynamics Calculations ---
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

            # --- 3. Quaternion Integration (Euler) ---
            d_q = self._quat_deriv(q0_curr, wb)
            q0_curr = q0_curr + d_q * self.dt
            q0_curr = q0_curr / torch.norm(q0_curr)

        # --- 4. Final Error Calculation ---
        dot_prod = torch.abs(torch.sum(q0_curr * q0_goal))
        dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = 2 * torch.acos(dot_prod)
        return angle_error ** 2

    # ------------------------------------------------------------------
    # Evaluation용 RK4 (dt=0.01, 3 폴더 공통 스펙과 동일)
    # ------------------------------------------------------------------
    def simulate_single_rk4(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Evaluation 전용 Physics Engine with RK4 integration]
        각속도 wb로부터 쿼터니언을 4차 Runge-Kutta로 적분하여 최종 자세 오차를 계산
        dt = 0.01초로 고정 (evaluation 전용)
        """
        dt_eval = 0.01  # Evaluation용 고정 dt
        num_steps_eval = int(self.total_time / dt_eval)

        R0 = torch.eye(3, device=self.device)
        r0 = torch.zeros(3, device=self.device)

        q0_curr = q0_init

        # 궤적을 더 세밀한 스텝으로 보간하기 위해 원본 궤적 인덱스 계산
        for t_eval in range(num_steps_eval):
            t_orig = t_eval * dt_eval / self.total_time  # 0 ~ 1로 정규화
            idx_orig = int(t_orig * (self.num_steps - 1))
            idx_orig = min(idx_orig, self.num_steps - 1)

            qm = q_traj[idx_orig]
            qd = q_dot_traj[idx_orig]

            # Dynamics (wb 계산까지는 기존과 동일)
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
            I0, Im = spart.inertia_projection(R0, RL, self.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)

            rhs = -H0m @ qd
            H0_damped = H0 + 1e-6 * torch.eye(6, device=self.device)
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]

            # RK4 integration for quaternion (dt_eval)
            k1 = self._quat_deriv(q0_curr, wb)
            k2 = self._quat_deriv(q0_curr + 0.5 * dt_eval * k1, wb)
            k3 = self._quat_deriv(q0_curr + 0.5 * dt_eval * k2, wb)
            k4 = self._quat_deriv(q0_curr + dt_eval * k3, wb)

            q0_curr = q0_curr + (dt_eval / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            q0_curr = q0_curr / torch.norm(q0_curr)

        dot_prod = torch.abs(torch.sum(q0_curr * q0_goal))
        dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = 2 * torch.acos(dot_prod)
        return angle_error ** 2

    # ------------------------------------------------------------------
    # Training/Optimization용 RK4 (dt=self.dt = 0.1)
    # ------------------------------------------------------------------
    def _simulate_single_rk4_train(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        학습/최적화에 사용하는 RK4 적분 (dt=self.dt).
        evaluate에서는 위의 simulate_single_rk4 (dt=0.01)를 사용.
        """
        dt = self.dt

        R0 = torch.eye(3, device=self.device)
        r0 = torch.zeros(3, device=self.device)

        q0_curr = q0_init

        for t in range(self.num_steps):
            qm = q_traj[t]
            qd = q_dot_traj[t]

            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
            I0, Im = spart.inertia_projection(R0, RL, self.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)

            rhs = -H0m @ qd
            H0_damped = H0 + 1e-6 * torch.eye(6, device=self.device)
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]

            # RK4 integration for quaternion (dt=self.dt)
            k1 = self._quat_deriv(q0_curr, wb)
            k2 = self._quat_deriv(q0_curr + 0.5 * dt * k1, wb)
            k3 = self._quat_deriv(q0_curr + 0.5 * dt * k2, wb)
            k4 = self._quat_deriv(q0_curr + dt * k3, wb)

            q0_curr = q0_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            q0_curr = q0_curr / torch.norm(q0_curr)

        dot_prod = torch.abs(torch.sum(q0_curr * q0_goal))
        dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = 2 * torch.acos(dot_prod)
        return angle_error ** 2

    # ------------------------------------------------------------------
    # Loss Functions
    # ------------------------------------------------------------------
    def calculate_loss(self, waypoints_flat, q0_init, q0_goal):
        """
        Batched Physics Simulation using vmap (RK4 기반 학습용)
        """
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat)
        batch_sim_fn = vmap(self._simulate_single_rk4_train, in_dims=(0, 0, 0, 0))
        loss_batch = batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)
        return loss_batch.mean()

