import torch
import sys
import os

# -----------------------------------------------------------------------------
# [경로 설정] 프로젝트 루트 및 워크스페이스 루트를 sys.path에 추가
#   - space_robot_planning/src/training/physics_layer.py 기준 구조:
#       <workspace_root>/CVAE/
#           src/...
#           space_robot_planning/src/...
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../space_robot_planning/src/training
src_dir = os.path.dirname(current_dir)                    # .../space_robot_planning/src
root_dir = os.path.dirname(src_dir)                       # .../space_robot_planning
workspace_root = os.path.dirname(root_dir)                # .../CVAE (상위 워크스페이스)

for p in (src_dir, root_dir, workspace_root):
    if p not in sys.path:
        sys.path.append(p)

# dynamics 모듈 로드 (워크스페이스 기준 src.dynamics)
import src.dynamics.spart_functions_torch as spart

from torch.func import vmap # [핵심] 자동 배칭(Auto-Batching)

class PhysicsLayer:
    def __init__(self, robot, num_waypoints, total_time, device):
        self.robot = robot
        self.n_q = robot['n_q']
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        # Simulation time step (fixed to 0.1s)
        self.dt = 0.1
        self.num_steps = int(total_time / self.dt)
        self.device = device

        # Pre-allocated constant tensors to avoid per-step allocations
        self.R0 = torch.eye(3, device=self.device)
        self.r0 = torch.zeros(3, device=self.device)
        self.eye6 = torch.eye(6, device=self.device)

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

    def simulate_single(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Core Physics Engine]
        단일 샘플에 대해 전체 시간을 적분합니다. (vmap에 의해 배치 병렬화됨)
        """
        # 초기 기저 상태 (기준 좌표계) - 사전 생성된 텐서 사용
        R0 = self.R0
        r0 = self.r0
        
        q0_curr = q0_init
        
        # Time Integration Loop (시간 순차 계산은 Loop 필수)
        for t in range(self.num_steps):
            qm = q_traj[t]
            qd = q_dot_traj[t]
            
            # --- 1. SPART Dynamics Calculations ---
            # 모든 함수는 In-place 연산 없이 구현되어야 함 (spart_functions_torch 수정본 사용)
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
            I0, Im = spart.inertia_projection(R0, RL, self.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)
            
            # --- 2. Non-holonomic Constraint Solver ---
            # Solve: H0 * wb = -H0m * qd
            rhs = -H0m @ qd
            
            # [안정성 추가] H0에 Damping을 주어 역행렬 계산 시 수치적 폭발 방지
            # 학습 초기에는 H0가 특이(Singular)해질 수 있음
            H0_damped = H0 + 1e-6 * self.eye6
            
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3] # Angular Velocity part
            
            # --- 3. Quaternion Integration ---
            # q_dot = 0.5 * q * wb
            xyz = q0_curr[:3]
            w = q0_curr[3]
            
            # torch.linalg.cross 사용 (Warning 해결됨)
            d_xyz = 0.5 * (w * wb + torch.linalg.cross(wb, xyz))
            d_w = -0.5 * torch.dot(wb, xyz)
            
            # Euler Integration
            q0_curr = q0_curr + torch.cat([d_xyz, d_w.unsqueeze(0)]) * self.dt
            
            # Normalization (Rotation 유효성 유지)
            q0_curr = q0_curr / torch.norm(q0_curr)
            
        # --- 4. Final Error Calculation (개선됨) ---
        # 단순 1-dot^2 보다, 각도(Angle) 자체를 줄이는 것이 학습에 훨씬 유리함
        
        # 쿼터니언 내적 (절댓값 취함, q와 -q는 같은 자세이므로)
        dot_prod = torch.abs(torch.sum(q0_curr * q0_goal))
        
        # 수치 안정성을 위한 Clamp (-1.0 ~ 1.0 벗어나면 acos에서 NaN 발생)
        dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 각도 차이 계산 (Theta)
        angle_error = 2 * torch.acos(dot_prod)
        
        # 오차를 제곱하여 큰 오차에 더 큰 페널티 (L2 Loss 성격)
        return angle_error ** 2

    # ------------------------------------------------------------------
    # RK4 기반 평가용 시뮬레이션 (최적화에는 사용하지 않고, 평가 전용)
    # ------------------------------------------------------------------
    def _quat_deriv(self, q, wb):
        """
        쿼터니언 q = [x, y, z, w]와 각속도 wb에 대한 dq/dt 계산
        """
        xyz = q[:3]
        w = q[3]
        d_xyz = 0.5 * (w * wb + torch.linalg.cross(wb, xyz))
        d_w = -0.5 * torch.dot(wb, xyz)
        return torch.cat([d_xyz, d_w.unsqueeze(0)])

    def simulate_single_rk4(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Evaluation 전용 Physics Engine with RK4 integration]
        각속도 wb로부터 쿼터니언을 4차 Runge-Kutta로 적분하여 최종 자세 오차를 계산
        dt = 0.01초로 고정 (evaluation 전용)
        """
        dt_eval = 0.01  # Evaluation용 고정 dt
        num_steps_eval = int(self.total_time / dt_eval)
        
        R0 = self.R0
        r0 = self.r0

        q0_curr = q0_init

        # 궤적을 더 세밀한 스텝으로 보간하기 위해 원본 궤적 인덱스 계산
        for t_eval in range(num_steps_eval):
            # 원본 궤적의 시간에 매핑 (0 ~ total_time)
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
            H0_damped = H0 + 1e-6 * self.eye6
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            wb = u0_sol[:3]

            # RK4 integration for quaternion
            k1 = self._quat_deriv(q0_curr, wb)
            k2 = self._quat_deriv(q0_curr + 0.5 * dt_eval * k1, wb)
            k3 = self._quat_deriv(q0_curr + 0.5 * dt_eval * k2, wb)
            k4 = self._quat_deriv(q0_curr + dt_eval * k3, wb)

            q0_curr = q0_curr + (dt_eval / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            q0_curr = q0_curr / torch.norm(q0_curr)

        # 최종 각도 오차 계산 (simulate_single과 동일한 정의)
        dot_prod = torch.abs(torch.sum(q0_curr * q0_goal))
        dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = 2 * torch.acos(dot_prod)
        return angle_error ** 2

    def calculate_loss(self, waypoints_flat, q0_init, q0_goal):
        """
        Batched Physics Simulation using vmap
        """
        # 1. Trajectory Generation
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat)
        
        # 2. Auto-Vectorization (Batch Parallelization)
        # simulate_single 함수를 배치 차원(0번 dim)에 대해 병렬화
        # in_dims=(0, 0, 0, 0) -> 입력 4개가 모두 배치 차원을 가짐
        batch_sim_fn = vmap(self.simulate_single, in_dims=(0, 0, 0, 0))
        
        # 3. Execute on GPU
        loss_batch = batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)
        
        # 4. Mean Loss
        return loss_batch.mean()

    def calculate_loss_rk4(self, waypoints_flat, q0_init, q0_goal):
        """
        RK4 기반 최종 자세 오차를 사용한 평가용 loss 계산 (최적화에는 사용 X)
        """
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat)
        batch_sim_fn = vmap(self.simulate_single_rk4, in_dims=(0, 0, 0, 0))
        loss_batch = batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)
        return loss_batch.mean()