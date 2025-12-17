import torch
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
        q 가 배치([B, 4])이든 단일( [4] )이든 모두 지원.
        vmap 에서도 안전하도록 in-place 연산을 사용하지 않는다.
        """
        orig_shape = q.shape
        # [4] -> [1, 4] 로 승격하여 공통 처리
        if q.dim() == 1:
            q = q.unsqueeze(0)

        x, y, z, w = q.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)

        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)

        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        R = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )  # [B, 3, 3]

        # 입력이 [4]였다면 [3, 3]으로 다시 줄여서 반환
        if len(orig_shape) == 1:
            R = R.squeeze(0)
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
        # vmap-safe 구현: 텐서 기반 분기 (no Python if on Tensor)
        theta = torch.linalg.norm(wb) * dt  # scalar tensor
        eps = 1e-8

        # 공통적으로 사용할 항들 계산
        axis = wb / (torch.linalg.norm(wb) + 1e-12)
        K = self._skew(axis)
        I = self.eye3.to(dtype=wb.dtype)

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # 일반적인 Rodrigues 회전 (finite theta)
        R_big = I + sin_theta * K + (1.0 - cos_theta) * (K @ K)

        # 매우 작은 회전: 1차 근사 (I + [w*dt]_x)
        K_small = self._skew(wb * dt)
        R_small = I + K_small

        small = theta < eps
        # where 는 브로드캐스트 가능해야 하므로, small 은 스칼라 bool 텐서
        R_delta = torch.where(small, R_small, R_big)
        return R_delta

    def _rot_to_quat(self, R):
        """
        회전행렬 R [..., 3, 3] -> 쿼터니언 q [..., 4] (x, y, z, w)
        vmap 호환성을 위해 torch.where 기반 분기 사용
        """
        r00 = R[..., 0, 0]
        r11 = R[..., 1, 1]
        r22 = R[..., 2, 2]
        trace = r00 + r11 + r22
        
        def safe_sqrt(x):
            return torch.sqrt(torch.clamp(x, min=1e-8))

        # Case 1: trace > 0
        S1 = safe_sqrt(trace + 1.0) * 2
        w1 = 0.25 * S1
        x1 = (R[..., 2, 1] - R[..., 1, 2]) / S1
        y1 = (R[..., 0, 2] - R[..., 2, 0]) / S1
        z1 = (R[..., 1, 0] - R[..., 0, 1]) / S1
        q1 = torch.stack([x1, y1, z1, w1], dim=-1)
        
        # Case 2: r00 is max
        S2 = safe_sqrt(1.0 + r00 - r11 - r22) * 2
        w2 = (R[..., 2, 1] - R[..., 1, 2]) / S2
        x2 = 0.25 * S2
        y2 = (R[..., 0, 1] + R[..., 1, 0]) / S2
        z2 = (R[..., 0, 2] + R[..., 2, 0]) / S2
        q2 = torch.stack([x2, y2, z2, w2], dim=-1)
        
        # Case 3: r11 is max
        S3 = safe_sqrt(1.0 + r11 - r00 - r22) * 2
        w3 = (R[..., 0, 2] - R[..., 2, 0]) / S3
        x3 = (R[..., 0, 1] + R[..., 1, 0]) / S3
        y3 = 0.25 * S3
        z3 = (R[..., 1, 2] + R[..., 2, 1]) / S3
        q3 = torch.stack([x3, y3, z3, w3], dim=-1)
        
        # Case 4: r22 is max
        S4 = safe_sqrt(1.0 + r22 - r00 - r11) * 2
        w4 = (R[..., 1, 0] - R[..., 0, 1]) / S4
        x4 = (R[..., 0, 2] + R[..., 2, 0]) / S4
        y4 = (R[..., 1, 2] + R[..., 2, 1]) / S4
        z4 = 0.25 * S4
        q4 = torch.stack([x4, y4, z4, w4], dim=-1)
        
        # Selection logic
        cond1 = trace > 0
        cond2 = (r00 > r11) & (r00 > r22)
        cond3 = (r11 > r22)
        
        # Unsqueeze for broadcasting with last dim (4)
        c1 = cond1.unsqueeze(-1)
        c2 = cond2.unsqueeze(-1)
        c3 = cond3.unsqueeze(-1)

        q_out = torch.where(c1, q1, torch.where(c2, q2, torch.where(c3, q3, q4)))
        
        # Normalize
        q_out = q_out / (torch.linalg.norm(q_out, dim=-1, keepdim=True) + 1e-8)
        return q_out

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
        
        # Return final quaternion as well
        q_final = self._rot_to_quat(R_curr)
        return angle_error ** 2, q_final

    def simulate_single_rk4(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [High-Fidelity Physics Engine]
        - RK4의 각 Sub-step마다 SPART 동역학을 새로 풀어 변화하는 w(각속도)를 반영
        - 입력 궤적(qm, qd)을 선형 보간(Linear Interpolation)하여 부드러운 입력 제공
        """
        dt_eval = 0.01
        num_steps_eval = int(self.total_time / dt_eval)
        
        # 초기화
        R0 = self.R0
        r0 = self.r0
        q_curr = q0_init.clone()
        q_goal = q0_goal.clone()

        def normalize_quat(q):
            return q / (torch.linalg.norm(q) + 1e-8)

        # --- Helper: 특정 시간 t에서의 입력(qm, qd) 보간 함수 ---
        def get_interpolated_input(t):
            # t는 현재 시뮬레이션 시간
            # 원본 궤적의 인덱스(float) 계산
            idx_float = t * (self.num_steps - 1) / self.total_time
            idx_floor = int(idx_float)
            idx_ceil = min(idx_floor + 1, self.num_steps - 1)
            alpha = idx_float - idx_floor  # 보간 가중치 (0~1)

            # 선형 보간 (Linear Interpolation)
            qm_interp = (1 - alpha) * q_traj[idx_floor] + alpha * q_traj[idx_ceil]
            qd_interp = (1 - alpha) * q_dot_traj[idx_floor] + alpha * q_dot_traj[idx_ceil]
            return qm_interp, qd_interp

        # --- Helper: 현재 쿼터니언(q)과 시간(t)에서 각속도(wb) 계산 ---
        # 핵심: RK4 단계마다 관성 행렬(H)이 바뀌므로 w도 다시 구해야 함
        def compute_omega(current_q, current_t):
            # 1. 쿼터니언 -> 회전행렬 변환
            # [수정] Rmat 버전(simulate_single)과 물리 동작을 일치시키기 위해
            # Dynamics 계산 시에는 현재 자세(R_curr)가 아닌 초기 자세(R0, Identity)를 사용합니다.
            # 이는 중력/관성이 Base Orientation에 의존하지 않도록(혹은 Body Frame 기준 고정) 함을 의미합니다.
            # R_curr_sub = self._quat_to_rot(current_q) 
            
            # 2. 입력 보간값 가져오기
            qm_sub, qd_sub = get_interpolated_input(current_t)

            # 3. SPART Dynamics 재계산
            # Rmat 버전과 동일하게 R0(Identity) 기준 동역학 풀이
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm_sub, self.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
            I0, Im = spart.inertia_projection(R0, RL, self.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)

            # 4. Constraint Solver
            rhs = -H0m @ qd_sub
            H0_damped = H0 + 1e-6 * self.eye6
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            
            return u0_sol[:3] # wb

        # --- Main Loop ---
        current_time = 0.0
        
        for _ in range(num_steps_eval):
            # RK4 Integration
            
            # k1: 현재 상태에서의 기울기
            w1 = compute_omega(q_curr, current_time)
            k1 = spart.quat_dot(q_curr, w1)

            # k2: 중간 상태 1에서의 기울기
            q_k2 = normalize_quat(q_curr + 0.5 * dt_eval * k1)
            w2 = compute_omega(q_k2, current_time + 0.5 * dt_eval)
            k2 = spart.quat_dot(q_k2, w2)

            # k3: 중간 상태 2에서의 기울기
            q_k3 = normalize_quat(q_curr + 0.5 * dt_eval * k2)
            w3 = compute_omega(q_k3, current_time + 0.5 * dt_eval)
            k3 = spart.quat_dot(q_k3, w3)

            # k4: 끝 상태에서의 기울기
            q_k4 = normalize_quat(q_curr + dt_eval * k3)
            w4 = compute_omega(q_k4, current_time + dt_eval)
            k4 = spart.quat_dot(q_k4, w4)

            # 최종 업데이트
            q_curr = normalize_quat(q_curr + (dt_eval / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
            current_time += dt_eval

        # Final Error Calculation
        R_curr = self._quat_to_rot(q_curr)
        R_goal = self._quat_to_rot(q_goal)
        R_err = R_goal.T @ R_curr
        trace = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        return (torch.acos(trace) ** 2), q_curr

    def calculate_loss(self, waypoints_flat, q0_init, q0_goal):
        """
        Batched Physics Simulation using vmap (Rotation Matrix Version)
        """
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat)

        # simulate_single 을 배치 차원에 대해 병렬화
        batch_sim_fn = vmap(self.simulate_single, in_dims=(0, 0, 0, 0))
        # Now returns (loss_batch, final_q_batch)
        loss_batch, _ = batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)
        return loss_batch.mean()


