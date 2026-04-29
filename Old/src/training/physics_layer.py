import torch
import src.dynamics.spart_functions_torch as spart
from torch.func import vmap  # Auto-Batching


def _so3_log_standalone(R):
    """SO(3) log map: rotation matrix → 3D rotation vector (axis-angle).
    autograd 호환 (in-place 연산 없음).
    """
    trace_val = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = torch.clamp((trace_val - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    # θ → 0 일 때 factor → 0.5  (Taylor 1차)
    factor = torch.where(
        theta.abs() < 1e-6,
        torch.ones_like(theta) * 0.5,
        theta / (2.0 * sin_theta + 1e-12),
    )

    omega = factor * torch.stack([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return omega  # [3]


def _skew_functional(v):
    """Skew-symmetric matrix from 3D vector (no in-place ops, autograd safe)."""
    vx, vy, vz = v[0], v[1], v[2]
    zero = torch.zeros_like(vx)
    row0 = torch.stack([zero, -vz, vy])
    row1 = torch.stack([vz, zero, -vx])
    row2 = torch.stack([-vy, vx, zero])
    return torch.stack([row0, row1, row2])


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
        self.num_segments = self.num_waypoints + 1
        self.steps_per_segment = self.num_steps // self.num_segments
        self.segment_remainder = self.num_steps % self.num_segments

        # Pre-allocated constant tensors to reduce per-step allocations
        self.R0 = torch.eye(3, device=self.device)
        self.r0 = torch.zeros(3, device=self.device)
        self.eye3 = torch.eye(3, device=self.device)
        self.eye6 = torch.eye(6, device=self.device)
        
        # Pre-computed constants to avoid recalculation
        self._damping_value = 1e-6
        self._damping_term = self._damping_value * self.eye6  # Pre-compute: 1e-6 * eye6
        self._dtype_runtime_cache = {}
        self._batch_sim_fn = vmap(self.simulate_single, in_dims=(0, 0, 0, 0))
        
        # Pre-allocate buffers for constraint solver (reused each step)
        # Note: These are single-use buffers, but pre-allocation helps with memory management
        self._rhs_buffer = torch.zeros(6, device=self.device, dtype=torch.float32)
        self._H0_damped_buffer = torch.zeros(6, 6, device=self.device, dtype=torch.float32)

    def _get_runtime_constants(self, dtype):
        cached = self._dtype_runtime_cache.get(dtype)
        if cached is not None:
            return cached

        segment_steps = tuple(
            self.steps_per_segment + (1 if seg < self.segment_remainder else 0)
            for seg in range(self.num_segments)
        )
        cached = {
            "R0": torch.eye(3, device=self.device, dtype=dtype),
            "r0": torch.zeros(3, device=self.device, dtype=dtype),
            "eye3": torch.eye(3, device=self.device, dtype=dtype),
            "damping_term": self._damping_value * torch.eye(6, device=self.device, dtype=dtype),
            "segment_times": tuple(
                torch.linspace(0, 1, steps, device=self.device, dtype=dtype)
                for steps in segment_steps
            ),
        }
        self._dtype_runtime_cache[dtype] = cached
        return cached

    # ------------------------------------------------------------------
    # Trajectory Generation (quintic polynomial)
    # ------------------------------------------------------------------
    def _quintic_segment(self, q_start, q_end, t_normalized):
        """
        5차 다항식(quintic) 분절 (ease-in-out):
          b(t) = 6 t^5 - 15 t^4 + 10 t^3,  t in [0, 1]
          q(t) = q_start + (q_end - q_start) * b(t)

        특징:
          - t = 0, 1 에서 속도, 가속도 모두 0 (b'(0)=b'(1)=0, b''(0)=b''(1)=0)

        q_start, q_end: [B, n_q]
        t_normalized: [seg_steps]
        Returns: [B, seg_steps, n_q]
        """
        t = t_normalized
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        basis = 6.0 * t5 - 15.0 * t4 + 10.0 * t3  # [seg_steps]
        dq = (q_end.unsqueeze(1) - q_start.unsqueeze(1))  # [B, 1, n_q]
        q = q_start.unsqueeze(1) + dq * basis.unsqueeze(0).unsqueeze(-1)
        return q

    def _quintic_derivative(self, q_start, q_end, t_normalized):
        """
        5차 다항식(quintic)의 1차 미분 (정규화 시간 기준):
          b'(t) = 30 t^4 - 60 t^3 + 30 t^2
          q'(t) = (q_end - q_start) * b'(t)

        q_start, q_end: [B, n_q]
        t_normalized: [seg_steps]
        Returns: [B, seg_steps, n_q]
        """
        t = t_normalized
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        d_basis = 30.0 * t4 - 60.0 * t3 + 30.0 * t2  # [seg_steps]
        dq = (q_end.unsqueeze(1) - q_start.unsqueeze(1))  # [B, 1, n_q]
        q_dot = dq * d_basis.unsqueeze(0).unsqueeze(-1)
        return q_dot

    def generate_trajectory(self, waypoints_flat, q_start=None, q_end=None):
        """
        [Batch, Waypoints*Joints] -> [Batch, Steps, Joints] (Pos, Vel)
        4분절 5차 다항식(quintic): 시작점 + 중간 waypoint 3개 + 끝점
        각 분절의 양 끝에서 속도, 가속도가 0
        
        Args:
            waypoints_flat: [batch_size, num_waypoints * n_q] flattened waypoints
            q_start: [batch_size, n_q] 시작 관절 각도 (None이면 0으로 설정)
            q_end: [batch_size, n_q] 목표 관절 각도 (None이면 0으로 설정)
        """
        batch_size = waypoints_flat.size(0)
        w_mid = waypoints_flat.view(batch_size, self.num_waypoints, self.n_q)
        runtime = self._get_runtime_constants(waypoints_flat.dtype)

        # 시작점과 끝점 설정
        if q_start is None:
            q_start = waypoints_flat.new_zeros(batch_size, self.n_q)
        if q_end is None:
            q_end = waypoints_flat.new_zeros(batch_size, self.n_q)
        
        # q_start와 q_end를 [B, 1, n_q] 형태로 변환
        q_start = q_start.unsqueeze(1) if q_start.dim() == 2 else q_start.view(batch_size, 1, self.n_q)
        q_end = q_end.unsqueeze(1) if q_end.dim() == 2 else q_end.view(batch_size, 1, self.n_q)
        
        w_full = torch.cat([q_start, w_mid, q_end], dim=1)  # [B, 5, n_q]: q0, w1, w2, w3, q4

        # 전체 시간을 4분절로 나눔
        num_segments = self.num_waypoints + 1  # 4분절
        num_segments = self.num_segments

        q_traj = waypoints_flat.new_zeros(batch_size, self.num_steps, self.n_q)
        q_dot_traj = waypoints_flat.new_zeros(batch_size, self.num_steps, self.n_q)

        step_idx = 0
        for seg, t_seg in enumerate(runtime["segment_times"]):
            q_start = w_full[:, seg, :]  # [B, n_q]
            q_end = w_full[:, seg + 1, :]  # [B, n_q]

            # 현재 분절의 스텝 수
            seg_steps = t_seg.numel()

            # 분절 내 정규화된 시간 [0, 1]

            # quintic polynomial으로 위치 계산: [B, seg_steps, n_q]
            q_seg = self._quintic_segment(q_start, q_end, t_seg)

            # quintic polynomial 1차 미분으로 속도 계산 (normalized time 기준): [B, seg_steps, n_q]
            q_dot_seg = self._quintic_derivative(q_start, q_end, t_seg)

            # 시간 스케일링 (전체 시간에 맞춤)
            segment_time = (self.total_time / self.num_segments)
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
        3D 벡터 v에 대한 skew-symmetric 행렬 [v]_x (vmap-safe, no in-place)
        """
        vx, vy, vz = v[0], v[1], v[2]
        zero = torch.zeros_like(vx)
        row0 = torch.stack([zero, -vz, vy])
        row1 = torch.stack([vz, zero, -vx])
        row2 = torch.stack([-vy, vx, zero])
        return torch.stack([row0, row1, row2])

    def _rot_from_omega(self, wb, dt):
        """
        각속도 wb 에 대해 dt 동안의 회전을 나타내는 회전행렬 R_delta 계산.
        Rodrigues 공식을 사용.
        Optimized: minimize intermediate tensor allocations.
        """
        # vmap-safe 구현: 텐서 기반 분기 (no Python if on Tensor)
        wb_norm = torch.linalg.norm(wb)
        # Clamp theta to prevent numerical instability from very large angular velocities
        # Maximum rotation per step: π radians (180 degrees)
        max_theta = 3.141592653589793  # π
        theta = torch.clamp(wb_norm * dt, max=max_theta)  # scalar tensor
        eps = 1e-8

        # 공통적으로 사용할 항들 계산 (reuse wb_norm)
        axis = wb / (wb_norm + 1e-12)  # Reuse norm calculation
        K = self._skew(axis)
        I = self._get_runtime_constants(wb.dtype)["eye3"]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # 일반적인 Rodrigues 회전 (finite theta)
        # Optimize: compute K @ K once and reuse
        K_squared = K @ K
        R_big = I + sin_theta * K + (1.0 - cos_theta) * K_squared

        # 매우 작은 회전: 1차 근사 (I + [w*dt]_x)
        wb_dt = wb * dt  # Compute once
        K_small = self._skew(wb_dt)
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
    # SPART dynamics → angular velocity
    # ------------------------------------------------------------------
    def _compute_wb(self, qm, qd):
        """Compute angular velocity wb from joint state (qm, qd) via SPART dynamics."""
        runtime = self._get_runtime_constants(qm.dtype)
        R0 = runtime["R0"]
        r0 = runtime["r0"]
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
        I0, Im = spart.inertia_projection(R0, RL, self.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)

        rhs = -H0m @ qd
        H0_damped = H0 + runtime["damping_term"]
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        return u0_sol[:3]  # Angular Velocity part

    # ------------------------------------------------------------------
    # 핵심 시뮬레이션 (회전행렬 기반, Euler 적분)
    # ------------------------------------------------------------------
    def simulate_single(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Core Physics Engine - Rotation Matrix + Euler Integration]

        vmap으로 100 timestep의 wb를 한 번에 계산한 뒤,
        순차적으로 R을 누적 (3x3 matmul만, 거의 공짜).

        Euler:
          wb_all = vmap(compute_wb)(q_traj, q_dot_traj)  # [T, 3]
          R_{t+1} = R_t @ exp(skew(wb_t * dt))
        """
        R_init = self._quat_to_rot(q0_init)
        R_goal = self._quat_to_rot(q0_goal)

        dt = self.dt

        # Batch wb computation: [num_steps, n_q] -> [num_steps, 3]
        all_wb = vmap(self._compute_wb)(q_traj, q_dot_traj)

        # Batch R_delta computation: [num_steps, 3] -> [num_steps, 3, 3]
        all_R_delta = vmap(lambda w: self._rot_from_omega(w, dt))(all_wb)

        # Sequential R accumulation (cheap 3x3 matmul)
        R_curr = R_init
        for t in range(self.num_steps):
            R_curr = R_curr @ all_R_delta[t]

        # --- Final Orientation Error ---
        R_diff = R_curr - R_goal
        R_diff_sq = R_diff.T @ R_diff
        trace_val = 0.5 * torch.trace(R_diff_sq)
        epsilon = 1e-8
        loss = torch.log(epsilon + trace_val)

        q_final = self._rot_to_quat(R_curr)
        return loss, q_final

    def simulate_single_rk4(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [High-Fidelity Physics Engine]
        - RK4의 각 Sub-step마다 SPART 동역학을 새로 풀어 변화하는 w(각속도)를 반영
        - 입력 궤적(qm, qd)을 선형 보간(Linear Interpolation)하여 부드러운 입력 제공
        """
        # Use 5000 steps for higher accuracy
        num_steps_eval = 5000
        dt_eval = self.total_time / num_steps_eval
        # 실제 시뮬레이션 시간을 정확히 total_time에 맞추기 위해 dt 조정
        actual_dt = self.total_time / num_steps_eval if num_steps_eval > 0 else dt_eval
        
        # 초기화
        R0 = self.R0
        r0 = self.r0
        q_curr = q0_init.clone()
        q_goal = q0_goal.clone()

        def normalize_quat(q):
            return q / (torch.linalg.norm(q) + 1e-8)

        # --- Helper: 특정 시간 t에서의 입력(qm, qd) 보간 함수 ---
        def get_interpolated_input(t):
            # t는 현재 시뮬레이션 시간 [0, total_time]
            # 경계 조건 처리: t가 total_time에 가까우면 마지막 인덱스 사용
            t_clamped = max(0.0, min(float(t), self.total_time - 1e-10))
            
            # 원본 궤적의 인덱스(float) 계산
            if self.num_steps > 1:
                idx_float = t_clamped * (self.num_steps - 1) / self.total_time
            else:
                idx_float = 0.0
            
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
            # [참고] Rmat 버전(simulate_single)과 물리 동작을 일치시키기 위해
            # Dynamics 계산 시에는 현재 자세(R_curr)가 아닌 초기 자세(R0, Identity)를 사용합니다.
            # 이는 중력/관성이 Base Orientation에 의존하지 않도록(혹은 Body Frame 기준 고정) 함을 의미합니다.
            
            # 입력 보간값 가져오기
            qm_sub, qd_sub = get_interpolated_input(current_t)

            # SPART Dynamics 재계산
            # Rmat 버전과 동일하게 R0(Identity) 기준 동역학 풀이
            RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm_sub, self.robot)
            Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
            I0, Im = spart.inertia_projection(R0, RL, self.robot)
            M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
            H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)

            # Constraint Solver
            rhs = -H0m @ qd_sub
            H0_damped = H0 + 1e-6 * self.eye6
            u0_sol = torch.linalg.solve(H0_damped, rhs)
            
            return u0_sol[:3] # wb

        # --- Main Loop ---
        current_time = 0.0
        
        for step in range(num_steps_eval):
            # 마지막 스텝에서는 남은 시간만큼만 적분
            dt_step = actual_dt
            if step == num_steps_eval - 1:
                remaining_time = self.total_time - current_time
                if remaining_time > 1e-10:
                    dt_step = remaining_time
            
            # RK4 Integration
            
            # k1: 현재 상태에서의 기울기
            w1 = compute_omega(q_curr, current_time)
            k1 = spart.quat_dot(q_curr, w1)

            # k2: 중간 상태 1에서의 기울기
            q_k2 = normalize_quat(q_curr + 0.5 * dt_step * k1)
            w2 = compute_omega(q_k2, current_time + 0.5 * dt_step)
            k2 = spart.quat_dot(q_k2, w2)

            # k3: 중간 상태 2에서의 기울기
            q_k3 = normalize_quat(q_curr + 0.5 * dt_step * k2)
            w3 = compute_omega(q_k3, current_time + 0.5 * dt_step)
            k3 = spart.quat_dot(q_k3, w3)

            # k4: 끝 상태에서의 기울기
            q_k4 = normalize_quat(q_curr + dt_step * k3)
            w4 = compute_omega(q_k4, current_time + dt_step)
            k4 = spart.quat_dot(q_k4, w4)

            # 최종 업데이트
            q_curr = normalize_quat(q_curr + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
            current_time += dt_step
            
            # 시간이 total_time에 도달했으면 종료
            if current_time >= self.total_time - 1e-10:
                break

        # Final Error Calculation - Angle error loss: log(epsilon + 1/2 * trace((Q - Q_d)^T * (Q - Q_d)))
        R_curr = self._quat_to_rot(q_curr)
        R_goal = self._quat_to_rot(q_goal)
        R_diff = R_curr - R_goal  # [3, 3]
        R_diff_T = R_diff.T  # [3, 3]
        R_diff_sq = R_diff_T @ R_diff  # [3, 3]
        trace_val = 0.5 * torch.trace(R_diff_sq)
        epsilon = 1e-8
        loss = torch.log(epsilon + trace_val)
        return loss, q_curr

    def calculate_loss(self, waypoints_flat, q0_init, q0_goal, q_start_joint=None, q_end_joint=None):
        """
        Batched Physics Simulation using vmap (Rotation Matrix Version)
        Returns only physics loss (for backward compatibility)
        
        Args:
            waypoints_flat: [batch_size, num_waypoints * n_q] flattened waypoints
            q0_init: [batch_size, 4] initial quaternion
            q0_goal: [batch_size, 4] goal quaternion
            q_start_joint: [batch_size, n_q] starting joint angles (optional)
            q_end_joint: [batch_size, n_q] goal joint angles (optional)
        """
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat, q_start=q_start_joint, q_end=q_end_joint)

        # simulate_single 을 배치 차원에 대해 병렬화
        loss_batch, _ = self._batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)
        return loss_batch.mean()

    def calculate_total_loss(self, waypoints_flat, q0_init, q0_goal, 
                            joint_squared_weight=0.01, joint_change_weight=0.01, 
                            max_joint_weight=0.01, return_mean=True, q_start_joint=None, q_end_joint=None):
        """
        Calculate total loss including physics loss, joint squared penalty, joint change penalty, and max joint angle penalty.
        
        Args:
            waypoints_flat: [batch_size, num_waypoints * n_q] flattened waypoints
            q0_init: [batch_size, 4] initial quaternion
            q0_goal: [batch_size, 4] goal quaternion
            joint_squared_weight: weight for mean of joint^2 penalty
            joint_change_weight: weight for joint change penalty between consecutive waypoints
            max_joint_weight: weight for maximum joint angle penalty
            return_mean: if True, return mean over batch (scalar); if False, return per-sample losses [batch_size]
            q_start_joint: [batch_size, n_q] starting joint angles (optional)
            q_end_joint: [batch_size, n_q] goal joint angles (optional)
            
        Returns:
            total_loss: scalar tensor (if return_mean=True) or [batch_size] tensor (if return_mean=False)
            loss_dict: dictionary with individual loss components for logging
                - 'physics_loss': physics simulation loss (scalar or [batch_size])
                - 'joint_squared_penalty': mean(waypoints^2) * joint_squared_weight (scalar or [batch_size])
                - 'joint_change_penalty': mean(diff^2) * joint_change_weight (scalar or [batch_size])
                - 'max_joint_penalty': max(|waypoints|) * max_joint_weight (scalar or [batch_size])
                - 'total_loss': total loss (scalar or [batch_size])
        """
        # Physics loss (per sample)
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat, q_start=q_start_joint, q_end=q_end_joint)
        physics_loss_batch, _ = self._batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)  # [batch_size]
        
        # Reshape waypoints
        batch_size = waypoints_flat.size(0)
        waypoints_reshaped = waypoints_flat.view(batch_size, self.num_waypoints, self.n_q)
        
        # Mean of joint^2 penalty (per sample: mean over waypoints and joints)
        joint_squared_per_sample = (waypoints_reshaped ** 2).mean(dim=(1, 2))  # [batch_size]
        joint_squared_penalty_batch = joint_squared_per_sample * joint_squared_weight  # [batch_size]
        
        # Joint change penalty (per sample: mean over waypoint pairs and joints)
        if self.num_waypoints > 1:
            joint_diff = waypoints_reshaped[:, 1:, :] - waypoints_reshaped[:, :-1, :]  # [batch_size, num_waypoints-1, n_q]
            joint_change_squared_per_sample = (joint_diff ** 2).mean(dim=(1, 2))  # [batch_size]
            joint_change_penalty_batch = joint_change_squared_per_sample * joint_change_weight  # [batch_size]
        else:
            joint_change_penalty_batch = torch.zeros(batch_size, device=waypoints_flat.device, dtype=waypoints_flat.dtype)
        
        # Maximum joint angle penalty (per sample: max over all waypoints and joints)
        max_joint_angle_per_sample = waypoints_reshaped.abs().view(batch_size, -1).max(dim=1)[0]  # [batch_size]
        max_joint_penalty_batch = max_joint_angle_per_sample * max_joint_weight  # [batch_size]
        
        # Total loss (per sample)
        total_loss_batch = (physics_loss_batch + joint_squared_penalty_batch + 
                           joint_change_penalty_batch + max_joint_penalty_batch)  # [batch_size]
        
        # Return mean or per-sample losses
        if return_mean:
            physics_loss = physics_loss_batch.mean()
            joint_squared_penalty = joint_squared_penalty_batch.mean()
            joint_change_penalty = joint_change_penalty_batch.mean()
            max_joint_penalty = max_joint_penalty_batch.mean()
            total_loss = total_loss_batch.mean()
        else:
            physics_loss = physics_loss_batch
            joint_squared_penalty = joint_squared_penalty_batch
            joint_change_penalty = joint_change_penalty_batch
            max_joint_penalty = max_joint_penalty_batch
            total_loss = total_loss_batch
        
        # Return loss dict for logging
        loss_dict = {
            'physics_loss': physics_loss,
            'joint_squared_penalty': joint_squared_penalty,
            'joint_change_penalty': joint_change_penalty,
            'max_joint_penalty': max_joint_penalty,
            'total_loss': total_loss
        }

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # SO(3) error vector (for Newton correction / null-space optimization)
    # ------------------------------------------------------------------
    def simulate_single_error_vec(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        Simulate trajectory and return 3D SO(3) error vector (log map).
        vmap으로 wb를 배치 계산.

        Returns:
            error_vec: [3] rotation vector representing R_goal^T @ R_final
                       (zero when orientation is achieved)
        """
        R_init = self._quat_to_rot(q0_init)
        R_goal = self._quat_to_rot(q0_goal)

        dt = self.dt
        all_wb = vmap(self._compute_wb)(q_traj, q_dot_traj)
        all_R_delta = vmap(lambda w: self._rot_from_omega(w, dt))(all_wb)

        R_curr = R_init
        for t in range(self.num_steps):
            R_curr = R_curr @ all_R_delta[t]

        R_err = R_goal.T @ R_curr
        return _so3_log_standalone(R_err)

    # ------------------------------------------------------------------
    # Torque computation & cost  (inertia + Coriolis/centrifugal)
    # ------------------------------------------------------------------
    def _compute_tau_single_step(self, qm, qd, qdd, runtime):
        """
        Compute joint torque at a single timestep.

        Full EOM (free-floating base, no external force on base):
          H0  ü0  + H0m  q̈m + C0  u0 + C0m q̇m = 0   … (base)
          H0m'ü0  + Hm   q̈m + Cm0 u0 + Cm  q̇m = τ   … (joints)

        Eliminate ü0 from base row:
          ü0 = -H0⁻¹ (H0m q̈m + C0 u0 + C0m q̇m)

        Substitute into joint row:
          τ = (Hm  - H0m' H0⁻¹ H0m ) q̈m        … inertia
            + (Cm  - H0m' H0⁻¹ C0m ) q̇m        … Coriolis (joint)
            + (Cm0 - H0m' H0⁻¹ C0  ) u0         … Coriolis (base-coupling)

        u0 is obtained from momentum conservation: u0 = -H0⁻¹ H0m q̇m
        """
        R0 = runtime["R0"]
        r0 = runtime["r0"]
        damping = runtime["damping_term"]

        # --- kinematics & inertia ---
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
        I0, Im = spart.inertia_projection(R0, RL, self.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
        H0, H0m, Hm = spart.generalized_inertia_matrix(
            M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )

        H0_damped = H0 + damping
        H0_inv = torch.linalg.inv(H0_damped)         # [6, 6]
        H0_inv_H0m = H0_inv @ H0m                    # [6, n_q]

        # --- base velocity (momentum conservation) ---
        u0 = -(H0_inv_H0m @ qd)                      # [6]

        # --- link twists (needed for Coriolis) ---
        t0, tL = spart.velocities(Bij, Bi0, P0, pm, u0, qd, self.robot)

        # --- Coriolis / centrifugal matrices ---
        C0, C0m, Cm0, Cm = spart.convective_inertia_matrix(
            t0, tL, I0, Im, M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )

        # --- torque ---
        H0m_T = H0m.T                                # [n_q, 6]
        H0m_T_H0inv = H0m_T @ H0_inv                 # [n_q, 6]

        tau_inertia  = (Hm  - H0m_T_H0inv @ H0m ) @ qdd
        tau_cor_joint = (Cm  - H0m_T_H0inv @ C0m ) @ qd
        tau_cor_base  = (Cm0 - H0m_T_H0inv @ C0  ) @ u0

        return tau_inertia + tau_cor_joint + tau_cor_base

    def compute_torque_cost(self, q_traj, q_dot_traj):
        """
        Compute quadratic torque cost:  cost = Σ_t ||τ_t||²

        Includes inertia + Coriolis/centrifugal terms.
        vmap으로 100 timestep의 tau를 한 번에 계산.

        Args:
            q_traj: [num_steps, n_q]
            q_dot_traj: [num_steps, n_q]
        Returns:
            torque_cost: scalar
        """
        runtime = self._get_runtime_constants(q_traj.dtype)

        # q_ddot via finite difference
        q_ddot = torch.zeros_like(q_dot_traj)
        q_ddot[:-1] = (q_dot_traj[1:] - q_dot_traj[:-1]) / self.dt
        if self.num_steps > 1:
            q_ddot[-1] = q_ddot[-2]

        # Batch all tau at once: [T, n_q]
        all_tau = vmap(
            lambda qm, qd, qdd: self._compute_tau_single_step(qm, qd, qdd, runtime)
        )(q_traj, q_dot_traj, q_ddot)

        return (all_tau * all_tau).sum()

    def compute_torques(self, q_traj, q_dot_traj):
        """
        Compute joint torque profile for a single trajectory.

        Includes inertia + Coriolis/centrifugal terms.
        vmap으로 100 timestep 배치 계산.

        Args:
            q_traj: [num_steps, n_q]
            q_dot_traj: [num_steps, n_q]
        Returns:
            torques: [num_steps, n_q]
        """
        runtime = self._get_runtime_constants(q_traj.dtype)

        q_ddot = torch.zeros_like(q_dot_traj)
        q_ddot[:-1] = (q_dot_traj[1:] - q_dot_traj[:-1]) / self.dt
        if self.num_steps > 1:
            q_ddot[-1] = q_ddot[-2]

        return vmap(
            lambda qm, qd, qdd: self._compute_tau_single_step(qm, qd, qdd, runtime)
        )(q_traj, q_dot_traj, q_ddot)


