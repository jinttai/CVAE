import torch
import sys
import os

# -----------------------------------------------------------------------------
# [경로 설정] 프로젝트 루트를 sys.path에 추가하여 모듈 import 에러 방지
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/training
src_dir = os.path.dirname(current_dir)                 # src
root_dir = os.path.dirname(src_dir)                    # space_robot_planning

if src_dir not in sys.path:
    sys.path.append(src_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# dynamics 모듈 로드 (폴더 구조에 맞게 import)
try:
    import dynamics.spart_functions_torch as spart
except ImportError:
    # 만약 src가 루트로 잡혀있는 경우 대비
    import dynamics.spart_functions_torch as spart

from torch.func import vmap # [핵심] 자동 배칭(Auto-Batching)

class PhysicsLayer:
    def __init__(self, robot, num_waypoints, total_time, device):
        self.robot = robot
        self.n_q = robot['n_q']
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        self.dt = 0.1  # Simulation step size (0.1s)
        self.num_steps = int(total_time / self.dt)
        self.device = device
        
        # Spline Basis 미리 생성 (GPU 메모리에 상주)
        self.basis = self._get_spline_basis(self.num_waypoints, self.num_steps).to(device)

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
        q_traj = torch.einsum('st,btj->bsj', self.basis, w_full)
        
        # 수치 미분으로 속도 계산
        q_dot_traj = torch.zeros_like(q_traj)
        q_dot_traj[:, 1:, :] = (q_traj[:, 1:, :] - q_traj[:, :-1, :]) / self.dt
        
        return q_traj, q_dot_traj

    def simulate_single(self, q_traj, q_dot_traj, q0_init, q0_goal):
        """
        [Core Physics Engine]
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
            H0_damped = H0 + 1e-6 * torch.eye(6, device=self.device) 
            
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
        """
        R0 = torch.eye(3, device=self.device)
        r0 = torch.zeros(3, device=self.device)

        q0_curr = q0_init

        for t in range(self.num_steps):
            qm = q_traj[t]
            qd = q_dot_traj[t]

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

            # RK4 integration for quaternion
            k1 = self._quat_deriv(q0_curr, wb)
            k2 = self._quat_deriv(q0_curr + 0.5 * self.dt * k1, wb)
            k3 = self._quat_deriv(q0_curr + 0.5 * self.dt * k2, wb)
            k4 = self._quat_deriv(q0_curr + self.dt * k3, wb)

            q0_curr = q0_curr + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
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