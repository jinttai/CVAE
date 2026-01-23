"""
CasADi-based DDP / iLQR implementation using SPART floating-base dynamics.

using CasADi for automatic differentiation.

State:  [q_joints (6), q_base_quat (4)]   -> dimension 10
Control: [joint_velocities (6)]           -> dimension 6
"""

import numpy as np
import casadi as ca

from src.dynamics.urdf2robot import urdf2robot
import src.dynamics.spart_casadi as spart


class CasadiSpaceRobotDynamics:
    """
    Floating-base space robot dynamics in CasADi using SPART formulation.
    """

    def __init__(self, robot: dict):
        self.robot = robot
        self.n_q = robot["n_q"]          # number of joints
        self.state_dim = self.n_q + 4    # joints + base quaternion
        self.control_dim = self.n_q      # joint velocities

        # Base pose is inertial frame (identity rotation, zero position)
        self.R0 = np.eye(3)
        self.r0 = np.zeros((3, 1))

        # Build symbolic dynamics f(x, u, dt)
        x = ca.SX.sym("x", self.state_dim)
        u = ca.SX.sym("u", self.control_dim)
        dt = ca.SX.sym("dt")

        x_next = self._step_symbolic(x, u, dt)
        self.f_fun = ca.Function("f_step", [x, u, dt], [x_next])

        # Linearizations
        A = ca.jacobian(x_next, x)
        B = ca.jacobian(x_next, u)
        self.fx_fun = ca.Function("fx", [x, u, dt], [A])
        self.fu_fun = ca.Function("fu", [x, u, dt], [B])

        # Second-order derivatives (Hessians) for Full DDP
        # Tensor structure handling in CasADi is tricky, so we create functions that return
        # specific slices or flattened versions, but here we can rely on CasADi's ability
        # to compute Hessians of scalar outputs. 
        # But dynamics is vector-valued. We need ∂²f_k / ∂x∂u etc. for each state dimension k.
        
        # We will create a function that takes an index 'k' and returns the Hessian of the k-th state update.
        # But passing 'k' as an argument to a CasADi function is not directly supported for indexing symbolic arrays in this way easily at runtime 
        # inside the compiled function without overhead.
        # Instead, we can generate all Hessians and stack them?
        # Or simpler: compute them on the fly in Python loop using A and B is not enough.
        # We need ∂²f/∂x², ∂²f/∂u², ∂²f/∂u∂x.
        
        # Let's pre-generate functions for Hessians of each state component.
        self.hessian_fns = []
        for k in range(self.state_dim):
            f_k = x_next[k]
            # Hessian w.r.t x (n_x, n_x)
            f_xx_k = ca.hessian(f_k, x)[0]
            # Hessian w.r.t u (n_u, n_u)
            f_uu_k = ca.hessian(f_k, u)[0]
            # Mixed Hessian ∂²f_k / ∂u∂x (n_u, n_x)  Note: CasADi gradient is column vector, Jacobian is (out, in)
            # jacobian(f_k, x) is (1, n_x). jacobian(jacobian(f_k, x), u) is (n_x, n_u)^T ? No.
            # Let's use gradient: g_x = grad(f_k, x). jacobian(g_x, u) -> (n_x, n_u). 
            # We usually want Q_ux term which comes from f_ux.
            # In standard DDP notation: f_ux[k] is ∂(∂f_k/∂x)/∂u. 
            # Let's just compute jacobian of gradient.
            g_x = ca.gradient(f_k, x)
            f_xu_k = ca.jacobian(g_x, u) # (n_x, n_u)
            
            self.hessian_fns.append({
                "f_xx": ca.Function(f"f_xx_{k}", [x, u, dt], [f_xx_k]),
                "f_uu": ca.Function(f"f_uu_{k}", [x, u, dt], [f_uu_k]),
                "f_xu": ca.Function(f"f_xu_{k}", [x, u, dt], [f_xu_k])
            })

    def _quat_multiply(self, q: ca.SX, p: ca.SX) -> ca.SX:
        """
        Quaternion multiplication q * p.
        q, p are [x, y, z, w].
        """
        q_xyz = q[0:3]
        q_w = q[3]
        p_xyz = p[0:3]
        p_w = p[3]
        
        r_xyz = q_w * p_xyz + p_w * q_xyz + ca.cross(q_xyz, p_xyz)
        r_w = q_w * p_w - ca.dot(q_xyz, p_xyz)
        return ca.vertcat(r_xyz, r_w)

    def _integrate_quat_exp(self, q: ca.SX, w: ca.SX, dt: ca.SX) -> ca.SX:
        """
        Exponential map integration: q_next = q * exp(w * dt / 2)
        w is angular velocity in body frame.
        """
        # Angle of rotation
        theta_sq = ca.dot(w, w) * dt**2 + 1e-16
        theta = ca.sqrt(theta_sq)
        
        # Half angle for quaternion
        a = theta / 2.0
        
        # Factors for sin(a)/a and cos(a)
        # Use if_else for numerical stability near 0
        # limit sin(x)/x as x->0 is 1.
        # Here we need sin(theta/2) / theta.
        # sin(theta/2) / theta = (theta/2 - (theta/2)^3/6) / theta = 1/2 - theta^2/48
        
        small_angle = theta < 1e-4
        
        # k = sin(theta/2) / theta
        k = ca.if_else(small_angle, 0.5 - theta_sq/48.0, ca.sin(a) / theta)
        
        # w_quat scalar part
        # qw = cos(theta/2)
        qw = ca.if_else(small_angle, 1.0 - theta_sq/8.0, ca.cos(a))
        
        dq_xyz = w * dt * k
        dq = ca.vertcat(dq_xyz, qw)
        
        return self._quat_multiply(q, dq)

    def _step_symbolic(self, x: ca.SX, u: ca.SX, dt: ca.SX) -> ca.SX:
        """
        One-step discrete dynamics using SPART equations of motion.
        x  = [q(6); q_base(4)]
        u  = [qd(6)]
        """
        n_q = self.n_q

        q_joints = x[0:n_q]
        q_base = x[n_q : n_q + 4]  # [x, y, z, w]

        # Normalize base quaternion
        q_norm = ca.sqrt(ca.dot(q_base, q_base) + 1e-8)
        q_base = q_base / q_norm

        # Base rotation matrix from quaternion
        R_base = spart.quat_dcm(q_base)

        # --- SPART kinematics and dynamics ---
        RJ, RL, rJ, rL, e, g = spart.kinematics(self.R0, self.r0, q_joints, self.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(self.R0, self.r0, rL, e, g, self.robot)
        I0, Im = spart.inertia_projection(self.R0, RL, self.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(
            M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )

        # Momentum conservation constraint:
        #   H0 * u0 + H0m * qd = 0  ->  u0 = - H0^{-1} H0m qd
        rhs = -H0m @ u
        # Use matrix inversion instead of ca.solve to avoid QR plugin dependency
        u0 = ca.mtimes(ca.inv(H0), rhs)

        # Base angular velocity (body-fixed) is first 3 components of u0
        wb = u0[0:3]

        # Quaternion integration using exponential map
        q_base_next = self._integrate_quat_exp(q_base, wb, dt)
        
        # Normalize to prevent numerical drift
        q_base_next = q_base_next / ca.sqrt(ca.dot(q_base_next, q_base_next) + 1e-8)

        # Joint integration
        q_joints_next = q_joints + dt * u

        x_next = ca.vertcat(q_joints_next, q_base_next)
        return x_next

    def rollout(self, x0: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
        """
        Rollout trajectory given an initial state and control sequence.
        x0 : [state_dim]
        U  : [T, control_dim]
        """
        x = np.asarray(x0, dtype=float).reshape(-1)
        T = U.shape[0]
        X = np.zeros((T + 1, self.state_dim), dtype=float)
        X[0] = x
        for t in range(T):
            u = U[t]
            x = np.array(self.f_fun(x, u, dt)).reshape(-1)
            X[t + 1] = x
        return X

    def linearize(self, x: np.ndarray, u: np.ndarray, dt: float):
        """
        Return (A, B) = (∂f/∂x, ∂f/∂u) at (x, u).
        """
        A = np.array(self.fx_fun(x, u, dt))
        B = np.array(self.fu_fun(x, u, dt))
        return A, B

    def hessians(self, x: np.ndarray, u: np.ndarray, dt: float):
        """
        Return (f_xx, f_uu, f_xu) tensors.
        f_xx: [n_x, n_x, n_x]
        f_uu: [n_x, n_u, n_u]
        f_xu: [n_x, n_x, n_u]  (Note: we computed f_xu as (n_x, n_u), so typically V_x * f_xu works out)
        
        Wait, standard DDP expansion:
        Q_xx += V_x * f_xx
        Q_uu += V_x * f_uu
        Q_ux += V_x * f_ux
        
        Here f_xx is tensor where f_xx[k] is Hessian of k-th state.
        """
        n_x = self.state_dim
        n_u = self.control_dim
        
        # We need to stack them.
        F_xx = np.zeros((n_x, n_x, n_x))
        F_uu = np.zeros((n_x, n_u, n_u))
        F_xu = np.zeros((n_x, n_x, n_u)) # d(df/dx)/du
        
        for k in range(n_x):
            fns = self.hessian_fns[k]
            F_xx[k] = np.array(fns["f_xx"](x, u, dt))
            F_uu[k] = np.array(fns["f_uu"](x, u, dt))
            F_xu[k] = np.array(fns["f_xu"](x, u, dt))
            
        return F_xx, F_uu, F_xu


class CasadiRunningCost:
    """
    Running cost L(x, u) = u^T R u  (plus optional soft joint limits if desired).
    For simplicity we only implement a quadratic control penalty here.
    """

    def __init__(self, R_weight: float | np.ndarray, n_u: int = 6):
        if np.isscalar(R_weight):
            self.R = float(R_weight) * np.eye(n_u)
        else:
            self.R = np.asarray(R_weight, dtype=float)
        
        self.n_u = n_u
        # n_x is n_u + 4 (joints + base quaternion)
        self.n_x = n_u + 4

        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)

        L = ca.mtimes([u.T, self.R, u])
        self.L_fun = ca.Function("L_running", [x, u], [L])

        Lx = ca.gradient(L, x)
        Lu = ca.gradient(L, u)
        Lxx = ca.hessian(L, x)[0]
        Luu = ca.hessian(L, u)[0]

        self.Lx_fun = ca.Function("Lx", [x, u], [Lx])
        self.Lu_fun = ca.Function("Lu", [x, u], [Lu])
        self.Lxx_fun = ca.Function("Lxx", [x, u], [Lxx])
        self.Luu_fun = ca.Function("Luu", [x, u], [Luu])

    def value(self, x: np.ndarray, u: np.ndarray) -> float:
        return float(self.L_fun(x, u))

    def derivatives(self, x: np.ndarray, u: np.ndarray):
        Lx = np.array(self.Lx_fun(x, u)).reshape(-1)
        Lu = np.array(self.Lu_fun(x, u)).reshape(-1)
        Lxx = np.array(self.Lxx_fun(x, u))
        Luu = np.array(self.Luu_fun(x, u))
        return Lx, Lu, Lxx, Luu


class CasadiTerminalCost:
    """
    Terminal cost:
        L_T(x) = w_orient * ||log(R(x)ᵀ R_goal)||² + w_joint * ||q - q_goal||² + w_velocity * ||u||²
    where orientation error is computed on SO(3).
    """

    def __init__(
        self,
        goal_quaternion: np.ndarray,
        goal_joints: np.ndarray | None = None,
        orientation_weight: float = 1.0,
        joint_weight: float = 0.0,
        velocity_weight: float = 0.0,
        n_u: int = 6,
    ):
        self.n_u = n_u
        self.n_x = n_u + 4

        self.goal_quat = np.asarray(goal_quaternion, dtype=float).reshape(4)
        self.goal_quat = self.goal_quat / (
            np.linalg.norm(self.goal_quat) + 1e-8
        )

        q_goal = np.zeros(self.n_u)
        self.has_joint_goal = goal_joints is not None
        if goal_joints is not None:
            q_goal = np.asarray(goal_joints, dtype=float).reshape(self.n_u)

        self.orientation_weight = float(orientation_weight)
        self.joint_weight = float(joint_weight)
        self.velocity_weight = float(velocity_weight)

        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)

        q = x[0:self.n_u]
        q_base = x[self.n_u : self.n_u + 4]

        # Normalize quaternion
        q_base = q_base / ca.sqrt(ca.dot(q_base, q_base) + 1e-8)

        # Rotation matrices
        R_current = spart.quat_dcm(q_base)
        R_goal = spart.quat_dcm(self.goal_quat)

        R_rel = R_current.T @ R_goal
        trace_R = ca.trace(R_rel)
        trace_clamped = ca.fmin(ca.fmax(trace_R, -1.0), 3.0)
        angle = ca.acos(
            ca.fmin(ca.fmax((trace_clamped - 1.0) / 2.0, -1.0), 1.0)
        )

        orient_cost = self.orientation_weight * angle**2

        joint_cost = 0
        if self.has_joint_goal:
            dq = q - q_goal
            joint_cost = self.joint_weight * ca.dot(dq, dq)
            
        velocity_cost = 0
        if self.velocity_weight > 0:
            velocity_cost = self.velocity_weight * ca.dot(u, u)

        L = orient_cost + joint_cost + velocity_cost

        self.LT_fun = ca.Function("L_terminal", [x, u], [L])
        Lx = ca.gradient(L, x)
        Lxx = ca.hessian(L, x)[0]
        self.Lx_fun = ca.Function("LT_x", [x, u], [Lx])
        self.Lxx_fun = ca.Function("LT_xx", [x, u], [Lxx])

    def value(self, x: np.ndarray, u: np.ndarray | None = None) -> float:
        if u is None:
            u = np.zeros(self.n_u)
        return float(self.LT_fun(x, u))

    def derivatives(self, x: np.ndarray):
        Lx = np.array(self.Lx_fun(x, np.zeros(self.n_u))).reshape(-1)
        Lxx = np.array(self.Lxx_fun(x, np.zeros(self.n_u)))
        return Lx, Lxx


class CasadiDDP:
    """
    iLQR / DDP-style solver using CasADi linearizations.
    Supports both iLQR (first-order dynamics approximation) and Full DDP (second-order).
    """

    def __init__(
        self,
        dynamics_model: CasadiSpaceRobotDynamics,
        running_cost: CasadiRunningCost,
        terminal_cost: CasadiTerminalCost,
        max_iter: int = 50,
        tol: float = 1e-4,
        reg_init: float = 1.0,
        reg_factor: float = 10.0,
        use_full_ddp: bool = False,
    ):
        self.dyn = dynamics_model
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.max_iter = max_iter
        self.tol = tol
        self.reg_init = reg_init
        self.reg_factor = reg_factor
        self.use_full_ddp = use_full_ddp

        self.n_x = self.dyn.state_dim
        self.n_u = self.dyn.control_dim

    def _rollout_cost(self, X: np.ndarray, U: np.ndarray, dt: float) -> float:
        T = U.shape[0]
        cost = 0.0
        for t in range(T):
            cost += self.running_cost.value(X[t], U[t])
        cost += self.terminal_cost.value(X[-1], U[-1])
        return float(cost)

    def backward_pass(self, X: np.ndarray, U: np.ndarray, dt: float, reg: float):
        T = U.shape[0]
        n_x, n_u = self.n_x, self.n_u

        k = np.zeros((T, n_u))
        K = np.zeros((T, n_u, n_x))

        # Terminal value derivatives
        V_x, V_xx = self.terminal_cost.derivatives(X[-1])

        for t in reversed(range(T)):
            x = X[t]
            u = U[t]

            A, B = self.dyn.linearize(x, u, dt)
            L_x, L_u, L_xx, L_uu = self.running_cost.derivatives(x, u)

            # Add terminal velocity cost to the last time step control cost
            if t == T - 1 and self.terminal_cost.velocity_weight > 0:
                # Cost is velocity_weight * ||u||^2
                # Gradient: 2 * weight * u
                # Hessian: 2 * weight * I
                w = self.terminal_cost.velocity_weight
                L_u += 2 * w * u
                L_uu += 2 * w * np.eye(n_u)

            Q_x = L_x + A.T @ V_x
            Q_u = L_u + B.T @ V_x
            Q_xx = L_xx + A.T @ V_xx @ A
            Q_uu = L_uu + B.T @ V_xx @ B
            Q_ux = B.T @ V_xx @ A

            # --- Full DDP: Add dynamics curvature terms ---
            if self.use_full_ddp:
                # F_xx: [n_x, n_x, n_x], F_uu: [n_x, n_u, n_u], F_xu: [n_x, n_x, n_u]
                F_xx, F_uu, F_xu = self.dyn.hessians(x, u, dt)
                
                # Tensor contractions with V_x
                # Q_xx += sum_k (V_x[k] * F_xx[k])
                # Q_uu += sum_k (V_x[k] * F_uu[k])
                # Q_ux += sum_k (V_x[k] * F_xu[k])
                
                # Efficient contraction using tensordot or manual loop
                # V_x is (n_x,)
                
                tens_xx = np.tensordot(V_x, F_xx, axes=([0], [0])) # (n_x, n_x)
                tens_uu = np.tensordot(V_x, F_uu, axes=([0], [0])) # (n_u, n_u)
                tens_xu = np.tensordot(V_x, F_xu, axes=([0], [0])) # (n_x, n_u)
                
                Q_xx += tens_xx
                Q_uu += tens_uu
                Q_ux += tens_xu.T # Note: F_xu was (n_x, n_u) stacked, so tens_xu is (n_x, n_u), but Q_ux should be (n_u, n_x)
                # Wait, F_xu from hessians() is [n_x, n_x, n_u]
                # self.hessian_fns returns f_xu_k as (n_x, n_u)
                # So F_xu[k] is (n_x, n_u).
                # tens_xu = sum_k V_x[k] * F_xu[k] -> (n_x, n_u).
                # This corresponds to ∂(∂f/∂x)/∂u. 
                # Q_ux corresponds to ∂²Q/∂u∂x = ∂(Q_u)/∂x = ∂(L_u + B^T V_x)/∂x
                # = L_ux + ∂(B^T V_x)/∂x 
                # = L_ux + ∂(sum B_ji V_x_j)/∂x
                # Term involving dynamics Hessian is sum_k V_x_k * ∂²f_k/∂u∂x.
                # ∂²f_k/∂u∂x has shape (n_u, n_x).
                # My F_xu[k] is computed as jacobian(gradient(f_k, x), u) -> (n_x, n_u).
                # So F_xu[k] is (∂²f_k/∂x∂u)^T = ∂²f_k/∂u∂x transposed.
                # So tens_xu is (n_x, n_u).
                # We need to add it to Q_ux which is (n_u, n_x).
                # So we add tens_xu.T.
                
            # Regularization and Inverse
            # We need Q_uu to be positive definite.
            # Using Cholesky decomposition to check and invert.
            
            # If use_full_ddp is True, Q_uu might be indefinite due to dynamics curvature.
            # We must ensure Q_uu_reg is PD.
            
            Q_uu_reg = Q_uu + reg * np.eye(n_u)
            
            try:
                # Explicit Cholesky check:
                L = np.linalg.cholesky(Q_uu_reg)
                
                # If success, solve for gains
                k[t] = -np.linalg.solve(Q_uu_reg, Q_u)
                K[t] = -np.linalg.solve(Q_uu_reg, Q_ux)
                
                # Update value function derivatives
                V_x = Q_x + K[t].T @ Q_uu @ k[t] + K[t].T @ Q_u + Q_ux.T @ k[t]
                V_xx = Q_xx + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_ux + Q_ux.T @ K[t]
                
                # Symmetrize V_xx to avoid numerical drift
                V_xx = 0.5 * (V_xx + V_xx.T)

            except np.linalg.LinAlgError:
                # Q_uu_reg is not Positive Definite
                # Return None to signal that we need to increase regularization
                return None, None

        return k, K

    def forward_pass(
        self,
        x0: np.ndarray,
        X_nom: np.ndarray,
        U_nom: np.ndarray,
        k: np.ndarray,
        K: np.ndarray,
        dt: float,
        alpha: float,
    ):
        T = U_nom.shape[0]
        n_x, n_u = self.n_x, self.n_u

        X_new = np.zeros((T + 1, n_x))
        U_new = np.zeros((T, n_u))
        X_new[0] = x0

        for t in range(T):
            dx = X_new[t] - X_nom[t]
            du = alpha * k[t] + K[t] @ dx
            U_new[t] = U_nom[t] + du
            X_new[t + 1] = np.array(self.dyn.f_fun(X_new[t], U_new[t], dt)).reshape(-1)

        J_new = self._rollout_cost(X_new, U_new, dt)
        return X_new, U_new, J_new

    def solve(self, x0: np.ndarray, U0: np.ndarray, dt: float):
        """
        Run iLQR/DDP optimization.
        Returns:
            X_opt, U_opt, cost_history
        """
        U = np.array(U0, dtype=float)
        X = self.dyn.rollout(x0, U, dt)
        J = self._rollout_cost(X, U, dt)

        cost_history = [J]
        reg = self.reg_init

        for it in range(self.max_iter):
            k, K = self.backward_pass(X, U, dt, reg)
            
            # If backward pass failed (non-PD Q_uu), increase regularization and restart
            if k is None:
                reg = max(reg * self.reg_factor, 1e-6)
                reg = min(reg, 1e9) # Cap max reg
                # print(f"  Backward pass failed (non-PD), increasing reg to {reg:.4e}")
                continue

            best_J = np.inf
            best_X = None
            best_U = None
            best_alpha = 0.0

            alpha = 1.0
            for _ in range(10):
                X_new, U_new, J_new = self.forward_pass(
                    x0, X, U, k, K, dt, alpha
                )
                if J_new < best_J:
                    best_J = J_new
                    best_X = X_new
                    best_U = U_new
                    best_alpha = alpha
                alpha *= 0.5

            if best_X is None:
                reg *= self.reg_factor
                continue

            improvement = J - best_J
            X, U, J = best_X, best_U, best_J
            cost_history.append(J)

            if improvement < self.tol:
                break

            if J < cost_history[-2]:
                reg = max(reg / self.reg_factor, 1e-8)
            else:
                reg *= self.reg_factor

        return X, U, cost_history


def load_robot_from_urdf(urdf_path: str) -> dict:
    robot, _ = urdf2robot(urdf_path, verbose_flag=False)
    return robot


