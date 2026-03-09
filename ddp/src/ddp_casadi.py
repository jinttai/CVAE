"""
CasADi-based DDP / iLQR implementation using SPART floating-base dynamics.

using CasADi for automatic differentiation.

State:  [q_joints (6), qd_joints (6), q_base_quat (4)]   -> dimension 16
Control: [qdd_joints (6)]                                -> dimension 6
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
        self.state_dim = 2 * self.n_q + 4 # joints + joint_vels + base quaternion
        self.control_dim = self.n_q      # joint accelerations

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
        self.hessian_fns = []
        for k in range(self.state_dim):
            f_k = x_next[k]
            # Hessian w.r.t x (n_x, n_x)
            f_xx_k = ca.hessian(f_k, x)[0]
            # Hessian w.r.t u (n_u, n_u)
            f_uu_k = ca.hessian(f_k, u)[0]
            
            # Mixed Hessian ∂²f_k / ∂u∂x (n_u, n_x)
            # We compute it as jacobian(gradient(f_k, x), u) -> (n_x, n_u) which is (∂²f/∂x∂u)^T
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
        x  = [q(6); qd(6); q_base(4)]
        u  = [qdd(6)]
        """
        n_q = self.n_q

        q_joints = x[0:n_q]
        qd_joints = x[n_q : 2*n_q]
        q_base = x[2*n_q : 2*n_q + 4]  # [x, y, z, w]

        # Normalize base quaternion
        q_norm = ca.sqrt(ca.dot(q_base, q_base) + 1e-8)
        q_base = q_base / q_norm

        # --- SPART kinematics and dynamics ---
        # Note: SPART kinematics typically depends on q_joints.
        # Momentum conservation depends on qd_joints.
        
        RJ, RL, rJ, rL, e, g = spart.kinematics(self.R0, self.r0, q_joints, self.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(self.R0, self.r0, rL, e, g, self.robot)
        I0, Im = spart.inertia_projection(self.R0, RL, self.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(
            M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )

        # Momentum conservation constraint:
        #   H0 * u0 + H0m * qd = 0  ->  u0 = - H0^{-1} H0m qd
        # Here qd is qd_joints (from state)
        rhs = -H0m @ qd_joints
        
        # Use ca.solve instead of inv for better stability
        u0 = ca.solve(H0, rhs)

        # Base angular velocity (body-fixed) is first 3 components of u0
        wb = u0[0:3]

        # Quaternion integration using exponential map
        q_base_next = self._integrate_quat_exp(q_base, wb, dt)
        
        # Normalize to prevent numerical drift
        q_base_next = q_base_next / ca.sqrt(ca.dot(q_base_next, q_base_next) + 1e-8)

        # Joint integration
        # q_next = q + qd * dt
        q_joints_next = q_joints + qd_joints * dt
        
        # qd_next = qd + qdd * dt (u is qdd)
        qd_joints_next = qd_joints + u * dt

        x_next = ca.vertcat(q_joints_next, qd_joints_next, q_base_next)
        return x_next

    def _build_inverse_dynamics(self):
        """
        Build CasADi function for joint torques: tau = inverse_dynamics(q, qd, qdd).
        EOM: [H0 H0m; H0m' Hm]*[u0_dot; qdd] + [C0 C0m; Cm0 Cm]*[u0; qd] = [0; tau].
        Base unactuated => u0_dot = -H0^{-1}*(H0m*qdd + C0*u0 + C0m*qd), u0 = -H0^{-1}*H0m*qd.
        """
        n_q = self.n_q
        q_joints = ca.SX.sym("q", n_q)
        qd_joints = ca.SX.sym("qd", n_q)
        qdd_joints = ca.SX.sym("qdd", n_q)

        RJ, RL, rJ, rL, e, g = spart.kinematics(self.R0, self.r0, q_joints, self.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(self.R0, self.r0, rL, e, g, self.robot)
        I0, Im = spart.inertia_projection(self.R0, RL, self.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
        H0, H0m, Hm = spart.generalized_inertia_matrix(
            M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )

        # Base velocity from momentum conservation: H0*u0 + H0m*qd = 0
        u0 = ca.solve(H0, -H0m @ qd_joints)

        t0, tL = spart.velocities(Bij, Bi0, P0, pm, u0, qd_joints, self.robot)
        C0, C0m, Cm0, Cm = spart.convective_inertia_matrix(
            t0, tL, I0, Im, M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )

        # u0_dot from base row: H0*u0_dot + H0m*qdd + C0*u0 + C0m*qd = 0
        u0_dot = ca.solve(H0, -(H0m @ qdd_joints + C0 @ u0 + C0m @ qd_joints))

        # Joint torques: H0m'*u0_dot + Hm*qdd + Cm0*u0 + Cm*qd = tau
        tau = H0m.T @ u0_dot + Hm @ qdd_joints + Cm0 @ u0 + Cm @ qd_joints

        self.tau_fun = ca.Function(
            "inverse_dynamics", [q_joints, qd_joints, qdd_joints], [tau]
        )

    def compute_torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Compute joint torques (inverse dynamics) at (q, qd, qdd).
        Returns tau of shape (n_q,).
        """
        if not hasattr(self, "tau_fun") or self.tau_fun is None:
            self._build_inverse_dynamics()
        q = np.asarray(q, dtype=float).reshape(-1)
        qd = np.asarray(qd, dtype=float).reshape(-1)
        qdd = np.asarray(qdd, dtype=float).reshape(-1)
        return np.array(self.tau_fun(q, qd, qdd)).flatten()

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
        """
        n_x = self.state_dim
        n_u = self.control_dim
        
        F_xx = np.zeros((n_x, n_x, n_x))
        F_uu = np.zeros((n_x, n_u, n_u))
        F_xu = np.zeros((n_x, n_x, n_u)) 
        
        for k in range(n_x):
            fns = self.hessian_fns[k]
            F_xx[k] = np.array(fns["f_xx"](x, u, dt))
            F_uu[k] = np.array(fns["f_uu"](x, u, dt))
            F_xu[k] = np.array(fns["f_xu"](x, u, dt))
            
        return F_xx, F_uu, F_xu


class CasadiRunningCost:
    """
    Running cost L(x, u) = u^T R u + Augmented Lagrangian penalty for joint limits.
    u is now joint acceleration.
    
    Uses Augmented Lagrangian Method (ALM) for joint limit constraints:
        g_lower[i] = q_min[i] - q[i] <= 0  (lower bound)
        g_upper[i] = q[i] - q_max[i] <= 0  (upper bound)
    
    ALM penalty for each constraint g <= 0:
        phi(g, lambda, mu) = (mu/2) * max(0, g + lambda/mu)^2 - lambda^2 / (2*mu)
    """

    def __init__(
        self, 
        R_weight: float | np.ndarray, 
        n_u: int = 6, 
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        mu_init: float = 1.0,
        lambda_init: float = 0.0,
    ):
        if np.isscalar(R_weight):
            self.R = float(R_weight) * np.eye(n_u)
        else:
            self.R = np.asarray(R_weight, dtype=float)
        
        self.n_u = n_u
        self.n_x = 2 * n_u + 4
        
        # Joint limits storage
        if joint_limits is not None:
            self.jl_lower = np.asarray(joint_limits[0], dtype=float)
            self.jl_upper = np.asarray(joint_limits[1], dtype=float)
            if self.jl_lower.shape[0] != n_u or self.jl_upper.shape[0] != n_u:
                raise ValueError(f"Joint limits must have size {n_u}")
            self.has_joint_limits = True
        else:
            self.jl_lower = -1e9 * np.ones(n_u)
            self.jl_upper = 1e9 * np.ones(n_u)
            self.has_joint_limits = False
        
        # Augmented Lagrangian parameters
        # mu: penalty parameter (increases to enforce constraints)
        # lambda_lower[i], lambda_upper[i]: Lagrange multipliers for each joint limit
        self.mu = mu_init
        self.lambda_lower = lambda_init * np.ones(n_u)  # multipliers for lower bounds
        self.lambda_upper = lambda_init * np.ones(n_u)  # multipliers for upper bounds
        
        # Build symbolic cost function with ALM
        self._build_cost_function()
    
    def _build_cost_function(self):
        """Build CasADi symbolic cost function with current ALM parameters."""
        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)
        
        # Base running cost: u^T R u
        L_base = ca.mtimes([u.T, self.R, u])
        
        # Augmented Lagrangian penalty for joint limits
        alm_penalty = 0.0
        
        if self.has_joint_limits:
            for i in range(self.n_u):
                q_i = x[i]
                
                # Lower bound constraint: g_lower = q_min - q <= 0
                g_lower = self.jl_lower[i] - q_i
                lambda_l = self.lambda_lower[i]
                
                # Upper bound constraint: g_upper = q - q_max <= 0
                g_upper = q_i - self.jl_upper[i]
                lambda_u = self.lambda_upper[i]
                
                # ALM penalty: phi(g, lambda, mu) = (mu/2) * max(0, g + lambda/mu)^2 - lambda^2/(2*mu)
                # For numerical stability, we use ca.fmax
                
                # Lower bound penalty
                z_lower = g_lower + lambda_l / self.mu
                phi_lower = (self.mu / 2.0) * ca.fmax(0, z_lower)**2 - (lambda_l**2) / (2.0 * self.mu)
                
                # Upper bound penalty
                z_upper = g_upper + lambda_u / self.mu
                phi_upper = (self.mu / 2.0) * ca.fmax(0, z_upper)**2 - (lambda_u**2) / (2.0 * self.mu)
                
                alm_penalty += phi_lower + phi_upper
        
        L = L_base + alm_penalty
        self.L_fun = ca.Function("L_running", [x, u], [L])
        
        # Compute gradients and Hessians
        Lx = ca.gradient(L, x)
        Lu = ca.gradient(L, u)
        Lxx = ca.hessian(L, x)[0]
        Luu = ca.hessian(L, u)[0]
        
        self.Lx_fun = ca.Function("Lx", [x, u], [Lx])
        self.Lu_fun = ca.Function("Lu", [x, u], [Lu])
        self.Lxx_fun = ca.Function("Lxx", [x, u], [Lxx])
        self.Luu_fun = ca.Function("Luu", [x, u], [Luu])
        
        # Constraint violation functions (for multiplier updates)
        if self.has_joint_limits:
            g_lower_vec = ca.SX.zeros(self.n_u)
            g_upper_vec = ca.SX.zeros(self.n_u)
            for i in range(self.n_u):
                g_lower_vec[i] = self.jl_lower[i] - x[i]  # q_min - q
                g_upper_vec[i] = x[i] - self.jl_upper[i]  # q - q_max
            
            self.g_lower_fun = ca.Function("g_lower", [x], [g_lower_vec])
            self.g_upper_fun = ca.Function("g_upper", [x], [g_upper_vec])
    
    def update_multipliers(self, X: np.ndarray):
        """
        Update Lagrange multipliers based on constraint violations.
        Called after each DDP solve iteration.
        
        Update rule: lambda_new = max(0, lambda + mu * g(x))
        
        Args:
            X: State trajectory [T+1, n_x]
        """
        if not self.has_joint_limits:
            return
        
        # Aggregate constraint violations over trajectory
        T = X.shape[0]
        max_g_lower = np.full(self.n_u, -np.inf)
        max_g_upper = np.full(self.n_u, -np.inf)
        
        for t in range(T):
            g_lower = np.array(self.g_lower_fun(X[t])).flatten()
            g_upper = np.array(self.g_upper_fun(X[t])).flatten()
            max_g_lower = np.maximum(max_g_lower, g_lower)
            max_g_upper = np.maximum(max_g_upper, g_upper)
        
        # Update multipliers: lambda = max(0, lambda + mu * g)
        self.lambda_lower = np.maximum(0.0, self.lambda_lower + self.mu * max_g_lower)
        self.lambda_upper = np.maximum(0.0, self.lambda_upper + self.mu * max_g_upper)
        
        # Rebuild cost function with new multipliers
        self._build_cost_function()
    
    def increase_penalty(self, factor: float = 10.0):
        """
        Increase penalty parameter mu.
        Called when constraint violations are not decreasing fast enough.
        """
        self.mu *= factor
        self._build_cost_function()
    
    def get_constraint_violations(self, X: np.ndarray) -> dict:
        """
        Compute maximum constraint violations over trajectory.
        
        Returns:
            dict with 'lower' and 'upper' violations for each joint
        """
        if not self.has_joint_limits:
            return {"lower": np.zeros(self.n_u), "upper": np.zeros(self.n_u)}
        
        T = X.shape[0]
        max_g_lower = np.full(self.n_u, -np.inf)
        max_g_upper = np.full(self.n_u, -np.inf)
        
        for t in range(T):
            g_lower = np.array(self.g_lower_fun(X[t])).flatten()
            g_upper = np.array(self.g_upper_fun(X[t])).flatten()
            max_g_lower = np.maximum(max_g_lower, g_lower)
            max_g_upper = np.maximum(max_g_upper, g_upper)
        
        # Positive values indicate violation
        return {
            "lower": np.maximum(0, max_g_lower),
            "upper": np.maximum(0, max_g_upper),
            "max_violation": max(np.max(np.maximum(0, max_g_lower)), 
                                  np.max(np.maximum(0, max_g_upper)))
        }
    
    def get_alm_info(self) -> dict:
        """Return current ALM parameters for logging."""
        return {
            "mu": self.mu,
            "lambda_lower": self.lambda_lower.copy(),
            "lambda_upper": self.lambda_upper.copy(),
        }

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
        L_T(x) = w_orient * (3 - trace(Rᵀ R_goal))  [base orientation SO3]
              + w_joint * ||q - q_goal||²           [arm joint angles]
              + w_joint_vel * ||qd||²               [arm joint velocity]
              + w_vel_idx11 * x[11]²                [특정 성분 강화: base linear vel index 11]
    """

    def __init__(
        self,
        goal_quaternion: np.ndarray,
        goal_joints: np.ndarray | None = None,
        orientation_weight: float = 1.0,
        joint_weight: float = 0.0,
        joint_vel_weight: float = 0.0,
        vel_idx11_weight: float = 0.0,
        n_u: int = 6,
    ):
        self.n_u = n_u
        self.n_x = 2 * n_u + 4

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
        self.joint_vel_weight = float(joint_vel_weight)
        self.vel_idx11_weight = float(vel_idx11_weight)

        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)

        q = x[0:self.n_u]
        qd = x[self.n_u : 2*self.n_u]
        q_base = x[2*self.n_u : 2*self.n_u + 4]

        # Normalize quaternion
        q_base = q_base / ca.sqrt(ca.dot(q_base, q_base) + 1e-8)

        # Rotation matrices
        R_current = spart.quat_dcm(q_base)
        R_goal = spart.quat_dcm(self.goal_quat)

        # SO(3) orientation error (same as src: log(eps + 0.5*trace((R-Rg)^T(R-Rg))))
        R_diff = R_current - R_goal
        trace_val = 0.5 * ca.trace(R_diff.T @ R_diff)
        orient_cost = self.orientation_weight * ca.log(1e-8 + trace_val)

        joint_cost = 0
        if self.has_joint_goal:
            dq = q - q_goal
            joint_cost = self.joint_weight * ca.dot(dq, dq)

        joint_vel_cost = 0
        if self.joint_vel_weight > 0:
            joint_vel_cost = self.joint_vel_weight * ca.dot(qd, qd)

        vel_idx11_cost = 0
        if self.vel_idx11_weight > 0:
            vel_idx11_cost = self.vel_idx11_weight * x[11] ** 2

        L = orient_cost + joint_cost + joint_vel_cost + vel_idx11_cost

        self.LT_fun = ca.Function("L_terminal", [x, u], [L])
        
        # --- Separate cost components for debugging ---
        self.orient_cost_fun = ca.Function("L_orient", [x], [orient_cost])
        self.joint_cost_fun = ca.Function("L_joint", [x], [joint_cost])
        self.joint_vel_cost_fun = ca.Function("L_joint_vel", [x], [joint_vel_cost])
        self.vel_idx11_cost_fun = ca.Function("L_vel_idx11", [x], [vel_idx11_cost])
        
        Lx = ca.gradient(L, x)
        Lxx = ca.hessian(L, x)[0]
        self.Lx_fun = ca.Function("LT_x", [x, u], [Lx])
        self.Lxx_fun = ca.Function("LT_xx", [x, u], [Lxx])

    def value(self, x: np.ndarray, u: np.ndarray | None = None) -> float:
        if u is None:
            u = np.zeros(self.n_u)
        return float(self.LT_fun(x, u))

    def get_cost_components(self, x: np.ndarray) -> dict:
        """
        Return individual cost components for analysis.
        """
        c_orient = float(self.orient_cost_fun(x))
        c_joint = float(self.joint_cost_fun(x))
        c_joint_vel = float(self.joint_vel_cost_fun(x))
        c_vel_idx11 = float(self.vel_idx11_cost_fun(x))
        return {
            "orientation": c_orient,
            "joint_pos": c_joint,
            "joint_vel": c_joint_vel,
            "vel_idx11": c_vel_idx11,
        }

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

            Q_x = L_x + A.T @ V_x
            Q_u = L_u + B.T @ V_x
            Q_xx = L_xx + A.T @ V_xx @ A
            Q_uu = L_uu + B.T @ V_xx @ B
            Q_ux = B.T @ V_xx @ A

            # --- Full DDP: Add dynamics curvature terms ---
            if self.use_full_ddp:
                F_xx, F_uu, F_xu = self.dyn.hessians(x, u, dt)
                
                tens_xx = np.tensordot(V_x, F_xx, axes=([0], [0])) # (n_x, n_x)
                tens_uu = np.tensordot(V_x, F_uu, axes=([0], [0])) # (n_u, n_u)
                tens_xu = np.tensordot(V_x, F_xu, axes=([0], [0])) # (n_x, n_u)
                
                Q_xx += tens_xx
                Q_uu += tens_uu
                Q_ux += tens_xu.T 
                
            # Regularization and Inverse
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
        Run iLQR/DDP optimization with Augmented Lagrangian outer loop for constraints.
        
        Returns:
            X_opt, U_opt, cost_history
        """
        U = np.array(U0, dtype=float)
        X = self.dyn.rollout(x0, U, dt)
        J = self._rollout_cost(X, U, dt)

        cost_history = [J]
        reg = self.reg_init

        print(f"{'Iter':<5} {'Cost':<12} {'Improvement':<12} {'Reg':<10} {'Alpha':<8} {'MaxViol':<10}")
        
        # Initial terminal cost breakdown
        term_comps = self.terminal_cost.get_cost_components(X[-1])
        print(f"      Initial Terminal Costs -> Orient: {term_comps['orientation']:.4f}, "
              f"JointPos(||q-q_goal||²): {term_comps['joint_pos']:.4f}, "
              f"JointVel(||qd||²→0): {term_comps['joint_vel']:.4f}, "
              f"VelIdx11: {term_comps['vel_idx11']:.4f}")
        
        # Initial constraint violation
        viol_info = self.running_cost.get_constraint_violations(X)
        print(f"      Initial Max Constraint Violation: {viol_info['max_violation']:.6f}")

        for it in range(self.max_iter):
            k, K = self.backward_pass(X, U, dt, reg)
            
            if k is None:
                reg = max(reg * self.reg_factor, 1e-6)
                reg = min(reg, 1e9) 
                print(f"{it:<5} {'REJECT (PD)':<12} {'-':<12} {reg:<10.2e} {'-':<8} {'-':<10}")
                continue

            best_J = np.inf
            best_X = None
            best_U = None
            
            alpha = 1.0
            for _ in range(10):
                X_new, U_new, J_new = self.forward_pass(
                    x0, X, U, k, K, dt, alpha
                )
                if J_new < best_J:
                    best_J = J_new
                    best_X = X_new
                    best_U = U_new
                alpha *= 0.5

            if best_X is None:
                reg *= self.reg_factor
                print(f"{it:<5} {'REJECT (LS)':<12} {'-':<12} {reg:<10.2e} {'-':<8} {'-':<10}")
                continue

            improvement = J - best_J
            X, U, J = best_X, best_U, best_J
            cost_history.append(J)
            
            # Get constraint violation
            viol_info = self.running_cost.get_constraint_violations(X)
            max_viol = viol_info['max_violation']

            print(f"{it:<5} {J:<12.6f} {improvement:<12.6f} {reg:<10.2e} {alpha:<8.4f} {max_viol:<10.6f}")

            if improvement < self.tol:
                print(f"Converged: Improvement < {self.tol}")
                break

            if J < cost_history[-2]:
                reg = max(reg / self.reg_factor, 1e-8)
            else:
                reg *= self.reg_factor

        # Final terminal cost breakdown
        term_comps = self.terminal_cost.get_cost_components(X[-1])
        print(f"\nFinal Terminal Costs -> Orient: {term_comps['orientation']:.4f}, "
              f"JointPos(||q-q_goal||²): {term_comps['joint_pos']:.4f}, "
              f"JointVel(||qd||²→0): {term_comps['joint_vel']:.4f}, "
              f"VelIdx11: {term_comps['vel_idx11']:.4f}")
        
        # Final constraint violation
        viol_info = self.running_cost.get_constraint_violations(X)
        print(f"Final Max Constraint Violation: {viol_info['max_violation']:.6f}")

        return X, U, cost_history
    
    def solve_alm(
        self, 
        x0: np.ndarray, 
        U0: np.ndarray, 
        dt: float,
        alm_max_iter: int = 10,
        constraint_tol: float = 1e-4,
        mu_increase_factor: float = 10.0,
    ):
        """
        Run DDP with Augmented Lagrangian outer loop for constraint handling.
        
        This method performs multiple DDP solves, updating Lagrange multipliers
        and penalty parameters between solves until constraints are satisfied.
        
        Args:
            x0: Initial state
            U0: Initial control sequence
            dt: Time step
            alm_max_iter: Maximum number of ALM outer iterations
            constraint_tol: Tolerance for constraint satisfaction
            mu_increase_factor: Factor to increase penalty parameter
        
        Returns:
            X_opt, U_opt, cost_history
        """
        U = np.array(U0, dtype=float)
        all_cost_history = []
        
        print("=" * 80)
        print("Augmented Lagrangian DDP Solver")
        print("=" * 80)
        
        for alm_iter in range(alm_max_iter):
            print(f"\n{'='*80}")
            print(f"ALM Outer Iteration {alm_iter + 1}/{alm_max_iter}")
            alm_info = self.running_cost.get_alm_info()
            print(f"  mu = {alm_info['mu']:.4e}")
            print(f"  lambda_lower = {alm_info['lambda_lower']}")
            print(f"  lambda_upper = {alm_info['lambda_upper']}")
            print("=" * 80)
            
            # Run inner DDP solve
            X, U, cost_history = self.solve(x0, U, dt)
            all_cost_history.extend(cost_history)
            
            # Check constraint violations
            viol_info = self.running_cost.get_constraint_violations(X)
            max_violation = viol_info['max_violation']
            
            print(f"\nALM Iter {alm_iter + 1}: Max Constraint Violation = {max_violation:.6e}")
            
            if max_violation < constraint_tol:
                print(f"\n*** Constraints satisfied (violation < {constraint_tol}) ***")
                break
            
            # Update Lagrange multipliers
            self.running_cost.update_multipliers(X)
            
            # Check if we need to increase penalty
            # Simple heuristic: increase penalty if violation not decreasing fast enough
            if alm_iter > 0 and max_violation > 0.5 * prev_violation:
                print(f"Increasing penalty parameter mu by factor {mu_increase_factor}")
                self.running_cost.increase_penalty(mu_increase_factor)
            
            prev_violation = max_violation
        
        # Final summary
        print("\n" + "=" * 80)
        print("ALM Optimization Complete")
        alm_info = self.running_cost.get_alm_info()
        print(f"  Final mu = {alm_info['mu']:.4e}")
        viol_info = self.running_cost.get_constraint_violations(X)
        print(f"  Final Max Constraint Violation = {viol_info['max_violation']:.6e}")
        print(f"  Lower bound violations: {viol_info['lower']}")
        print(f"  Upper bound violations: {viol_info['upper']}")
        print("=" * 80)
        
        return X, U, all_cost_history


def load_robot_from_urdf(urdf_path: str) -> dict:
    robot, _ = urdf2robot(urdf_path, verbose_flag=False)
    return robot
