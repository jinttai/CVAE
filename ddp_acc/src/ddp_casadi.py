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
        
        # Use matrix inversion 
        u0 = ca.mtimes(ca.inv(H0), rhs)

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
    Running cost L(x, u) = u^T R u.
    u is now joint acceleration.
    """

    def __init__(self, R_weight: float | np.ndarray, n_u: int = 6):
        if np.isscalar(R_weight):
            self.R = float(R_weight) * np.eye(n_u)
        else:
            self.R = np.asarray(R_weight, dtype=float)
        
        self.n_u = n_u
        # n_x = 2*n_u + 4
        self.n_x = 2 * n_u + 4

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
    
    Note: 'velocity_weight' here applies to the control input 'u', which is now acceleration.
    Ideally we should also penalize terminal joint velocity (which is in state).
    But preserving structure for now.
    """

    def __init__(
        self,
        goal_quaternion: np.ndarray,
        goal_joints: np.ndarray | None = None,
        orientation_weight: float = 1.0,
        joint_weight: float = 0.0,
        joint_vel_weight: float = 0.0,
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

        R_rel = R_current.T @ R_goal
        trace_R = ca.trace(R_rel)
        
        # Use simple trace cost: 1 - trace(R_rel)/3? Or just 3 - trace.
        # Max trace is 3. Min trace is -1.
        # Cost = w * (3 - trace(R_rel))  -> ranges from 0 to 4*w
        
        orient_cost = self.orientation_weight * (3.0 - trace_R)

        joint_cost = 0
        if self.has_joint_goal:
            dq = q - q_goal
            joint_cost = self.joint_weight * ca.dot(dq, dq)
            
        joint_vel_cost = 0
        if self.joint_vel_weight > 0:
            joint_vel_cost = self.joint_vel_weight * ca.dot(qd, qd)

        L = orient_cost + joint_cost + joint_vel_cost

        self.LT_fun = ca.Function("L_terminal", [x, u], [L])
        
        # --- Separate cost components for debugging ---
        self.orient_cost_fun = ca.Function("L_orient", [x], [orient_cost])
        self.joint_cost_fun = ca.Function("L_joint", [x], [joint_cost])
        self.joint_vel_cost_fun = ca.Function("L_joint_vel", [x], [joint_vel_cost])
        
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
        return {
            "orientation": c_orient,
            "joint_pos": c_joint,
            "joint_vel": c_joint_vel
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
        Run iLQR/DDP optimization.
        Returns:
            X_opt, U_opt, cost_history
        """
        U = np.array(U0, dtype=float)
        X = self.dyn.rollout(x0, U, dt)
        J = self._rollout_cost(X, U, dt)

        cost_history = [J]
        reg = self.reg_init

        print(f"{'Iter':<5} {'Cost':<12} {'Improvement':<12} {'Reg':<10} {'Alpha':<8}")
        
        # Initial terminal cost breakdown
        term_comps = self.terminal_cost.get_cost_components(X[-1])
        print(f"      Initial Terminal Costs -> Orient: {term_comps['orientation']:.4f}, "
              f"JointPos: {term_comps['joint_pos']:.4f}, JointVel: {term_comps['joint_vel']:.4f}")

        for it in range(self.max_iter):
            k, K = self.backward_pass(X, U, dt, reg)
            
            if k is None:
                reg = max(reg * self.reg_factor, 1e-6)
                reg = min(reg, 1e9) 
                print(f"{it:<5} {'REJECT (PD)':<12} {'-':<12} {reg:<10.2e} {'-':<8}")
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
                print(f"{it:<5} {'REJECT (LS)':<12} {'-':<12} {reg:<10.2e} {'-':<8}")
                continue

            improvement = J - best_J
            X, U, J = best_X, best_U, best_J
            cost_history.append(J)

            print(f"{it:<5} {J:<12.6f} {improvement:<12.6f} {reg:<10.2e} {alpha:<8.4f}")

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
              f"JointPos: {term_comps['joint_pos']:.4f}, JointVel: {term_comps['joint_vel']:.4f}")

        return X, U, cost_history


def load_robot_from_urdf(urdf_path: str) -> dict:
    robot, _ = urdf2robot(urdf_path, verbose_flag=False)
    return robot
