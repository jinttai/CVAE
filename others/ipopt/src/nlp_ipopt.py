"""
CasADi NLP formulation for space robot trajectory optimization using IPOPT.

Solves the same problem as ddp_acc but via direct transcription (multiple shooting):
- State: [q_joints(6), qd_joints(6), q_base_quat(4)]  -> 16
- Control: [qdd_joints(6)]  -> 6

NLP variables: z = [x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T]
Objective: sum_t L(x_t, u_t) + L_T(x_T)
Equality constraints: x_0 = x_init,  x_{t+1} = f(x_t, u_t, dt)
Bounds: joint limits on q for each x_t
"""

import numpy as np
import casadi as ca

from src.dynamics.urdf2robot import urdf2robot
import src.dynamics.spart_casadi as spart


class CasadiSpaceRobotDynamics:
    """
    Floating-base space robot dynamics in CasADi (SPART).
    Same as ddp_acc for compatibility.
    """

    def __init__(self, robot: dict):
        self.robot = robot
        self.n_q = robot["n_q"]
        self.state_dim = 2 * self.n_q + 4
        self.control_dim = self.n_q
        self.R0 = np.eye(3)
        self.r0 = np.zeros((3, 1))

        x = ca.SX.sym("x", self.state_dim)
        u = ca.SX.sym("u", self.control_dim)
        dt = ca.SX.sym("dt")
        x_next = self._step_symbolic(x, u, dt)
        self.f_fun = ca.Function("f_step", [x, u, dt], [x_next])

    def _quat_multiply(self, q: ca.SX, p: ca.SX) -> ca.SX:
        q_xyz = q[0:3]
        q_w = q[3]
        p_xyz = p[0:3]
        p_w = p[3]
        r_xyz = q_w * p_xyz + p_w * q_xyz + ca.cross(q_xyz, p_xyz)
        r_w = q_w * p_w - ca.dot(q_xyz, p_xyz)
        return ca.vertcat(r_xyz, r_w)

    def _integrate_quat_exp(self, q: ca.SX, w: ca.SX, dt: ca.SX) -> ca.SX:
        theta_sq = ca.dot(w, w) * dt**2 + 1e-16
        theta = ca.sqrt(theta_sq)
        a = theta / 2.0
        small_angle = theta < 1e-4
        k = ca.if_else(small_angle, 0.5 - theta_sq/48.0, ca.sin(a) / theta)
        qw = ca.if_else(small_angle, 1.0 - theta_sq/8.0, ca.cos(a))
        dq_xyz = w * dt * k
        dq = ca.vertcat(dq_xyz, qw)
        return self._quat_multiply(q, dq)

    def _step_symbolic(self, x: ca.SX, u: ca.SX, dt: ca.SX) -> ca.SX:
        n_q = self.n_q
        q_joints = x[0:n_q]
        qd_joints = x[n_q : 2*n_q]
        q_base = x[2*n_q : 2*n_q + 4]
        q_norm = ca.sqrt(ca.dot(q_base, q_base) + 1e-8)
        q_base = q_base / q_norm

        RJ, RL, rJ, rL, e, g = spart.kinematics(self.R0, self.r0, q_joints, self.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(self.R0, self.r0, rL, e, g, self.robot)
        I0, Im = spart.inertia_projection(self.R0, RL, self.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(
            M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )
        rhs = -H0m @ qd_joints
        u0 = ca.solve(H0, rhs)
        wb = u0[0:3]
        q_base_next = self._integrate_quat_exp(q_base, wb, dt)
        q_base_next = q_base_next / ca.sqrt(ca.dot(q_base_next, q_base_next) + 1e-8)
        q_joints_next = q_joints + qd_joints * dt
        qd_joints_next = qd_joints + u * dt
        return ca.vertcat(q_joints_next, qd_joints_next, q_base_next)


def build_terminal_cost_casadi(goal_quat, goal_joints, orientation_weight, joint_weight, joint_vel_weight, n_u):
    """Build CasADi terminal cost function L_T(x)."""
    n_x = 2 * n_u + 4
    x = ca.SX.sym("x", n_x)
    q = x[0:n_u]
    qd = x[n_u : 2*n_u]
    q_base = x[2*n_u : 2*n_u + 4]
    q_base = q_base / ca.sqrt(ca.dot(q_base, q_base) + 1e-8)

    R_current = spart.quat_dcm(q_base)
    R_goal = spart.quat_dcm(goal_quat)
    R_rel = R_current.T @ R_goal
    trace_R = ca.trace(R_rel)
    orient_cost = orientation_weight * (3.0 - trace_R)

    q_goal = np.asarray(goal_joints, dtype=float).reshape(n_u)
    joint_cost = joint_weight * ca.dot(q - q_goal, q - q_goal)
    joint_vel_cost = joint_vel_weight * ca.dot(qd, qd)
    L_T = orient_cost + joint_cost + joint_vel_cost
    return ca.Function("L_terminal", [x], [L_T])


def build_ipopt_nlp(
    dyn: CasadiSpaceRobotDynamics,
    x0: np.ndarray,
    goal_quat: np.ndarray,
    goal_joints: np.ndarray,
    T: int,
    dt: float,
    R_weight: float = 0.01,
    orientation_weight: float = 20.0,
    joint_weight: float = 1.0,
    joint_vel_weight: float = 1.0,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
):
    """
    Build CasADi NLP for trajectory optimization.
    Returns: nlp_dict, nlp_solver, variable bounds/indices.
    """
    n_x = dyn.state_dim
    n_u = dyn.control_dim
    n_q = dyn.n_q
    f_fun = dyn.f_fun

    # NLP variable layout: [x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T]
    # Total: (T+1)*n_x + T*n_u
    n_z = (T + 1) * n_x + T * n_u

    lbz = np.full(n_z, -np.inf)
    ubz = np.full(n_z, np.inf)

    # Running cost: L(x,u) = u' R u
    R = R_weight * np.eye(n_u)
    LT_fun = build_terminal_cost_casadi(
        goal_quat, goal_joints, orientation_weight, joint_weight, joint_vel_weight, n_q
    )

    # Create symbolic variables for each stage
    x_syms = [ca.SX.sym(f"x_{t}", n_x) for t in range(T + 1)]
    u_syms = [ca.SX.sym(f"u_{t}", n_u) for t in range(T)]

    # Objective
    J = 0
    for t in range(T):
        u = u_syms[t]
        J += ca.mtimes([u.T, R, u])
    J += LT_fun(x_syms[T])

    # Dynamics equality constraints: x_{t+1} = f(x_t, u_t, dt)
    g_list = []
    for t in range(T):
        x_next_pred = f_fun(x_syms[t], u_syms[t], dt)
        g_list.append(x_syms[t + 1] - x_next_pred)

    g_dyn = ca.vertcat(*g_list)

    # Initial condition: x_0 = x0
    g_init = x_syms[0] - x0
    g = ca.vertcat(g_init, g_dyn)

    # Decision variable: z = [x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T]
    z_alt = []
    for t in range(T):
        z_alt.append(x_syms[t])
        z_alt.append(u_syms[t])
    z_alt.append(x_syms[T])
    z = ca.vertcat(*z_alt)

    # Bounds on z: z = [x_0,u_0, x_1,u_1, ..., x_{T-1},u_{T-1}, x_T]
    block_size = n_x + n_u
    for t in range(T):
        x_start = t * block_size
        for i in range(n_q):
            if joint_limits is not None:
                lbz[x_start + i] = joint_limits[0][i]
                ubz[x_start + i] = joint_limits[1][i]
    # x_T
    x_T_start = T * block_size
    for i in range(n_q):
        if joint_limits is not None:
            lbz[x_T_start + i] = joint_limits[0][i]
            ubz[x_T_start + i] = joint_limits[1][i]

    nlp = {
        "x": z,
        "f": J,
        "g": g,
    }
    solver = ca.nlpsol("ipopt_solver", "ipopt", nlp)

    # Pack for solve: we need lbx, ubx, lbg, ubg
    lbg = np.zeros(g.size1())
    ubg = np.zeros(g.size1())

    return {
        "nlp": nlp,
        "solver": solver,
        "lbz": lbz,
        "ubz": ubz,
        "lbg": lbg,
        "ubg": ubg,
        "n_x": n_x,
        "n_u": n_u,
        "T": T,
    }


def solve_ipopt(
    dyn: CasadiSpaceRobotDynamics,
    x0: np.ndarray,
    goal_quat: np.ndarray,
    goal_joints: np.ndarray,
    T: int = 100,
    dt: float = 0.1,
    R_weight: float = 0.01,
    orientation_weight: float = 20.0,
    joint_weight: float = 1.0,
    joint_vel_weight: float = 1.0,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
):
    """
    Solve trajectory optimization using IPOPT.
    Returns X_opt [T+1, n_x], U_opt [T, n_u].
    """
    n_x = dyn.state_dim
    n_u = dyn.control_dim

    build = build_ipopt_nlp(
        dyn, x0, goal_quat, goal_joints, T, dt,
        R_weight, orientation_weight, joint_weight, joint_vel_weight, joint_limits
    )
    solver = build["solver"]
    lbz = build["lbz"]
    ubz = build["ubz"]
    lbg = build["lbg"]
    ubg = build["ubg"]

    # Initial guess: linear interpolation in state, zero control
    block_size = n_x + n_u
    z0 = np.zeros(lbz.shape[0])
    x_goal = np.concatenate([goal_joints, np.zeros(n_u), goal_quat])
    for t in range(T):
        alpha = t / T
        x_t = (1 - alpha) * x0 + alpha * x_goal
        start = t * block_size
        z0[start : start + n_x] = x_t
        z0[start + n_x : start + block_size] = 0  # u_t = 0
    z0[T * block_size : T * block_size + n_x] = x_goal

    sol = solver(x0=z0, lbx=lbz, ubx=ubz, lbg=lbg, ubg=ubg)
    z_opt = np.array(sol["x"]).flatten()

    # Extract X_opt, U_opt from z_opt: [x_0,u_0, x_1,u_1, ..., x_{T-1},u_{T-1}, x_T]
    X_opt = np.zeros((T + 1, n_x))
    U_opt = np.zeros((T, n_u))
    block_size = n_x + n_u
    for t in range(T):
        start = t * block_size
        X_opt[t] = z_opt[start : start + n_x]
        U_opt[t] = z_opt[start + n_x : start + block_size]
    X_opt[T] = z_opt[T * block_size : T * block_size + n_x]

    return X_opt, U_opt


def load_robot_from_urdf(urdf_path: str) -> dict:
    robot, _ = urdf2robot(urdf_path, verbose_flag=False)
    return robot
