"""
Run SQP (Sequential Quadratic Programming) using CasADi + SPART dynamics.

This script solves the trajectory optimization problem using a custom Python-based
SQP solver that utilizes CasADi for derivatives and Scipy for sparse linear algebra.
This approach is robust even if CasADi's IPOPT/SNOPT plugins are missing.
"""

import os
import sys
import time
import numpy as np
import casadi as ca
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Add project root to sys.path based on current file location
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ddp.src.ddp_casadi import (
    load_robot_from_urdf,
    CasadiSpaceRobotDynamics,
    CasadiRunningCost,
    CasadiTerminalCost,
)
from ddp.src.trajectory_utils import save_trajectory_csv


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to quaternion [x, y, z, w].
    roll, pitch, yaw are in radians (floats).
    """
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q) + 1e-8
    return q


def quat_to_euler(q):
    """
    Convert quaternion [x, y, z, w] to ZYX Euler angles [roll, pitch, yaw].
    Returns angles in radians.
    """
    x, y, z, w = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def quat_to_rot(q):
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix (NumPy).
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])


def geodesic_distance_so3(R1, R2):
    """
    Compute geodesic distance between two rotation matrices.
    theta = arccos((trace(R1^T @ R2) - 1) / 2)
    """
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    val = (trace - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)


class CustomSQPSolver:
    """
    A pure Python SQP solver for Trajectory Optimization.
    Uses CasADi for derivatives and Scipy for KKT system solution.
    Solves equality constrained optimization:
        min f(z)
        s.t. g(z) = 0
    where z = [x0, u0, x1, u1, ..., xT]
    """
    def __init__(self, dyn, running_cost, terminal_cost, T, dt):
        self.dyn = dyn
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.T = T
        self.dt = dt
        
        self.n_x = dyn.state_dim
        self.n_u = dyn.control_dim
        
        # Build symbolic NLP
        self._build_nlp()

    def _build_nlp(self):
        T = self.T
        dt = self.dt
        n_x = self.n_x
        n_u = self.n_u
        
        # Total variables: (T+1)*n_x + T*n_u
        # Structure: x0, u0, x1, u1, ..., xT
        
        # Create symbolic variables
        X = [ca.SX.sym(f"x_{t}", n_x) for t in range(T + 1)]
        U = [ca.SX.sym(f"u_{t}", n_u) for t in range(T)]
        
        # Objective function
        J = 0
        for t in range(T):
            J += self.running_cost.L_fun(X[t], U[t])
        # Pass dummy control to LT_fun
        J += self.terminal_cost.LT_fun(X[T], ca.DM.zeros(n_u))
        
        # Constraints (Dynamics + Initial Condition placeholder)
        # g(z) = 0
        g = []
        
        # Note: Initial condition constraint will be handled by fixing x0 bounds 
        # or adding explicit constraint. Here we add explicit constraint x0 - x_init = 0
        # But x_init is a parameter. 
        # Let's make x_init a parameter 'p'
        p = ca.SX.sym("x_init", n_x)
        g.append(X[0] - p)
        
        for t in range(T):
            x_next = self.dyn.f_fun(X[t], U[t], dt)
            g.append(X[t+1] - x_next)
            
        g_all = ca.vertcat(*g)
        
        # Flatten decision variables
        # z = [x0, u0, x1, u1, ..., x_{T-1}, u_{T-1}, xT]
        z_list = []
        for t in range(T):
            z_list.append(X[t])
            z_list.append(U[t])
        z_list.append(X[T])
        z = ca.vertcat(*z_list)
        
        self.n_z = z.shape[0]
        self.n_g = g_all.shape[0]
        
        # Create functions for SQP
        # 1. Objective gradient and Hessian (approximated or exact)
        # Using exact Hessian of Lagrangian is better for SQP
        lam = ca.SX.sym("lam", self.n_g)
        L = J + ca.dot(lam, g_all)
        
        hess_L = ca.hessian(L, z)[0]
        grad_J = ca.gradient(J, z)
        jac_g = ca.jacobian(g_all, z)
        
        self.f_cost = ca.Function("f_cost", [z], [J])
        self.f_grad_J = ca.Function("f_grad_J", [z], [grad_J])
        self.f_g = ca.Function("f_g", [z, p], [g_all])
        self.f_jac_g = ca.Function("f_jac_g", [z], [jac_g])
        self.f_hess_L = ca.Function("f_hess_L", [z, lam], [hess_L])
        
    def solve(self, x0_val, max_iter=20, tol=1e-4):
        # Initial guess
        z_curr = np.zeros(self.n_z)
        lam_curr = np.zeros(self.n_g)
        
        # Regularization parameter
        reg = 1e-3
        
        # Set x parts of initial guess to x0 (simple warm start)
        # Helper to set/get z
        def set_x(z_vec, t, x_val):
            idx = t * (self.n_x + self.n_u)
            z_vec[idx : idx + self.n_x] = x_val
            
        def get_vars(z_vec):
            # Extract X and U
            X_out = np.zeros((self.T + 1, self.n_x))
            U_out = np.zeros((self.T, self.n_u))
            for t in range(self.T):
                idx = t * (self.n_x + self.n_u)
                X_out[t] = z_vec[idx : idx + self.n_x]
                U_out[t] = z_vec[idx + self.n_x : idx + self.n_x + self.n_u]
            X_out[self.T] = z_vec[self.T * (self.n_x + self.n_u) :]
            return X_out, U_out

        # Initialize trajectory with rollout (better than zeros)
        # But for SQP, sometimes simple initialization is fine.
        # Let's do a rollout with zero controls.
        curr_x = x0_val
        for t in range(self.T + 1):
            set_x(z_curr, t, curr_x)
            if t < self.T:
                # u is zero, so just dynamics
                curr_x = np.array(self.dyn.f_fun(curr_x, np.zeros(self.n_u), self.dt)).flatten()
        
        print(f"  Iter |   Cost   |  ||g||  |  ||dz|| | alpha | reg")
        print(f"----------------------------------------------------------")
        
        success = False
        start_time = time.time()
        
        for i in range(max_iter):
            # Evaluate NLP functions
            J_val = float(self.f_cost(z_curr))
            g_val = np.array(self.f_g(z_curr, x0_val)).flatten()
            grad_J_val = np.array(self.f_grad_J(z_curr)).flatten()
            jac_g_val = np.array(self.f_jac_g(z_curr)) 
            hess_L_val = np.array(self.f_hess_L(z_curr, lam_curr)) 
            
            # Form KKT matrix
            # Ensure Hessian is positive definite (regularization)
            H = hess_L_val + reg * np.eye(self.n_z) # Regularization
            A = jac_g_val
            
            # Assemble sparse KKT matrix
            # Using scipy sparse for speed
            zeros_br = np.zeros((self.n_g, self.n_g))
            
            # Construct KKT matrix [ H   A.T ]
            #                      [ A    0  ]
            KKT = sp.bmat([
                [sp.csc_matrix(H), sp.csc_matrix(A.T)],
                [sp.csc_matrix(A), sp.csc_matrix(zeros_br)]
            ], format='csc')
            
            rhs = np.concatenate([-grad_J_val - A.T @ lam_curr, -g_val])
            
            # Solve
            try:
                sol = spla.spsolve(KKT, rhs)
            except Exception as e:
                print(f"Linear solver failed: {e}")
                reg *= 10.0 # Increase regularization if solve fails
                continue
                
            dz = sol[:self.n_z]
            dlam = sol[self.n_z:]
            
            # Line Search (Backtracking with Merit Function)
            alpha = 1.0
            
            # L1 Merit Function: phi(z) = J(z) + mu * ||g(z)||1
            # Heuristic for mu: mu >= ||lambda||_inf
            mu = np.max(np.abs(lam_curr + dlam)) * 1.5 + 1.0
            if i == 0: mu = 1000.0 # Initial large penalty
            
            phi_curr = J_val + mu * np.sum(np.abs(g_val))
            
            z_new = z_curr
            accept = False
            
            for ls_iter in range(12):
                z_try = z_curr + alpha * dz
                
                J_try = float(self.f_cost(z_try))
                g_try = np.array(self.f_g(z_try, x0_val)).flatten()
                phi_try = J_try + mu * np.sum(np.abs(g_try))
                
                # Simple check for decrease
                if phi_try < phi_curr:
                    z_new = z_try
                    lam_curr = lam_curr + alpha * dlam
                    accept = True
                    break
                
                alpha *= 0.5
            
            if not accept:
                # If line search fails, increase regularization and try taking a small step
                reg *= 5.0
                reg = min(reg, 1e2)
                z_curr = z_curr + 1e-4 * dz
            else:
                z_curr = z_new
                # Decrease regularization if step accepted easily (large alpha)
                if alpha > 0.1:
                    reg *= 0.5
                    reg = max(reg, 1e-6)

            g_norm = np.linalg.norm(g_val)
            dz_norm = np.linalg.norm(dz)
            
            print(f"  {i+1:3d}  | {J_val:.4e} | {g_norm:.4e} | {dz_norm:.4e} | {alpha:.2e} | {reg:.2e}")
            
            if g_norm < tol and dz_norm < tol:
                success = True
                break
                
        solve_time = time.time() - start_time
        X_opt, U_opt = get_vars(z_curr)
        return X_opt, U_opt, solve_time, success


def solve_sqp_custom(dyn, running_cost, terminal_cost, x0_val, T, dt):
    """
    Wrapper for CustomSQPSolver
    """
    solver = CustomSQPSolver(dyn, running_cost, terminal_cost, T, dt)
    return solver.solve(x0_val, max_iter=30, tol=1e-3)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    urdf_path = os.path.join(root_dir, "assets", "SC_ur10e.urdf")

    robot = load_robot_from_urdf(urdf_path)

    # Time horizon
    T = 100
    dt = 0.1
    total_time = T * dt

    print(f"Time horizon: {total_time}s ({T} steps, dt={dt}s)")

    # Dynamics
    dyn = CasadiSpaceRobotDynamics(robot)

    # Initial state: joints zero, base identity quaternion
    q0 = np.zeros(6)
    q_base0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    x0 = np.concatenate([q0, q_base0])

    # Goal orientation & joints
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    q_goal = euler_to_quaternion(roll, pitch, yaw)
    goal_joints = np.zeros(6)

    print(f"Initial orientation: Identity")
    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")

    # Costs
    running_cost = CasadiRunningCost(R_weight=0.01, n_u=6)
    terminal_cost = CasadiTerminalCost(
        goal_quaternion=q_goal,
        goal_joints=goal_joints,
        orientation_weight=20.0,
        joint_weight=5.0,
    )

    print("\nStarting Custom SQP optimization (Python + CasADi + Scipy)...")
    print("-" * 50)
    
    # Use Custom Solver
    X_opt, U_opt, elapsed_time, success = solve_sqp_custom(
        dyn, running_cost, terminal_cost, x0, T, dt
    )

    print("-" * 50)
    print(f"Optimization completed! (Success: {success})")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # Compute cost manually for display
    final_cost = 0
    for t in range(T):
        final_cost += running_cost.value(X_opt[t], U_opt[t])
    final_cost += terminal_cost.value(X_opt[-1])
    print(f"Final cost: {final_cost:.6f}")

    # --- Compute final errors ---
    final_q_base = X_opt[-1, 6:]
    final_q_base /= np.linalg.norm(final_q_base) + 1e-8
    final_R = quat_to_rot(final_q_base)
    goal_R = quat_to_rot(q_goal)
    
    orient_err_rad = geodesic_distance_so3(final_R, goal_R)
    orient_err_deg = np.rad2deg(orient_err_rad)
    
    final_joints = X_opt[-1, :6]
    joint_err_norm = np.linalg.norm(final_joints - goal_joints)

    print(f"\nFinal orientation error: {orient_err_deg:.2f}°")
    print(f"Final joint error: {joint_err_norm:.4f} rad")

    # --- Save Results ---
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save NPY
    np.save(os.path.join(results_dir, "trajectory_casadi_sqp_states.npy"), X_opt)
    np.save(os.path.join(results_dir, "trajectory_casadi_sqp_controls.npy"), U_opt)
    
    # Save CSV
    csv_path = os.path.join(results_dir, "trajectory_casadi_sqp.csv")
    save_trajectory_csv(X_opt, U_opt, dt, csv_path, method_name="casadi_sqp")

    print(f"\nSaved trajectories to: {results_dir}")

    # --- Plotting ---
    plt.figure(figsize=(14, 10))
    time_steps = np.arange(T + 1) * dt

    # Joint angles
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, X_opt[:, :6])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.title('Joint Angles (SQP)')
    plt.grid(True, alpha=0.3)
    plt.legend([f'J{i+1}' for i in range(6)])

    # Controls
    plt.subplot(2, 2, 2)
    plt.plot(time_steps[:-1], U_opt)
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Velocity (rad/s)')
    plt.title('Controls (Joint Velocities)')
    plt.grid(True, alpha=0.3)
    plt.legend([f'J{i+1}' for i in range(6)])

    # Quaternion (now Euler)
    plt.subplot(2, 2, 3)
    
    euler_angles = []
    for t in range(T + 1):
        qt = X_opt[t, 6:]
        qt /= np.linalg.norm(qt) + 1e-8
        euler_angles.append(np.rad2deg(quat_to_euler(qt)))
    euler_angles = np.array(euler_angles)
    
    plt.plot(time_steps, euler_angles)
    plt.xlabel('Time (s)')
    plt.ylabel('Euler Angles (deg)')
    plt.title('Base Orientation (Euler)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Roll', 'Pitch', 'Yaw'])
    # Goal lines
    goals = [roll_deg, pitch_deg, yaw_deg]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, val in enumerate(goals):
        plt.axhline(y=val, color=colors[i], linestyle='--', alpha=0.5)

    # Orientation Error
    errors = []
    for t in range(T + 1):
        qt = X_opt[t, 6:]
        qt /= np.linalg.norm(qt) + 1e-8
        Rt = quat_to_rot(qt)
        err = geodesic_distance_so3(Rt, goal_R)
        errors.append(np.rad2deg(err))
    
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, errors, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (deg)')
    plt.title('Orientation Error')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "trajectory_casadi_sqp.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved plots to results directory.")


if __name__ == "__main__":
    main()
