"""
Differentiable Dynamics for DDP
JAX implementation of space robot dynamics using SPART formulation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
import functools

# Add parent directory to path to import SPART functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import src.dynamics.spart_functions_jax as spart


def quat_to_rot(q):
    """
    Convert quaternion [x, y, z, w] to rotation matrix R (3x3).
    Always returns [3, 3] matrix.
    """
    # Ensure q is a 1D array [x, y, z, w]
    q = jnp.asarray(q).flatten()
    
    # Extract components
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # Compute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    # Build rotation matrix
    R = jnp.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]
    ])
    
    return R


def normalize_quat(q):
    """Normalize quaternion to unit length."""
    norm = jnp.linalg.norm(q)
    return q / (norm + 1e-8)


def skew_symmetric(v):
    """3D vector to skew-symmetric matrix [v]_x."""
    vx, vy, vz = v[0], v[1], v[2]
    
    return jnp.array([
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0]
    ])


def rot_from_omega_exponential(wb, dt):
    """
    Calculate rotation matrix from angular velocity using exponential map.
    R_delta = exp([ω]_× * dt)
    Uses Rodrigues' formula: exp([ω]_× * dt) = I + sin(θ)*[ω̂]_× + (1-cos(θ))*[ω̂]_×²
    
    Args:
        wb: [3] angular velocity vector
        dt: scalar time step
    
    Returns:
        R_delta: [3, 3] rotation matrix
    """
    # Calculate rotation angle
    wb_norm = jnp.linalg.norm(wb)
    # Clamp theta to prevent numerical instability from very large angular velocities
    # Maximum rotation per step: π radians (180 degrees)
    max_theta = 3.141592653589793  # π
    theta = jnp.clip(wb_norm * dt, a_max=max_theta)
    
    # Small angle approximation for numerical stability
    eps = 1e-8
    I = jnp.eye(3)
    
    # Compute axis (normalized) - reuse wb_norm calculation
    axis = wb / (wb_norm + 1e-12)
    K = skew_symmetric(axis)
    
    # General Rodrigues' formula
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    K_squared = K @ K
    R_big = I + sin_theta * K + (1.0 - cos_theta) * K_squared
    
    # Small angle approximation: R ≈ I + [ω*dt]_×
    wb_dt = wb * dt
    K_small = skew_symmetric(wb_dt)
    R_small = I + K_small
    
    # Select based on angle size
    small = theta < eps
    R_delta = jnp.where(small, R_small, R_big)
    
    return R_delta


def rot_to_quat(R):
    """
    Convert rotation matrix R (3x3) to quaternion [x, y, z, w].
    JAX-compatible version using jnp.where for conditional logic.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    def safe_sqrt(x):
        return jnp.sqrt(jnp.clip(x, a_min=1e-8))
    
    # Case 1: trace > 0
    S1 = safe_sqrt(trace + 1.0) * 2
    w1 = 0.25 * S1
    x1 = (R[2, 1] - R[1, 2]) / S1
    y1 = (R[0, 2] - R[2, 0]) / S1
    z1 = (R[1, 0] - R[0, 1]) / S1
    q1 = jnp.array([x1, y1, z1, w1])
    
    # Case 2: r00 is max
    S2 = safe_sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
    w2 = (R[2, 1] - R[1, 2]) / S2
    x2 = 0.25 * S2
    y2 = (R[0, 1] + R[1, 0]) / S2
    z2 = (R[0, 2] + R[2, 0]) / S2
    q2 = jnp.array([x2, y2, z2, w2])
    
    # Case 3: r11 is max
    S3 = safe_sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
    w3 = (R[0, 2] - R[2, 0]) / S3
    x3 = (R[0, 1] + R[1, 0]) / S3
    y3 = 0.25 * S3
    z3 = (R[1, 2] + R[2, 1]) / S3
    q3 = jnp.array([x3, y3, z3, w3])
    
    # Case 4: r22 is max
    S4 = safe_sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    w4 = (R[1, 0] - R[0, 1]) / S4
    x4 = (R[0, 2] + R[2, 0]) / S4
    y4 = (R[1, 2] + R[2, 1]) / S4
    z4 = 0.25 * S4
    q4 = jnp.array([x4, y4, z4, w4])
    
    # Determine which case to use (JAX-friendly)
    cond1 = trace > 0
    cond2 = (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2])
    cond3 = R[1, 1] > R[2, 2]
    
    # Use nested jnp.where
    q = jnp.where(cond1, q1,
           jnp.where(cond2, q2,
              jnp.where(cond3, q3, q4)))
    
    return normalize_quat(q)


class SpaceRobotDynamics:
    """
    Differentiable dynamics model for space robot.
    State: [joint_angles (6), base_quaternion (4)] = [10]
    Control: [joint_velocities (6)] = [6]
    """
    
    def __init__(self, robot):
        # urdf2robot_jax already returns JAX arrays, so we can use it directly
        # If it's a tuple, extract the robot dict
        if isinstance(robot, tuple):
            robot = robot[0]
        
        self.robot = robot
        self.n_q = self.robot['n_q']  # Number of joints (6)
        self.state_dim = self.n_q + 4  # 6 joints + 4 quaternion
        self.control_dim = self.n_q  # 6 joint velocities
        
        # Base frame (identity)
        self.R0 = jnp.eye(3)
        self.r0 = jnp.zeros(3)
        
        # Damping term for constraint solver
        self.damping_value = 1e-6
        self.eye6 = jnp.eye(6)
        self.damping_term = self.damping_value * self.eye6

        # Simple joint angle bound to prevent extremely large configurations
        self.q_max = 2.0 * jnp.pi
    
    @functools.partial(jax.jit, static_argnames=('self',))
    def step(self, state, control, dt):
        """
        Single step dynamics: x_{k+1} = f(x_k, u_k)
        
        Args:
            state: [10] array [q1...q6, qx, qy, qz, qw]
            control: [6] array [qd1...qd6] (joint velocities)
            dt: time step (scalar)
        
        Returns:
            next_state: [10] array
        """
        # Extract joint angles and quaternion
        q_joints = state[:self.n_q]  # [6]
        q_base = state[self.n_q:]  # [4] quaternion [x, y, z, w]
        
        # Normalize quaternion
        q_base = normalize_quat(q_base)
        
        # Convert quaternion to rotation matrix
        R_base = quat_to_rot(q_base)  # [3, 3]
        
        # --- SPART Dynamics Calculation ---
        # Forward kinematics
        RJ, RL, rJ, rL, e, g = spart.kinematics(
            self.R0, self.r0, q_joints, self.robot
        )
        
        # Differential kinematics
        Bij, Bi0, P0, pm = spart.diff_kinematics(
            self.R0, self.r0, rL, e, g, self.robot
        )
        
        # Inertia projection
        I0, Im = spart.inertia_projection(
            self.R0, RL, self.robot
        )
        
        # Composite mass matrices
        M0_t, Mm_t = spart.mass_composite_body(
            I0, Im, Bij, Bi0, self.robot
        )
        
        # Generalized inertia matrix
        H0, H0m, _ = spart.generalized_inertia_matrix(
            M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot
        )

        # Sanitize inertia matrices to avoid NaNs/Infs propagating into solve
        H0 = jnp.nan_to_num(H0, nan=0.0, posinf=1e6, neginf=-1e6)
        H0m = jnp.nan_to_num(H0m, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # --- Non-holonomic Constraint Solver ---
        # Momentum conservation: H0 * u0 + H0m * qd = 0
        rhs = -H0m @ control  # [6]
        rhs = jnp.nan_to_num(rhs, nan=0.0, posinf=1e6, neginf=-1e6)
        H0_damped = H0 + self.damping_term
        u0_sol = jnp.linalg.solve(H0_damped, rhs)  # [6]
        # Numerical safety: sanitize solution to avoid NaNs/Infs
        u0_sol = jnp.nan_to_num(u0_sol, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Extract angular velocity (first 3 elements)
        wb = u0_sol[:3]  # [3] angular velocity in body frame
        # Clamp angular velocity magnitude to avoid extreme rotations
        wb_norm = jnp.linalg.norm(wb)
        wb_max = 10.0  # rad/s
        scale = jnp.minimum(1.0, wb_max / (wb_norm + 1e-8))
        wb = wb * scale
        
        # --- Update rotation using exponential map ---
        # R_new = R_old @ exp([ω]_× * dt)
        R_delta = rot_from_omega_exponential(wb, dt)  # [3, 3]
        R_base_next = R_base @ R_delta  # [3, 3]
        
        # Convert rotation matrix back to quaternion
        q_base_next = rot_to_quat(R_base_next)  # [4]
        
        # --- Update joint angles (simple integration) ---
        q_joints_next = q_joints + dt * control
        # Clamp joint angles to a reasonable range to avoid numerical issues
        q_joints_next = jnp.clip(q_joints_next, -self.q_max, self.q_max)
        
        # Combine next state
        next_state = jnp.concatenate([q_joints_next, q_base_next], axis=0)
        
        return next_state
    
    @functools.partial(jax.jit, static_argnames=('self',))
    def rollout(self, initial_state, controls, dt):
        """
        Rollout trajectory given initial state and control sequence.
        
        Args:
            initial_state: [10] initial state
            controls: [T, 6] control sequence
            dt: time step (scalar)
        
        Returns:
            states: [T+1, 10] state trajectory
        """
        T = controls.shape[0]

        # Use scan for better JAX compatibility
        def scan_fun(carry, t):
            next_state = self.step(carry, controls[t], dt)
            return next_state, next_state

        _, states_array = jax.lax.scan(
            scan_fun, initial_state, jnp.arange(T)
        )

        # Stack initial state with trajectory
        states = jnp.vstack([initial_state, states_array])

        return states

