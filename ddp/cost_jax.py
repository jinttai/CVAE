"""
Cost Functions for DDP/iLQR
Running cost and terminal cost implementations (JAX version).
"""

import jax
import jax.numpy as jnp
import ddp.dynamics_jax as dynamics


def log_map_so3(R):
    """
    Logarithmic map on SO(3): R -> log(R) in so(3) (3D vector).
    Returns the rotation axis * angle representation.
    """
    # Trace of R
    trace = jnp.trace(R)
    
    # Clamp trace for numerical stability
    trace_clamped = jnp.clip(trace, a_min=-1.0, a_max=3.0)
    
    # Angle: theta = arccos((trace(R) - 1) / 2)
    theta = jnp.arccos(jnp.clip((trace_clamped - 1.0) / 2.0, a_min=-1.0, a_max=1.0))
    
    # Handle small angles (use Taylor expansion)
    eps = 1e-6
    small_angle = theta < eps
    
    # For small angles: log(R) ≈ (R - R^T) / 2
    R_minus_RT = R - R.T
    log_small = jnp.array([
        R_minus_RT[2, 1],
        R_minus_RT[0, 2],
        R_minus_RT[1, 0]
    ]) / 2.0
    
    # For larger angles: log(R) = (theta / (2*sin(theta))) * (R - R^T)
    sin_theta = jnp.sin(theta)
    coeff = jnp.where(small_angle, 1.0, theta / (2.0 * sin_theta + 1e-8))
    log_large = coeff * log_small
    
    return jnp.where(small_angle, log_small, log_large)


def geodesic_distance_so3(R1, R2):
    """
    Geodesic distance on SO(3) between two rotation matrices.
    Returns the angle of rotation needed to align R1 with R2.
    """
    # Ensure R1 and R2 are [3, 3] matrices
    if len(R1.shape) > 2:
        R1 = jnp.squeeze(R1)
    if len(R2.shape) > 2:
        R2 = jnp.squeeze(R2)
    
    # Ensure they are exactly [3, 3]
    if R1.shape != (3, 3):
        R1 = R1.reshape(3, 3)
    if R2.shape != (3, 3):
        R2 = R2.reshape(3, 3)
    
    R_diff = R1.T @ R2  # Relative rotation [3, 3]
    trace = jnp.trace(R_diff)
    trace_clamped = jnp.clip(trace, a_min=-1.0, a_max=3.0)
    angle = jnp.arccos(jnp.clip((trace_clamped - 1.0) / 2.0, a_min=-1.0, a_max=1.0))
    return angle


class RunningCost:
    """
    Running cost L(x, u) = u^T R u + soft_joint_limits(x)
    """
    
    def __init__(self, R_weight, joint_limits, barrier_weight=1.0):
        """
        Args:
            R_weight: [6, 6] or scalar control cost weight matrix
            joint_limits: [6, 2] array with [lower, upper] limits for each joint
            barrier_weight: weight for joint limit barrier function
        """
        # Control cost weight
        if isinstance(R_weight, (int, float)):
            self.R = R_weight * jnp.eye(6)
        else:
            self.R = jnp.array(R_weight)
        
        # Joint limits
        self.joint_limits = jnp.array(joint_limits)  # [6, 2]
        self.barrier_weight = barrier_weight
        
        # Soft limit parameters
        self.margin = 0.1  # margin from hard limits for soft penalty
        self.steepness = 10.0  # steepness of barrier function
    
    def __call__(self, state, control):
        """
        Compute running cost L(x, u).
        
        Args:
            state: [10] array [q1...q6, qx, qy, qz, qw]
            control: [6] array [qd1...qd6]
        
        Returns:
            cost: scalar array
        """
        # Control effort: u^T R u
        if len(control.shape) == 1:
            # u^T R u where u is [6], R is [6, 6]
            control_cost = control @ self.R @ control
        else:
            control_cost = control.T @ self.R @ control
        
        # Joint limit penalty (soft barrier)
        q_joints = state[:6]
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        
        # Barrier function: penalty when approaching limits
        # Penalty = weight * exp(-steepness * (q - lower + margin))
        #         + weight * exp(-steepness * (upper - margin - q))
        
        # Lower limit penalty
        dist_lower = q_joints - lower + self.margin
        penalty_lower = self.barrier_weight * jnp.sum(
            jnp.exp(-self.steepness * jnp.clip(dist_lower, a_min=0.0))
        )
        
        # Upper limit penalty
        dist_upper = upper - self.margin - q_joints
        penalty_upper = self.barrier_weight * jnp.sum(
            jnp.exp(-self.steepness * jnp.clip(dist_upper, a_min=0.0))
        )
        
        joint_limit_cost = penalty_lower + penalty_upper
        
        return control_cost + joint_limit_cost


class TerminalCost:
    """
    Terminal cost L_final(x) = orientation_cost + joint_cost
    - orientation_cost: geodesic distance on SO(3) to goal orientation
    - joint_cost: squared distance to goal joint angles
    """
    
    def __init__(self, goal_quaternion, goal_joints=None, 
                 orientation_weight=1.0, joint_weight=1.0):
        """
        Args:
            goal_quaternion: [4] array [x, y, z, w] target orientation
            goal_joints: [6] array target joint angles (optional)
            orientation_weight: weight for orientation cost
            joint_weight: weight for joint cost
        """
        self.goal_quat = jnp.array(goal_quaternion)
        self.goal_quat = dynamics.normalize_quat(self.goal_quat)
        self.goal_R = dynamics.quat_to_rot(self.goal_quat)  # [3, 3]
        
        # Joint goal (optional)
        if goal_joints is not None:
            self.goal_joints = jnp.array(goal_joints)
            self.has_joint_goal = True
        else:
            self.goal_joints = None
            self.has_joint_goal = False
        
        self.orientation_weight = orientation_weight
        self.joint_weight = joint_weight
    
    def __call__(self, state):
        """
        Compute terminal cost L_final(x).
        
        Args:
            state: [10] array [q1...q6, qx, qy, qz, qw]
        
        Returns:
            cost: scalar array
        """
        total_cost = 0.0
        
        # Orientation cost: geodesic distance on SO(3)
        q_base = state[6:]  # [4]
        q_base = dynamics.normalize_quat(q_base)
        R_current = dynamics.quat_to_rot(q_base)  # [3, 3]
        orientation_distance = geodesic_distance_so3(R_current, self.goal_R)
        orientation_cost = self.orientation_weight * (orientation_distance ** 2)
        total_cost = total_cost + orientation_cost
        
        # Joint cost: squared distance to goal joints
        if self.has_joint_goal:
            q_joints = state[:6]  # [6]
            joint_error = q_joints - self.goal_joints
            joint_cost = self.joint_weight * jnp.sum(joint_error ** 2)
            total_cost = total_cost + joint_cost
        
        return total_cost

