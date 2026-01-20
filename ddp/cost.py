"""
Cost Functions for DDP/iLQR
Running cost and terminal cost implementations.
"""

import torch
import ddp.dynamics_torch as dynamics


def log_map_so3(R):
    """
    Logarithmic map on SO(3): R -> log(R) in so(3) (3D vector).
    Returns the rotation axis * angle representation.
    """
    # Trace of R
    trace = torch.trace(R)
    
    # Clamp trace for numerical stability
    trace_clamped = torch.clamp(trace, min=-1.0, max=3.0)
    
    # Angle: theta = arccos((trace(R) - 1) / 2)
    theta = torch.acos(torch.clamp((trace_clamped - 1.0) / 2.0, min=-1.0, max=1.0))
    
    # Handle small angles (use Taylor expansion)
    eps = 1e-6
    small_angle = theta < eps
    
    # For small angles: log(R) ≈ (R - R^T) / 2
    R_minus_RT = R - R.T
    log_small = torch.stack([
        R_minus_RT[2, 1],
        R_minus_RT[0, 2],
        R_minus_RT[1, 0]
    ]) / 2.0
    
    # For larger angles: log(R) = (theta / (2*sin(theta))) * (R - R^T)
    sin_theta = torch.sin(theta)
    coeff = torch.where(small_angle, torch.ones_like(theta), theta / (2.0 * sin_theta + 1e-8))
    log_large = coeff * log_small
    
    return torch.where(small_angle.unsqueeze(0).expand(3), log_small, log_large)


def geodesic_distance_so3(R1, R2):
    """
    Geodesic distance on SO(3) between two rotation matrices.
    Returns the angle of rotation needed to align R1 with R2.
    """
    # Ensure R1 and R2 are [3, 3] matrices
    if R1.dim() > 2:
        R1 = R1.squeeze()
    if R2.dim() > 2:
        R2 = R2.squeeze()
    
    # Ensure they are exactly [3, 3]
    if R1.shape != (3, 3):
        R1 = R1.view(3, 3)
    if R2.shape != (3, 3):
        R2 = R2.view(3, 3)
    
    # Ensure same dtype and device
    dtype = R1.dtype
    device = R1.device
    R2 = R2.to(dtype=dtype, device=device)
    
    R_diff = R1.T @ R2  # Relative rotation [3, 3]
    trace = torch.trace(R_diff)
    trace_clamped = torch.clamp(trace, min=-1.0, max=3.0)
    angle = torch.acos(torch.clamp((trace_clamped - 1.0) / 2.0, min=-1.0, max=1.0))
    return angle


class RunningCost:
    """
    Running cost L(x, u) = u^T R u + soft_joint_limits(x)
    """
    
    def __init__(self, R_weight, joint_limits, barrier_weight=1.0, device='cpu'):
        """
        Args:
            R_weight: [6, 6] or scalar control cost weight matrix
            joint_limits: [6, 2] tensor with [lower, upper] limits for each joint
            barrier_weight: weight for joint limit barrier function
            device: device for tensors
        """
        self.device = device
        
        # Control cost weight
        if isinstance(R_weight, (int, float)):
            self.R = R_weight * torch.eye(6, device=device)
        else:
            self.R = R_weight.to(device)
        
        # Joint limits
        self.joint_limits = joint_limits.to(device)  # [6, 2]
        self.barrier_weight = barrier_weight
        
        # Soft limit parameters
        self.margin = 0.1  # margin from hard limits for soft penalty
        self.steepness = 10.0  # steepness of barrier function
    
    def __call__(self, state, control):
        """
        Compute running cost L(x, u).
        
        Args:
            state: [10] tensor [q1...q6, qx, qy, qz, qw]
            control: [6] tensor [qd1...qd6]
        
        Returns:
            cost: scalar tensor
        """
        # Control effort: u^T R u
        # Use unsqueeze instead of .T for 1D tensors
        if control.dim() == 1:
            # u^T R u where u is [6], R is [6, 6]
            # Result: [1, 6] @ [6, 6] @ [6, 1] = [1, 1] -> squeeze to scalar
            control_cost = (control.unsqueeze(0) @ self.R @ control.unsqueeze(-1)).squeeze()
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
        penalty_lower = self.barrier_weight * torch.sum(
            torch.exp(-self.steepness * torch.clamp(dist_lower, min=0.0))
        )
        
        # Upper limit penalty
        dist_upper = upper - self.margin - q_joints
        penalty_upper = self.barrier_weight * torch.sum(
            torch.exp(-self.steepness * torch.clamp(dist_upper, min=0.0))
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
                 orientation_weight=1.0, joint_weight=1.0, device='cpu'):
        """
        Args:
            goal_quaternion: [4] tensor [x, y, z, w] target orientation
            goal_joints: [6] tensor target joint angles (optional)
            orientation_weight: weight for orientation cost
            joint_weight: weight for joint cost
            device: device for tensors
        """
        self.device = device
        self.goal_quat = goal_quaternion.to(device=device, dtype=torch.float32)
        self.goal_quat = dynamics.normalize_quat(self.goal_quat)
        self.goal_R = dynamics.quat_to_rot(self.goal_quat)  # [3, 3]
        # Ensure goal_R is float32
        if self.goal_R.dtype != torch.float32:
            self.goal_R = self.goal_R.to(dtype=torch.float32)
        
        # Joint goal (optional)
        if goal_joints is not None:
            self.goal_joints = goal_joints.to(device=device, dtype=torch.float32)
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
            state: [10] tensor [q1...q6, qx, qy, qz, qw]
        
        Returns:
            cost: scalar tensor
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
            joint_cost = self.joint_weight * torch.sum(joint_error ** 2)
            total_cost = total_cost + joint_cost
        
        return total_cost

