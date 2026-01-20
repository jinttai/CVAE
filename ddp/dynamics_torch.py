"""
Differentiable Dynamics for DDP
PyTorch implementation of space robot dynamics using SPART formulation.
"""

import torch
import sys
import os

# Add parent directory to path to import SPART functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import src.dynamics.spart_functions_torch as spart
from src.dynamics.urdf2robot_torch import urdf2robot


def quat_to_rot(q):
    """
    Convert quaternion [x, y, z, w] to rotation matrix R (3x3).
    Always returns [3, 3] matrix.
    """
    orig_shape = q.shape
    was_1d = q.dim() == 1
    
    if was_1d:
        q = q.unsqueeze(0)  # [1, 4]
    
    x, y, z, w = q.unbind(-1)  # Each is [1] if was_1d, or [B] if batch
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
    
    # Stack into [B, 3, 3] or [3, 3]
    if was_1d:
        # Single quaternion: create [3, 3] directly
        R = torch.stack([
            torch.stack([r00.squeeze(), r01.squeeze(), r02.squeeze()]),
            torch.stack([r10.squeeze(), r11.squeeze(), r12.squeeze()]),
            torch.stack([r20.squeeze(), r21.squeeze(), r22.squeeze()])
        ])
    else:
        # Batch: [B, 3, 3]
        R = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ], dim=-2)
    
    return R


def normalize_quat(q):
    """Normalize quaternion to unit length."""
    norm = torch.linalg.norm(q)
    return q / (norm + 1e-8)


def skew_symmetric(v):
    """3D vector to skew-symmetric matrix [v]_x."""
    vx, vy, vz = v[0], v[1], v[2]
    device = v.device
    dtype = v.dtype
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    
    return torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])


class SpaceRobotDynamics:
    """
    Differentiable dynamics model for space robot.
    State: [joint_angles (6), base_quaternion (4)] = [10]
    Control: [joint_velocities (6)] = [6]
    """
    
    def __init__(self, robot, device='cpu'):
        self.robot = robot
        self.device = device
        self.n_q = robot['n_q']  # Number of joints (6)
        self.state_dim = self.n_q + 4  # 6 joints + 4 quaternion
        self.control_dim = self.n_q  # 6 joint velocities
        
        # Base frame (identity)
        self.R0 = torch.eye(3, device=device)
        self.r0 = torch.zeros(3, device=device)
        
        # Damping term for constraint solver
        self.damping_value = 1e-6
        self.eye6 = torch.eye(6, device=device)
        self.damping_term = self.damping_value * self.eye6
    
    def step(self, state, control, dt):
        """
        Single step dynamics: x_{k+1} = f(x_k, u_k)
        
        Args:
            state: [10] tensor [q1...q6, qx, qy, qz, qw]
            control: [6] tensor [qd1...qd6] (joint velocities)
            dt: time step (scalar)
        
        Returns:
            next_state: [10] tensor
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
        
        # --- Non-holonomic Constraint Solver ---
        # Momentum conservation: H0 * u0 + H0m * qd = 0
        rhs = -H0m @ control  # [6]
        H0_damped = H0 + self.damping_term
        u0_sol = torch.linalg.solve(H0_damped, rhs)  # [6]
        
        # Extract angular velocity (first 3 elements)
        wb = u0_sol[:3]  # [3] angular velocity in body frame
        
        # --- Update quaternion using angular velocity ---
        # Quaternion derivative: dq/dt = 0.5 * q * [0, wx, wy, wz]
        q_dot = spart.quat_dot(q_base, wb)  # [4]
        q_base_next = normalize_quat(q_base + dt * q_dot)
        
        # --- Update joint angles (simple integration) ---
        q_joints_next = q_joints + dt * control
        
        # Combine next state
        next_state = torch.cat([q_joints_next, q_base_next], dim=0)
        
        return next_state
    
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
        states = torch.zeros(T + 1, self.state_dim, device=self.device)
        states[0] = initial_state
        
        for t in range(T):
            states[t + 1] = self.step(states[t], controls[t], dt)
        
        return states

