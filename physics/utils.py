"""
Physics Utility Functions

Common rotation and matrix operations used across the project.
Provides both NumPy and PyTorch implementations.
"""

import numpy as np
import torch
import math

# =============================================================================
# NumPy Implementations
# =============================================================================

def euler_to_quaternion_np(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert ZYX Euler angles to quaternion [x, y, z, w].

    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)

    Returns:
        Quaternion as numpy array [x, y, z, w], normalized
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


def quat_to_euler_np(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to ZYX Euler angles [roll, pitch, yaw].

    Args:
        q: Quaternion as numpy array [x, y, z, w]

    Returns:
        Euler angles as numpy array [roll, pitch, yaw] in radians
    """
    x, y, z, w = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def quat_to_rot_np(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.

    Args:
        q: Quaternion as numpy array [x, y, z, w]

    Returns:
        3x3 rotation matrix as numpy array
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])


def geodesic_distance_so3_np(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute geodesic distance between two rotation matrices.
    theta = arccos((trace(R1^T @ R2) - 1) / 2)

    Args:
        R1: First 3x3 rotation matrix
        R2: Second 3x3 rotation matrix

    Returns:
        Geodesic distance in radians
    """
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    val = (trace - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)


def skew_np(v: np.ndarray) -> np.ndarray:
    """
    Compute skew-symmetric matrix from 3D vector.

    Args:
        v: 3D vector as numpy array

    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


# =============================================================================
# PyTorch Implementations
# =============================================================================

def euler_to_quaternion_torch(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
    Using ZYX convention (yaw around Z, pitch around Y, roll around X).

    Args:
        roll: Rotation around X-axis (radians), shape [batch] or scalar
        pitch: Rotation around Y-axis (radians), shape [batch] or scalar
        yaw: Rotation around Z-axis (radians), shape [batch] or scalar

    Returns:
        Quaternion tensor [x, y, z, w], shape [..., 4]
    """
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return torch.stack([qx, qy, qz, qw], dim=-1)


def quat_to_rot_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    Supports both single input [4] and batch input [B, 4].

    Args:
        q: Quaternion tensor, shape [4] or [B, 4]

    Returns:
        Rotation matrix tensor, shape [3, 3] or [B, 3, 3]
    """
    if q.dim() == 1:
        x, y, z, w = q
    else:
        x, y, z, w = q.unbind(dim=-1)

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

    if q.dim() == 1:
        R = torch.stack([
            torch.stack([r00, r01, r02]),
            torch.stack([r10, r11, r12]),
            torch.stack([r20, r21, r22]),
        ], dim=0)
    else:
        R = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ], dim=-2)

    return R


def skew_torch(v: torch.Tensor) -> torch.Tensor:
    """
    Compute skew-symmetric matrix from 3D vector (PyTorch).

    Args:
        v: 3D vector tensor, shape [3] or components as tuple

    Returns:
        3x3 skew-symmetric matrix tensor
    """
    if isinstance(v, (tuple, list)):
        vx, vy, vz = v
    else:
        vx, vy, vz = v[0], v[1], v[2]

    zero = torch.zeros_like(vx)
    M = torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])
    return M


def rot_from_omega_torch(wb: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Compute rotation matrix from angular velocity using Rodrigues' formula.

    Args:
        wb: Angular velocity vector, shape [3]
        dt: Time step

    Returns:
        3x3 rotation matrix tensor
    """
    device = wb.device
    dtype = wb.dtype

    theta = torch.linalg.norm(wb) * dt
    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew_torch(axis)
    I = torch.eye(3, device=device, dtype=dtype)

    # Rodrigues' rotation formula
    R_delta = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R_delta


def rot_to_euler_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix (3x3) to Euler angles (ZYX order: yaw-pitch-roll).

    Args:
        R: 3x3 rotation matrix tensor

    Returns:
        Euler angles tensor [yaw, pitch, roll] in radians
    """
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        yaw = torch.atan2(R[1, 0], R[0, 0])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.atan2(R[2, 1], R[2, 2])
    else:
        yaw = torch.atan2(-R[0, 1], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.zeros_like(yaw)

    return torch.stack([yaw, pitch, roll])


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to [-pi, pi] range.

    Args:
        x: Angle tensor in radians

    Returns:
        Wrapped angle tensor in [-pi, pi]
    """
    two_pi = 2.0 * math.pi
    return torch.remainder(x + math.pi, two_pi) - math.pi


def geodesic_distance_so3_torch(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance between two rotation matrices (PyTorch).

    Args:
        R1: First 3x3 rotation matrix tensor
        R2: Second 3x3 rotation matrix tensor

    Returns:
        Geodesic distance tensor in radians
    """
    R_diff = R1.T @ R2
    trace = torch.trace(R_diff)
    val = (trace - 1.0) / 2.0
    val = torch.clamp(val, -1.0, 1.0)
    return torch.acos(val)


def generate_random_quaternion_from_euler_torch(
    batch_size: int,
    max_angle_deg: float = 30.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate random quaternions from Euler angles within specified range.

    Args:
        batch_size: Number of quaternions to generate
        max_angle_deg: Maximum angle in degrees for each axis
        device: PyTorch device

    Returns:
        Quaternion tensor, shape [batch_size, 4]
    """
    max_angle_rad = math.radians(max_angle_deg)

    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad

    return euler_to_quaternion_torch(roll, pitch, yaw)


# =============================================================================
# Unified Interface (auto-detect numpy vs torch)
# =============================================================================

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion. Auto-detects numpy vs torch.
    """
    if isinstance(roll, torch.Tensor):
        return euler_to_quaternion_torch(roll, pitch, yaw)
    else:
        return euler_to_quaternion_np(roll, pitch, yaw)


def quat_to_rot(q):
    """
    Convert quaternion to rotation matrix. Auto-detects numpy vs torch.
    """
    if isinstance(q, torch.Tensor):
        return quat_to_rot_torch(q)
    else:
        return quat_to_rot_np(q)


def quat_to_euler(q):
    """
    Convert quaternion to Euler angles (NumPy only for now).
    """
    if isinstance(q, torch.Tensor):
        q_np = q.detach().cpu().numpy()
        return quat_to_euler_np(q_np)
    else:
        return quat_to_euler_np(q)


def geodesic_distance_so3(R1, R2):
    """
    Compute geodesic distance between rotation matrices. Auto-detects numpy vs torch.
    """
    if isinstance(R1, torch.Tensor):
        return geodesic_distance_so3_torch(R1, R2)
    else:
        return geodesic_distance_so3_np(R1, R2)


def skew(v):
    """
    Compute skew-symmetric matrix. Auto-detects numpy vs torch.
    """
    if isinstance(v, torch.Tensor) or (isinstance(v, (tuple, list)) and isinstance(v[0], torch.Tensor)):
        return skew_torch(v)
    else:
        return skew_np(v)
