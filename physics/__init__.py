"""
Physics module for space robot dynamics and utility functions.
"""

from .utils import (
    # NumPy functions
    euler_to_quaternion_np,
    quat_to_euler_np,
    quat_to_rot_np,
    geodesic_distance_so3_np,
    skew_np,
    # PyTorch functions
    euler_to_quaternion_torch,
    quat_to_rot_torch,
    skew_torch,
    rot_from_omega_torch,
    rot_to_euler_torch,
    wrap_to_pi_torch,
    geodesic_distance_so3_torch,
    generate_random_quaternion_from_euler_torch,
    # Unified interface
    euler_to_quaternion,
    quat_to_rot,
    quat_to_euler,
    geodesic_distance_so3,
    skew,
)

__all__ = [
    # NumPy
    'euler_to_quaternion_np',
    'quat_to_euler_np',
    'quat_to_rot_np',
    'geodesic_distance_so3_np',
    'skew_np',
    # PyTorch
    'euler_to_quaternion_torch',
    'quat_to_rot_torch',
    'skew_torch',
    'rot_from_omega_torch',
    'rot_to_euler_torch',
    'wrap_to_pi_torch',
    'geodesic_distance_so3_torch',
    'generate_random_quaternion_from_euler_torch',
    # Unified
    'euler_to_quaternion',
    'quat_to_rot',
    'quat_to_euler',
    'geodesic_distance_so3',
    'skew',
]
