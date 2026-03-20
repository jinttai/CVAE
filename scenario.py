"""
Common test scenario shared by DDP and src (CVAE/MLP training).
Import from both sides to ensure identical setup.

Usage:
    from scenario import SCENARIO, get_goal_quaternion
"""
import numpy as np


def euler_to_quaternion(roll, pitch, yaw):
    """Euler angles (rad) -> quaternion [x, y, z, w]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


SCENARIO = {
    # ── Robot ──
    "urdf": "assets/SC_ur10e.urdf",
    "n_q": 6,

    # ── Time horizon ──
    "T": 100,          # number of steps
    "dt": 0.1,         # step size (s)
    "total_time": 10.0, # T * dt

    # ── Initial state ──
    "q0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # joint angles (rad)
    "qd0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # joint velocities (rad/s)
    "q_base0": [0.0, 0.0, 0.0, 1.0],              # base quaternion [x,y,z,w] (identity)

    # ── Goal ──
    "goal_euler_deg": [15.0, 15.0, -15.0],         # roll, pitch, yaw (deg)
    "goal_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    # ── src specific ──
    "num_waypoints": 3,

    # ── Cost weights (orientation cost: log(eps + 0.5*trace((R-Rg)^T(R-Rg)))) ──
    "orientation_weight": 10.0,
    "joint_weight": 100.0,
    "joint_vel_weight": 50.0,
    "vel_idx11_weight": 100.0,
    "R_weight": 1e-2,

    # ── src regularization ──
    "joint_squared_weight": 0.01,
    "joint_change_weight": 0.01,
    "max_joint_weight": 0.1,
}


def get_goal_quaternion():
    """Return goal quaternion [x, y, z, w] from SCENARIO euler angles."""
    r, p, y = [np.deg2rad(a) for a in SCENARIO["goal_euler_deg"]]
    return euler_to_quaternion(r, p, y)


def get_initial_state():
    """Return full DDP initial state vector x0 (16D)."""
    return np.array(
        SCENARIO["q0"] + SCENARIO["qd0"] + SCENARIO["q_base0"],
        dtype=float,
    )
