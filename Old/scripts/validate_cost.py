"""
Validate that DDP and src orientation costs produce identical values
for the same scenarios, including the shared SCENARIO config.
"""
import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import src.dynamics.spart_casadi as spart_ca
from scenario import SCENARIO, get_goal_quaternion


# -- helpers ------------------------------------------------------------------
def euler_to_quat_np(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def cost_casadi(q_curr, q_goal):
    """DDP orientation cost (CasADi)."""
    R_curr = np.array(spart_ca.quat_dcm(q_curr))
    R_goal = np.array(spart_ca.quat_dcm(q_goal))
    R_diff = R_curr - R_goal
    trace_val = 0.5 * np.trace(R_diff.T @ R_diff)
    return float(np.log(1e-8 + trace_val))


def cost_torch(q_curr, q_goal):
    """src orientation cost (PyTorch)."""
    def quat_to_rot(q):
        x, y, z, w = q
        return torch.tensor([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=torch.float64)

    R_curr = quat_to_rot(torch.tensor(q_curr, dtype=torch.float64))
    R_goal = quat_to_rot(torch.tensor(q_goal, dtype=torch.float64))
    R_diff = R_curr - R_goal
    trace_val = 0.5 * torch.trace(R_diff.T @ R_diff)
    return float(torch.log(torch.tensor(1e-8) + trace_val))


# -- scenarios ----------------------------------------------------------------
scenarios = [
    ("SCENARIO default (15,15,-15 deg)",
     np.array(SCENARIO["q_base0"]),
     get_goal_quaternion()),
    ("Identity (error=0)",
     euler_to_quat_np(0, 0, 0),
     euler_to_quat_np(0, 0, 0)),
    ("Small rotation (5,5,-5 deg)",
     euler_to_quat_np(0, 0, 0),
     euler_to_quat_np(np.radians(5), np.radians(5), np.radians(-5))),
    ("Large rotation (45,30,-60 deg)",
     euler_to_quat_np(0, 0, 0),
     euler_to_quat_np(np.radians(45), np.radians(30), np.radians(-60))),
    ("Near 180 deg rotation",
     euler_to_quat_np(0, 0, 0),
     euler_to_quat_np(np.radians(170), 0, 0)),
    ("Both non-identity",
     euler_to_quat_np(np.radians(10), np.radians(-20), np.radians(30)),
     euler_to_quat_np(np.radians(-15), np.radians(25), np.radians(-10))),
]

print("=" * 75)
print(f"{'Scenario':<35} {'DDP(CasADi)':>14} {'src(Torch)':>14} {'diff':>12}")
print("=" * 75)

all_pass = True
for name, q_curr, q_goal in scenarios:
    c_ddp = cost_casadi(q_curr, q_goal)
    c_src = cost_torch(q_curr, q_goal)
    diff = abs(c_ddp - c_src)
    status = "OK" if diff < 1e-6 else "MISMATCH"
    if diff >= 1e-6:
        all_pass = False
    print(f"{name:<35} {c_ddp:>14.8f} {c_src:>14.8f} {diff:>12.2e} {status}")

print("=" * 75)
if all_pass:
    print("ALL PASSED: DDP and src orientation costs are identical.")
else:
    print("SOME FAILED: costs differ -- check formulas.")
