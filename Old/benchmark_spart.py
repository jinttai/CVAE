"""Benchmark SPART dynamics forward+backward pass."""
import torch
import time
import math
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.dynamics.urdf2robot_torch import urdf2robot
from src.training.physics_layer import PhysicsLayer


def benchmark(batch_size=256, n_iters=20, warmup=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    robot, _ = urdf2robot('assets/SC_ur10e.urdf', verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    TOTAL_TIME = 10.0
    n_q = robot['n_q']
    JOINT_MIN = math.radians(-140.0)
    JOINT_MAX = math.radians(140.0)

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    # Fixed inputs
    torch.manual_seed(42)
    q_start = torch.rand(batch_size, n_q, device=device) * (JOINT_MAX - JOINT_MIN) + JOINT_MIN
    q_goal = torch.rand(batch_size, n_q, device=device) * (JOINT_MAX - JOINT_MIN) + JOINT_MIN
    q0_start = torch.zeros(batch_size, 4, device=device); q0_start[:, 3] = 1.0
    rand_axis = torch.randn(batch_size, 3, device=device)
    rand_axis = rand_axis / torch.norm(rand_axis, dim=1, keepdim=True)
    rand_theta = torch.rand(batch_size, 1, device=device) * math.radians(10.0)
    q0_goal = torch.cat([rand_axis * torch.sin(rand_theta/2), torch.cos(rand_theta/2)], dim=1)

    times = []
    for i in range(warmup + n_iters):
        wp = (torch.rand(batch_size, NUM_WAYPOINTS * n_q, device=device) * 2 - 1).requires_grad_(True)

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss, _ = physics.calculate_total_loss(
            wp, q0_start, q0_goal, return_mean=True,
            q_start_joint=q_start, q_end_joint=q_goal
        )
        loss.backward()

        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            times.append(t1 - t0)

    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    print(f"Batch={batch_size}, {n_iters} iters: {avg*1000:.1f} ± {std*1000:.1f} ms/iter")
    return avg


if __name__ == '__main__':
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    for bs in [32, 128, 256]:
        benchmark(batch_size=bs)
