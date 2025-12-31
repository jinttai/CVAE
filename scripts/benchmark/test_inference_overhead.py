"""
CVAE inference 단계의 시간 오버헤드(특히 CUDA lazy init / vmap 첫 호출 / 캐시) 측정용 벤치 스크립트.

측정 항목(각 반복):
- decode 시간 (CVAE.decode)
- trajectory 생성 시간 (PhysicsLayer.generate_trajectory)
- simulate(vmap) 시간 (vmap(PhysicsLayer.simulate_single))
- total 시간

사용 예:
  python3 scripts/benchmark/test_inference_overhead.py --device cuda --num-samples 1000 --repeats 10 --warmup 1
  python3 scripts/benchmark/test_inference_overhead.py --device cpu --num-samples 100 --repeats 5
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

import torch

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.dynamics.urdf2robot_torch import urdf2robot
from src.models.cvae import CVAE
from src.training.physics_layer import PhysicsLayer


def euler_to_quaternion(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Euler (roll, pitch, yaw) -> quaternion (x, y, z, w), ZYX (yaw-Z, pitch-Y, roll-X)
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


def load_cvae(weights_path: str, cond_dim: int, output_dim: int, latent_dim: int, device: str) -> CVAE:
    model = CVAE(cond_dim, output_dim, latent_dim).to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[Warning] strict=True 로드 실패, strict=False로 재시도합니다.\n  {e}")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(x: float) -> float:
    return x * 1000.0


@torch.no_grad()
def run_once(
    *,
    model: CVAE,
    physics: PhysicsLayer,
    batch_sim_fn,
    condition: torch.Tensor,
    q0_start: torch.Tensor,
    q0_goal: torch.Tensor,
    num_samples: int,
    latent_dim: int,
    device: str,
) -> Dict[str, float]:
    """
    단일 측정 반복: decode -> generate_trajectory -> simulate(vmap)
    반환: 초 단위 딕셔너리
    """
    z = torch.randn(num_samples, latent_dim, device=device, dtype=torch.float32)
    cond_batch = condition.repeat(num_samples, 1)

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    candidates = model.decode(cond_batch, z)
    _sync_if_cuda(device)
    t1 = time.perf_counter()

    q_traj, q_dot_traj = physics.generate_trajectory(candidates)
    _sync_if_cuda(device)
    t2 = time.perf_counter()

    sim_out = batch_sim_fn(
        q_traj,
        q_dot_traj,
        q0_start.repeat(num_samples, 1),
        q0_goal.repeat(num_samples, 1),
    )
    # simulate_single 은 (loss, q_final) 을 반환하므로 loss만 분리
    losses = sim_out[0] if isinstance(sim_out, (tuple, list)) else sim_out
    losses = torch.where(torch.isfinite(losses), losses, torch.full_like(losses, float("inf")))
    _ = torch.argmin(losses)
    _sync_if_cuda(device)
    t3 = time.perf_counter()

    return {
        "decode": t1 - t0,
        "traj": t2 - t1,
        "simulate": t3 - t2,
        "total": t3 - t0,
    }


def summarize(times: List[float]) -> Tuple[float, float, float]:
    if not times:
        return (float("nan"), float("nan"), float("nan"))
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) >= 2 else 0.0
    p50 = statistics.median(times)
    return mean, stdev, p50


def main() -> None:
    parser = argparse.ArgumentParser(description="CVAE inference overhead benchmark (decode + vmap simulate)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", type=str, default=os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v3.pth"))
    parser.add_argument("--urdf", type=str, default=os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"))
    parser.add_argument("--num-waypoints", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--cond-dim", type=int, default=8)
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1, help="warmup 반복 횟수(측정에서 제외)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roll-deg", type=float, default=15.0)
    parser.add_argument("--pitch-deg", type=float, default=15.0)
    parser.add_argument("--yaw-deg", type=float, default=-15.0)
    parser.add_argument(
        "--show-per-iter",
        action="store_true",
        help="각 반복의 decode/traj/simulate/total 시간을 ms로 출력",
    )
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"=== CVAE Inference Overhead Benchmark on {device} ===")
    print(f"num_samples={args.num_samples}, repeats={args.repeats}, warmup={args.warmup}")

    robot, _ = urdf2robot(args.urdf, verbose_flag=False, device=device)
    output_dim = args.num_waypoints * robot["n_q"]
    physics = PhysicsLayer(robot, args.num_waypoints, args.total_time, device)

    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = euler_to_quaternion(
        torch.tensor([math.radians(args.roll_deg)], device=device),
        torch.tensor([math.radians(args.pitch_deg)], device=device),
        torch.tensor([math.radians(args.yaw_deg)], device=device),
    ).to(dtype=torch.float32)
    condition = torch.cat([q0_start, q0_goal], dim=1)

    model = load_cvae(args.weights, args.cond_dim, output_dim, args.latent_dim, device)

    # vmap 함수 생성
    batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))

    # Warm-up (CUDA lazy init + 캐시 + 첫 호출 비용 제거)
    for _ in range(max(0, args.warmup)):
        _ = run_once(
            model=model,
            physics=physics,
            batch_sim_fn=batch_sim_fn,
            condition=condition,
            q0_start=q0_start,
            q0_goal=q0_goal,
            num_samples=1,
            latent_dim=args.latent_dim,
            device=device,
        )

    # 측정
    results: List[Dict[str, float]] = []
    for i in range(args.repeats):
        out = run_once(
            model=model,
            physics=physics,
            batch_sim_fn=batch_sim_fn,
            condition=condition,
            q0_start=q0_start,
            q0_goal=q0_goal,
            num_samples=args.num_samples,
            latent_dim=args.latent_dim,
            device=device,
        )
        results.append(out)
        if args.show_per_iter:
            print(
                f"[iter {i:02d}] "
                f"decode={_ms(out['decode']):8.3f} ms | "
                f"traj={_ms(out['traj']):8.3f} ms | "
                f"simulate={_ms(out['simulate']):8.3f} ms | "
                f"total={_ms(out['total']):8.3f} ms"
            )

    # 요약
    keys = ["decode", "traj", "simulate", "total"]
    print("\n--- Summary (ms) ---")
    for k in keys:
        arr = [r[k] for r in results]
        mean, stdev, p50 = summarize(arr)
        print(f"{k:8s}: mean={_ms(mean):.3f}  std={_ms(stdev):.3f}  p50={_ms(p50):.3f}")


if __name__ == "__main__":
    main()


