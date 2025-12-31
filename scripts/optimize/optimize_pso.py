"""
PSO(Particle Swarm Optimization) 기반 trajectory waypoint(=knots joint 파라미터) 최적화.

- 최적화 변수: waypoints_flat, shape [NUM_WAYPOINTS * n_q]
- Trajectory 생성: PhysicsLayer.generate_trajectory (batched)
- Loss 평가: torch.func.vmap(PhysicsLayer.simulate_single) (batched dynamics)

사용 예:
  python3 scripts/optimize/optimize_pso.py --device cuda --particles 512 --iters 200 --eval-batch-size 512
  python3 scripts/optimize/optimize_pso.py --device cuda --particles 1024 --iters 300 --w 0.6 --c1 1.4 --c2 1.4
  python3 scripts/optimize/optimize_pso.py --device cpu --particles 128 --iters 100

주의:
- 이 스크립트는 조인트 limit을 URDF의 <limit lower/upper>에서 읽어 bounds로 사용합니다.
  limit이 없는 joint(continuous 등)는 기본 [-pi, pi]를 사용합니다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.dynamics.urdf2robot_torch import urdf2robot
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


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_urdf_limits(urdf_path: str) -> Dict[str, Tuple[Optional[float], Optional[float], str]]:
    """
    URDF에서 joint limit(lower/upper)을 파싱.
    반환: joint_name -> (lower, upper, joint_type)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: Dict[str, Tuple[Optional[float], Optional[float], str]] = {}
    for j in root.findall("joint"):
        name = j.attrib.get("name", "")
        jtype = j.attrib.get("type", "fixed")
        lower = None
        upper = None
        limit = j.find("limit")
        if limit is not None:
            if "lower" in limit.attrib:
                try:
                    lower = float(limit.attrib["lower"])
                except ValueError:
                    lower = None
            if "upper" in limit.attrib:
                try:
                    upper = float(limit.attrib["upper"])
                except ValueError:
                    upper = None
        out[name] = (lower, upper, jtype)
    return out


def build_bounds_from_urdf(
    *,
    urdf_path: str,
    robot: dict,
    robot_keys: dict,
    num_waypoints: int,
    device: str,
    default_lower: float,
    default_upper: float,
    continuous_lower: float,
    continuous_upper: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q_id mapping(robot_keys['q_id'])을 이용해 (n_q,) bounds를 만들고,
    waypoint dimension(D = num_waypoints*n_q)으로 확장하여 반환.
    """
    n_q = int(robot["n_q"])
    lower_q = torch.full((n_q,), float(default_lower), device=device, dtype=torch.float32)
    upper_q = torch.full((n_q,), float(default_upper), device=device, dtype=torch.float32)

    limits = parse_urdf_limits(urdf_path)
    q_id_map = robot_keys.get("q_id", {})
    for jname, (lo, hi, jtype) in limits.items():
        q_id = q_id_map.get(jname, None)
        if q_id is None:
            continue
        idx = int(q_id) - 1  # SPART uses q_id-1 indexing
        if idx < 0 or idx >= n_q:
            continue

        if jtype == "continuous":
            lower_q[idx] = float(continuous_lower)
            upper_q[idx] = float(continuous_upper)
            continue

        # revolute/prismatic: limit이 있으면 사용, 없으면 default 유지
        if lo is not None:
            lower_q[idx] = float(lo)
        if hi is not None:
            upper_q[idx] = float(hi)

    # Expand to waypoints
    lower = lower_q.repeat(num_waypoints)
    upper = upper_q.repeat(num_waypoints)

    # 안전장치: lower>upper인 경우 스왑
    lo2 = torch.minimum(lower, upper)
    hi2 = torch.maximum(lower, upper)
    return lo2, hi2


@dataclass
class PSOState:
    x: torch.Tensor  # [P, D]
    v: torch.Tensor  # [P, D]
    pbest_x: torch.Tensor  # [P, D]
    pbest_f: torch.Tensor  # [P]
    gbest_x: torch.Tensor  # [D]
    gbest_f: float


@torch.no_grad()
def evaluate_particles(
    *,
    physics: PhysicsLayer,
    batch_sim_fn,
    x: torch.Tensor,  # [P, D]
    q0_start: torch.Tensor,  # [1, 4]
    q0_goal: torch.Tensor,  # [1, 4]
    eval_batch_size: int,
) -> torch.Tensor:
    """
    P개의 particle을 batch로 나눠 batched dynamics로 loss를 계산.
    반환: loss [P] (각 particle별)
    """
    P = x.shape[0]
    losses_all = []
    for s in range(0, P, eval_batch_size):
        xb = x[s : s + eval_batch_size]
        B = xb.shape[0]
        q_traj, q_dot_traj = physics.generate_trajectory(xb)  # [B, T, n_q]
        sim_out = batch_sim_fn(
            q_traj,
            q_dot_traj,
            q0_start.repeat(B, 1),
            q0_goal.repeat(B, 1),
        )
        loss_b = sim_out[0] if isinstance(sim_out, (tuple, list)) else sim_out
        loss_b = torch.where(torch.isfinite(loss_b), loss_b, torch.full_like(loss_b, float("inf")))
        losses_all.append(loss_b)
    return torch.cat(losses_all, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="PSO optimization for trajectory waypoints using batched dynamics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--urdf", type=str, default=os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"))
    parser.add_argument("--save-dir", type=str, default=os.path.join(ROOT_DIR, "outputs/results/opt_pso"))

    # problem config (same style as optimize_cvae/mlp)
    parser.add_argument("--cond-dim", type=int, default=8)
    parser.add_argument("--num-waypoints", type=int, default=3)
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--roll-deg", type=float, default=15.0)
    parser.add_argument("--pitch-deg", type=float, default=15.0)
    parser.add_argument("--yaw-deg", type=float, default=-15.0)

    # PSO hyperparams
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.6, help="inertia")
    parser.add_argument("--c1", type=float, default=1.5, help="cognitive")
    parser.add_argument("--c2", type=float, default=1.5, help="social")
    parser.add_argument("--tol", type=float, default=1e-6, help="early stop threshold for best loss")
    parser.add_argument("--vmax-frac", type=float, default=0.2, help="velocity clamp as fraction of (upper-lower)")
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)

    # bounds
    parser.add_argument("--default-lower", type=float, default=-math.pi)
    parser.add_argument("--default-upper", type=float, default=math.pi)
    parser.add_argument("--continuous-lower", type=float, default=-math.pi)
    parser.add_argument("--continuous-upper", type=float, default=math.pi)

    # logging
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"=== PSO Optimization on {device} ===")
    print(f"particles={args.particles}, iters={args.iters}, eval_batch_size={args.eval_batch_size}")

    # robot + physics
    robot, robot_keys = urdf2robot(args.urdf, verbose_flag=False, device=device)
    n_q = int(robot["n_q"])
    D = args.num_waypoints * n_q
    physics = PhysicsLayer(robot, args.num_waypoints, args.total_time, device)
    dt = float(physics.dt)

    # fixed start/goal (quaternion)
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = euler_to_quaternion(
        torch.tensor([math.radians(args.roll_deg)], device=device),
        torch.tensor([math.radians(args.pitch_deg)], device=device),
        torch.tensor([math.radians(args.yaw_deg)], device=device),
    ).to(dtype=torch.float32)

    # bounds from URDF
    lower, upper = build_bounds_from_urdf(
        urdf_path=args.urdf,
        robot=robot,
        robot_keys=robot_keys,
        num_waypoints=args.num_waypoints,
        device=device,
        default_lower=args.default_lower,
        default_upper=args.default_upper,
        continuous_lower=args.continuous_lower,
        continuous_upper=args.continuous_upper,
    )

    # init swarm
    P = int(args.particles)
    _sync_if_cuda(device)
    x = lower + (upper - lower) * torch.rand((P, D), device=device)
    v = torch.zeros_like(x)

    # vmap batch simulator (batched dynamics)
    batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))

    # warm-up (첫 호출 컴파일/캐시 오버헤드 감소)
    with torch.no_grad():
        dummy = torch.zeros((1, D), device=device)
        q_traj_d, q_dot_d = physics.generate_trajectory(dummy)
        _ = batch_sim_fn(q_traj_d, q_dot_d, q0_start, q0_goal)
    _sync_if_cuda(device)

    # evaluate initial
    t0 = time.time()
    f = evaluate_particles(
        physics=physics,
        batch_sim_fn=batch_sim_fn,
        x=x,
        q0_start=q0_start,
        q0_goal=q0_goal,
        eval_batch_size=args.eval_batch_size,
    )
    _sync_if_cuda(device)
    t1 = time.time()
    print(f"[init] eval_time={t1 - t0:.3f}s  best={float(f.min().item()):.6f}")

    pbest_x = x.clone()
    pbest_f = f.clone()
    gbest_idx = int(torch.argmin(pbest_f).item())
    gbest_x = pbest_x[gbest_idx].clone()
    gbest_f = float(pbest_f[gbest_idx].item())

    state = PSOState(
        x=x,
        v=v,
        pbest_x=pbest_x,
        pbest_f=pbest_f,
        gbest_x=gbest_x,
        gbest_f=gbest_f,
    )

    vmax = (upper - lower) * float(args.vmax_frac)
    hist_best = []
    hist_mean = []
    hist_eval_s = []
    hist_iter_s = []

    # PSO loop
    opt_start = time.time()
    for it in range(1, int(args.iters) + 1):
        iter_start = time.perf_counter()
        # update velocity / position
        r1 = torch.rand_like(state.x)
        r2 = torch.rand_like(state.x)
        state.v = (
            float(args.w) * state.v
            + float(args.c1) * r1 * (state.pbest_x - state.x)
            + float(args.c2) * r2 * (state.gbest_x.unsqueeze(0) - state.x)
        )
        state.v = torch.clamp(state.v, -vmax, vmax)
        state.x = state.x + state.v
        state.x = torch.clamp(state.x, lower, upper)

        # evaluate
        eval_start = time.perf_counter()
        f = evaluate_particles(
            physics=physics,
            batch_sim_fn=batch_sim_fn,
            x=state.x,
            q0_start=q0_start,
            q0_goal=q0_goal,
            eval_batch_size=args.eval_batch_size,
        )
        eval_end = time.perf_counter()
        hist_eval_s.append(eval_end - eval_start)

        # personal best update
        improved = f < state.pbest_f
        if improved.any():
            state.pbest_f = torch.where(improved, f, state.pbest_f)
            state.pbest_x = torch.where(improved.unsqueeze(1), state.x, state.pbest_x)

        # global best update
        cur_best_idx = int(torch.argmin(state.pbest_f).item())
        cur_best_f = float(state.pbest_f[cur_best_idx].item())
        if cur_best_f < state.gbest_f:
            state.gbest_f = cur_best_f
            state.gbest_x = state.pbest_x[cur_best_idx].clone()

        hist_best.append(state.gbest_f)
        hist_mean.append(float(torch.mean(f).item()))
        hist_iter_s.append(time.perf_counter() - iter_start)

        if it == 1 or (args.print_every > 0 and it % int(args.print_every) == 0):
            print(
                f"[iter {it:04d}] best={state.gbest_f:.6e}  mean={hist_mean[-1]:.6e}  "
                f"iter={hist_iter_s[-1]*1000.0:.1f}ms  eval={hist_eval_s[-1]*1000.0:.1f}ms"
            )

        # early stop
        if state.gbest_f <= float(args.tol):
            print(f"[early-stop] reached tol={float(args.tol):.1e} at iter={it}, best={state.gbest_f:.6e}")
            break

    opt_end = time.time()
    best_deg = float(np.rad2deg(np.sqrt(state.gbest_f))) if state.gbest_f > 0 else 0.0
    print(f"\n[done] best_loss={state.gbest_f:.10f}  (~{best_deg:.4f} deg)")
    total_s = opt_end - opt_start
    mean_iter_ms = (1000.0 * float(np.mean(hist_iter_s))) if hist_iter_s else 0.0
    mean_eval_ms = (1000.0 * float(np.mean(hist_eval_s))) if hist_eval_s else 0.0
    print(
        f"[done] total_time={total_s:.3f}s  mean_iter={mean_iter_ms:.1f}ms  mean_eval={mean_eval_ms:.1f}ms  "
        f"dt={dt}  n_q={n_q}  D={D}"
    )

    # save artifacts
    # 1) history
    hist_path = os.path.join(args.save_dir, "pso_history.csv")
    iters_done = len(hist_best)
    hist_mat = np.column_stack(
        [
            np.arange(1, iters_done + 1),
            np.array(hist_best),
            np.array(hist_mean),
            np.array(hist_iter_s) if hist_iter_s else np.zeros((iters_done,)),
            np.array(hist_eval_s) if hist_eval_s else np.zeros((iters_done,)),
        ]
    )
    np.savetxt(
        hist_path,
        hist_mat,
        delimiter=",",
        header="iter,best_loss,mean_loss,iter_time_s,eval_time_s",
        comments="",
    )

    # 2) best waypoints
    wp_path = os.path.join(args.save_dir, "waypoints_pso.csv")
    np.savetxt(wp_path, state.gbest_x.detach().cpu().numpy().reshape(1, -1), delimiter=",", comments="")

    # 2.5) bounds used
    np.savetxt(
        os.path.join(args.save_dir, "bounds_lower.csv"),
        lower.detach().cpu().numpy().reshape(1, -1),
        delimiter=",",
        comments="",
    )
    np.savetxt(
        os.path.join(args.save_dir, "bounds_upper.csv"),
        upper.detach().cpu().numpy().reshape(1, -1),
        delimiter=",",
        comments="",
    )

    # 2.6) start/goal quaternion
    np.savetxt(
        os.path.join(args.save_dir, "q0_start.csv"),
        q0_start.detach().cpu().numpy(),
        delimiter=",",
        header="qx,qy,qz,qw",
        comments="",
    )
    np.savetxt(
        os.path.join(args.save_dir, "q0_goal.csv"),
        q0_goal.detach().cpu().numpy(),
        delimiter=",",
        header="qx,qy,qz,qw",
        comments="",
    )

    # 2.7) meta
    with open(os.path.join(args.save_dir, "meta.csv"), "w") as f:
        f.write("dt,total_time,num_waypoints,n_q,D,particles,iters_done,iters_max,tol,w,c1,c2,vmax_frac,eval_batch_size,seed\n")
        f.write(
            f"{dt},{float(args.total_time)},{int(args.num_waypoints)},{n_q},{D},{P},{iters_done},{int(args.iters)},"
            f"{float(args.tol)},{float(args.w)},{float(args.c1)},{float(args.c2)},{float(args.vmax_frac)},"
            f"{int(args.eval_batch_size)},{int(args.seed)}\n"
        )

    # 3) optional: final trajectory (CPU-friendly CSV)
    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(state.gbest_x.unsqueeze(0))
        q_traj_np = q_traj[0].detach().cpu().numpy()
        q_dot_np = q_dot_traj[0].detach().cpu().numpy()
        t = np.linspace(0.0, float(args.total_time), q_traj_np.shape[0])
        np.savetxt(
            os.path.join(args.save_dir, "q_traj_pso.csv"),
            np.column_stack([t, q_traj_np]),
            delimiter=",",
            header="t," + ",".join([f"J{i+1}" for i in range(n_q)]),
            comments="",
        )
        np.savetxt(
            os.path.join(args.save_dir, "q_dot_traj_pso.csv"),
            np.column_stack([t, q_dot_np]),
            delimiter=",",
            header="t," + ",".join([f"dJ{i+1}" for i in range(n_q)]),
            comments="",
        )

    print(f"Saved results to: {args.save_dir}")


if __name__ == "__main__":
    main()


