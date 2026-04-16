"""
Multi-seed batch optimization으로 대규모 trajectory dataset 생성 (GPU).

2-Phase 전략:
  Phase 1: 대규모 random seeds → Adam + LBFGS → seed solutions 확보
  Phase 2: seed solutions를 perturbation → 짧은 LBFGS → 대량 수집

Target orientation: roll=15, pitch=15, yaw=-15
Output: [5000, 100, 6] joint angle trajectories (.npy)
용도: PCA 분석 → 2D embedding + timestep 3D 시각화
"""

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

import torch
import torch.optim as optim
import os
import sys
import time
import numpy as np
import math
from torch.func import vmap

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2); sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2); sy = torch.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return torch.stack([qx, qy, qz, qw], dim=-1)


# =====================================================================
# Phase 1: Random seeds → Adam + LBFGS → seed solutions
# =====================================================================
def find_seed_solutions(physics, num_seeds, output_dim, q0_start, q0_goal,
                        robot, num_waypoints, max_joint_weight, device,
                        adam_steps=300, adam_lr=0.05, init_scale=0.5,
                        lbfgs_outer=3, lbfgs_inner=50, threshold_deg=1.0):
    """
    Random initialization → Adam warm-up → LBFGS polish.
    Returns converged waypoints as seed solutions.
    """
    n_q = robot["n_q"]
    q0s = q0_start.expand(num_seeds, -1)
    q0g = q0_goal.expand(num_seeds, -1)

    wp = (torch.randn(num_seeds, output_dim, device=device) * init_scale)
    wp = wp.requires_grad_(True)

    # ---- Adam warm-up ----
    opt_adam = optim.Adam([wp], lr=adam_lr)
    for step in range(adam_steps):
        opt_adam.zero_grad()
        qt, qdt = physics.generate_trajectory(wp)
        lb, _ = physics._batch_sim_fn(qt, qdt, q0s, q0g)
        w = wp.view(num_seeds, num_waypoints, n_q)
        mj = w.abs().view(num_seeds, -1).max(dim=1)[0]
        loss = (lb + max_joint_weight * mj).mean()
        loss.backward()
        opt_adam.step()
        if step % 100 == 0:
            print(f"    Adam {step:3d}/{adam_steps}  loss={loss.item():.4f}")

    # ---- LBFGS polish ----
    opt_lbfgs = optim.LBFGS([wp], lr=1.0, max_iter=lbfgs_inner,
                             history_size=50, line_search_fn="strong_wolfe")
    for outer in range(lbfgs_outer):
        def closure():
            opt_lbfgs.zero_grad()
            qt, qdt = physics.generate_trajectory(wp)
            lb, _ = physics._batch_sim_fn(qt, qdt, q0s, q0g)
            w = wp.view(num_seeds, num_waypoints, n_q)
            mj = w.abs().view(num_seeds, -1).max(dim=1)[0]
            loss = (lb + max_joint_weight * mj).mean()
            loss.backward()
            return loss
        loss = opt_lbfgs.step(closure)
        print(f"    LBFGS outer {outer+1}/{lbfgs_outer}  loss={loss.item():.4f}")

    # ---- Filter converged ----
    with torch.no_grad():
        qt, qdt = physics.generate_trajectory(wp)
        _, qf = physics._batch_sim_fn(qt, qdt, q0s, q0g)
        dots = torch.sum(qf * q0g, dim=-1).abs().clamp(-1.0, 1.0)
        errs = 2.0 * torch.acos(dots) * 180.0 / math.pi
        mask = errs < threshold_deg
        seeds_wp = wp[mask].detach()
        seeds_traj = qt[mask]

    return seeds_wp, seeds_traj, errs


# =====================================================================
# Phase 2: Perturb seed solutions → short LBFGS → collect
# =====================================================================
def perturb_and_optimize(physics, seed_waypoints, num_per_seed, output_dim,
                         q0_start, q0_goal, robot, num_waypoints,
                         max_joint_weight, device,
                         perturb_scale=0.3, lbfgs_outer=2, lbfgs_inner=50,
                         threshold_deg=1.0):
    """
    Perturb each seed solution, re-optimize with brief LBFGS.
    Returns converged trajectories.
    """
    n_q = robot["n_q"]
    n_seeds = seed_waypoints.shape[0]
    total = n_seeds * num_per_seed

    # Expand seeds and add noise: [n_seeds * num_per_seed, output_dim]
    expanded = seed_waypoints.unsqueeze(1).expand(-1, num_per_seed, -1)
    expanded = expanded.reshape(total, output_dim)
    noise = torch.randn(total, output_dim, device=device) * perturb_scale
    wp = (expanded + noise).detach().requires_grad_(True)

    q0s = q0_start.expand(total, -1)
    q0g = q0_goal.expand(total, -1)

    opt = optim.LBFGS([wp], lr=1.0, max_iter=lbfgs_inner,
                       history_size=50, line_search_fn="strong_wolfe")

    for outer in range(lbfgs_outer):
        def closure():
            opt.zero_grad()
            qt, qdt = physics.generate_trajectory(wp)
            lb, _ = physics._batch_sim_fn(qt, qdt, q0s, q0g)
            w = wp.view(total, num_waypoints, n_q)
            mj = w.abs().view(total, -1).max(dim=1)[0]
            loss = (lb + max_joint_weight * mj).mean()
            loss.backward()
            return loss
        loss = opt.step(closure)

    # Evaluate
    with torch.no_grad():
        qt, qdt = physics.generate_trajectory(wp)
        _, qf = physics._batch_sim_fn(qt, qdt, q0s, q0g)
        dots = torch.sum(qf * q0g, dim=-1).abs().clamp(-1.0, 1.0)
        errs = 2.0 * torch.acos(dots) * 180.0 / math.pi
        mask = errs < threshold_deg
        conv_traj = qt[mask]
        conv_wp = wp[mask].detach()

    return conv_traj, conv_wp, errs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Trajectory Dataset Generation on {device} ===")
    if device == "cpu":
        print("WARNING: No GPU detected. This will be very slow on CPU.")
    print()

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    MAX_JOINT_WEIGHT = 0.01
    TARGET_TOTAL = 5000
    THRESHOLD_DEG = 1.0

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/trajectory_dataset")
    os.makedirs(save_dir, exist_ok=True)

    # Target: roll=15, pitch=15, yaw=-15
    roll_deg, pitch_deg, yaw_deg = 15, 15, -15
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = euler_to_quaternion(
        torch.tensor([math.radians(roll_deg)], device=device),
        torch.tensor([math.radians(pitch_deg)], device=device),
        torch.tensor([math.radians(yaw_deg)], device=device),
    )

    print(f"Target  : roll={roll_deg}, pitch={pitch_deg}, yaw={yaw_deg}")
    print(f"Dataset : {TARGET_TOTAL} trajectories, shape [5000, 100, 6]")
    print(f"Threshold: < {THRESHOLD_DEG} deg\n")

    t_start = time.time()

    # ==================================================================
    # Phase 1: Find seed solutions (1000 random → Adam+LBFGS)
    # ==================================================================
    PHASE1_SEEDS = 1000
    print(f"{'='*60}")
    print(f"Phase 1: Find seed solutions ({PHASE1_SEEDS} random seeds)")
    print(f"{'='*60}")

    t1 = time.time()
    torch.manual_seed(42)
    seed_wp, seed_traj, phase1_errs = find_seed_solutions(
        physics, PHASE1_SEEDS, OUTPUT_DIM, q0_start, q0_goal,
        robot, NUM_WAYPOINTS, MAX_JOINT_WEIGHT, device,
        adam_steps=300, adam_lr=0.05, init_scale=0.5,
        lbfgs_outer=3, lbfgs_inner=50, threshold_deg=THRESHOLD_DEG,
    )
    t1_elapsed = time.time() - t1

    n_seeds_found = seed_wp.shape[0]
    print(f"\n  Phase 1 result: {n_seeds_found}/{PHASE1_SEEDS} converged "
          f"({n_seeds_found/PHASE1_SEEDS*100:.1f}%) in {t1_elapsed:.1f}s")

    if n_seeds_found == 0:
        print("ERROR: No seed solutions found. Try increasing PHASE1_SEEDS or adam_steps.")
        return

    # Collect phase 1 converged trajectories
    all_trajs = [seed_traj.cpu()]
    collected = seed_traj.shape[0]
    print(f"  Collected so far: {collected}/{TARGET_TOTAL}")

    # ==================================================================
    # Phase 2: Perturb + re-optimize in batches until 5000
    # ==================================================================
    PERTURB_BATCH = 1000   # perturbations per batch (GPU-friendly)
    perturb_scale = 0.3
    batch_idx = 0

    while collected < TARGET_TOTAL:
        remaining = TARGET_TOTAL - collected
        # 각 seed에서 균등하게 perturbation 생성
        num_per_seed = max(1, PERTURB_BATCH // n_seeds_found)
        actual_batch = n_seeds_found * num_per_seed

        print(f"\n{'='*60}")
        print(f"Phase 2 Batch {batch_idx}  |  {n_seeds_found} seeds x {num_per_seed} perturbs "
              f"= {actual_batch}  |  collected={collected}/{TARGET_TOTAL}")
        print(f"{'='*60}")

        t2 = time.time()
        torch.manual_seed(batch_idx * 7777 + 13)
        conv_traj, conv_wp, errs = perturb_and_optimize(
            physics, seed_wp, num_per_seed, OUTPUT_DIM,
            q0_start, q0_goal, robot, NUM_WAYPOINTS,
            MAX_JOINT_WEIGHT, device,
            perturb_scale=perturb_scale,
            lbfgs_outer=2, lbfgs_inner=50,
            threshold_deg=THRESHOLD_DEG,
        )
        t2_elapsed = time.time() - t2

        n_conv = conv_traj.shape[0]
        take = min(n_conv, remaining)
        if take > 0:
            all_trajs.append(conv_traj[:take].cpu())
            collected += take

        conv_rate = n_conv / actual_batch * 100
        print(f"  -> converged: {n_conv}/{actual_batch} ({conv_rate:.1f}%)  "
              f"| took {take}  | total: {collected}/{TARGET_TOTAL}  "
              f"| time: {t2_elapsed:.1f}s")

        # Merge newly found solutions into seed pool for more diversity
        if conv_wp.shape[0] > 0:
            seed_wp = torch.cat([seed_wp, conv_wp], dim=0)
            # Deduplicate: keep unique (L2 > 0.01)
            if seed_wp.shape[0] > 500:
                perm = torch.randperm(seed_wp.shape[0], device=device)[:500]
                seed_wp = seed_wp[perm]
            n_seeds_found = seed_wp.shape[0]

        # Increase perturbation scale if convergence rate is high (explore more)
        if conv_rate > 80:
            perturb_scale = min(perturb_scale * 1.2, 1.0)
            print(f"  (increasing perturb_scale to {perturb_scale:.2f} for diversity)")

        batch_idx += 1

    t_total = time.time() - t_start

    # ==================================================================
    # Assemble & Save dataset [5000, 100, 6]
    # ==================================================================
    dataset = torch.cat(all_trajs, dim=0)[:TARGET_TOTAL]
    assert dataset.shape == (TARGET_TOTAL, physics.num_steps, robot["n_q"]), \
        f"Expected ({TARGET_TOTAL}, {physics.num_steps}, {robot['n_q']}), got {dataset.shape}"

    dataset_np = dataset.numpy()

    npy_path = os.path.join(save_dir, "trajectories_5000x100x6.npy")
    np.save(npy_path, dataset_np)

    meta = {
        "shape": list(dataset_np.shape),
        "target_euler_deg": [roll_deg, pitch_deg, yaw_deg],
        "q0_start": q0_start.cpu().numpy().tolist(),
        "q0_goal": q0_goal.cpu().numpy().tolist(),
        "total_time": TOTAL_TIME,
        "dt": physics.dt,
        "num_waypoints": NUM_WAYPOINTS,
        "num_steps": physics.num_steps,
        "n_q": robot["n_q"],
        "convergence_threshold_deg": THRESHOLD_DEG,
        "generation_time_s": t_total,
        "phase1_seeds": PHASE1_SEEDS,
        "phase2_batches": batch_idx,
    }
    meta_path = os.path.join(save_dir, "metadata.npy")
    np.save(meta_path, meta)

    # ---- Stats ----
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Shape       : {dataset_np.shape}  (trajectories x timesteps x joints)")
    print(f"Saved to    : {npy_path}")
    print(f"Target      : roll={roll_deg}, pitch={pitch_deg}, yaw={yaw_deg}")
    print(f"Total time  : {t_total:.1f}s ({t_total/60:.1f}min)")
    print()

    print(f"Trajectory stats (rad):")
    print(f"  mean abs  : {np.abs(dataset_np).mean():.4f}")
    print(f"  max abs   : {np.abs(dataset_np).max():.4f}")
    print(f"  std       : {dataset_np.std():.4f}")

    print(f"\nPer-joint range (rad):")
    for j in range(robot["n_q"]):
        jdata = dataset_np[:, :, j]
        print(f"  J{j+1}: [{jdata.min():+.3f}, {jdata.max():+.3f}]  std={jdata.std():.4f}")

    # Diversity check: pairwise distances at midpoint
    mid = dataset_np[:, 50, :]  # [5000, 6] at t=5s
    sample_idx = np.random.choice(TARGET_TOTAL, size=min(500, TARGET_TOTAL), replace=False)
    mid_sample = mid[sample_idx]
    dists = np.linalg.norm(mid_sample[:, None, :] - mid_sample[None, :, :], axis=-1)
    triu_idx = np.triu_indices(len(sample_idx), k=1)
    triu_dists = dists[triu_idx]
    print(f"\nDiversity (L2 at t=5s, 500 sample pairs):")
    print(f"  mean={triu_dists.mean():.4f}  min={triu_dists.min():.4f}  "
          f"max={triu_dists.max():.4f}  std={triu_dists.std():.4f}")


if __name__ == "__main__":
    main()
