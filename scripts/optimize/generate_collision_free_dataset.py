"""
Collision-free trajectory dataset generation.

Strategy:
  - Differentiable joint limit penalty (from collision analysis: safe joint ranges)
  - 2-Phase optimization: Adam warm-up + LBFGS polish
  - Post-hoc MuJoCo collision filter for validation

Target: roll=15, pitch=15, yaw=-15
Output: [5000, 100, 6] collision-free joint trajectories
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
from src.utils.collision import CollisionChecker


def euler_to_quaternion(roll, pitch, yaw):
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2); sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2); sy = torch.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return torch.stack([qx, qy, qz, qw], dim=-1)


# Safe joint ranges from collision analysis
# (slightly expanded from observed collision-free ranges for more diversity)
JOINT_LOWER = torch.tensor([-2.0, -2.0, -2.5, -2.5, -2.0, -3.0])
JOINT_UPPER = torch.tensor([3.5,   2.0,  3.0,  2.5,  2.2,  3.0])


def joint_limit_penalty(q_traj, joint_lower, joint_upper):
    """
    Differentiable soft joint limit penalty.
    Quadratic penalty for exceeding safe ranges.

    Args:
        q_traj: [batch, T, n_q] joint angle trajectories
        joint_lower: [n_q] lower limits
        joint_upper: [n_q] upper limits
    Returns:
        penalty: [batch] per-sample penalty
    """
    lower_viol = torch.clamp(joint_lower - q_traj, min=0) ** 2  # [B, T, n_q]
    upper_viol = torch.clamp(q_traj - joint_upper, min=0) ** 2  # [B, T, n_q]
    return (lower_viol + upper_viol).sum(dim=(1, 2))  # [batch]


def find_seed_solutions(physics, num_seeds, output_dim, q0_start, q0_goal,
                        robot, num_waypoints, max_joint_weight, device,
                        joint_limit_weight=1.0,
                        adam_steps=300, adam_lr=0.05, init_scale=0.3,
                        lbfgs_outer=3, lbfgs_inner=50, threshold_deg=1.0):
    """Phase 1: Random init -> Adam + LBFGS with collision penalty."""
    n_q = robot["n_q"]
    q0s = q0_start.expand(num_seeds, -1)
    q0g = q0_goal.expand(num_seeds, -1)

    jl = JOINT_LOWER.to(device)
    ju = JOINT_UPPER.to(device)

    wp = (torch.randn(num_seeds, output_dim, device=device) * init_scale)
    # Clamp initial waypoints to safe range for better starting point
    wp_clamped = wp.view(num_seeds, num_waypoints, n_q)
    wp_clamped = torch.clamp(wp_clamped, jl, ju)
    wp = wp_clamped.view(num_seeds, output_dim).detach().requires_grad_(True)

    def compute_loss(waypoints):
        qt, qdt = physics.generate_trajectory(waypoints)
        lb, _ = physics._batch_sim_fn(qt, qdt, q0s, q0g)  # [N]

        w = waypoints.view(num_seeds, num_waypoints, n_q)
        mj = w.abs().view(num_seeds, -1).max(dim=1)[0]  # [N]

        jl_pen = joint_limit_penalty(qt, jl, ju)  # [N]

        total = lb + max_joint_weight * mj + joint_limit_weight * jl_pen
        return total, qt

    # Adam warm-up
    opt_adam = optim.Adam([wp], lr=adam_lr)
    for step in range(adam_steps):
        opt_adam.zero_grad()
        total, _ = compute_loss(wp)
        loss = total.mean()
        loss.backward()
        opt_adam.step()
        if step % 100 == 0:
            print(f"    Adam {step:3d}/{adam_steps}  loss={loss.item():.4f}")

    # LBFGS polish
    opt_lbfgs = optim.LBFGS([wp], lr=1.0, max_iter=lbfgs_inner,
                             history_size=50, line_search_fn="strong_wolfe")
    for outer in range(lbfgs_outer):
        def closure():
            opt_lbfgs.zero_grad()
            total, _ = compute_loss(wp)
            loss = total.mean()
            loss.backward()
            return loss
        loss = opt_lbfgs.step(closure)
        print(f"    LBFGS outer {outer+1}/{lbfgs_outer}  loss={loss.item():.4f}")

    # Filter converged
    with torch.no_grad():
        total, qt = compute_loss(wp)
        _, qf = physics._batch_sim_fn(
            *physics.generate_trajectory(wp), q0s, q0g
        )
        dots = torch.sum(qf * q0g, dim=-1).abs().clamp(-1.0, 1.0)
        errs = 2.0 * torch.acos(dots) * 180.0 / math.pi
        mask = errs < threshold_deg
        seeds_wp = wp[mask].detach()
        seeds_traj = qt[mask]

    return seeds_wp, seeds_traj, errs


def perturb_and_optimize(physics, seed_waypoints, num_per_seed, output_dim,
                         q0_start, q0_goal, robot, num_waypoints,
                         max_joint_weight, device,
                         joint_limit_weight=1.0,
                         perturb_scale=0.2, lbfgs_outer=2, lbfgs_inner=50,
                         threshold_deg=1.0):
    """Phase 2: Perturb seeds + short LBFGS with collision penalty."""
    n_q = robot["n_q"]
    n_seeds = seed_waypoints.shape[0]
    total_n = n_seeds * num_per_seed

    jl = JOINT_LOWER.to(device)
    ju = JOINT_UPPER.to(device)

    expanded = seed_waypoints.unsqueeze(1).expand(-1, num_per_seed, -1)
    expanded = expanded.reshape(total_n, output_dim)
    noise = torch.randn(total_n, output_dim, device=device) * perturb_scale

    # Clamp perturbed waypoints to safe range
    wp_init = (expanded + noise).view(total_n, num_waypoints, n_q)
    wp_init = torch.clamp(wp_init, jl, ju)
    wp = wp_init.view(total_n, output_dim).detach().requires_grad_(True)

    q0s = q0_start.expand(total_n, -1)
    q0g = q0_goal.expand(total_n, -1)

    opt = optim.LBFGS([wp], lr=1.0, max_iter=lbfgs_inner,
                       history_size=50, line_search_fn="strong_wolfe")

    for outer in range(lbfgs_outer):
        def closure():
            opt.zero_grad()
            qt, qdt = physics.generate_trajectory(wp)
            lb, _ = physics._batch_sim_fn(qt, qdt, q0s, q0g)
            w = wp.view(total_n, num_waypoints, n_q)
            mj = w.abs().view(total_n, -1).max(dim=1)[0]
            jl_pen = joint_limit_penalty(qt, jl, ju)
            loss = (lb + max_joint_weight * mj + joint_limit_weight * jl_pen).mean()
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


def mujoco_filter(data_np, checker):
    """Post-hoc MuJoCo collision filter. Returns collision-free mask."""
    N, T, _ = data_np.shape
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        for t in range(T):
            if checker.check(data_np[i, t]):
                mask[i] = False
                break  # early exit: one collision = reject
    return mask


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Collision-Free Trajectory Dataset Generation on {device} ===")
    if device == "cpu":
        print("WARNING: No GPU detected. This will be slow.")
    print()

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    MAX_JOINT_WEIGHT = 0.01
    JOINT_LIMIT_WEIGHT = 1.0
    TARGET_TOTAL = 5000
    THRESHOLD_DEG = 1.0

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    checker = CollisionChecker(os.path.join(ROOT_DIR, "assets/spacerobot_collision.xml"))

    save_dir = os.path.join(ROOT_DIR, "outputs/results/collision_free_dataset")
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
    print(f"Dataset : {TARGET_TOTAL} collision-free trajectories")
    print(f"Joint limits: {JOINT_LOWER.tolist()} ~ {JOINT_UPPER.tolist()}")
    print(f"Joint limit weight: {JOINT_LIMIT_WEIGHT}\n")

    t_start = time.time()

    # ==================================================================
    # Phase 1: Find seed solutions (1000 random seeds)
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
        joint_limit_weight=JOINT_LIMIT_WEIGHT,
        adam_steps=300, adam_lr=0.05, init_scale=0.3,
        lbfgs_outer=3, lbfgs_inner=50, threshold_deg=THRESHOLD_DEG,
    )
    t1_elapsed = time.time() - t1

    n_seeds_found = seed_wp.shape[0]
    print(f"\n  Phase 1: {n_seeds_found}/{PHASE1_SEEDS} converged "
          f"({n_seeds_found/PHASE1_SEEDS*100:.1f}%) in {t1_elapsed:.1f}s")

    # MuJoCo filter on phase 1
    if n_seeds_found > 0:
        seed_traj_np = seed_traj.cpu().numpy()
        col_mask = mujoco_filter(seed_traj_np, checker)
        n_col_free = col_mask.sum()
        print(f"  MuJoCo filter: {n_col_free}/{n_seeds_found} collision-free")
        seed_wp = seed_wp[torch.from_numpy(col_mask).to(device)]
        seed_traj = seed_traj[torch.from_numpy(col_mask).to(device)]
        n_seeds_found = seed_wp.shape[0]

    if n_seeds_found == 0:
        print("ERROR: No collision-free seed solutions. Try increasing PHASE1_SEEDS.")
        return

    all_trajs = [seed_traj.cpu()]
    collected = seed_traj.shape[0]
    print(f"  Collected: {collected}/{TARGET_TOTAL}")

    # ==================================================================
    # Phase 2: Perturb + re-optimize in batches
    # ==================================================================
    PERTURB_BATCH = 1000
    perturb_scale = 0.2
    batch_idx = 0

    while collected < TARGET_TOTAL:
        remaining = TARGET_TOTAL - collected
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
            joint_limit_weight=JOINT_LIMIT_WEIGHT,
            perturb_scale=perturb_scale,
            lbfgs_outer=2, lbfgs_inner=50,
            threshold_deg=THRESHOLD_DEG,
        )
        t2_elapsed = time.time() - t2

        n_conv = conv_traj.shape[0]
        print(f"  Converged: {n_conv}/{actual_batch}")

        # MuJoCo filter
        if n_conv > 0:
            conv_traj_np = conv_traj.cpu().numpy()
            col_mask = mujoco_filter(conv_traj_np, checker)
            n_col_free = col_mask.sum()
            conv_traj = conv_traj[torch.from_numpy(col_mask).to(device)]
            conv_wp = conv_wp[torch.from_numpy(col_mask).to(device)]
            n_conv = conv_traj.shape[0]
            print(f"  MuJoCo filter: {n_col_free} collision-free")

        take = min(n_conv, remaining)
        if take > 0:
            all_trajs.append(conv_traj[:take].cpu())
            collected += take

        print(f"  -> took {take}  | total: {collected}/{TARGET_TOTAL}  | time: {t2_elapsed:.1f}s")

        # Grow seed pool
        if conv_wp.shape[0] > 0:
            seed_wp = torch.cat([seed_wp, conv_wp], dim=0)
            if seed_wp.shape[0] > 500:
                perm = torch.randperm(seed_wp.shape[0], device=device)[:500]
                seed_wp = seed_wp[perm]
            n_seeds_found = seed_wp.shape[0]

        if perturb_scale < 0.5 and n_conv > 0 and n_col_free / max(n_conv, 1) > 0.8:
            perturb_scale = min(perturb_scale * 1.1, 0.5)

        batch_idx += 1

    t_total = time.time() - t_start

    # ==================================================================
    # Assemble & Save
    # ==================================================================
    dataset = torch.cat(all_trajs, dim=0)[:TARGET_TOTAL]
    dataset_np = dataset.numpy()

    assert dataset_np.shape == (TARGET_TOTAL, physics.num_steps, robot["n_q"]), \
        f"Expected ({TARGET_TOTAL}, {physics.num_steps}, {robot['n_q']}), got {dataset_np.shape}"

    npy_path = os.path.join(save_dir, "trajectories_collision_free_5000x100x6.npy")
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
        "joint_limit_weight": JOINT_LIMIT_WEIGHT,
        "joint_lower": JOINT_LOWER.tolist(),
        "joint_upper": JOINT_UPPER.tolist(),
        "generation_time_s": t_total,
        "collision_checked": True,
    }
    np.save(os.path.join(save_dir, "metadata.npy"), meta)

    # ---- Final verification: re-check all with MuJoCo ----
    print(f"\n--- Final MuJoCo Verification ---")
    final_mask = mujoco_filter(dataset_np, checker)
    n_verified = final_mask.sum()
    print(f"Verified collision-free: {n_verified}/{TARGET_TOTAL} ({n_verified/TARGET_TOTAL*100:.1f}%)")

    # ---- Stats ----
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Shape       : {dataset_np.shape}")
    print(f"Saved to    : {npy_path}")
    print(f"Total time  : {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Collision-free verified: {n_verified}/{TARGET_TOTAL}")

    print(f"\nPer-joint range (rad):")
    for j in range(robot["n_q"]):
        jd = dataset_np[:, :, j]
        print(f"  J{j+1}: [{jd.min():+.3f}, {jd.max():+.3f}]  std={jd.std():.4f}")

    # Diversity
    mid = dataset_np[:, 50, :]
    idx = np.random.default_rng(42).choice(TARGET_TOTAL, size=min(500, TARGET_TOTAL), replace=False)
    sample = mid[idx]
    dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    triu = dists[np.triu_indices(len(idx), k=1)]
    print(f"\nDiversity (L2 at t=5s, 500 pairs):")
    print(f"  mean={triu.mean():.4f}  std={triu.std():.4f}")


if __name__ == "__main__":
    main()
