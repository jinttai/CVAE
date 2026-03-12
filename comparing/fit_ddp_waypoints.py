"""
Fit waypoints to match the DDP trajectory using least squares,
then evaluate using CVAE's physics loss.

1. Load DDP trajectory from results_ddp/
2. Optimize 3 waypoints so that quintic spline ≈ DDP joint trajectory
3. Evaluate fitted waypoints with PhysicsLayer (same cost as CVAE)
"""
import os
import sys
import math
import numpy as np
import torch

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from scenario import SCENARIO, get_goal_quaternion


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load robot
    urdf_path = os.path.join(ROOT_DIR, SCENARIO["urdf"])
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    n_q = robot["n_q"]

    # Scenario
    total_time = SCENARIO["total_time"]
    NUM_WAYPOINTS = SCENARIO["num_waypoints"]
    q_goal_np = get_goal_quaternion()

    q_start_joint = torch.tensor([SCENARIO["q0"]], device=device, dtype=torch.float32)
    q_goal_joint = torch.tensor([SCENARIO["goal_joints"]], device=device, dtype=torch.float32)
    q0_start = torch.tensor([SCENARIO["q_base0"]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([q_goal_np.tolist()], device=device, dtype=torch.float32)

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, total_time, device)
    num_steps = physics.num_steps  # 100

    # ── Load DDP trajectory ──
    ddp_dir = os.path.join(os.path.dirname(__file__), "results_ddp")
    ddp_data = np.genfromtxt(os.path.join(ddp_dir, "q_traj.csv"), delimiter=",", names=True)

    # DDP has T+1=101 points, PhysicsLayer has 100 steps -> take first 100
    ddp_q = np.zeros((101, n_q))
    for i in range(n_q):
        ddp_q[:, i] = ddp_data[f"J{i+1}"]
    ddp_q_target = torch.tensor(ddp_q[:num_steps], device=device, dtype=torch.float32)  # [100, 6]

    # DDP의 실제 최종 joint 값을 spline 끝점으로 사용
    # (DDP는 goal_joints에 정확히 도달하지 않으므로, 실제 끝점을 사용해야 공정한 비교)
    ddp_q_end = torch.tensor([ddp_q[-1].tolist()], device=device, dtype=torch.float32)  # [1, 6]
    print(f"DDP actual end joints: {ddp_q[-1]}")
    print(f"Goal joints (SCENARIO): {SCENARIO['goal_joints']}")
    print(f"  -> Using DDP actual end joints as spline boundary")

    print(f"=== Fitting waypoints to DDP trajectory ===")
    print(f"DDP trajectory: {ddp_q.shape[0]} points, using first {num_steps}")
    print(f"Waypoints to fit: {NUM_WAYPOINTS} (quintic spline, 4 segments)")

    # ── Initialize waypoints from DDP trajectory at segment boundaries ──
    # Segments: [0, 25, 50, 75, 100] -> waypoints at steps 25, 50, 75
    seg_len = num_steps // (NUM_WAYPOINTS + 1)
    init_wp = []
    for i in range(NUM_WAYPOINTS):
        idx = seg_len * (i + 1)
        init_wp.append(ddp_q_target[min(idx, num_steps-1)])
    init_wp = torch.stack(init_wp)  # [3, 6]
    waypoints = init_wp.reshape(1, -1).clone().requires_grad_(True)  # [1, 18]

    print(f"Initial waypoints (from DDP at t={[seg_len*(i+1)*total_time/num_steps for i in range(NUM_WAYPOINTS)]}s):")
    print(f"  {waypoints.detach().cpu().numpy()}")

    # ── Optimize: minimize ||q_traj_spline - q_traj_ddp||^2 ──
    optimizer = torch.optim.LBFGS([waypoints], lr=0.1, max_iter=50, line_search_fn='strong_wolfe')
    iter_count = [0]

    def closure():
        optimizer.zero_grad()
        q_traj, _ = physics.generate_trajectory(waypoints, q_start=q_start_joint, q_end=ddp_q_end)
        q_traj_s = q_traj[0]  # [100, 6]
        loss = ((q_traj_s - ddp_q_target) ** 2).mean()
        loss.backward()
        iter_count[0] += 1
        if iter_count[0] <= 5 or iter_count[0] % 10 == 0:
            print(f"  iter {iter_count[0]}: MSE={loss.item():.8f}")
        return loss

    print("\nLBFGS fitting...")
    optimizer.step(closure)

    # Final MSE
    with torch.no_grad():
        q_traj_fit, q_dot_fit = physics.generate_trajectory(waypoints, q_start=q_start_joint, q_end=ddp_q_end)
        mse = ((q_traj_fit[0] - ddp_q_target) ** 2).mean().item()
        max_err = (q_traj_fit[0] - ddp_q_target).abs().max().item()

    print(f"\nFitting result:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Max error: {max_err:.6f} rad ({math.degrees(max_err):.4f} deg)")
    print(f"  Fitted waypoints: {waypoints.detach().cpu().numpy()}")

    # ── Evaluate with CVAE's physics loss ──
    print("\n=== Evaluating fitted waypoints with CVAE loss ===")
    with torch.no_grad():
        total_loss, loss_dict = physics.calculate_total_loss(
            waypoints, q0_start, q0_goal,
            joint_squared_weight=SCENARIO["joint_squared_weight"],
            joint_change_weight=SCENARIO["joint_change_weight"],
            max_joint_weight=SCENARIO["max_joint_weight"],
            q_start_joint=q_start_joint,
            q_end_joint=ddp_q_end,
        )
        physics_loss = loss_dict['physics_loss'].item()
        total_loss_val = loss_dict['total_loss'].item()
        joint_sq = loss_dict['joint_squared_penalty'].item()
        joint_ch = loss_dict['joint_change_penalty'].item()
        max_j = loss_dict['max_joint_penalty'].item()

        # Angle error
        sim_out = physics.simulate_single(
            q_traj_fit[0], q_dot_fit[0], q0_start[0], q0_goal[0]
        )
        q_final = sim_out[1]
        dot_val = torch.sum(q_final * q0_goal[0]).abs().clamp(-1, 1)
        angle_err = float(2.0 * torch.acos(dot_val) * 180.0 / math.pi)

    # ── Load CVAE results for comparison ──
    cvae_dir = os.path.join(os.path.dirname(__file__), "results_cvae")
    cvae_meta = {}
    with open(os.path.join(cvae_dir, "meta.csv")) as f:
        next(f)
        for line in f:
            k, v = line.strip().split(",", 1)
            try:
                cvae_meta[k] = float(v)
            except ValueError:
                cvae_meta[k] = v

    ddp_meta = {}
    with open(os.path.join(os.path.dirname(__file__), "results_ddp", "meta.csv")) as f:
        next(f)
        for line in f:
            k, v = line.strip().split(",", 1)
            try:
                ddp_meta[k] = float(v)
            except ValueError:
                ddp_meta[k] = v

    print(f"\n{'='*65}")
    print(f"{'':>25} {'DDP(raw)':>12} {'DDP->WP':>12} {'CVAE':>12}")
    print(f"{'='*65}")
    print(f"{'Orient cost':>25} {ddp_meta['orient_cost']:>12.6f} {physics_loss:>12.6f} {cvae_meta['physics_loss']:>12.6f}")
    print(f"{'Total loss':>25} {'--':>12} {total_loss_val:>12.6f} {cvae_meta['total_loss']:>12.6f}")
    print(f"{'Angle err (deg)':>25} {'~0':>12} {angle_err:>12.4f} {cvae_meta['angle_err_deg']:>12.4f}")
    print(f"{'Joint sq penalty':>25} {'--':>12} {joint_sq:>12.6f} {'--':>12}")
    print(f"{'Joint change penalty':>25} {'--':>12} {joint_ch:>12.6f} {'--':>12}")
    print(f"{'Max joint penalty':>25} {'--':>12} {max_j:>12.6f} {'--':>12}")
    print(f"{'Traj fit MSE':>25} {'--':>12} {mse:>12.8f} {'--':>12}")
    print(f"{'='*65}")

    # Save fitted waypoints
    save_dir = os.path.join(os.path.dirname(__file__), "results_ddp_fit")
    os.makedirs(save_dir, exist_ok=True)
    wp_np = waypoints.detach().cpu().numpy()
    np.savetxt(os.path.join(save_dir, "waypoints.csv"), wp_np, delimiter=",",
               header=",".join([f"w{i}" for i in range(wp_np.shape[1])]), comments="")

    # Save fitted trajectory
    t_vec = np.linspace(0, total_time, num_steps)
    q_fit_np = q_traj_fit[0].cpu().numpy()
    header_q = "t," + ",".join([f"J{i+1}" for i in range(n_q)])
    np.savetxt(os.path.join(save_dir, "q_traj.csv"),
               np.column_stack([t_vec, q_fit_np]),
               delimiter=",", header=header_q, comments="")

    with open(os.path.join(save_dir, "meta.csv"), "w") as f:
        f.write("key,value\n")
        f.write(f"physics_loss,{physics_loss:.8f}\n")
        f.write(f"total_loss,{total_loss_val:.8f}\n")
        f.write(f"angle_err_deg,{angle_err:.4f}\n")
        f.write(f"fit_mse,{mse:.8f}\n")
        f.write(f"fit_max_err_rad,{max_err:.8f}\n")

    print(f"\nFitted results saved to {save_dir}/")


if __name__ == "__main__":
    main()
