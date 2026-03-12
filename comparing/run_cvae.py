"""
Run CVAE joint optimization for the shared SCENARIO and save results to comparing/results_cvae/.
Uses the newest .pth weight file from outputs/weights/cvae_joint/.
"""
import os
import sys
import time
import math
import glob
import numpy as np
import torch
from torch.func import vmap

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.cvae import CVAE
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart
from scenario import SCENARIO, get_goal_quaternion


def find_newest_pth(weight_dir):
    """Find the newest .pth file in a directory."""
    pth_files = glob.glob(os.path.join(weight_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files in {weight_dir}")
    newest = max(pth_files, key=os.path.getmtime)
    return newest


def quat_to_rot(q):
    x, y, z, w = q
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]),
    ])


def rot_to_euler(R):
    """Rotation matrix -> euler [yaw, pitch, roll] (ZYX)."""
    pitch = torch.arcsin(-torch.clamp(R[2, 0], -1, 1))
    if torch.abs(torch.cos(pitch)) > 1e-6:
        yaw = torch.atan2(R[1, 0], R[0, 0])
        roll = torch.atan2(R[2, 1], R[2, 2])
    else:
        yaw = torch.atan2(-R[0, 1], R[1, 1])
        roll = torch.zeros_like(yaw)
    return torch.stack([yaw, pitch, roll])


def compute_orientation_traj(physics, q_traj, q_dot_traj, q0_init):
    """Integrate body orientation and return euler trajectory [T, 3]."""
    device = physics.device
    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)
    R_curr = quat_to_rot(q0_init)

    def compute_wb(qm, qd):
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
        rhs = -H0m @ qd
        H0d = H0 + 1e-6 * torch.eye(6, device=device)
        u0 = torch.linalg.solve(H0d, rhs)
        return u0[:3]

    batch_wb = vmap(compute_wb, in_dims=(0, 0))
    wb_all = batch_wb(q_traj, q_dot_traj)

    def rot_from_omega(wb, dt):
        theta = torch.linalg.norm(wb) * dt
        axis = wb / (torch.linalg.norm(wb) + 1e-12)
        K = torch.stack([
            torch.stack([torch.zeros_like(theta), -axis[2], axis[1]]),
            torch.stack([axis[2], torch.zeros_like(theta), -axis[0]]),
            torch.stack([-axis[1], axis[0], torch.zeros_like(theta)]),
        ])
        I = torch.eye(3, device=device)
        return I + torch.sin(theta)*K + (1-torch.cos(theta))*(K@K)

    batch_rot = vmap(rot_from_omega, in_dims=(0, None))
    R_delta_all = batch_rot(wb_all, physics.dt)

    eulers = []
    for t in range(physics.num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))
    return torch.stack(eulers, dim=0)


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "results_cvae")
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load robot
    urdf_path = os.path.join(ROOT_DIR, SCENARIO["urdf"])
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    n_q = robot["n_q"]

    # Scenario params
    total_time = SCENARIO["total_time"]
    NUM_WAYPOINTS = SCENARIO["num_waypoints"]
    q_goal_np = get_goal_quaternion()

    # Condition: start_joint(6) + goal_joint(6) + goal_quat(4) = 16
    COND_DIM = n_q + n_q + 4
    OUTPUT_DIM = NUM_WAYPOINTS * n_q
    LATENT_DIM = 3

    q_start_joint = torch.tensor([SCENARIO["q0"]], device=device, dtype=torch.float32)
    q_goal_joint = torch.tensor([SCENARIO["goal_joints"]], device=device, dtype=torch.float32)
    q0_start = torch.tensor([SCENARIO["q_base0"]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([q_goal_np.tolist()], device=device, dtype=torch.float32)

    condition = torch.cat([q_start_joint, q_goal_joint, q0_goal], dim=1)  # [1, 16]

    print(f"=== CVAE Joint Optimization (SCENARIO) on {device} ===")
    print(f"Goal euler (deg): {SCENARIO['goal_euler_deg']}")
    print(f"Goal quat: {q_goal_np}")
    print(f"Start joints: {SCENARIO['q0']}")
    print(f"Goal joints: {SCENARIO['goal_joints']}")

    # Physics layer
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, total_time, device)

    # Load newest CVAE model
    weight_dir = os.path.join(ROOT_DIR, "outputs/weights/cvae_joint")
    weights_path = find_newest_pth(weight_dir)
    print(f"Loading weights: {weights_path}")
    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM, joint_limits=robot['joint_limits']).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Phase 1: CVAE Sampling ──
    num_samples = 1024
    print(f"\nSampling {num_samples} candidates...")
    t_start = time.time()

    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM, device=device)
        cond_batch = condition.repeat(num_samples, 1)
        candidates = model.decode(cond_batch, z)

        total_loss, _ = physics.calculate_total_loss(
            candidates,
            q0_start.repeat(num_samples, 1),
            q0_goal.repeat(num_samples, 1),
            joint_squared_weight=SCENARIO["joint_squared_weight"],
            joint_change_weight=SCENARIO["joint_change_weight"],
            max_joint_weight=SCENARIO["max_joint_weight"],
            return_mean=False,
            q_start_joint=q_start_joint.repeat(num_samples, 1),
            q_end_joint=q_goal_joint.repeat(num_samples, 1),
        )
        losses = torch.where(torch.isfinite(total_loss), total_loss, torch.full_like(total_loss, float("inf")))
        best_idx = torch.argmin(losses)
        best_waypoints = candidates[best_idx:best_idx+1].clone()
        best_loss = losses[best_idx].item()

    if device == "cuda":
        torch.cuda.synchronize()
    sample_time = time.time() - t_start
    print(f"Best sample loss: {best_loss:.8f} (took {sample_time:.3f}s)")

    # ── Phase 2: LBFGS Refinement ──
    print("\nLBFGS refinement...")
    waypoints_param = best_waypoints.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([waypoints_param], lr=1e-3, max_iter=20, line_search_fn='strong_wolfe')
    iter_count = [0]

    def closure():
        optimizer.zero_grad()
        loss, _ = physics.calculate_total_loss(
            waypoints_param, q0_start, q0_goal,
            joint_squared_weight=SCENARIO["joint_squared_weight"],
            joint_change_weight=SCENARIO["joint_change_weight"],
            max_joint_weight=SCENARIO["max_joint_weight"],
            q_start_joint=q_start_joint,
            q_end_joint=q_goal_joint,
        )
        loss.backward()
        iter_count[0] += 1
        if iter_count[0] % 5 == 0:
            print(f"  iter {iter_count[0]}: loss={loss.item():.8f}")
        return loss

    t_opt_start = time.time()
    optimizer.step(closure)
    if device == "cuda":
        torch.cuda.synchronize()
    opt_time = time.time() - t_opt_start

    # ── Evaluate final result ──
    with torch.no_grad():
        total_loss_final, loss_dict = physics.calculate_total_loss(
            waypoints_param, q0_start, q0_goal,
            joint_squared_weight=SCENARIO["joint_squared_weight"],
            joint_change_weight=SCENARIO["joint_change_weight"],
            max_joint_weight=SCENARIO["max_joint_weight"],
            q_start_joint=q_start_joint,
            q_end_joint=q_goal_joint,
        )
        physics_loss = loss_dict['physics_loss'].item()
        total_loss_val = loss_dict['total_loss'].item()

        # Generate trajectory
        q_traj, q_dot_traj = physics.generate_trajectory(
            waypoints_param, q_start=q_start_joint, q_end=q_goal_joint
        )
        q_traj_s = q_traj[0]       # [T, n_q]
        q_dot_s = q_dot_traj[0]    # [T, n_q]

        # Final orientation error
        sim_out = physics.simulate_single(q_traj_s, q_dot_s, q0_start[0], q0_goal[0])
        q_final_sim = sim_out[1]
        dot_val = torch.sum(q_final_sim * q0_goal[0]).abs().clamp(-1, 1)
        angle_err_deg = float(2.0 * torch.acos(dot_val) * 180.0 / math.pi)

        # Euler trajectory
        euler_traj = compute_orientation_traj(physics, q_traj_s, q_dot_s, q0_start[0])

    total_elapsed = sample_time + opt_time
    print(f"\n=== Results ===")
    print(f"Physics loss (orientation): {physics_loss:.8f}")
    print(f"Total loss: {total_loss_val:.8f}")
    print(f"Angle error: {angle_err_deg:.4f} deg")
    print(f"Time: sampling {sample_time:.3f}s + LBFGS {opt_time:.3f}s = {total_elapsed:.3f}s")

    # ── Save results ──
    num_steps = q_traj_s.shape[0]
    t_vec = np.linspace(0, total_time, num_steps)
    q_np = q_traj_s.cpu().numpy()
    qd_np = q_dot_s.cpu().numpy()
    euler_np = euler_traj.cpu().numpy()

    # Joint trajectory
    header_q = "t," + ",".join([f"J{i+1}" for i in range(n_q)])
    np.savetxt(os.path.join(save_dir, "q_traj.csv"),
               np.column_stack([t_vec, q_np]),
               delimiter=",", header=header_q, comments="")

    # Joint velocity
    header_qd = "t," + ",".join([f"dJ{i+1}" for i in range(n_q)])
    np.savetxt(os.path.join(save_dir, "qd_traj.csv"),
               np.column_stack([t_vec, qd_np]),
               delimiter=",", header=header_qd, comments="")

    # Euler trajectory
    np.savetxt(os.path.join(save_dir, "euler_traj.csv"),
               np.column_stack([t_vec, euler_np]),
               delimiter=",", header="t,yaw,pitch,roll", comments="")

    # Meta
    with open(os.path.join(save_dir, "meta.csv"), "w") as f:
        f.write("key,value\n")
        f.write(f"elapsed_s,{total_elapsed:.4f}\n")
        f.write(f"sample_time_s,{sample_time:.4f}\n")
        f.write(f"opt_time_s,{opt_time:.4f}\n")
        f.write(f"physics_loss,{physics_loss:.8f}\n")
        f.write(f"total_loss,{total_loss_val:.8f}\n")
        f.write(f"angle_err_deg,{angle_err_deg:.4f}\n")
        f.write(f"total_time,{total_time}\n")
        f.write(f"weights,{os.path.basename(weights_path)}\n")

    print(f"Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
