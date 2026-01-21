"""
DDP/iLQR with Neural Warm-Start (CVAE + PhysicsLayer).

Pipeline:
1) Load trained CVAE (v5_joint_change.pth) and PhysicsLayer (R-mat dynamics).
2) Sample many waypoint candidates conditioned on (q0_start, q0_goal) quaternions.
3) Evaluate physics-based total loss for each candidate and pick the best.
4) Generate a 10s (100-step) joint trajectory from that best waypoint set,
   with start / end joint angles explicitly fixed to zero.
5) Use the resulting joint velocity trajectory as the initial control sequence
   for DDP/iLQR, instead of zero controls.
"""

import os
import sys
import time as time_module

import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================================
# SWITCH: Choose between iLQR and DDP
# ============================================================================
USE_ILQR = True  # Set to True for iLQR (faster), False for full DDP
# ============================================================================

# Add project root to sys.path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.models.cvae import CVAE  # noqa: E402
from src.training.physics_layer import PhysicsLayer  # noqa: E402
from src.dynamics.urdf2robot_torch import urdf2robot  # noqa: E402
import ddp.dynamics_torch as dynamics  # noqa: E402
import ddp.cost as cost  # noqa: E402
import ddp.solver as solver  # noqa: E402


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (ZYX) to quaternion [x, y, z, w]."""
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

    return torch.stack([qx, qy, qz, qw])


def load_cvae(weights_path, condition_dim, output_dim, latent_dim, joint_limits, device):
    """
    Utility to construct CVAE and load weights with strict=False fallback.
    """
    model = CVAE(condition_dim, output_dim, latent_dim, joint_limits=joint_limits).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    print(f"[CVAE] Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[CVAE] strict=True load failed, retrying with strict=False.\n  {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def main():
    # ------------------------------------------------------------------
    # 0. Device and robot setup
    # ------------------------------------------------------------------
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    print(f"Using device: {device}")

    urdf_path = os.path.join(ROOT_DIR, "assets", "SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    # Time horizon (match PhysicsLayer and original DDP)
    T = 100
    dt = 0.1
    total_time = T * dt  # 10 seconds
    print(f"Time horizon: {total_time}s ({T} steps, dt={dt}s)")

    # ------------------------------------------------------------------
    # 1. DDP dynamics and cost definitions (same as run_ddp.py)
    # ------------------------------------------------------------------
    dynamics_model = dynamics.SpaceRobotDynamics(robot, device=device)

    # Initial state: [joint_angles (6), base_quaternion (4)]
    q_joints_init = torch.zeros(6, device=device)  # all joints at zero
    q_base_init = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    initial_state = torch.cat([q_joints_init, q_base_init], dim=0)  # [10]

    # Goal orientation: 15 deg roll, 15 deg pitch, -15 deg yaw
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll_rad = torch.tensor(np.deg2rad(roll_deg), device=device)
    pitch_rad = torch.tensor(np.deg2rad(pitch_deg), device=device)
    yaw_rad = torch.tensor(np.deg2rad(yaw_deg), device=device)
    goal_quat = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

    # Goal joint angles: zero (for now)
    goal_joints = torch.zeros(6, device=device)

    print(f"Initial orientation: Identity")
    print(f"Initial joints: {q_joints_init.cpu().numpy()}")
    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")
    print(f"Goal joints: {goal_joints.cpu().numpy()}")

    # Running and terminal cost (same hyperparameters as run_ddp.py)
    R_weight = 0.01
    joint_limits = robot["joint_limits"].to(device)
    running_cost = cost.RunningCost(
        R_weight=R_weight,
        joint_limits=joint_limits,
        barrier_weight=0.1,
        device=device,
    )

    terminal_cost = cost.TerminalCost(
        goal_quaternion=goal_quat,
        goal_joints=goal_joints,
        orientation_weight=20.0,
        joint_weight=5.0,
        device=device,
    )

    # ------------------------------------------------------------------
    # 2. CVAE + PhysicsLayer warm-start (joint start/end = 0)
    # ------------------------------------------------------------------
    COND_DIM = 8  # [q0_start (4), q0_goal (4)]
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 3

    JOINT_SQUARED_WEIGHT = 0.01
    JOINT_CHANGE_WEIGHT = 0.01
    MAX_JOINT_WEIGHT = 0.1

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, total_time, device)

    cvae_weights_path = os.path.join(
        ROOT_DIR, "outputs", "weights", "cvae_debug", "v5_joint_change.pth"
    )
    cvae_model = load_cvae(
        cvae_weights_path,
        COND_DIM,
        OUTPUT_DIM,
        LATENT_DIM,
        joint_limits=robot["joint_limits"],
        device=device,
    )

    # Condition: start/goal quaternions (start: identity, goal: target orientation)
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    q0_goal = goal_quat.unsqueeze(0)  # [1, 4]
    condition = torch.cat([q0_start, q0_goal], dim=1)  # [1, 8]

    num_samples = 1024
    print(f"\n[Warm-Start] CVAE sampling: {num_samples} candidates ...")

    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM, device=device)
        cond_batch = condition.repeat(num_samples, 1)
        candidates = cvae_model.decode(cond_batch, z)  # [num_samples, OUTPUT_DIM]

        # Joint start/end are explicitly zero for all samples
        q_start_joint_batch = torch.zeros(num_samples, robot["n_q"], device=device)
        q_end_joint_batch = torch.zeros(num_samples, robot["n_q"], device=device)

        q0_start_batch = q0_start.repeat(num_samples, 1)
        q0_goal_batch = q0_goal.repeat(num_samples, 1)

        total_loss_vec, _ = physics.calculate_total_loss(
            candidates,
            q0_start_batch,
            q0_goal_batch,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False,
            q_start_joint=q_start_joint_batch,
            q_end_joint=q_end_joint_batch,
        )

        safe_losses = torch.where(
            torch.isfinite(total_loss_vec),
            total_loss_vec,
            torch.full_like(total_loss_vec, float("inf")),
        )
        best_idx = torch.argmin(safe_losses)
        best_waypoints = candidates[best_idx : best_idx + 1].clone()  # [1, OUTPUT_DIM]
        best_loss = safe_losses[best_idx].item()

    print(f"[Warm-Start] Selected best CVAE sample index {best_idx.item()} with loss {best_loss:.6f}")

    # Generate single joint trajectory (start/end joint = 0)
    with torch.no_grad():
        q_start_joint_single = torch.zeros(1, robot["n_q"], device=device)
        q_end_joint_single = torch.zeros(1, robot["n_q"], device=device)
        q_traj, q_dot_traj = physics.generate_trajectory(
            best_waypoints,
            q_start=q_start_joint_single,
            q_end=q_end_joint_single,
        )

        q_traj_single = q_traj[0]  # [T, 6]
        q_dot_traj_single = q_dot_traj[0]  # [T, 6]

    if q_dot_traj_single.shape[0] != T:
        raise RuntimeError(
            f"PhysicsLayer produced {q_dot_traj_single.shape[0]} steps, "
            f"but DDP expects T={T}."
        )

    print(f"[Warm-Start] Using PhysicsLayer joint velocity trajectory as initial controls.")

    # ------------------------------------------------------------------
    # 3. Setup DDP/iLQR solver and solve with NN warm-start
    # ------------------------------------------------------------------
    initial_controls = q_dot_traj_single.clone()  # [T, 6]

    method_name = "iLQR" if USE_ILQR else "DDP"
    use_full_ddp = not USE_ILQR

    ddp_solver = solver.DDP(
        dynamics_model=dynamics_model,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        max_iter=50,
        tol=1e-4,
        reg_init=1.0,
        use_full_ddp=use_full_ddp,
        terminal_control_weight=1.0,
        device=device,
    )

    print(f"\nStarting {method_name} optimization with NN warm-start...")
    print("-" * 50)
    if USE_ILQR:
        print("Mode: iLQR (dynamics curvature terms disabled - faster)")
    else:
        print("Mode: Full DDP (dynamics curvature terms enabled - more accurate)")
    print("Initial controls from CVAE+PhysicsLayer (start/end joints = 0).")
    print("-" * 50)

    start_time = time_module.time()
    states, controls, cost_history = ddp_solver.solve(
        initial_state=initial_state,
        initial_controls=initial_controls,
        dt=dt,
    )
    elapsed_time = time_module.time() - start_time

    print(f"\nOptimization completed!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Initial cost: {cost_history[0]:.6f}")
    print(f"Cost reduction: {cost_history[0] - cost_history[-1]:.6f}")
    print(f"Iterations: {len(cost_history) - 1}")
    if len(cost_history) > 1:
        avg_time_per_iter = elapsed_time / (len(cost_history) - 1)
        print(f"Average time per iteration: {avg_time_per_iter:.3f} seconds")

    # Final orientation error
    final_quat = states[-1, 6:].detach()
    final_quat = dynamics.normalize_quat(final_quat)
    final_R = dynamics.quat_to_rot(final_quat)
    goal_R = dynamics.quat_to_rot(goal_quat.detach())
    orientation_error = cost.geodesic_distance_so3(final_R, goal_R)
    print(f"\nFinal orientation error: {torch.rad2deg(orientation_error.detach()):.2f}°")

    # Final joint error
    final_joints = states[-1, :6].detach()
    joint_error = final_joints - goal_joints
    joint_error_norm = torch.linalg.norm(joint_error)
    print(f"Final joint error: {joint_error_norm.item():.4f} rad (L2 norm)")
    print(f"Final joint angles: {final_joints.cpu().numpy()}")
    print(f"Goal joint angles: {goal_joints.cpu().numpy()}")

    # ------------------------------------------------------------------
    # 4. Save results and plots (suffix _nn to distinguish)
    # ------------------------------------------------------------------
    save_dir = os.path.dirname(__file__)
    os.makedirs(save_dir, exist_ok=True)

    results_filename = f"results_{method_name.lower()}_nn.pt"
    results_path = os.path.join(save_dir, results_filename)
    torch.save(
        {
            "states": states.cpu(),
            "controls": controls.cpu(),
            "cost_history": cost_history,
            "initial_state": initial_state.cpu(),
            "goal_quaternion": goal_quat.cpu(),
            "dt": dt,
            "T": T,
            "method": method_name,
            "use_full_ddp": use_full_ddp,
        },
        results_path,
    )
    print(f"\nResults saved to: {results_path}")

    # Plot cost history
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, "b-", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Total Cost", fontsize=12)
    plt.title(f"{method_name} Cost History (NN Warm-Start)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    cost_history_filename = f"cost_history_{method_name.lower()}_nn.png"
    cost_history_path = os.path.join(save_dir, cost_history_filename)
    plt.savefig(cost_history_path, dpi=150, bbox_inches="tight")
    print(f"Cost history plot saved to: {cost_history_path}")

    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_arr = np.arange(T + 1) * dt

    states_np = states.detach().cpu().numpy()
    controls_np = controls.detach().cpu().numpy()
    goal_quat_np = goal_quat.detach().cpu().numpy()

    # Joint angles
    axes[0, 0].plot(time_arr, states_np[:, :6])
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Joint Angle (rad)")
    axes[0, 0].set_title("Joint Angles")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend([f"Joint {i+1}" for i in range(6)])

    # Joint velocities (controls)
    axes[0, 1].plot(time_arr[:-1], controls_np)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Joint Velocity (rad/s)")
    axes[0, 1].set_title("Joint Velocities (Controls)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend([f"Joint {i+1}" for i in range(6)])

    # Base quaternion
    axes[1, 0].plot(time_arr, states_np[:, 6:])
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Quaternion Component")
    axes[1, 0].set_title("Base Orientation (Quaternion)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(["qx", "qy", "qz", "qw"])
    axes[1, 0].axhline(y=goal_quat_np[0], color="r", linestyle="--", alpha=0.5, label="Goal qx")
    axes[1, 0].axhline(y=goal_quat_np[1], color="g", linestyle="--", alpha=0.5, label="Goal qy")
    axes[1, 0].axhline(y=goal_quat_np[2], color="b", linestyle="--", alpha=0.5, label="Goal qz")
    axes[1, 0].axhline(y=goal_quat_np[3], color="m", linestyle="--", alpha=0.5, label="Goal qw")

    # Orientation error over time
    orientation_errors = []
    goal_R_detached = goal_R.detach()
    for t_idx in range(T + 1):
        q_t = dynamics.normalize_quat(states[t_idx, 6:].detach())
        R_t = dynamics.quat_to_rot(q_t)
        err = cost.geodesic_distance_so3(R_t, goal_R_detached)
        orientation_errors.append(torch.rad2deg(err.detach()).item())

    axes[1, 1].plot(time_arr, orientation_errors, "r-", linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Orientation Error (deg)")
    axes[1, 1].set_title("Orientation Error vs Goal")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    trajectory_filename = f"trajectory_{method_name.lower()}_nn.png"
    trajectory_path = os.path.join(save_dir, trajectory_filename)
    plt.savefig(trajectory_path, dpi=150, bbox_inches="tight")
    print(f"Trajectory plot saved to: {trajectory_path}")

    print("\n" + "=" * 50)
    print(f"{method_name} Optimization with NN Warm-Start Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()


