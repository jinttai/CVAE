import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import math
from torch.func import vmap

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from NN_opt.model.cvae import CVAE, MLP
from NN_opt.training.physics_layer import PhysicsLayer   # default
from physics.dynamics.urdf2robot_torch import urdf2robot
import physics.dynamics.spart_functions_torch as spart
from physics.utils import euler_to_quaternion_torch as euler_to_quaternion, quat_to_rot_torch as quat_to_rot, skew_torch as skew, rot_from_omega_torch as rot_from_omega, rot_to_euler_torch as rot_to_euler, wrap_to_pi_torch as wrap_to_pi, generate_random_quaternion_from_euler_torch as generate_random_quaternion_from_euler


def compute_orientation_traj(physics, q_traj, q_dot_traj, q0_init):
    """
    PhysicsLayer에서 사용하는 동역학과 동일하게 body orientation 궤적을 적분하여
    각 스텝의 Euler angle (yaw, pitch, roll)을 반환.
    Vectorized version: wb 계산은 vmap으로 병렬 처리 (PhysicsLayer와 동일)

    Args:
        physics: PhysicsLayer 인스턴스
        q_traj: [num_steps, n_q]
        q_dot_traj: [num_steps, n_q]
        q0_init: [4]
    Returns:
        euler_traj: [num_steps, 3] (rad)
    """
    device = physics.device
    num_steps = physics.num_steps

    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)

    R_curr = quat_to_rot(q0_init)

    # Vectorized: 모든 step의 wb를 한번에 계산 (PhysicsLayer.simulate_single과 동일)
    def compute_wb_single_step(qm, qd):
        """Single step의 wb 계산"""
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)

        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]  # Angular Velocity part
        return wb
    
    # vmap으로 모든 step을 병렬 처리
    batch_compute_wb = vmap(compute_wb_single_step, in_dims=(0, 0))
    wb_all = batch_compute_wb(q_traj, q_dot_traj)  # [num_steps, 3]
    
    # 모든 step의 R_delta를 한번에 계산
    batch_rot_from_omega = vmap(rot_from_omega, in_dims=(0, None))
    R_delta_all = batch_rot_from_omega(wb_all, physics.dt)  # [num_steps, 3, 3]
    
    # 순차적으로 R_curr 업데이트 및 euler 계산 (각 스텝마다 euler가 필요하므로)
    eulers = []
    for t in range(num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))

    euler_traj = torch.stack(eulers, dim=0)
    return euler_traj


def plot_trajectory(q_traj, q_dot_traj, euler_traj, title, save_path, total_time, target_euler=None):
    """
    joint trajectory는 PhysicsLayer.generate_trajectory의 5차 다항식(quintic) 결과를 그대로 사용하고,
    body orientation 궤적은 Euler angle 로 함께 plot.
    """
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    euler_traj = euler_traj.detach().cpu().numpy()  # [T, 3], rad

    # Optional target Euler angle (single 3-vector, rad)
    target_deg = None
    if target_euler is not None:
        target_deg = np.rad2deg(target_euler.detach().cpu().numpy())  # [3]

    num_steps = q_traj.shape[0]
    t = np.linspace(0.0, total_time, num_steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # 1) Joint Angles
    for i in range(q_traj.shape[1]):
        axes[0].plot(t, q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"{title} - Joint Angles (Half-cosine)")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", fontsize=8)

    # 2) Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(t, q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    # 3) Body Orientation (Euler, deg)
    euler_deg = np.rad2deg(euler_traj)
    labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    for i in range(3):
        axes[2].plot(t, euler_deg[:, i], label=labels[i])
        if target_deg is not None:
            axes[2].axhline(target_deg[i], linestyle="--", linewidth=1.5, label=f"Target {labels[i]}")
    axes[2].set_title("Body Orientation (Euler)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Angle [deg]")
    axes[2].grid(True)
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Random Initialization + LBFGS Start on {device} ===")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=device)

    # 파라미터 (원본 프로젝트 구조와 동일하게 정리)
    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 8
    TOTAL_TIME = 10.0  # 10초 trajectory
    MAX_JOINT_WEIGHT = 0.01  # Weight for maximum joint angle regularization (same as training)

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/opt_nn_lbfgs")
    os.makedirs(save_dir, exist_ok=True)

    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    # Fixed desired orientation: roll=15deg, pitch=15deg, yaw=-15deg (within angle limits)
    roll_deg, pitch_deg, yaw_deg = np.random.randint(0, 40), np.random.randint(0, 40), np.random.randint(0, 40)
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    q0_goal = euler_to_quaternion(
        torch.tensor([roll_rad], device=device),
        torch.tensor([pitch_rad], device=device),
        torch.tensor([yaw_rad], device=device),
    )
    condition = torch.cat([q0_start, q0_goal], dim=1)


    # 1. LBFGS Refinement (zero initial guess 사용)
    waypoints_param = torch.randn(1, OUTPUT_DIM, device=device, dtype=torch.float32)
    waypoints_param.requires_grad = True
    print(f"Initial waypoints: {waypoints_param}")

    optimizer = optim.LBFGS(
        [waypoints_param],
        lr=1.0,
        max_iter=50,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_history = []
    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        physics_loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        
        # Maximum joint angle penalty (same as training)
        waypoints_reshaped = waypoints_param.view(1, NUM_WAYPOINTS, robot["n_q"])
        max_joint_angle = waypoints_reshaped.abs().view(1, -1).max(dim=1)[0]  # [1]
        max_joint_penalty = max_joint_angle.mean()  # Scalar
        
        # Combined loss (same as training)
        loss = physics_loss + MAX_JOINT_WEIGHT * max_joint_penalty
        
        loss.backward()
        loss_value = loss.item()
        loss_history.append(loss_value)
        iteration_count[0] += 1

        if iteration_count[0] <= 20 or iteration_count[0] % 10 == 0:
            print(f"[Iter] Iter [{iteration_count[0]}] Loss: {loss_value:.6f} (physics: {physics_loss.item():.6f}, joint_penalty: {max_joint_penalty.item():.6f})")

        return loss

    opt_start = time.time()
    optimizer.step(closure)
    opt_end = time.time()

    # 결과 확인
    # Use physics loss only for angle error calculation (exclude joint penalty)
    physics_loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal).item()
    
    # Calculate joint penalty separately for reporting
    waypoints_reshaped = waypoints_param.view(1, NUM_WAYPOINTS, robot["n_q"])
    max_joint_angle = waypoints_reshaped.abs().view(1, -1).max(dim=1)[0].item()
    max_joint_penalty = max_joint_angle * MAX_JOINT_WEIGHT
    total_loss = physics_loss + max_joint_penalty
    
    # Calculate actual quaternion error for display
    with torch.no_grad():
        q_traj_temp, q_dot_traj_temp = physics.generate_trajectory(waypoints_param)
        sim_out_temp = physics.simulate_single(q_traj_temp[0], q_dot_traj_temp[0], q0_start[0], q0_goal[0])
        q_final_temp = sim_out_temp[1]
        q1 = q_final_temp
        q2 = q0_goal[0]
        dot = torch.sum(q1 * q2).abs().clamp(-1.0, 1.0)
        angle_rad = 2.0 * torch.acos(dot)
        final_deg = angle_rad * 180.0 / math.pi

    print(f"Optimization Finished (LBFGS). Time: {opt_end - opt_start:.4f}s")
    print(f"Total Loss: {total_loss:.2e} (physics: {physics_loss:.2e}, joint_penalty: {max_joint_penalty:.2e})")
    print(f"Angle Error: {final_deg.item():.2e}°")
    print(f"Iterations: {len(loss_history)}")
    print(f"Final waypoints: {waypoints_param}")

    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        q_traj_single = q_traj[0]
        q_dot_traj_single = q_dot_traj[0]
        euler_traj = compute_orientation_traj(physics, q_traj_single, q_dot_traj_single, q0_start[0])

        # Target body orientation in Euler angles (rad)
        R_goal = quat_to_rot(q0_goal[0])
        target_euler = rot_to_euler(R_goal)

        # --------------------------------------------------------------
        # Debug: compare final vs desired orientation (quat + Euler)
        # --------------------------------------------------------------
        final_euler = euler_traj[-1]                     # [3] (yaw, pitch, roll)
        target_euler_vec = target_euler                  # [3] (yaw, pitch, roll)

        # Convert to degrees for readability
        final_euler_deg = final_euler * 180.0 / math.pi
        target_euler_deg = target_euler_vec * 180.0 / math.pi

        # Reconstruct quaternion from final Euler to check Euler<->quat consistency.
        # NOTE: rot_to_euler returns [yaw, pitch, roll] (ZYX),
        #       while euler_to_quaternion expects (roll, pitch, yaw).
        yaw_f, pitch_f, roll_f = final_euler[0], final_euler[1], final_euler[2]
        q_final = euler_to_quaternion(
            roll_f.unsqueeze(0),
            pitch_f.unsqueeze(0),
            yaw_f.unsqueeze(0),
        )  # [1, 4]

        print("\n=== Orientation Check ===")
        print("Final Euler (rad)   [yaw, pitch, roll]:", final_euler)
        print("Target Euler (rad)  [yaw, pitch, roll]:", target_euler_vec)
        print("Final Euler (deg)   [yaw, pitch, roll]:", final_euler_deg)
        print("Target Euler (deg)  [yaw, pitch, roll]:", target_euler_deg)
        print("Final quaternion (from Euler) :", q_final)
        print("Target quaternion (q0_goal)   :", q0_goal)

        plot_trajectory(
            q_traj_single,
            q_dot_traj_single,
            euler_traj,
            f"CVAE+LBFGS (Err: {physics_loss:.6f}, Angle: {final_deg.item():.2e}°)",
            os.path.join(save_dir, "cvae_lbfgs_traj.png"),
            TOTAL_TIME,
            target_euler=target_euler,
        )

        # ------------------------------------------------------------------
        # Save data for external (e.g., MATLAB) plotting as CSV files
        # ------------------------------------------------------------------
        dt = float(physics.dt)
        num_steps = q_traj_single.shape[0]
        t = np.linspace(0.0, TOTAL_TIME, num_steps)

        q_traj_np = q_traj_single.detach().cpu().numpy()       # [T, n_q]
        q_dot_np = q_dot_traj_single.detach().cpu().numpy()    # [T, n_q]
        euler_np = euler_traj.detach().cpu().numpy()           # [T, 3] (rad)
        waypoints_np = waypoints_param.detach().cpu().numpy()  # [1, W]
        q0_start_np = q0_start.detach().cpu().numpy()          # [1, 4]
        q0_goal_np = q0_goal.detach().cpu().numpy()            # [1, 4]
        target_euler_np = target_euler.detach().cpu().numpy()  # [3] (rad)

        # 1) Joint position trajectory: time + J1..Jn
        n_q = robot["n_q"]
        header_q = "t," + ",".join([f"J{i+1}" for i in range(n_q)])
        q_traj_mat = np.column_stack([t, q_traj_np])
        np.savetxt(
            os.path.join(save_dir, "q_traj.csv"),
            q_traj_mat,
            delimiter=",",
            header=header_q,
            comments="",
        )

        # 2) Joint velocity trajectory: time + dJ1..dJn
        header_qdot = "t," + ",".join([f"dJ{i+1}" for i in range(n_q)])
        q_dot_mat = np.column_stack([t, q_dot_np])
        np.savetxt(
            os.path.join(save_dir, "q_dot_traj.csv"),
            q_dot_mat,
            delimiter=",",
            header=header_qdot,
            comments="",
        )

        # 3) Body orientation (Euler) and target orientation (rad)
        #    Columns: t, yaw, pitch, roll, yaw_target, pitch_target, roll_target
        target_tile = np.tile(target_euler_np.reshape(1, 3), (num_steps, 1))
        body_mat = np.column_stack([t, euler_np, target_tile])
        header_body = "t,yaw,pitch,roll,yaw_target,pitch_target,roll_target"
        np.savetxt(
            os.path.join(save_dir, "body_orientation.csv"),
            body_mat,
            delimiter=",",
            header=header_body,
            comments="",
        )

        # 4) Waypoints (single row)
        header_wp = ",".join([f"W{i+1}" for i in range(waypoints_np.shape[1])])
        np.savetxt(
            os.path.join(save_dir, "waypoints.csv"),
            waypoints_np,
            delimiter=",",
            header=header_wp,
            comments="",
        )

        # 5) Start / goal quaternion (each as separate CSV)
        np.savetxt(
            os.path.join(save_dir, "q0_start.csv"),
            q0_start_np,
            delimiter=",",
            header="qx,qy,qz,qw",
            comments="",
        )
        np.savetxt(
            os.path.join(save_dir, "q0_goal.csv"),
            q0_goal_np,
            delimiter=",",
            header="qx,qy,qz,qw",
            comments="",
        )

        # 6) Meta info (dt, total_time)
        meta_path = os.path.join(save_dir, "meta.csv")
        with open(meta_path, "w") as f:
            f.write("dt,total_time\n")
            f.write(f"{dt},{TOTAL_TIME}\n")

        print(f"Saved CSV trajectory data to {save_dir}")


if __name__ == "__main__":
    main()


