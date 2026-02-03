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

from NN_opt.training.physics_layer import PhysicsLayer
from physics.dynamics.urdf2robot_torch import urdf2robot
from physics.utils import (
    euler_to_quaternion,
    quat_to_rot,
    skew,
    rot_from_omega_torch,
    rot_to_euler_torch,
)

# Need to import spart for compute_orientation_traj
import physics.dynamics.spart_functions_torch as spart

def compute_orientation_traj(physics, q_traj, q_dot_traj, q0_init):
    device = physics.device
    num_steps = physics.num_steps

    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)

    R_curr = quat_to_rot(q0_init)

    eulers = []
    for t in range(num_steps):
        qm = q_traj[t]
        qd = q_dot_traj[t]

        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)

        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]

        R_delta = rot_from_omega_torch(wb, physics.dt)
        R_curr = R_curr @ R_delta

        eulers.append(rot_to_euler_torch(R_curr))

    euler_traj = torch.stack(eulers, dim=0)
    return euler_traj

def plot_trajectory(q_traj, q_dot_traj, euler_traj, title, save_path, total_time, target_euler=None):
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    euler_traj = euler_traj.detach().cpu().numpy()  # [T, 3], rad

    target_deg = None
    if target_euler is not None:
        target_deg = np.rad2deg(target_euler.detach().cpu().numpy())  # [3]

    num_steps = q_traj.shape[0]
    t = np.linspace(0.0, total_time, num_steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # 1) Joint Angles
    for i in range(q_traj.shape[1]):
        axes[0].plot(t, q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"{title} - Joint Angles (Cubic Spline)")
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
    print(f"=== PSO Optimization Start on {device} ===")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=device)

    # Parameters
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    
    # PSO Parameters
    POP_SIZE = 128
    MAX_ITER = 50
    W = 0.7  # Inertia weight
    C1 = 1.5 # Cognitive (particle best)
    C2 = 1.5 # Social (global best)

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/opt_pso")
    os.makedirs(save_dir, exist_ok=True)

    # Target Setup
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    q0_goal = euler_to_quaternion(
        torch.tensor([roll_rad], device=device),
        torch.tensor([pitch_rad], device=device),
        torch.tensor([yaw_rad], device=device),
    )
    
    # Expand start/goal for batch processing
    q0_start_batch = q0_start.repeat(POP_SIZE, 1)
    q0_goal_batch = q0_goal.repeat(POP_SIZE, 1)

    # Initialize Particle Swarm
    # Position: [POP_SIZE, OUTPUT_DIM]
    particles = torch.randn(POP_SIZE, OUTPUT_DIM, device=device)
    velocities = torch.zeros_like(particles)
    
    pbest_pos = particles.clone()
    pbest_loss = torch.full((POP_SIZE,), float('inf'), device=device)
    
    gbest_pos = particles[0].clone()
    gbest_loss = float('inf')

    loss_history = []
    
    # Pre-compile simulate_single with vmap
    batch_sim_fn = vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))

    start_time = time.time()
    
    print(f"Starting PSO with {POP_SIZE} particles, {MAX_ITER} iterations...")

    for i in range(MAX_ITER):
        # 1. Evaluate Fitness (Batch Loss Calculation)
        # Generate trajectories for all particles
        q_traj, q_dot_traj = physics.generate_trajectory(particles)
        
        # Calculate loss for each particle
        # Returns: (loss_batch, final_q_batch)
        loss_batch, _ = batch_sim_fn(q_traj, q_dot_traj, q0_start_batch, q0_goal_batch)
        
        # 2. Update Personal Best
        improved_mask = loss_batch < pbest_loss
        pbest_pos[improved_mask] = particles[improved_mask]
        pbest_loss[improved_mask] = loss_batch[improved_mask]
        
        # 3. Update Global Best
        min_loss, min_idx = torch.min(loss_batch, dim=0)
        if min_loss < gbest_loss:
            gbest_loss = min_loss.item()
            gbest_pos = particles[min_idx].clone()
        
        loss_history.append(gbest_loss)
        
        if (i+1) % 10 == 0:
            print(f"Iter [{i+1}/{MAX_ITER}] Best Loss: {gbest_loss:.6f}")
            
        # 4. Update Velocities and Positions
        r1 = torch.rand_like(particles)
        r2 = torch.rand_like(particles)
        
        velocities = (W * velocities + 
                      C1 * r1 * (pbest_pos - particles) + 
                      C2 * r2 * (gbest_pos.unsqueeze(0) - particles))
        
        particles = particles + velocities

    end_time = time.time()
    
    # Final Result
    physics_loss = gbest_loss  # PSO uses physics loss only (no joint penalty)
    # loss is log(chordal_dist_sq), so we need to reverse the log and sqrt
    # chordal_dist = 2 * sin(angle/2), so angle = 2 * arcsin(chordal_dist/2)
    # For small angles: chordal_dist ≈ angle (rad)
    chordal_dist_sq = np.exp(physics_loss)
    chordal_dist = np.sqrt(chordal_dist_sq)
    # Convert chordal distance to angle (rad), then to degrees
    # For small angles, chordal_dist ≈ angle_rad
    angle_rad = 2.0 * np.arcsin(np.clip(chordal_dist / 2.0, -1.0, 1.0))
    final_deg = np.rad2deg(angle_rad)
    
    print(f"Optimization Finished (PSO). Time: {end_time - start_time:.4f}s")
    print(f"Final Error: {physics_loss:.2e} ({final_deg:.2e}°)")
    print(f"Best waypoints: {gbest_pos}")

    # Visualize Best Result
    with torch.no_grad():
        best_wp = gbest_pos.unsqueeze(0) # [1, Dim]
        q_traj, q_dot_traj = physics.generate_trajectory(best_wp)
        q_traj_single = q_traj[0]
        q_dot_traj_single = q_dot_traj[0]
        
        euler_traj = compute_orientation_traj(physics, q_traj_single, q_dot_traj_single, q0_start[0])

        R_goal = quat_to_rot(q0_goal[0])
        target_euler = rot_to_euler_torch(R_goal)

        plot_trajectory(
            q_traj_single,
            q_dot_traj_single,
            euler_traj,
            f"PSO (Err: {physics_loss:.6f}, Angle: {final_deg:.2e}°)",
            os.path.join(save_dir, "pso_traj.png"),
            TOTAL_TIME,
            target_euler=target_euler,
        )

        # Save Data (CSV)
        dt = float(physics.dt)
        num_steps = q_traj_single.shape[0]
        t = np.linspace(0.0, TOTAL_TIME, num_steps)

        q_traj_np = q_traj_single.cpu().numpy()
        q_dot_np = q_dot_traj_single.cpu().numpy()
        euler_np = euler_traj.cpu().numpy()
        waypoints_np = best_wp.cpu().numpy()
        q0_start_np = q0_start.cpu().numpy()
        q0_goal_np = q0_goal.cpu().numpy()
        target_euler_np = target_euler.cpu().numpy()

        # 1) Joint positions
        n_q = robot["n_q"]
        header_q = "t," + ",".join([f"J{i+1}" for i in range(n_q)])
        q_traj_mat = np.column_stack([t, q_traj_np])
        np.savetxt(os.path.join(save_dir, "q_traj.csv"), q_traj_mat, delimiter=",", header=header_q, comments="")

        # 2) Joint velocities
        header_qdot = "t," + ",".join([f"dJ{i+1}" for i in range(n_q)])
        q_dot_mat = np.column_stack([t, q_dot_np])
        np.savetxt(os.path.join(save_dir, "q_dot_traj.csv"), q_dot_mat, delimiter=",", header=header_qdot, comments="")

        # 3) Body orientation
        target_tile = np.tile(target_euler_np.reshape(1, 3), (num_steps, 1))
        body_mat = np.column_stack([t, euler_np, target_tile])
        header_body = "t,yaw,pitch,roll,yaw_target,pitch_target,roll_target"
        np.savetxt(os.path.join(save_dir, "body_orientation.csv"), body_mat, delimiter=",", header=header_body, comments="")

        # 4) Waypoints
        header_wp = ",".join([f"W{i+1}" for i in range(waypoints_np.shape[1])])
        np.savetxt(os.path.join(save_dir, "waypoints.csv"), waypoints_np, delimiter=",", header=header_wp, comments="")
        
        # 5) Start/Goal
        np.savetxt(os.path.join(save_dir, "q0_start.csv"), q0_start_np, delimiter=",", header="qx,qy,qz,qw", comments="")
        np.savetxt(os.path.join(save_dir, "q0_goal.csv"), q0_goal_np, delimiter=",", header="qx,qy,qz,qw", comments="")

        # 6) Meta
        with open(os.path.join(save_dir, "meta.csv"), "w") as f:
            f.write("dt,total_time\n")
            f.write(f"{dt},{TOTAL_TIME}\n")
            
        print(f"Saved CSV trajectory data to {save_dir}")

if __name__ == "__main__":
    main()

