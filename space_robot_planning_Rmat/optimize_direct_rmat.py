import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np

# === 경로 설정 ===
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../space_robot_planning_Rmat
project_root = os.path.dirname(current_dir)               # .../CVAE

orig_src_dir = os.path.join(project_root, "space_robot_planning", "src")
if orig_src_dir not in sys.path:
    sys.path.append(orig_src_dir)

rmat_src_dir = os.path.join(current_dir, "src")
if rmat_src_dir not in sys.path:
    sys.path.append(rmat_src_dir)

from training.physics_layer import PhysicsLayer   # Rmat 버전
from dynamics.urdf2robot_torch import urdf2robot


def plot_trajectory(q_traj, q_dot_traj, title, save_path):
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"{title} - Joint Angles")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="right", fontsize="small")

    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Direct Optimization (Rmat Physics) Start on {device} ===")

    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)

    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = "results_rmat/direct_opt"
    os.makedirs(save_dir, exist_ok=True)

    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0.0, 0.0, 0.7071, 0.7071]], device=device, dtype=torch.float32)

    print("\n--- [Task 1] Fixed Goal Optimization (Rmat Physics) ---")

    waypoints_param = torch.zeros(1, OUTPUT_DIM, device=device)
    torch.nn.init.normal_(waypoints_param, mean=0.0, std=0.1)
    waypoints_param.requires_grad = True

    optimizer = optim.Adam([waypoints_param], lr=0.05)

    start_time = time.time()
    iterations = 200

    loss_history = []
    stop_threshold = 1e-4

    for i in range(iterations):
        optimizer.zero_grad()

        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        if (i + 1) <= 20 or (i + 1) % 10 == 0:
            print(f"[Rmat] Iter [{i+1}/{iterations}] Loss: {loss_value:.6f}")

        if loss_value < stop_threshold:
            print(f"[Rmat] Loss {loss_value:.6f} < {stop_threshold:.6f}. Early stopping at iter {i+1}.")
            break

    end_time = time.time()
    print(f"[Rmat] Optimization Finished. Time: {end_time - start_time:.4f}s")

    final_error = loss.item()
    final_deg = np.rad2deg(np.sqrt(final_error))
    print(f"[Rmat] Final Error: {final_error:.6f} (approx {final_deg:.2f}°)")

    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        plot_trajectory(
            q_traj[0],
            q_dot_traj[0],
            f"Direct Opt Rmat (Err: {final_error:.4f})",
            os.path.join(save_dir, "fixed_goal_traj_rmat.png"),
        )


if __name__ == "__main__":
    main()


