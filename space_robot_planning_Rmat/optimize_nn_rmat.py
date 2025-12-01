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

from models.cvae import CVAE
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


def load_cvae(device, robot, weights_path="weights/cvae_debug/v1.pth"):
    COND_DIM = 8
    NUM_WAYPOINTS = 4
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 8

    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, NUM_WAYPOINTS, OUTPUT_DIM, LATENT_DIM


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== NN-based Initialization + LBFGS (Rmat Physics) Start on {device} ===")

    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)

    # 파라미터
    cvae_model, NUM_WAYPOINTS, OUTPUT_DIM, LATENT_DIM = load_cvae(
        device, robot, weights_path="weights/cvae_debug/v1.pth"
    )
    TOTAL_TIME = 1.0

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = "results_rmat/opt_nn_lbfgs"
    os.makedirs(save_dir, exist_ok=True)

    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0.0, 0.0, 0.7071, 0.7071]], device=device, dtype=torch.float32)
    condition = torch.cat([q0_start, q0_goal], dim=1)

    print("\n--- [Task 1] Fixed Goal Optimization with CVAE Init (LBFGS, Rmat Physics) ---")

    # 1. CVAE Inference (Warm Start)
    inference_start = time.time()

    with torch.no_grad():
        num_samples = 10
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1)

        candidates = cvae_model.decode(cond_batch, z)

        q_traj, q_dot_traj = physics.generate_trajectory(candidates)
        batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        losses = batch_sim_fn(
            q_traj,
            q_dot_traj,
            q0_start.repeat(num_samples, 1),
            q0_goal.repeat(num_samples, 1),
        )

        best_idx = torch.argmin(losses)
        best_waypoints = candidates[best_idx].unsqueeze(0).clone()
        best_loss = losses[best_idx].item()

    inference_end = time.time()
    print(f"[Rmat CVAE Init] Selected best of {num_samples} samples with loss {best_loss:.8f}")

    # 2. LBFGS Refinement (Rmat PhysicsLayer 사용)
    waypoints_param = best_waypoints.detach().clone()
    waypoints_param.requires_grad = True

    optimizer = optim.LBFGS(
        [waypoints_param],
        lr=1.0,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    loss_history = [best_loss]
    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        loss.backward()
        loss_value = loss.item()
        loss_history.append(loss_value)
        iteration_count[0] += 1

        if iteration_count[0] <= 20 or iteration_count[0] % 10 == 0:
            print(f"[Rmat] Iter [{iteration_count[0]}] Loss: {loss_value:.6f}")

        return loss

    opt_start = time.time()
    optimizer.step(closure)
    opt_end = time.time()

    # 결과 확인 (Euler 기반 loss, Rmat 물리)
    final_loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal).item()
    final_deg = np.rad2deg(np.sqrt(final_loss)) if final_loss > 0 else 0.0

    print(f"Inference Finished (Rmat CVAE warm start). Time: {inference_end - inference_start:.4f}s")
    print(f"Optimization Finished (Rmat LBFGS). Time: {opt_end - opt_start:.4f}s")
    print(f"[Rmat] Final Error: {final_loss:.10f} ({final_deg:.4f}°)")
    print(f"[Rmat] Iterations: {len(loss_history)}")

    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        plot_trajectory(
            q_traj[0],
            q_dot_traj[0],
            f"CVAE+LBFGS Rmat (Err: {final_loss:.6f})",
            os.path.join(save_dir, "cvae_lbfgs_traj_rmat.png"),
        )


if __name__ == "__main__":
    main()


