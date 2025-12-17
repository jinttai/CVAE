import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.models.cvae import CVAE, MLP
from src.training.physics_layer import PhysicsLayer   # default
from src.dynamics.urdf2robot_torch import urdf2robot


def load_model(model_class, weights_path, input_dim, output_dim, latent_dim=None, device="cpu"):
    if model_class == CVAE:
        model = CVAE(input_dim, output_dim, latent_dim).to(device)
    else:
        model = MLP(input_dim, output_dim).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def plot_best_trajectories(q_trajs, losses, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_idx = np.argsort(losses)
    colors = ["r", "g", "b"]

    for i in range(min(3, len(losses))):
        idx = sorted_idx[i]
        traj = q_trajs[idx]
        loss = losses[idx]
        ax.plot(traj[:, 0], color=colors[i], label=f"Rank {i+1} (Loss: {loss:.4f})", alpha=0.8, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Joint 1 Angle (Rad)")
    ax.legend()
    ax.grid(True)

    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Evaluation Start on {device} ===")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=device)

    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 8
    TOTAL_TIME = 1.0

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    cvae_weights_dir = os.path.join(ROOT_DIR, "outputs/weights/cvae_debug")
    mlp_weights_dir = os.path.join(ROOT_DIR, "outputs/weights/mlp_debug")

    def get_latest_weight(dir_path):
        if not os.path.exists(dir_path):
            return None
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".pth")]
        if not files:
            return None
        return max(files, key=os.path.getctime)

    cvae_path = get_latest_weight(cvae_weights_dir)
    mlp_path = get_latest_weight(mlp_weights_dir)

    print(f"[CVAE] CVAE weights: {cvae_path}")
    print(f"[MLP] MLP  weights: {mlp_path}")

    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    goal_rpy = [np.pi / 100, np.pi / 100, np.pi / 100]
    goal_quat = R.from_euler("xyz", goal_rpy).as_quat().reshape(1, 4)
    q0_goal = torch.tensor(goal_quat, device=device, dtype=torch.float32)
    condition = torch.cat([q0_start, q0_goal], dim=1)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/evaluation")
    os.makedirs(save_dir, exist_ok=True)

    # === CVAE 평가 (PhysicsLayer) ===
    if cvae_path:
        cvae = load_model(CVAE, cvae_path, COND_DIM, OUTPUT_DIM, LATENT_DIM, device)

        print("\n--- [CVAE] CVAE Evaluation (100 Samples) ---")

        num_samples = 100
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1)
        start_batch = q0_start.repeat(num_samples, 1)
        goal_batch = q0_goal.repeat(num_samples, 1)

        with torch.no_grad():
            waypoints = cvae.decode(cond_batch, z)
            q_traj, q_dot_traj = physics.generate_trajectory(waypoints)

            batch_sim_fn = torch.func.vmap(physics.simulate_single_rk4, in_dims=(0, 0, 0, 0))
            errors = batch_sim_fn(q_traj, q_dot_traj, start_batch, goal_batch)

        errors_np = errors.cpu().numpy()
        q_traj_np = q_traj.cpu().numpy()

        sorted_indices = np.argsort(errors_np)

        print(f"{'Rank':<5} | {'Error (Rad^2)':<15} | {'Error (Deg)':<15}")
        print("-" * 40)

        for i, idx in enumerate(sorted_indices):
            err_rad = np.sqrt(errors_np[idx])
            err_deg = np.rad2deg(err_rad)
            print(f"{i+1:<5} | {errors_np[idx]:.6f}        | {err_deg:.4f}°")

        plot_best_trajectories(
            q_traj_np,
            errors_np,
            "[CVAE] Top-3 Trajectories",
            os.path.join(save_dir, "cvae_top3.png"),
        )

    # === MLP 평가 (PhysicsLayer) ===
    if mlp_path:
        mlp = load_model(MLP, mlp_path, COND_DIM, OUTPUT_DIM, device=device)
        print("\n--- [MLP] MLP Evaluation ---")

        with torch.no_grad():
            wp = mlp(condition)
            q_traj, q_dot_traj = physics.generate_trajectory(wp)
            error = physics.simulate_single_rk4(q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])

        err_val = error.item()
        err_deg = np.rad2deg(np.sqrt(err_val))
        print(f"[MLP] MLP Error: {err_val:.6f} ({err_deg:.4f}°)")

        plt.figure()
        plt.plot(q_traj[0].cpu().numpy()[:, 0], label="J1")
        plt.title(f"[MLP] MLP Trajectory (Error: {err_val:.4f})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "mlp_traj.png"))
        print("[MLP] Saved MLP plot.")


if __name__ == "__main__":
    main()


