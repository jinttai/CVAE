"""
CVAE Training with 9 waypoints / 200 timesteps.

- 9 waypoints → 10 segments (start + 9 wp + end)
- dt = 0.1s, total_time = 20.0s → 200 steps
- 20 timesteps per segment
- output_dim = 9 * 6 = 54
"""
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys
import csv
import math

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from torch.utils.tensorboard import SummaryWriter
from src.models.cvae import CVAE
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    cr = torch.cos(roll / 2); sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2); sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2); sy = torch.sin(yaw / 2)
    return torch.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dim=-1)


def plot_trajectory(q_traj, q_dot_traj, epoch):
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"Joint Angles (Epoch {epoch})")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="right", fontsize="small")

    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CVAE 9-Waypoint Training on {device} ===")

    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    log_dir = os.path.join(ROOT_DIR, "outputs/logs/cvae_9wp")
    writer = SummaryWriter(log_dir=log_dir)

    # ==========================================
    # Parameters: 9 waypoints, 200 timesteps
    # ==========================================
    COND_DIM = 8           # start_quat(4) + goal_quat(4)
    NUM_WAYPOINTS = 9
    N_Q = robot["n_q"]     # 6
    OUTPUT_DIM = NUM_WAYPOINTS * N_Q  # 54
    LATENT_DIM = 3
    TOTAL_TIME = 20.0      # 200 steps * 0.1s = 20s
    # 10 segments, 20 steps each

    BATCH_SIZE = 256
    NUM_EPOCHS = 2000
    LR = 1e-3

    JOINT_SQUARED_WEIGHT = 0.01
    JOINT_CHANGE_WEIGHT = 0.01
    MAX_JOINT_WEIGHT = 0.1

    # Model & Physics
    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM,
                 joint_limits=robot['joint_limits']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    print(f"  Waypoints: {NUM_WAYPOINTS}, Steps: {physics.num_steps}, "
          f"Segments: {physics.num_segments}, Steps/seg: {physics.steps_per_segment}")

    # Fixed test condition for visualization
    fixed_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    max_angle_rad = math.radians(10.0)
    roll = (2 * max_angle_rad) * torch.rand(1, device=device) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(1, device=device) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(1, device=device) - max_angle_rad
    fixed_goal = euler_to_quaternion(roll, pitch, yaw)
    fixed_cond = torch.cat([fixed_start, fixed_goal], dim=1)

    # Training loop
    total_start_time = time.time()
    epoch_start_time = time.time()

    train_losses = []
    val_losses = []
    epoch_durations = []

    for epoch in range(NUM_EPOCHS):
        # Start pose (identity)
        q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(BATCH_SIZE, 1)

        # Random goal (axis-angle, up to 60 deg)
        rand_axis = torch.randn(BATCH_SIZE, 3, device=device)
        rand_axis = rand_axis / torch.norm(rand_axis, dim=1, keepdim=True)
        max_angle = math.radians(60.0)
        rand_theta = torch.rand(BATCH_SIZE, 1, device=device) * max_angle
        half_theta = rand_theta / 2.0
        sin_half = torch.sin(half_theta)
        cos_half = torch.cos(half_theta)
        q_xyz = rand_axis * sin_half
        q_w = cos_half
        q0_goal = torch.cat([q_xyz, q_w], dim=1)

        condition = torch.cat([q0_start, q0_goal], dim=1)

        optimizer.zero_grad()

        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        waypoints_pred = model.decode(condition, z)

        loss, loss_dict = physics.calculate_total_loss(
            waypoints_pred, q0_start, q0_goal,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"   >>> Warning: NaN/Inf at epoch {epoch+1}, skipping")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = loss.item()
        train_losses.append(loss_value)
        writer.add_scalar("Loss/train", loss_value, epoch)

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss_value:.6f} | Time: {epoch_duration:.2f}s")
        epoch_start_time = time.time()

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z_vis = torch.randn(1, LATENT_DIM, device=device)
                wp_vis = model.decode(fixed_cond, z_vis)
                q_traj, q_dot_traj = physics.generate_trajectory(wp_vis)
                val_loss = physics.calculate_loss(wp_vis, fixed_start, fixed_goal)

                val_value = val_loss.item()
                val_losses.append((epoch + 1, val_value))

                fig = plot_trajectory(q_traj[0], q_dot_traj[0], epoch + 1)
                writer.add_figure("Trajectory/Fixed_Goal", fig, epoch)
                plt.close(fig)
                print(f"   >>> Validation Loss: {val_value:.6f}")

    print(f"Training Finished. Total Time: {time.time() - total_start_time:.2f}s")

    # Save training curves
    plots_dir = os.path.join(ROOT_DIR, "outputs/plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(train_losses) > 0:
        epochs = list(range(1, len(train_losses) + 1))

        csv_dir = os.path.join(plots_dir, "cvae_training_curve")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "9wp.csv")

        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["epoch", "train_loss", "epoch_duration", "val_loss"])
            val_dict = {e: v for e, v in val_losses}
            for ep, train_loss, duration in zip(epochs, train_losses, epoch_durations):
                val_loss = val_dict.get(ep, "")
                csv_writer.writerow([ep, train_loss, duration, val_loss])
        print(f"Training data saved to: {csv_path}")

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")
        if len(val_losses) > 0:
            val_epochs = [e for (e, _) in val_losses]
            val_values = [v for (_, v) in val_losses]
            plt.plot(val_epochs, val_values, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CVAE 9-Waypoint Training Curve")
        plt.grid(True)
        plt.legend()
        save_dir = os.path.join(plots_dir, "cvae_training_curve")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "9wp.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Save model weights
    save_dir = os.path.join(ROOT_DIR, "outputs/weights/cvae_9wp")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "v1.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    writer.close()


if __name__ == "__main__":
    main()
