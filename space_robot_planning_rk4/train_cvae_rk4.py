import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys
import csv
import math

# === 경로 설정 ===
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../space_robot_planning_rk4
project_root = os.path.dirname(current_dir)               # .../CVAE

rk4_src_dir = os.path.join(current_dir, "src")
if rk4_src_dir not in sys.path:
    sys.path.append(rk4_src_dir)

orig_src_dir = os.path.join(project_root, "space_robot_planning", "src")
if orig_src_dir not in sys.path:
    sys.path.append(orig_src_dir)

from torch.utils.tensorboard import SummaryWriter

from models.cvae import CVAE
from training.physics_layer import PhysicsLayer          # RK4 래퍼
from dynamics.urdf2robot_torch import urdf2robot


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    Using ZYX convention (yaw around Z, pitch around Y, roll around X)
    """
    # Half angles
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    
    # Quaternion components
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    
    return torch.stack([qx, qy, qz, qw], dim=-1)


def generate_random_quaternion_from_euler(batch_size, max_angle_deg=10.0, device='cpu'):
    """
    Generate random quaternions from Euler angles within specified range
    Args:
        batch_size: Number of quaternions to generate
        max_angle_deg: Maximum angle in degrees for each Euler angle (default: 10 degrees)
        device: Device to create tensors on
    Returns:
        quaternions: [batch_size, 4] tensor of quaternions (x, y, z, w)
    """
    max_angle_rad = math.radians(max_angle_deg)
    
    # Generate random Euler angles in [-max_angle_deg, max_angle_deg]
    # Using torch.rand to generate uniform distribution in [0, 1], then scale to [-max, max]
    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    
    # Convert to quaternion
    quaternions = euler_to_quaternion(roll, pitch, yaw)
    
    return quaternions


def plot_trajectory(q_traj, q_dot_traj, epoch):
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"[RK4] Joint Angles (Epoch {epoch})")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="right", fontsize="small")

    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("[RK4] Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CVAE Training (Quaternion RK4 Physics) Start on {device} ===")

    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)

    writer = SummaryWriter(log_dir="runs/cvae_rk4_v1")

    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 8

    BATCH_SIZE = 512
    TOTAL_TIME = 1.0
    NUM_EPOCHS = 15000

    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)  # RK4 loss 사용

    fixed_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    fixed_goal = torch.tensor([[0.0, 0.0, 0.7071, 0.7071]], device=device)
    fixed_cond = torch.cat([fixed_start, fixed_goal], dim=1)

    total_start_time = time.time()
    epoch_start_time = time.time()

    train_losses = []
    val_losses = []
    epoch_durations = []

    for epoch in range(NUM_EPOCHS):
        q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(BATCH_SIZE, 1)
        # Generate random quaternions from Euler angles within ±10 degrees
        q0_goal = generate_random_quaternion_from_euler(BATCH_SIZE, max_angle_deg=10.0, device=device)

        condition = torch.cat([q0_start, q0_goal], dim=1)

        optimizer.zero_grad()

        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        waypoints_pred = model.decode(condition, z)

        # === 여기서부터 loss 는 RK4 기반 PhysicsLayer.calculate_loss 를 사용 ===
        loss = physics.calculate_loss(waypoints_pred, q0_start, q0_goal)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        train_losses.append(loss_value)
        writer.add_scalar("Loss/train", loss_value, epoch)

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        print(f"[RK4] Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss_value:.6f} | Time: {epoch_duration:.2f}s")
        epoch_start_time = time.time()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z_vis = torch.randn(1, LATENT_DIM, device=device)
                wp_vis = model.decode(fixed_cond, z_vis)

                q_traj, q_dot_traj = physics.generate_trajectory(wp_vis)
                val_loss = physics.calculate_loss(wp_vis, fixed_start, fixed_goal)

                val_value = val_loss.item()
                val_losses.append((epoch + 1, val_value))

                fig = plot_trajectory(q_traj[0], q_dot_traj[0], epoch + 1)
                writer.add_figure("[RK4] Trajectory/Fixed_Goal", fig, epoch)
                plt.close(fig)

                print(f"   >>> [RK4] Validation Loss: {val_value:.6f}")

    print(f"[RK4] Training Finished. Total Time: {time.time() - total_start_time:.2f}s")

    # === 학습 곡선 저장 ===
    plots_dir = "plots_rk4"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if len(train_losses) > 0:
        epochs = list(range(1, len(train_losses) + 1))

        csv_dir = os.path.join(plots_dir, "cvae_training_curve_rk4")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "v1.csv")

        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["epoch", "train_loss", "epoch_duration", "val_loss"])

            val_dict = {e: v for e, v in val_losses}

            for epoch, train_loss, duration in zip(epochs, train_losses, epoch_durations):
                val_loss = val_dict.get(epoch, "")
                csv_writer.writerow([epoch, train_loss, duration, val_loss])

        print(f"[RK4] Training data saved to: {csv_path}")

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss (RK4)")

        if len(val_losses) > 0:
            val_epochs = [e for (e, _) in val_losses]
            val_values = [v for (_, v) in val_losses]
            plt.plot(val_epochs, val_values, label="Val Loss (RK4)")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CVAE Training Curve (Quaternion RK4 Physics)")
        plt.grid(True)
        plt.legend()
        save_dir = os.path.join(plots_dir, "cvae_training_curve_rk4")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "v1.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 모델 저장
    save_dir = os.path.join("weights", "cvae_rk4_debug")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "v1.pth")
    torch.save(model.state_dict(), save_path)
    writer.close()


if __name__ == "__main__":
    main()


