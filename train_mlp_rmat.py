import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import csv
import math

from torch.utils.tensorboard import SummaryWriter

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.models.cvae import MLP
from src.training.physics_layer import PhysicsLayer   # Rmat 버전
from src.dynamics.urdf2robot_torch import urdf2robot


def plot_trajectory(q_traj, q_dot_traj, epoch):
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"[Rmat] MLP Joint Angles (Epoch {epoch})")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="right", fontsize="small")

    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("[Rmat] MLP Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== MLP (Baseline, Rmat Physics) Training Start on {device} ===")

    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)

    writer = SummaryWriter(log_dir="runs/mlp_rmat_v2")

    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]

    BATCH_SIZE = 512
    TOTAL_TIME = 1.0
    NUM_EPOCHS = 10000

    model = MLP(COND_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    fixed_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    fixed_goal = torch.tensor([[0.0, 0.0, 0.7071, 0.7071]], device=device)
    fixed_cond = torch.cat([fixed_start, fixed_goal], dim=1)

    total_start_time = time.time()
    epoch_start_time = time.time()

    train_losses = []
    val_losses = []
    epoch_durations = []

    for epoch in range(NUM_EPOCHS):
        # --- Training Step ---
        
        # 1. 시작 자세 (Identity Fixed)
        q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(BATCH_SIZE, 1)
        
        # 2. 목표 자세 (Random Axis + Angle Limit 60 deg)
        # (1) 랜덤 회전축 생성 (Unit Vector)
        rand_axis = torch.randn(BATCH_SIZE, 3, device=device)
        rand_axis = rand_axis / torch.norm(rand_axis, dim=1, keepdim=True)
        
        # (2) 회전 각도 생성 (0 ~ 60도)
        # math.radians(60) = 1.0472 rad
        max_angle = math.radians(60.0)
        rand_theta = torch.rand(BATCH_SIZE, 1, device=device) * max_angle
        
        # (3) Axis-Angle -> Quaternion 변환 [x, y, z, w]
        # q = [sin(theta/2)*ux, sin(theta/2)*uy, sin(theta/2)*uz, cos(theta/2)]
        half_theta = rand_theta / 2.0
        sin_half = torch.sin(half_theta)
        cos_half = torch.cos(half_theta)
        
        q_xyz = rand_axis * sin_half
        q_w = cos_half
        
        q0_goal = torch.cat([q_xyz, q_w], dim=1) # [B, 4]

        condition = torch.cat([q0_start, q0_goal], dim=1)

        optimizer.zero_grad()

        waypoints_pred = model(condition)

        loss = physics.calculate_loss(waypoints_pred, q0_start, q0_goal)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        train_losses.append(loss_value)
        writer.add_scalar("Loss/train", loss_value, epoch)

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        print(
            f"[Rmat] MLP Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss_value:.6f} | Time: {epoch_duration:.2f}s"
        )
        epoch_start_time = time.time()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                wp_vis = model(fixed_cond)
                q_traj, q_dot_traj = physics.generate_trajectory(wp_vis)
                val_loss = physics.calculate_loss(wp_vis, fixed_start, fixed_goal)
                val_value = val_loss.item()
                val_losses.append((epoch + 1, val_value))

                fig = plot_trajectory(q_traj[0], q_dot_traj[0], epoch + 1)
                writer.add_figure("[Rmat] Trajectory/Fixed_Goal", fig, epoch)
                plt.close(fig)

                print(f"   >>> [Rmat] Validation Loss: {val_value:.6f}")

    print(f"[Rmat] Training Finished. Total Time: {time.time() - total_start_time:.2f}s")

    plots_dir = "plots_rmat"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if len(train_losses) > 0:
        epochs = list(range(1, len(train_losses) + 1))

        csv_dir = os.path.join(plots_dir, "mlp_training_curve_rmat")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_path = os.path.join(csv_dir, "v2.csv")

        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["epoch", "train_loss", "epoch_duration", "val_loss"])

            val_dict = {e: v for e, v in val_losses}

            for epoch, train_loss, duration in zip(epochs, train_losses, epoch_durations):
                val_loss = val_dict.get(epoch, "")
                csv_writer.writerow([epoch, train_loss, duration, val_loss])

        print(f"[Rmat] Training data saved to: {csv_path}")

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss (Rmat)")

        if len(val_losses) > 0:
            val_epochs = [e for (e, _) in val_losses]
            val_values = [v for (_, v) in val_losses]
            plt.plot(val_epochs, val_values, label="Val Loss (Rmat)")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MLP Training Curve (Rmat Physics)")
        plt.grid(True)
        plt.legend()
        save_dir = os.path.join(plots_dir, "mlp_training_curve_rmat")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "v2.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    save_dir = os.path.join("weights", "mlp_rmat_debug")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "v2.pth")
    torch.save(model.state_dict(), save_path)
    writer.close()


if __name__ == "__main__":
    main()


