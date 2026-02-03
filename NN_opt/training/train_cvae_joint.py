import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys
import csv
import math

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from torch.utils.tensorboard import SummaryWriter

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from NN_opt.model.cvae import CVAE
from NN_opt.training.physics_layer import PhysicsLayer   # default
from physics.dynamics.urdf2robot_torch import urdf2robot
from physics.utils import euler_to_quaternion_torch as euler_to_quaternion


def generate_random_quaternion_from_euler(batch_size, max_angle_deg=30.0, device='cpu'):
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


# --- 시각화 헬퍼 함수 ---
def plot_trajectory(q_traj, q_dot_traj, epoch):
    """
    생성된 궤적을 Matplotlib 그림으로 변환하여 TensorBoard에 기록
    """
    q_traj = q_traj.detach().cpu().numpy()  # [Steps, Joints]
    q_dot_traj = q_dot_traj.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # 1. Joint Positions
    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"Joint Angles (Epoch {epoch})")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="right", fontsize="small")

    # 2. Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def main():
    # 1. 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CVAE Training Start on {device} ===")

    # 로봇 로드 (원본 dynamics 사용)
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    # TensorBoard Writer (로그 디렉토리)
    log_dir = os.path.join(ROOT_DIR, "outputs/logs/cvae_joint_v1")
    writer = SummaryWriter(log_dir=log_dir)

    # ==========================================
    # [중요] 파라미터 설정 (Joint-based training)
    # ==========================================
    # Condition: start_joint (6) + desired_joint (6) + desired_quaternion (4) = 16
    COND_DIM = robot["n_q"] + robot["n_q"] + 4  # 6 + 6 + 4 = 16
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 3

    BATCH_SIZE = 256
    TOTAL_TIME = 10.0
    NUM_EPOCHS = 10000  
    
    # Joint angle range: -140deg to 140deg
    JOINT_MIN_DEG = -140.0
    JOINT_MAX_DEG = 140.0
    JOINT_MIN_RAD = math.radians(JOINT_MIN_DEG)
    JOINT_MAX_RAD = math.radians(JOINT_MAX_DEG)
    
    # Joint regularization weights
    JOINT_SQUARED_WEIGHT = 0.01  # Weight for mean of joint^2 regularization
    JOINT_CHANGE_WEIGHT = 0.01  # Weight for joint change penalty between consecutive waypoints
    MAX_JOINT_WEIGHT = 0.1  # Weight for maximum joint angle penalty

    # 2. 모델 및 물리 엔진 준비
    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM, joint_limits=robot['joint_limits']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    # 시각화용 고정 테스트 셋 (관절 각도)
    fixed_start_joint = torch.zeros(1, robot["n_q"], device=device)  # [1, n_q]
    fixed_goal_joint = (torch.rand(1, robot["n_q"], device=device) * (JOINT_MAX_RAD - JOINT_MIN_RAD) + JOINT_MIN_RAD)  # [1, n_q]
    # 시각화용 quaternion (condition용)
    fixed_goal_quat = generate_random_quaternion_from_euler(1, max_angle_deg=10.0, device=device)
    # Condition: start_joint + desired_joint + desired_quaternion
    fixed_cond = torch.cat([fixed_start_joint, fixed_goal_joint, fixed_goal_quat], dim=1)  # [1, 16]

    # 3. 학습 루프
    total_start_time = time.time()
    epoch_start_time = time.time()

    train_losses = []
    val_losses = []
    epoch_durations = []

    for epoch in range(NUM_EPOCHS):
        # --- Training Step ---
        
        # 1. 시작 관절 각도 (-140deg ~ 140deg)
        q_start_joint = (torch.rand(BATCH_SIZE, robot["n_q"], device=device) * 
                        (JOINT_MAX_RAD - JOINT_MIN_RAD) + JOINT_MIN_RAD)  # [B, n_q]
        
        # 2. 목표 관절 각도 (-140deg ~ 140deg)
        q_goal_joint = (torch.rand(BATCH_SIZE, robot["n_q"], device=device) * 
                        (JOINT_MAX_RAD - JOINT_MIN_RAD) + JOINT_MIN_RAD)  # [B, n_q]
        
        # 3. 목표 자세 quaternion (Random Axis + Angle Limit 10 deg)
        rand_axis = torch.randn(BATCH_SIZE, 3, device=device)
        rand_axis = rand_axis / torch.norm(rand_axis, dim=1, keepdim=True)
        max_angle = math.radians(10.0)  # Changed from 60.0 to 10.0
        rand_theta = torch.rand(BATCH_SIZE, 1, device=device) * max_angle
        half_theta = rand_theta / 2.0
        sin_half = torch.sin(half_theta)
        cos_half = torch.cos(half_theta)
        q_xyz = rand_axis * sin_half
        q_w = cos_half
        q0_goal = torch.cat([q_xyz, q_w], dim=1)  # [B, 4]

        # Condition: start_joint + desired_joint + desired_quaternion
        condition = torch.cat([q_start_joint, q_goal_joint, q0_goal], dim=1)  # [B, 16]

        # Physics loss 계산을 위한 quaternion (identity start)
        q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(BATCH_SIZE, 1)

        optimizer.zero_grad()

        # Inference (Decoder Only for Physics Loss)
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        waypoints_pred = model.decode(condition, z)

        # Total loss calculation (physics loss + penalties) from PhysicsLayer
        # Note: q0_start and q0_goal are quaternions for physics loss, but we use joint angles for trajectory generation
        loss, loss_dict = physics.calculate_total_loss(
            waypoints_pred, q0_start, q0_goal,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            q_start_joint=q_start_joint,
            q_end_joint=q_goal_joint
        )

        # Check for NaN and skip update if found
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"   >>> Warning: NaN/Inf loss detected at epoch {epoch+1}, skipping update")
            optimizer.zero_grad()
            continue

        loss.backward()
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = loss.item()
        train_losses.append(loss_value)
        writer.add_scalar("Loss/train", loss_value, epoch)

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss_value:.6f} | Time: {epoch_duration:.2f}s"
        )
        epoch_start_time = time.time()

        # --- Validation & Visualization (10 에폭마다) ---
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z_vis = torch.randn(1, LATENT_DIM, device=device)
                wp_vis = model.decode(fixed_cond, z_vis)

                q_traj, q_dot_traj = physics.generate_trajectory(
                    wp_vis, 
                    q_start=fixed_start_joint, 
                    q_end=fixed_goal_joint
                )
                # Validation loss 계산을 위한 quaternion (identity start, goal quaternion)
                fixed_start_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
                val_loss = physics.calculate_loss(wp_vis, fixed_start_quat, fixed_goal_quat)

                val_value = val_loss.item()
                val_losses.append((epoch + 1, val_value))

                fig = plot_trajectory(q_traj[0], q_dot_traj[0], epoch + 1)
                writer.add_figure("Trajectory/Fixed_Goal", fig, epoch)
                plt.close(fig)

                print(f"   >>> Validation Loss: {val_value:.6f}")

    print(f"Training Finished. Total Time: {time.time() - total_start_time:.2f}s")

    # === 학습 곡선 저장 (전용 디렉토리) ===
    plots_dir = os.path.join(ROOT_DIR, "outputs/plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if len(train_losses) > 0:
        epochs = list(range(1, len(train_losses) + 1))

        csv_dir = os.path.join(plots_dir, "cvae_joint_training_curve")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_path = os.path.join(csv_dir, "v1.csv")

        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["epoch", "train_loss", "epoch_duration", "val_loss"])

            val_dict = {e: v for e, v in val_losses}

            for epoch, train_loss, duration in zip(epochs, train_losses, epoch_durations):
                val_loss = val_dict.get(epoch, "")
                csv_writer.writerow([epoch, train_loss, duration, val_loss])

        print(f"Training data saved to: {csv_path}")

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")

        if len(val_losses) > 0:
            val_epochs = [e for (e, _) in val_losses]
            val_values = [v for (_, v) in val_losses]
            plt.plot(val_epochs, val_values, label="Val Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CVAE Training Curve")
        plt.grid(True)
        plt.legend()
        save_dir = os.path.join(plots_dir, "cvae_joint_training_curve")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "v1.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 모델 저장 (전용 디렉토리)
    save_dir = os.path.join(ROOT_DIR, "outputs/weights/cvae_joint")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "v1.pth")
    torch.save(model.state_dict(), save_path)
    writer.close()


if __name__ == "__main__":
    main()


