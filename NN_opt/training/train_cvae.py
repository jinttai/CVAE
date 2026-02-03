"""
Training script for ConditionalDecoder (formerly CVAE) model.

Trains a latent-conditioned trajectory generator using physics-based loss.
"""

import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import csv
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# Project imports
from NN_opt.config import (
    ROOT_DIR,
    training_config,
    model_config,
    trajectory_config,
    path_config,
)
from NN_opt.model.cvae import ConditionalDecoder
from NN_opt.training.physics_layer import PhysicsLayer
from NN_opt.utils.logging_utils import TrainingLogger
from physics.dynamics.urdf2robot_torch import urdf2robot
from physics.utils import generate_random_quaternion_from_euler_torch


def plot_trajectory(q_traj: torch.Tensor, q_dot_traj: torch.Tensor, epoch: int):
    """
    Convert trajectory to matplotlib figure for TensorBoard logging.

    Args:
        q_traj: [num_steps, n_joints] Joint positions
        q_dot_traj: [num_steps, n_joints] Joint velocities
        epoch: Current epoch number

    Returns:
        matplotlib Figure object
    """
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Joint Positions
    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"Joint Angles (Epoch {epoch})")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="right", fontsize="small")

    # Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def generate_random_goal_quaternion(batch_size: int, max_angle_deg: float, device: str):
    """
    Generate random goal quaternions using axis-angle representation.

    Args:
        batch_size: Number of quaternions to generate
        max_angle_deg: Maximum rotation angle in degrees
        device: Torch device

    Returns:
        Quaternions tensor [batch_size, 4]
    """
    # Random rotation axis (unit vector)
    rand_axis = torch.randn(batch_size, 3, device=device)
    rand_axis = rand_axis / torch.norm(rand_axis, dim=1, keepdim=True)

    # Random rotation angle (0 to max_angle)
    max_angle = math.radians(max_angle_deg)
    rand_theta = torch.rand(batch_size, 1, device=device) * max_angle

    # Axis-Angle to Quaternion: q = [sin(θ/2)*axis, cos(θ/2)]
    half_theta = rand_theta / 2.0
    q_xyz = rand_axis * torch.sin(half_theta)
    q_w = torch.cos(half_theta)

    return torch.cat([q_xyz, q_w], dim=1)


def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = TrainingLogger(__name__)
    logger.log_training_start(device, "ConditionalDecoder")

    # Ensure output directories exist
    path_config.ensure_dirs()

    # Load robot
    urdf_path = ROOT_DIR / "assets/SC_ur10e.urdf"
    robot, _ = urdf2robot(str(urdf_path), verbose_flag=False, device=device)

    # TensorBoard writer
    log_dir = path_config.log_dir / "decoder_v1"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Model configuration
    output_dim = trajectory_config.num_waypoints * robot["n_q"]

    # Initialize model
    model = ConditionalDecoder(
        condition_dim=model_config.condition_dim,
        output_dim=output_dim,
        latent_dim=model_config.latent_dim,
        joint_limits=robot['joint_limits'],
        hidden_dim=model_config.hidden_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)

    # Physics layer
    physics = PhysicsLayer(
        robot,
        trajectory_config.num_waypoints,
        trajectory_config.total_time,
        device
    )

    # Fixed test condition for validation
    fixed_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    fixed_goal = generate_random_quaternion_from_euler_torch(
        1, max_angle_deg=10.0, device=device
    )
    fixed_cond = torch.cat([fixed_start, fixed_goal], dim=1)

    # Training loop
    total_start_time = time.time()
    epoch_start_time = time.time()

    train_losses = []
    val_losses = []
    epoch_durations = []

    for epoch in range(training_config.num_epochs):
        # Generate training data
        q0_start = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], device=device
        ).repeat(training_config.batch_size, 1)

        q0_goal = generate_random_goal_quaternion(
            training_config.batch_size,
            training_config.max_goal_angle_deg,
            device
        )

        condition = torch.cat([q0_start, q0_goal], dim=1)

        # Forward pass
        optimizer.zero_grad()
        z = torch.randn(
            training_config.batch_size,
            model_config.latent_dim,
            device=device
        )
        waypoints_pred = model.decode(condition, z)

        # Calculate loss
        loss, loss_dict = physics.calculate_total_loss(
            waypoints_pred, q0_start, q0_goal,
            joint_squared_weight=training_config.joint_squared_weight,
            joint_change_weight=training_config.joint_change_weight,
            max_joint_weight=training_config.max_joint_weight
        )

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.log_nan_detected(epoch + 1)
            optimizer.zero_grad()
            continue

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=training_config.max_grad_norm
        )
        optimizer.step()

        # Logging
        loss_value = loss.item()
        train_losses.append(loss_value)
        writer.add_scalar("Loss/train", loss_value, epoch)

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        logger.log_epoch(
            epoch + 1,
            training_config.num_epochs,
            loss_value,
            epoch_duration
        )
        epoch_start_time = time.time()

        # Validation
        if (epoch + 1) % training_config.val_interval == 0:
            with torch.no_grad():
                z_vis = torch.randn(1, model_config.latent_dim, device=device)
                wp_vis = model.decode(fixed_cond, z_vis)

                q_traj, q_dot_traj = physics.generate_trajectory(wp_vis)
                val_loss = physics.calculate_loss(wp_vis, fixed_start, fixed_goal)

                val_value = val_loss.item()
                val_losses.append((epoch + 1, val_value))

                fig = plot_trajectory(q_traj[0], q_dot_traj[0], epoch + 1)
                writer.add_figure("Trajectory/Fixed_Goal", fig, epoch)
                plt.close(fig)

                logger.log_validation(epoch + 1, val_value)

    total_time = time.time() - total_start_time
    logger.log_training_end(total_time)

    # Save training curves
    if len(train_losses) > 0:
        epochs = list(range(1, len(train_losses) + 1))

        csv_dir = path_config.plot_dir / "decoder_training_curve"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "v1.csv"

        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["epoch", "train_loss", "epoch_duration", "val_loss"])

            val_dict = {e: v for e, v in val_losses}
            for ep, train_loss, duration in zip(epochs, train_losses, epoch_durations):
                val_loss = val_dict.get(ep, "")
                csv_writer.writerow([ep, train_loss, duration, val_loss])

        # Plot training curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")

        if len(val_losses) > 0:
            val_epochs = [e for (e, _) in val_losses]
            val_values = [v for (_, v) in val_losses]
            plt.plot(val_epochs, val_values, label="Val Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ConditionalDecoder Training Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(csv_dir / "v1.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Save model
    weight_dir = path_config.decoder_weight_dir
    weight_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_dir / "v1.pth")

    writer.close()


if __name__ == "__main__":
    main()
