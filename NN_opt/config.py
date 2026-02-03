"""
Configuration for NN-based trajectory optimization.

Centralized configuration to avoid hardcoded values across files.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent


@dataclass
class RobotConfig:
    """Robot configuration."""
    urdf_path: str = str(ROOT_DIR / "assets/SC_ur10e.urdf")
    n_joints: int = 6


@dataclass
class TrajectoryConfig:
    """Trajectory generation configuration."""
    num_waypoints: int = 3
    total_time: float = 10.0
    dt: float = 0.1

    @property
    def num_steps(self) -> int:
        return int(self.total_time / self.dt)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 256
    num_epochs: int = 2000
    learning_rate: float = 1e-3

    # Loss weights
    joint_squared_weight: float = 0.01
    joint_change_weight: float = 0.01
    max_joint_weight: float = 0.1

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Validation
    val_interval: int = 10

    # Goal angle range (degrees)
    max_goal_angle_deg: float = 60.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Common
    condition_dim: int = 8  # start_quat(4) + goal_quat(4)

    # ConditionalDecoder (formerly CVAE)
    latent_dim: int = 3
    hidden_dim: int = 256

    # MLP baseline
    mlp_hidden_dim: int = 128


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    num_samples: int = 1024

    # LBFGS parameters
    lbfgs_lr: float = 1e-3
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 10

    # Loss weights (same as training for consistency)
    joint_squared_weight: float = 0.01
    joint_change_weight: float = 0.01
    max_joint_weight: float = 0.1


@dataclass
class PathConfig:
    """Path configuration for outputs."""
    # Weights
    weight_dir: Path = field(default_factory=lambda: ROOT_DIR / "NN_opt/weight")
    decoder_weight_dir: Path = field(default_factory=lambda: ROOT_DIR / "NN_opt/weight/decoder_debug")
    mlp_weight_dir: Path = field(default_factory=lambda: ROOT_DIR / "NN_opt/weight/mlp_debug")

    # Results
    result_dir: Path = field(default_factory=lambda: ROOT_DIR / "NN_opt/result")

    # Logs
    log_dir: Path = field(default_factory=lambda: ROOT_DIR / "NN_opt/logs")

    # Plots
    plot_dir: Path = field(default_factory=lambda: ROOT_DIR / "NN_opt/plots")

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for path in [self.weight_dir, self.decoder_weight_dir, self.mlp_weight_dir,
                     self.result_dir, self.log_dir, self.plot_dir]:
            path.mkdir(parents=True, exist_ok=True)


# Default configurations
robot_config = RobotConfig()
trajectory_config = TrajectoryConfig()
training_config = TrainingConfig()
model_config = ModelConfig()
optimization_config = OptimizationConfig()
path_config = PathConfig()
