"""
Neural network models for trajectory generation.

ConditionalDecoder: Latent-conditioned trajectory generator (formerly CVAE)
MLP: Deterministic baseline model
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import math

# Default output range (used when joint_limits not provided): -140 to 140 degrees
OUTPUT_MAX_DEG = 140.0
OUTPUT_MAX_RAD = math.radians(OUTPUT_MAX_DEG)


class ResidualBlock(nn.Module):
    """ResNet-style residual block for fully connected layers."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        return out + residual


class MLP(nn.Module):
    """
    Deterministic baseline model for trajectory generation.

    Input: Condition (start/goal quaternions) -> Output: Waypoints
    Output is scaled by joint_limits or fixed range (-140~140 deg).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        joint_limits: Optional[Tensor] = None,
        hidden_dim: int = 128,
        num_residual_blocks: int = 2
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

        self._setup_joint_limits(joint_limits)

    def _setup_joint_limits(self, joint_limits: Optional[Tensor]) -> None:
        """Setup joint limit scaling parameters."""
        if joint_limits is not None:
            self.register_buffer('joint_limits', joint_limits)
            lower = joint_limits[:, 0]
            upper = joint_limits[:, 1]
            self.register_buffer('scale', (upper - lower) / 2.0)
            self.register_buffer('mid', (upper + lower) / 2.0)
            self.n_q = joint_limits.shape[0]
        else:
            self.joint_limits = None
            self.n_q = None

    def _scale_output(self, x: Tensor) -> Tensor:
        """Scale tanh output to joint limits or default range."""
        if self.joint_limits is not None:
            batch_size = x.size(0)
            x_reshaped = x.view(batch_size, -1, self.n_q)
            x_scaled = self.mid + x_reshaped * self.scale
            return x_scaled.view(batch_size, -1)
        return x * OUTPUT_MAX_RAD

    def forward(self, condition: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            condition: [batch, condition_dim] Start and goal quaternions

        Returns:
            waypoints: [batch, num_waypoints * n_joints] Trajectory waypoints
        """
        x = self.relu(self.input_proj(condition))

        for block in self.residual_blocks:
            x = block(x)

        x = self.tanh(self.output_proj(x))
        return self._scale_output(x)


class ConditionalDecoder(nn.Module):
    """
    Latent-conditioned trajectory generator.

    Generates diverse trajectories based on condition and latent vector z.
    The encoder is only used during training for learning the latent space.
    At inference time, z is sampled from standard normal distribution.

    Note: This model uses only the decoder for physics-based training,
    not the full VAE reconstruction loss with KL divergence.
    """

    def __init__(
        self,
        condition_dim: int,
        output_dim: int,
        latent_dim: int = 8,
        joint_limits: Optional[Tensor] = None,
        hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # Encoder (for potential future use with KL loss)
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_input_proj = nn.Linear(condition_dim + latent_dim, hidden_dim)
        self.decoder_relu = nn.ReLU()

        self.decoder_residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(2)
        ])

        self.decoder_output_proj = nn.Linear(hidden_dim, output_dim)
        self.decoder_tanh = nn.Tanh()

        self._setup_joint_limits(joint_limits)

    def _setup_joint_limits(self, joint_limits: Optional[Tensor]) -> None:
        """Setup joint limit scaling parameters."""
        if joint_limits is not None:
            self.register_buffer('joint_limits', joint_limits)
            lower = joint_limits[:, 0]
            upper = joint_limits[:, 1]
            self.register_buffer('scale', (upper - lower) / 2.0)
            self.register_buffer('mid', (upper + lower) / 2.0)
            self.n_q = joint_limits.shape[0]
        else:
            self.joint_limits = None
            self.n_q = None

    def _scale_output(self, x: Tensor) -> Tensor:
        """Scale tanh output to joint limits or default range."""
        if self.joint_limits is not None:
            batch_size = x.size(0)
            x_reshaped = x.view(batch_size, -1, self.n_q)
            x_scaled = self.mid + x_reshaped * self.scale
            return x_scaled.view(batch_size, -1)
        return x * OUTPUT_MAX_RAD

    def encode(self, condition: Tensor, trajectory: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode condition and trajectory to latent parameters.

        Args:
            condition: [batch, condition_dim]
            trajectory: [batch, output_dim]

        Returns:
            mu: [batch, latent_dim] Mean of latent distribution
            logvar: [batch, latent_dim] Log variance of latent distribution
        """
        x = torch.cat([condition, trajectory], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, condition: Tensor, z: Tensor) -> Tensor:
        """
        Decode condition and latent vector to trajectory.

        Args:
            condition: [batch, condition_dim]
            z: [batch, latent_dim]

        Returns:
            waypoints: [batch, output_dim] Generated trajectory waypoints
        """
        x = torch.cat([condition, z], dim=1)
        x = self.decoder_relu(self.decoder_input_proj(x))

        for block in self.decoder_residual_blocks:
            x = block(x)

        x = self.decoder_tanh(self.decoder_output_proj(x))
        return self._scale_output(x)

    def forward(
        self,
        condition: Tensor,
        trajectory: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass (for training with reconstruction).

        Args:
            condition: [batch, condition_dim]
            trajectory: [batch, output_dim] Ground truth trajectory

        Returns:
            recon_traj: Reconstructed trajectory
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(condition, trajectory)
        z = self.reparameterize(mu, logvar)
        recon_traj = self.decode(condition, z)
        return recon_traj, mu, logvar

    def inference(self, condition: Tensor) -> Tensor:
        """
        Inference mode with random latent sampling.

        Args:
            condition: [batch, condition_dim]

        Returns:
            waypoints: [batch, output_dim] Generated trajectory
        """
        batch_size = condition.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=condition.device)
        return self.decode(condition, z)


# Backward compatibility alias
CVAE = ConditionalDecoder
