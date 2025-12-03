import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 출력 범위: -140도 ~ 140도 (라디안으로 변환)
OUTPUT_MAX_DEG = 140.0
OUTPUT_MAX_RAD = math.radians(OUTPUT_MAX_DEG)  # 약 2.4435 라디안


class ResidualBlock(nn.Module):
    """
    ResNet-style residual block for fully connected layers
    """
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        # Skip connection
        out = out + residual
        return out


class MLP(nn.Module):
    """
    [Baseline] Simple Deterministic Policy with ResNet structure
    Input: Condition (Start/Goal Base Pose) -> Output: Waypoints
    Output is limited to -140deg ~ 140deg using tanh activation
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_residual_blocks=2):
        super(MLP, self).__init__()
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, condition):
        # Initial projection
        x = self.input_proj(condition)
        x = self.relu(x)
        
        # Residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x)
        
        # Output projection with tanh
        x = self.output_proj(x)
        x = self.tanh(x)
        
        # tanh 출력 (-1 ~ 1)을 -140deg ~ 140deg 범위로 스케일링
        return x * OUTPUT_MAX_RAD

class CVAE(nn.Module):
    """
    [Proposed] Conditional Variational Autoencoder
    Generates diverse trajectories based on condition and latent z.
    """
    def __init__(self, condition_dim, output_dim, latent_dim=8, hidden_dim=256):
        super(CVAE, self).__init__()
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # --- Encoder (Training Only) ---
        # Input: Condition + Ground Truth Trajectory
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder (Inference / Generator) ---
        # Input: Condition + Latent z
        # Output is limited to -140deg ~ 140deg using tanh activation
        # ResNet structure with residual blocks
        self.decoder_input_proj = nn.Linear(condition_dim + latent_dim, hidden_dim)
        self.decoder_relu = nn.ReLU()
        
        # Residual blocks for decoder
        num_decoder_blocks = 2
        self.decoder_residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_decoder_blocks)
        ])
        
        # Output projection
        self.decoder_output_proj = nn.Linear(hidden_dim, output_dim)
        self.decoder_tanh = nn.Tanh()

    def encode(self, condition, trajectory):
        x = torch.cat([condition, trajectory], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, condition, z):
        x = torch.cat([condition, z], dim=1)
        
        # Initial projection
        x = self.decoder_input_proj(x)
        x = self.decoder_relu(x)
        
        # Residual blocks
        for res_block in self.decoder_residual_blocks:
            x = res_block(x)
        
        # Output projection with tanh
        x = self.decoder_output_proj(x)
        x = self.decoder_tanh(x)
        
        # tanh 출력 (-1 ~ 1)을 -140deg ~ 140deg 범위로 스케일링
        return x * OUTPUT_MAX_RAD

    def forward(self, condition, trajectory):
        # Forward pass for training (reconstruction)
        mu, logvar = self.encode(condition, trajectory)
        z = self.reparameterize(mu, logvar)
        recon_traj = self.decode(condition, z)
        return recon_traj, mu, logvar

    def inference(self, condition):
        # Inference mode (random sampling)
        batch_size = condition.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=condition.device)
        return self.decode(condition, z)