import torch
import torch.nn as nn
from src.models.cvae import ResidualBlock


class PolicyNetwork(nn.Module):
    """
    Gaussian policy for single-step bandit: condition -> waypoints.
    Output is bounded by joint limits via Tanh scaling.
    """

    def __init__(self, condition_dim, action_dim, joint_limits, hidden_dim=256, num_residual_blocks=2):
        super().__init__()
        self.action_dim = action_dim

        # Joint limits for Tanh scaling (same pattern as CVAE/MLP)
        n_q = joint_limits.shape[0]
        self.n_q = n_q
        self.register_buffer('joint_limits', joint_limits)
        lower = joint_limits[:, 0]
        upper = joint_limits[:, 1]
        self.register_buffer('scale', (upper - lower) / 2.0)
        self.register_buffer('mid', (upper + lower) / 2.0)

        # Shared backbone
        self.input_proj = nn.Linear(condition_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])

        # Mean head
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

        # State-independent learnable log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, condition):
        """Return raw mean (before tanh scaling) — mainly for debugging."""
        x = self.relu(self.input_proj(condition))
        for block in self.residual_blocks:
            x = block(x)
        return self.mean_head(x)

    def _scale_action(self, tanh_out):
        """Scale tanh output [-1,1] to joint limits, per-joint."""
        batch_size = tanh_out.size(0)
        x = tanh_out.view(batch_size, -1, self.n_q)
        x = self.mid + x * self.scale
        return x.view(batch_size, -1)

    def sample(self, condition):
        """Sample action from Gaussian, squash through tanh, scale to joint limits."""
        x = self.relu(self.input_proj(condition))
        for block in self.residual_blocks:
            x = block(x)
        mean_raw = self.mean_head(x)  # unbounded

        std = torch.exp(self.log_std).expand_as(mean_raw)
        noise = torch.randn_like(mean_raw)
        raw_action = mean_raw + std * noise  # Gaussian sample (unbounded)

        # Squash through tanh
        tanh_action = torch.tanh(raw_action)
        action = self._scale_action(tanh_action)

        # Log prob with tanh correction
        # log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
        var = std ** 2
        log_prob = -0.5 * ((raw_action - mean_raw) ** 2 / var + 2 * self.log_std + 1.8378770664093453)
        # Tanh squashing correction
        log_prob = log_prob - torch.log(1 - tanh_action ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # [batch]

        return action, log_prob

    def evaluate_actions(self, condition, actions):
        """Compute log_prob and entropy for given actions (used in PPO update)."""
        x = self.relu(self.input_proj(condition))
        for block in self.residual_blocks:
            x = block(x)
        mean_raw = self.mean_head(x)
        std = torch.exp(self.log_std).expand_as(mean_raw)

        # Invert scaling: action -> tanh_out -> raw_action
        batch_size = actions.size(0)
        a_reshaped = actions.view(batch_size, -1, self.n_q)
        tanh_action = (a_reshaped - self.mid) / self.scale  # [-1, 1]
        tanh_action = tanh_action.view(batch_size, -1)
        # Clamp to avoid atanh explosion at boundaries
        tanh_action = torch.clamp(tanh_action, -0.999, 0.999)
        raw_action = torch.atanh(tanh_action)

        # Log prob
        var = std ** 2
        log_prob = -0.5 * ((raw_action - mean_raw) ** 2 / var + 2 * self.log_std + 1.8378770664093453)
        log_prob = log_prob - torch.log(1 - tanh_action ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        # Entropy of the Gaussian (before squashing — standard approximation)
        entropy = 0.5 * (1 + 2 * self.log_std + 1.8378770664093453)
        entropy = entropy.sum()  # scalar

        return log_prob, entropy


class ValueNetwork(nn.Module):
    """State value function V(condition)."""

    def __init__(self, condition_dim, hidden_dim=256, num_residual_blocks=2):
        super().__init__()
        self.input_proj = nn.Linear(condition_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, condition):
        x = self.relu(self.input_proj(condition))
        for block in self.residual_blocks:
            x = block(x)
        return self.value_head(x).squeeze(-1)  # [batch]
