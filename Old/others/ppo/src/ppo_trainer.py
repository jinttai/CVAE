import torch
import torch.nn as nn
import math


class RunningMeanStd:
    """Track running mean and variance for reward normalization."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        """Update with a batch of values (1D tensor)."""
        batch_mean = x.mean().item()
        batch_var = x.var().item()
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (math.sqrt(self.var) + 1e-8)


class PPOTrainer:
    """PPO trainer for single-step bandit (condition -> waypoints -> reward)."""

    def __init__(
        self,
        policy,
        value_net,
        physics,
        robot,
        device,
        policy_lr=3e-4,
        value_lr=1e-3,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        value_loss_coeff=0.5,
        ppo_epochs=4,
        num_minibatches=4,
        max_grad_norm=0.5,
        joint_squared_weight=0.01,
        joint_change_weight=0.01,
        max_joint_weight=0.1,
    ):
        self.policy = policy
        self.value_net = value_net
        self.physics = physics
        self.robot = robot
        self.device = device

        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)

        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.max_grad_norm = max_grad_norm

        # Physics loss weights
        self.joint_squared_weight = joint_squared_weight
        self.joint_change_weight = joint_change_weight
        self.max_joint_weight = max_joint_weight

        # Reward normalization
        self.reward_rms = RunningMeanStd()

        # Joint angle range (same as train_cvae_joint.py)
        self.joint_min_rad = math.radians(-140.0)
        self.joint_max_rad = math.radians(140.0)

    def _generate_conditions(self, batch_size):
        """Generate random conditions (same pattern as train_cvae_joint.py)."""
        n_q = self.robot["n_q"]

        # Random start joint angles
        q_start_joint = (
            torch.rand(batch_size, n_q, device=self.device)
            * (self.joint_max_rad - self.joint_min_rad)
            + self.joint_min_rad
        )

        # Random goal joint angles
        q_goal_joint = (
            torch.rand(batch_size, n_q, device=self.device)
            * (self.joint_max_rad - self.joint_min_rad)
            + self.joint_min_rad
        )

        # Random goal quaternion (axis-angle, max 10 deg)
        rand_axis = torch.randn(batch_size, 3, device=self.device)
        rand_axis = rand_axis / torch.norm(rand_axis, dim=1, keepdim=True)
        max_angle = math.radians(10.0)
        rand_theta = torch.rand(batch_size, 1, device=self.device) * max_angle
        half_theta = rand_theta / 2.0
        q_xyz = rand_axis * torch.sin(half_theta)
        q_w = torch.cos(half_theta)
        q0_goal = torch.cat([q_xyz, q_w], dim=1)  # [B, 4]

        # Condition vector
        condition = torch.cat([q_start_joint, q_goal_joint, q0_goal], dim=1)  # [B, 16]

        # Identity start quaternion for physics
        q0_start = torch.zeros(batch_size, 4, device=self.device)
        q0_start[:, 3] = 1.0

        return condition, q_start_joint, q_goal_joint, q0_start, q0_goal

    @torch.no_grad()
    def collect_rollouts(self, batch_size):
        """Collect a batch of (condition, action, log_prob, reward, value)."""
        self.policy.eval()
        self.value_net.eval()

        condition, q_start_joint, q_goal_joint, q0_start, q0_goal = self._generate_conditions(batch_size)

        # Sample actions from policy
        actions, log_probs = self.policy.sample(condition)

        # Compute reward = -total_loss (per sample)
        total_loss, loss_dict = self.physics.calculate_total_loss(
            actions, q0_start, q0_goal,
            joint_squared_weight=self.joint_squared_weight,
            joint_change_weight=self.joint_change_weight,
            max_joint_weight=self.max_joint_weight,
            return_mean=False,
            q_start_joint=q_start_joint,
            q_end_joint=q_goal_joint,
        )
        rewards = -total_loss  # [batch_size]

        # Filter NaN samples
        valid_mask = ~(torch.isnan(rewards) | torch.isinf(rewards))
        if valid_mask.sum() == 0:
            return None  # All samples are NaN

        condition = condition[valid_mask]
        actions = actions[valid_mask]
        log_probs = log_probs[valid_mask]
        rewards = rewards[valid_mask]

        # Update reward normalization stats
        self.reward_rms.update(rewards)

        # Value estimates
        values = self.value_net(condition)

        # Advantages (normalized reward - value baseline)
        normalized_rewards = self.reward_rms.normalize(rewards)
        advantages = normalized_rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Returns for value function training
        returns = normalized_rewards

        self.policy.train()
        self.value_net.train()

        return {
            'conditions': condition,
            'actions': actions,
            'old_log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'advantages': advantages,
            'returns': returns,
        }

    def update(self, rollout):
        """PPO update with clipped surrogate objective."""
        conditions = rollout['conditions']
        actions = rollout['actions']
        old_log_probs = rollout['old_log_probs']
        advantages = rollout['advantages']
        returns = rollout['returns']

        batch_size = conditions.size(0)
        minibatch_size = batch_size // self.num_minibatches

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if end > batch_size:
                    break
                mb_idx = indices[start:end]

                mb_cond = conditions[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Policy loss
                new_log_probs, entropy = self.policy.evaluate_actions(mb_cond, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Value loss
                new_values = self.value_net(mb_cond)
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                # Total loss
                loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy

                # Update policy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                # Track stats
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl.item()
                num_updates += 1

        if num_updates == 0:
            return {}

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'approx_kl': total_kl / num_updates,
        }
