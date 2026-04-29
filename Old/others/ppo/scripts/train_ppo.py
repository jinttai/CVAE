import torch
import time
import os
import sys
import csv

# Add root directory to sys.path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from torch.utils.tensorboard import SummaryWriter

from src.dynamics.urdf2robot_torch import urdf2robot
from src.training.physics_layer import PhysicsLayer
from ppo.src.ppo_agent import PolicyNetwork, ValueNetwork
from ppo.src.ppo_trainer import PPOTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== PPO Training Start on {device} ===")

    # --- Robot setup (same as train_cvae_joint.py) ---
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    # --- Hyperparameters ---
    COND_DIM = robot["n_q"] + robot["n_q"] + 4  # 16
    NUM_WAYPOINTS = 3
    ACTION_DIM = NUM_WAYPOINTS * robot["n_q"]  # 18
    TOTAL_TIME = 10.0

    BATCH_SIZE = 1024
    NUM_ITERATIONS = 5000
    EVAL_INTERVAL = 50
    HIDDEN_DIM = 256

    # --- TensorBoard ---
    log_dir = os.path.join(ROOT_DIR, "outputs/logs/ppo_v1")
    writer = SummaryWriter(log_dir=log_dir)

    # --- Physics engine ---
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    # --- Networks ---
    policy = PolicyNetwork(COND_DIM, ACTION_DIM, robot['joint_limits'], hidden_dim=HIDDEN_DIM).to(device)
    value_net = ValueNetwork(COND_DIM, hidden_dim=HIDDEN_DIM).to(device)

    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value params:  {sum(p.numel() for p in value_net.parameters()):,}")

    # --- Trainer ---
    trainer = PPOTrainer(
        policy=policy,
        value_net=value_net,
        physics=physics,
        robot=robot,
        device=device,
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
    )

    # --- Fixed eval conditions ---
    eval_batch_size = 64
    torch.manual_seed(42)
    eval_cond, eval_q_start, eval_q_goal, eval_q0_start, eval_q0_goal = trainer._generate_conditions(eval_batch_size)
    torch.manual_seed(int(time.time()))

    # --- Training loop ---
    total_start = time.time()
    train_log = []

    for iteration in range(1, NUM_ITERATIONS + 1):
        iter_start = time.time()

        # Collect rollouts
        rollout = trainer.collect_rollouts(BATCH_SIZE)
        if rollout is None:
            print(f"Iter {iteration}: all NaN, skipping")
            continue

        mean_reward = rollout['rewards'].mean().item()

        # PPO update
        stats = trainer.update(rollout)
        if not stats:
            continue

        iter_time = time.time() - iter_start

        # Log to TensorBoard
        writer.add_scalar("Reward/mean", mean_reward, iteration)
        writer.add_scalar("Loss/policy", stats['policy_loss'], iteration)
        writer.add_scalar("Loss/value", stats['value_loss'], iteration)
        writer.add_scalar("Stats/entropy", stats['entropy'], iteration)
        writer.add_scalar("Stats/approx_kl", stats['approx_kl'], iteration)
        writer.add_scalar("Stats/std_mean", torch.exp(policy.log_std).mean().item(), iteration)

        log_entry = {
            'iteration': iteration,
            'reward': mean_reward,
            'policy_loss': stats['policy_loss'],
            'value_loss': stats['value_loss'],
            'entropy': stats['entropy'],
            'approx_kl': stats['approx_kl'],
            'time': iter_time,
        }
        train_log.append(log_entry)

        print(
            f"Iter [{iteration}/{NUM_ITERATIONS}] | "
            f"Reward: {mean_reward:.4f} | "
            f"PL: {stats['policy_loss']:.4f} | "
            f"VL: {stats['value_loss']:.4f} | "
            f"Ent: {stats['entropy']:.2f} | "
            f"KL: {stats['approx_kl']:.4f} | "
            f"Time: {iter_time:.2f}s"
        )

        # --- Evaluation ---
        if iteration % EVAL_INTERVAL == 0:
            with torch.no_grad():
                policy.eval()
                # Deterministic eval: use mean (tanh scaled)
                mean_raw = policy.forward(eval_cond)
                tanh_out = torch.tanh(mean_raw)
                eval_actions = policy._scale_action(tanh_out)

                eval_loss, eval_loss_dict = physics.calculate_total_loss(
                    eval_actions, eval_q0_start, eval_q0_goal,
                    joint_squared_weight=0.01,
                    joint_change_weight=0.01,
                    max_joint_weight=0.1,
                    return_mean=True,
                    q_start_joint=eval_q_start,
                    q_end_joint=eval_q_goal,
                )
                eval_reward = -eval_loss.item()
                policy.train()

            writer.add_scalar("Eval/reward", eval_reward, iteration)
            writer.add_scalar("Eval/physics_loss", eval_loss_dict['physics_loss'].item(), iteration)
            print(f"   >>> Eval Reward: {eval_reward:.4f} | Physics Loss: {eval_loss_dict['physics_loss'].item():.4f}")

            log_entry['eval_reward'] = eval_reward
            log_entry['eval_physics_loss'] = eval_loss_dict['physics_loss'].item()

    total_time = time.time() - total_start
    print(f"Training Finished. Total Time: {total_time:.2f}s")

    # --- Save training curve CSV ---
    csv_dir = os.path.join(ROOT_DIR, "outputs/plots/ppo_training_curve")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "v1.csv")

    with open(csv_path, "w", newline="") as f:
        fieldnames = ['iteration', 'reward', 'policy_loss', 'value_loss',
                      'entropy', 'approx_kl', 'time', 'eval_reward', 'eval_physics_loss']
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for entry in train_log:
            w.writerow(entry)
    print(f"Training curve saved to: {csv_path}")

    # --- Save checkpoint ---
    save_dir = os.path.join(ROOT_DIR, "outputs/weights/ppo")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "v1.pth")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_net.state_dict(),
        'policy_optimizer': trainer.policy_optimizer.state_dict(),
        'value_optimizer': trainer.value_optimizer.state_dict(),
        'iteration': NUM_ITERATIONS,
    }, save_path)
    print(f"Checkpoint saved to: {save_path}")

    writer.close()


if __name__ == "__main__":
    main()
