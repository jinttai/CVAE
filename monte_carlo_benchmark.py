import torch
import torch.optim as optim
import numpy as np
import math
import time
from typing import Tuple

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.models.cvae import CVAE, MLP
from src.training.physics_layer import PhysicsLayer   # Rmat 버전
from src.dynamics.urdf2robot_torch import urdf2robot


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


def load_model(model_class, weights_path, input_dim, output_dim, latent_dim=None, device="cpu"):
    """
    원본 프로젝트와 동일한 방식으로 CVAE/MLP를 로드하는 유틸 함수.
    """
    if model_class == CVAE:
        model = CVAE(input_dim, output_dim, latent_dim).to(device)
    else:
        model = MLP(input_dim, output_dim).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def optimize_zero_lbfgs(physics: PhysicsLayer, q0_start: torch.Tensor, q0_goal: torch.Tensor) -> float:
    """
    Random initialization + LBFGS optimization
    Returns: calculation time in seconds
    """
    OUTPUT_DIM = physics.num_waypoints * physics.n_q
    device = physics.device
    
    # Random initialization (randn)
    waypoints_param = torch.randn(1, OUTPUT_DIM, device=device, dtype=torch.float32)
    waypoints_param.requires_grad = True

    optimizer = optim.LBFGS(
        [waypoints_param],
        max_iter=50,
        history_size=100,
        tolerance_grad=1e-6,
        tolerance_change=1e-6,
        line_search_fn="strong_wolfe"
    )

    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        loss.backward()
        iteration_count[0] += 1
        return loss

    # Measure only optimization time
    opt_start = time.time()
    optimizer.step(closure)
    opt_end = time.time()
    
    return opt_end - opt_start


def optimize_mlp_lbfgs(physics: PhysicsLayer, q0_start: torch.Tensor, q0_goal: torch.Tensor, 
                       mlp_model: MLP) -> float:
    """
    MLP initialization + LBFGS optimization
    Returns: calculation time in seconds (including MLP inference)
    """
    COND_DIM = 8
    OUTPUT_DIM = physics.num_waypoints * physics.n_q
    device = physics.device
    
    condition = torch.cat([q0_start, q0_goal], dim=1)
    
    # Measure time from MLP inference to end of optimization
    opt_start = time.time()
    
    # MLP Inference (Warm Start)
    with torch.no_grad():
        best_waypoints = mlp_model(condition)

    # LBFGS Refinement
    waypoints_param = best_waypoints.detach().clone()
    waypoints_param.requires_grad = True

    optimizer = optim.LBFGS(
        [waypoints_param],
        max_iter=50,
        history_size=100,
        tolerance_grad=1e-6,
        tolerance_change=1e-6,
        line_search_fn="strong_wolfe"
    )

    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        loss.backward()
        iteration_count[0] += 1
        return loss

    optimizer.step(closure)
    opt_end = time.time()
    
    return opt_end - opt_start


def optimize_cvae_lbfgs(physics: PhysicsLayer, q0_start: torch.Tensor, q0_goal: torch.Tensor,
                        cvae_model: CVAE) -> float:
    """
    CVAE initialization + LBFGS optimization
    Returns: calculation time in seconds (including CVAE inference)
    """
    COND_DIM = 8
    LATENT_DIM = 8
    OUTPUT_DIM = physics.num_waypoints * physics.n_q
    device = physics.device
    
    condition = torch.cat([q0_start, q0_goal], dim=1)
    
    # Measure time from CVAE inference to end of optimization
    opt_start = time.time()
    
    # CVAE Inference (Warm Start)
    with torch.no_grad():
        num_samples = 1000
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1)

        candidates = cvae_model.decode(cond_batch, z)

        q_traj, q_dot_traj = physics.generate_trajectory(candidates)
        batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        losses = batch_sim_fn(
            q_traj,
            q_dot_traj,
            q0_start.repeat(num_samples, 1),
            q0_goal.repeat(num_samples, 1),
        )

        best_idx = torch.argmin(losses)
        best_waypoints = candidates[best_idx].unsqueeze(0).clone()

    # LBFGS Refinement
    waypoints_param = best_waypoints.detach().clone()
    waypoints_param.requires_grad = True

    optimizer = optim.LBFGS(
        [waypoints_param],
        max_iter=50,
        history_size=100,
        tolerance_grad=1e-6,
        tolerance_change=1e-6,
        line_search_fn="strong_wolfe"
    )

    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        loss.backward()
        iteration_count[0] += 1
        return loss

    optimizer.step(closure)
    opt_end = time.time()
    
    return opt_end - opt_start


def generate_random_goal(max_angle_deg: float = 40.0, device: str = "cpu") -> torch.Tensor:
    """
    Generate random quaternion goal from Euler angles within specified range
    """
    max_angle_rad = math.radians(max_angle_deg)
    
    # Generate random Euler angles in [-max_angle_deg, max_angle_deg]
    roll = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(1, device=device).item() - max_angle_rad
    
    # Convert to quaternion
    roll_t = torch.tensor([roll], device=device)
    pitch_t = torch.tensor([pitch], device=device)
    yaw_t = torch.tensor([yaw], device=device)
    
    q0_goal = euler_to_quaternion(roll_t, pitch_t, yaw_t)
    # euler_to_quaternion already returns [1, 4] shape, so no need for unsqueeze
    return q0_goal  # [1, 4]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Monte Carlo Benchmark Start on {device} ===")
    print("Testing 3 optimization methods with random goals (40 deg limit)")
    print("100 iterations per method\n")

    # Setup (done once, not included in timing)
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0
    
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    # Load models once (not included in timing)
    print("Loading models...")
    COND_DIM = 8
    LATENT_DIM = 8
    
    mlp_weights_path = "weights/mlp_rmat_debug/v3.pth"
    cvae_weights_path = "weights/cvae_rmat_debug/v3.pth"
    
    mlp_model = None
    cvae_model = None
    
    try:
        mlp_model = load_model(MLP, mlp_weights_path, COND_DIM, OUTPUT_DIM, None, device)
        print(f"MLP model loaded from {mlp_weights_path}")
    except Exception as e:
        print(f"Warning: Could not load MLP model: {e}")
        print("MLP optimization will be skipped")
    
    try:
        cvae_model = load_model(CVAE, cvae_weights_path, COND_DIM, OUTPUT_DIM, LATENT_DIM, device)
        print(f"CVAE model loaded from {cvae_weights_path}")
    except Exception as e:
        print(f"Warning: Could not load CVAE model: {e}")
        print("CVAE optimization will be skipped")
    
    print("\nStarting Monte Carlo simulation...\n")
    
    # Monte Carlo parameters
    num_iterations = 10
    max_angle_deg = 40.0
    
    # Storage for timing results
    times_zero_lbfgs = []
    times_mlp_lbfgs = []
    times_cvae_lbfgs = []
    
    # Run Monte Carlo
    for i in range(num_iterations):
        # Generate random goal
        q0_goal = generate_random_goal(max_angle_deg, device)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_iterations}")
        
        # 1. Random + LBFGS
        try:
            t = optimize_zero_lbfgs(physics, q0_start, q0_goal)
            times_zero_lbfgs.append(t)
        except Exception as e:
            print(f"Error in random_lbfgs iteration {i+1}: {e}")
        
        # 2. MLP + LBFGS
        if mlp_model is not None:
            try:
                t = optimize_mlp_lbfgs(physics, q0_start, q0_goal, mlp_model)
                times_mlp_lbfgs.append(t)
            except Exception as e:
                print(f"Error in mlp_lbfgs iteration {i+1}: {e}")
        
        # 3. CVAE + LBFGS
        if cvae_model is not None:
            try:
                t = optimize_cvae_lbfgs(physics, q0_start, q0_goal, cvae_model)
                times_cvae_lbfgs.append(t)
            except Exception as e:
                print(f"Error in cvae_lbfgs iteration {i+1}: {e}")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("Monte Carlo Results (Calculation Time)")
    print("="*60)
    
    def print_stats(name: str, times: list):
        if len(times) == 0:
            print(f"{name:20s}: No valid results")
            return
        
        times_array = np.array(times)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        
        print(f"{name:20s}: Mean = {mean_time:.6f}s, Std = {std_time:.6f}s ({len(times)}/{num_iterations} successful)")
    
    print_stats("Random + LBFGS", times_zero_lbfgs)
    print_stats("MLP + LBFGS", times_mlp_lbfgs)
    print_stats("CVAE + LBFGS", times_cvae_lbfgs)
    
    # Save results to file
    results = {
        "random_lbfgs": {
            "times": times_zero_lbfgs,
            "mean": np.mean(times_zero_lbfgs) if times_zero_lbfgs else None,
            "std": np.std(times_zero_lbfgs) if times_zero_lbfgs else None,
        },
        "mlp_lbfgs": {
            "times": times_mlp_lbfgs,
            "mean": np.mean(times_mlp_lbfgs) if times_mlp_lbfgs else None,
            "std": np.std(times_mlp_lbfgs) if times_mlp_lbfgs else None,
        },
        "cvae_lbfgs": {
            "times": times_cvae_lbfgs,
            "mean": np.mean(times_cvae_lbfgs) if times_cvae_lbfgs else None,
            "std": np.std(times_cvae_lbfgs) if times_cvae_lbfgs else None,
        },
    }
    
    import json
    output_file = "monte_carlo_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    
    # Also save as CSV for easy analysis
    import csv
    csv_file = "monte_carlo_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Iteration", "Time (s)"])
        
        for i, t in enumerate(times_zero_lbfgs):
            writer.writerow(["Random+LBFGS", i+1, t])
        for i, t in enumerate(times_mlp_lbfgs):
            writer.writerow(["MLP+LBFGS", i+1, t])
        for i, t in enumerate(times_cvae_lbfgs):
            writer.writerow(["CVAE+LBFGS", i+1, t])
    
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
