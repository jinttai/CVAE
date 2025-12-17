import torch
import numpy as np
import os
import sys
import pandas as pd

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot

def load_csv_tensor(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    tensor = torch.tensor(data, device=device, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def load_meta(path):
    if not os.path.exists(path):
        print(f"Meta file not found: {path}. Using default total_time=10.0")
        return 10.0
    
    df = pd.read_csv(path)
    if 'total_time' in df.columns:
        return float(df['total_time'].iloc[0])
    return 10.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RK4 Loss Evaluation from Optimized Results ({device}) ===")

    # 1. Load Data
    result_dir = os.path.join(ROOT_DIR, "outputs/results/opt_nn_lbfgs")
    print(f"Loading results from: {result_dir}")

    try:
        waypoints = load_csv_tensor(os.path.join(result_dir, "waypoints.csv"), device)
        q0_start = load_csv_tensor(os.path.join(result_dir, "q0_start.csv"), device)
        q0_goal = load_csv_tensor(os.path.join(result_dir, "q0_goal.csv"), device)
        total_time = load_meta(os.path.join(result_dir, "meta.csv"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Setup Robot & Physics
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    n_q = robot["n_q"]

    # Deduce num_waypoints from waypoints shape [1, N*n_q]
    num_elements = waypoints.numel()
    if num_elements % n_q != 0:
        print(f"Error: Waypoints size {num_elements} is not divisible by n_q={n_q}")
        return
    num_waypoints = num_elements // n_q
    
    print(f"Configuration: Waypoints={num_waypoints}, Total Time={total_time}s")

    physics = PhysicsLayer(robot, num_waypoints, total_time, device)

    # 3. Generate Trajectory
    # generate_trajectory expects [Batch, Waypoints*Joints]
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)

    # 4. Compute RK4 Loss
    # simulate_single_rk4 expects single trajectory inputs (not batched for internal logic if called directly without vmap, 
    # but based on previous code it takes single tensors). 
    # q_traj: [Steps, n_q], q_dot_traj: [Steps, n_q], q0_start: [4], q0_goal: [4]
    
    print("Computing RK4 loss...")
    loss_rk4 = physics.simulate_single(q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])
    
    loss_val = loss_rk4[0].item()
    q_final = loss_rk4[1].detach().cpu().numpy()
    
    error_rad = np.sqrt(loss_val)
    error_deg = np.rad2deg(error_rad)

    print("\n" + "="*40)
    print(f"RK4 Loss (Rad^2) : {loss_val:.8f}")
    print(f"Error (Rad)      : {error_rad:.8f}")
    print(f"Error (Deg)      : {error_deg:.4f}°")
    print(f"Final Quaternion : {q_final}")
    print("="*40)

if __name__ == "__main__":
    main()
