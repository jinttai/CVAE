import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import math
import argparse
from torch.func import vmap  # For batch simulation

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.models.cvae import CVAE, MLP # MLP는 사용하지 않지만 원본 구조 유지를 위해 import
from src.training.physics_layer import PhysicsLayer  # default
from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart


# === Utility Functions ===

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


def generate_random_quaternion_from_euler(batch_size, max_angle_deg=30.0, device='cpu'):
    """
    Generate random quaternions from Euler angles within specified range
    """
    max_angle_rad = math.radians(max_angle_deg)
    
    # Generate random Euler angles in [-max_angle_deg, max_angle_deg]
    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    
    # Convert to quaternion
    quaternions = euler_to_quaternion(roll, pitch, yaw)
    
    return quaternions


# === Orientation & Trajectory Helpers (Rmat Physics) ===

def quat_to_rot(q):
    """
    쿼터니언 q = [x, y, z, w] 를 회전행렬 R (3x3) 로 변환.
    - 단일 입력: [4] 또는 배치 입력: [B, 4] 모두 지원
    - 행(row) 기준으로 쌓아 올바른 회전행렬을 반환 (전치 방지)
    """
    if q.dim() == 1:
        x, y, z, w = q
    else:
        x, y, z, w = q.unbind(dim=-1)
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    if q.dim() == 1:
        R = torch.stack(
            [
                torch.stack([r00, r01, r02]),
                torch.stack([r10, r11, r12]),
                torch.stack([r20, r21, r22]),
            ],
            dim=0,
        )
    else:
        R = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )

    return R


def skew(v):
    vx, vy, vz = v
    zero = torch.zeros_like(vx)
    M = torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])
    return M


def rot_from_omega(wb, dt):
    device = wb.device
    dtype = wb.dtype
    
    # wb가 스칼라가 아닐 경우 linalg.norm은 스칼라를 반환해야 함
    theta = torch.linalg.norm(wb) * dt

    # 특이점 방지를 위해 1e-12 더함
    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew(axis)
    I = torch.eye(3, device=device, dtype=dtype)
    
    # Rodrigues' rotation formula
    R_delta = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R_delta


def rot_to_euler(R):
    """
    회전 행렬 R (3x3)을 Euler angle (ZYX 순서, yaw-pitch-roll)로 변환.
    """
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        yaw = torch.atan2(R[1, 0], R[0, 0])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.atan2(R[2, 1], R[2, 2])
    else:
        yaw = torch.atan2(-R[0, 1], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.zeros_like(yaw)

    return torch.stack([yaw, pitch, roll])

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    각도를 [-pi, pi] 범위로 래핑 (Euler 출력 시 점프 현상 완화)
    """
    two_pi = 2.0 * math.pi
    return torch.remainder(x + math.pi, two_pi) - math.pi


def compute_orientation_traj(physics, q_traj, q_dot_traj, q0_init):
    """
    PhysicsLayer에서 사용하는 Rmat 동역학과 동일하게 body orientation 궤적을 적분하여
    각 스텝의 Euler angle (yaw, pitch, roll)을 반환.
    Vectorized version: wb 계산은 vmap으로 병렬 처리 (PhysicsLayer와 동일)
    """
    device = physics.device
    num_steps = physics.num_steps

    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)

    R_curr = quat_to_rot(q0_init)

    # Vectorized: 모든 step의 wb를 한번에 계산 (PhysicsLayer.simulate_single과 동일)
    def compute_wb_single_step(qm, qd):
        """Single step의 wb 계산"""
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)

        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]  # Angular Velocity part
        return wb
    
    # vmap으로 모든 step을 병렬 처리
    batch_compute_wb = vmap(compute_wb_single_step, in_dims=(0, 0))
    wb_all = batch_compute_wb(q_traj, q_dot_traj)  # [num_steps, 3]
    
    # 모든 step의 R_delta를 한번에 계산
    batch_rot_from_omega = vmap(rot_from_omega, in_dims=(0, None))
    R_delta_all = batch_rot_from_omega(wb_all, physics.dt)  # [num_steps, 3, 3]
    
    # 순차적으로 R_curr 업데이트 및 euler 계산 (각 스텝마다 euler가 필요하므로)
    eulers = []
    for t in range(num_steps):
        R_curr = R_curr @ R_delta_all[t]
        eulers.append(rot_to_euler(R_curr))

    euler_traj = torch.stack(eulers, dim=0)
    return euler_traj


# === Visualization and Load Helpers ===

def plot_trajectory(q_traj, q_dot_traj, euler_traj, title, save_path, total_time, target_euler=None):
    """
    Joint trajectory 및 Body orientation 궤적을 Matplotlib으로 Plot하고 저장.
    """
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    euler_traj = euler_traj.detach().cpu().numpy()  # [T, 3], rad

    target_deg = None
    if target_euler is not None:
        target_deg = np.rad2deg(target_euler.detach().cpu().numpy())  # [3]

    num_steps = q_traj.shape[0]
    t = np.linspace(0.0, total_time, num_steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # 1) Joint Angles
    for i in range(q_traj.shape[1]):
        axes[0].plot(t, q_traj[:, i], label=f"J{i+1}")
    axes[0].set_title(f"{title} - Joint Angles (Cubic Spline)")
    axes[0].set_ylabel("Rad")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", fontsize=8)

    # 2) Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(t, q_dot_traj[:, i], label=f"J{i+1}")
    axes[1].set_title("Joint Velocities")
    axes[1].set_ylabel("Rad/s")
    axes[1].grid(True)

    # 3) Body Orientation (Euler, deg)
    euler_deg = np.rad2deg(euler_traj)
    labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    for i in range(3):
        axes[2].plot(t, euler_deg[:, i], label=labels[i])
        if target_deg is not None:
            axes[2].axhline(target_deg[i], color=axes[2].lines[-1].get_color(), linestyle="--", linewidth=1.5, label=f"Target {labels[i]}")
    axes[2].set_title("Body Orientation (Euler, ZYX)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Angle [deg]")
    axes[2].grid(True)
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def load_model(model_class, weights_path, input_dim, output_dim, latent_dim=None, device="cpu", joint_limits=None):
    """
    CVAE/MLP 모델 가중치를 로드하는 유틸 함수.
    """
    if model_class == CVAE:
        model = CVAE(input_dim, output_dim, latent_dim, joint_limits=joint_limits).to(device)
    else:
        model = MLP(input_dim, output_dim, joint_limits=joint_limits).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[Warning] strict=True 로드 실패, strict=False로 재시도합니다.\n  {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model



# === Main Execution ===

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CVAE Joint-based trajectory optimization with optional LBFGS refinement")
    parser.add_argument("--no-optimize", action="store_true", 
                        help="Skip LBFGS optimization after CVAE sampling (only use best sample)")
    parser.add_argument("--num-samples", type=int, default=1024,
                        help="Number of CVAE samples to generate and evaluate")
    parser.add_argument("--start-joint", type=str, default=None,
                        help="Starting joint angles (comma-separated, in radians). If not provided, random in [-140deg, 140deg]")
    parser.add_argument("--goal-joint", type=str, default=None,
                        help="Goal joint angles (comma-separated, in radians). If not provided, random in [-140deg, 140deg]")
    args = parser.parse_args()
    
    # 1. 초기 설정 (CVAE Inference는 CUDA 사용)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    optimize_flag = not args.no_optimize
    print(f"=== NN-based Joint Initialization {'+ LBFGS' if optimize_flag else '(Sampling Only)'} Start on {device} ===")

    # 로봇 로드 (CUDA용)
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    # 파라미터
    # Condition: start_joint (6) + desired_joint (6) + desired_quaternion (4) = 16
    COND_DIM = robot["n_q"] + robot["n_q"] + 4  # 6 + 6 + 4 = 16
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    LATENT_DIM = 3
    TOTAL_TIME = 10.0  # 10초 trajectory
    JOINT_SQUARED_WEIGHT = 0.01  # Weight for mean of joint^2 regularization
    JOINT_CHANGE_WEIGHT = 0.01  # Weight for joint change penalty between consecutive waypoints
    MAX_JOINT_WEIGHT = 0.1  # Weight for maximum joint angle penalty
    
    # Joint angle range: -140deg to 140deg
    JOINT_MIN_DEG = -140.0
    JOINT_MAX_DEG = 140.0
    JOINT_MIN_RAD = math.radians(JOINT_MIN_DEG)
    JOINT_MAX_RAD = math.radians(JOINT_MAX_DEG)

    # PhysicsLayer (GPU only)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/opt_cvae_joint")
    os.makedirs(save_dir, exist_ok=True)

    # 시작 및 목표 관절 각도 설정
    n_q = robot["n_q"]
    if args.start_joint is not None:
        start_joint_list = [float(x) for x in args.start_joint.split(",")]
        if len(start_joint_list) != n_q:
            raise ValueError(f"Start joint angles must have {n_q} values, got {len(start_joint_list)}")
        q_start_joint = torch.tensor([start_joint_list], device=device, dtype=torch.float32)
    else:
        # Random in [-140deg, 140deg]
        q_start_joint = (torch.rand(1, n_q, device=device) * (JOINT_MAX_RAD - JOINT_MIN_RAD) + JOINT_MIN_RAD)
        print(f"Generated random start joint: {q_start_joint.cpu().numpy() * 180.0 / math.pi} deg")
    
    if args.goal_joint is not None:
        goal_joint_list = [float(x) for x in args.goal_joint.split(",")]
        if len(goal_joint_list) != n_q:
            raise ValueError(f"Goal joint angles must have {n_q} values, got {len(goal_joint_list)}")
        q_goal_joint = torch.tensor([goal_joint_list], device=device, dtype=torch.float32)
    else:
        # Random in [-140deg, 140deg]
        q_goal_joint = (torch.rand(1, n_q, device=device) * (JOINT_MAX_RAD - JOINT_MIN_RAD) + JOINT_MIN_RAD)
        print(f"Generated random goal joint: {q_goal_joint.cpu().numpy() * 180.0 / math.pi} deg")
    
    # 목표 자세 quaternion: x 방향으로 10도 회전 (roll=10deg, pitch=0, yaw=0)
    roll_rad = math.radians(20.0)
    pitch_rad = 0.0
    yaw_rad = 0.0
    q0_goal_temp = euler_to_quaternion(
        torch.tensor([roll_rad], device=device),
        torch.tensor([pitch_rad], device=device),
        torch.tensor([yaw_rad], device=device),
    )  # Could be [4] or [1, 4]
    # Ensure q0_goal is [1, 4] shape
    if q0_goal_temp.dim() == 1:
        q0_goal = q0_goal_temp.unsqueeze(0)  # [4] -> [1, 4]
    else:
        q0_goal = q0_goal_temp.view(1, 4)  # Reshape to [1, 4]
    
    # 시작 자세 quaternion (identity)
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    # Debug: Check shapes before concatenation
    print(f"\n[Debug] Before concatenation:")
    print(f"  q_start_joint shape: {q_start_joint.shape}")
    print(f"  q_goal_joint shape: {q_goal_joint.shape}")
    print(f"  q0_goal shape: {q0_goal.shape}")
    
    # Condition: start_joint + desired_joint + desired_quaternion
    condition = torch.cat([q_start_joint, q_goal_joint, q0_goal], dim=1)  # [1, 16]

    # Condition 검증
    print("\n--- [Task 1] Joint-based Optimization with CVAE Init (LBFGS) ---")
    print(f"Condition structure:")
    print(f"  - q_start_joint shape: {q_start_joint.shape}, values (deg): {q_start_joint.cpu().numpy() * 180.0 / math.pi}")
    print(f"  - q_goal_joint shape: {q_goal_joint.shape}, values (deg): {q_goal_joint.cpu().numpy() * 180.0 / math.pi}")
    print(f"  - q0_goal shape: {q0_goal.shape}, values: {q0_goal.cpu().numpy()}")
    print(f"  - condition shape: {condition.shape}, expected: [1, {COND_DIM}]")
    assert condition.shape == (1, COND_DIM), f"Condition shape mismatch! Got {condition.shape}, expected (1, {COND_DIM})"
    print(f"  ✓ Condition shape is correct!")

    # 1. CVAE Inference (Warm Start) - CUDA에서 수행
    inference_start = time.time()
    
    # CVAE inference parameters
    num_samples = args.num_samples  # Number of samples to generate and evaluate
    
    cvae_weights_path = os.path.join(ROOT_DIR, "outputs/weights/cvae_joint/v1.pth")
    cvae_model = load_model(
        CVAE,
        cvae_weights_path,
        COND_DIM,
        OUTPUT_DIM,
        LATENT_DIM,
        device, # CUDA 모델 로드
        joint_limits=robot['joint_limits']
    )

    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1)
        print(f"\n[CVAE Inference] Condition batch shape: {cond_batch.shape}, expected: [{num_samples}, {COND_DIM}]")
        assert cond_batch.shape == (num_samples, COND_DIM), f"Condition batch shape mismatch! Got {cond_batch.shape}, expected ({num_samples}, {COND_DIM})"
        candidates = cvae_model.decode(cond_batch, z)
        
        # Calculate total loss for all candidates using PhysicsLayer (batch processing)
        q0_start_batch = q0_start.repeat(num_samples, 1)
        q0_goal_batch = q0_goal.repeat(num_samples, 1)
        q_start_joint_batch = q_start_joint.repeat(num_samples, 1)
        q_goal_joint_batch = q_goal_joint.repeat(num_samples, 1)
        
        total_loss, _ = physics.calculate_total_loss(
            candidates, q0_start_batch, q0_goal_batch,
            joint_squared_weight=JOINT_SQUARED_WEIGHT,
            joint_change_weight=JOINT_CHANGE_WEIGHT,
            max_joint_weight=MAX_JOINT_WEIGHT,
            return_mean=False,  # Return per-sample losses
            q_start_joint=q_start_joint_batch,
            q_end_joint=q_goal_joint_batch
        )
        
        # total_loss is now [num_samples] tensor
        losses = torch.where(torch.isfinite(total_loss), total_loss, torch.full_like(total_loss, float("inf")))
        
        best_idx = torch.argmin(losses)
        best_waypoints = candidates[best_idx:best_idx + 1].clone()
        best_loss = losses[best_idx].item()

    torch.cuda.synchronize()  # GPU 작업 완료 대기 (정확한 측정)
    inference_end = time.time()
    print(f"[CVAE Init] Selected best of {num_samples} samples with loss {best_loss:.8f}")

    # =================================================================
    # 2. LBFGS Refinement (Optional)
    # =================================================================
    if not optimize_flag:
        # Skip optimization, use best sample directly
        print("\n--- Skipping LBFGS optimization (using best sample directly) ---")
        waypoints_param = best_waypoints.clone()
        
        # Evaluate final loss without optimization (GPU)
        with torch.no_grad():
            total_loss, loss_dict = physics.calculate_total_loss(
                waypoints_param, q0_start, q0_goal,
                joint_squared_weight=JOINT_SQUARED_WEIGHT,
                joint_change_weight=JOINT_CHANGE_WEIGHT,
                max_joint_weight=MAX_JOINT_WEIGHT,
                q_start_joint=q_start_joint,
                q_end_joint=q_goal_joint
            )
            physics_loss = loss_dict['physics_loss'].item()
            joint_squared_penalty = loss_dict['joint_squared_penalty'].item()
            joint_change_penalty = loss_dict['joint_change_penalty'].item()
            max_joint_penalty = loss_dict['max_joint_penalty'].item()
            total_loss_val = loss_dict['total_loss'].item()
            
            # Get final quaternion for angle error calculation
            q_traj, q_dot_traj = physics.generate_trajectory(
                waypoints_param, 
                q_start=q_start_joint, 
                q_end=q_goal_joint
            )
            sim_out = physics.simulate_single(q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])
            q_final_from_sim = sim_out[1]  # Final quaternion from simulate_single
            
            # Calculate actual quaternion error from simulate_single's q_final
            q1 = q_final_from_sim
            q2 = q0_goal[0]
            dot = torch.sum(q1 * q2).abs().clamp(-1.0, 1.0)
            quat_angle_err_from_sim = 2.0 * torch.acos(dot) * 180.0 / math.pi
        
        print(f"\nInference Finished (CVAE sampling only). Time: {inference_end - inference_start:.4f}s")
        print(f"Physics Loss (GPU): {physics_loss:.8f}")
        print(f"Total Loss: {total_loss_val:.2e} (physics: {physics_loss:.2e}, joint_sq: {joint_squared_penalty:.2e}, joint_change: {joint_change_penalty:.2e}, max_joint: {max_joint_penalty:.2e})")
        print(f"Quat angle error (from simulate_single q_final): {quat_angle_err_from_sim.item():.2e}°")
        print(f"Final waypoints (on GPU): {waypoints_param}")
        
        opt_start = inference_end
        opt_end = inference_end
    else:
        # =================================================================
        # 2. LBFGS Refinement (GPU)
        # =================================================================
        print(f"\n--- LBFGS Refinement on {device} ---")
        
        waypoints_param = best_waypoints.clone()
        waypoints_param.requires_grad = True
        
        print(f"Initial waypoints (on GPU): {waypoints_param}")

        # LBFGS optimization (GPU)
        optimizer = optim.LBFGS(
            [waypoints_param],
            lr=1e-3,
            max_iter=20,
            line_search_fn='strong_wolfe'
        )

        loss_history = [best_loss]
        iteration_count = [0]

        def closure():
            optimizer.zero_grad()
            loss, loss_dict = physics.calculate_total_loss(
                waypoints_param, q0_start, q0_goal,
                joint_squared_weight=JOINT_SQUARED_WEIGHT,
                joint_change_weight=JOINT_CHANGE_WEIGHT,
                max_joint_weight=MAX_JOINT_WEIGHT,
                q_start_joint=q_start_joint,
                q_end_joint=q_goal_joint
            )
            
            loss.backward()
            loss_value = loss.item()
            loss_history.append(loss_value)
            iteration_count[0] += 1

            if iteration_count[0] <= 20 or iteration_count[0] % 10 == 0:
                physics_loss_val = loss_dict['physics_loss'].item()
                joint_sq_val = loss_dict['joint_squared_penalty'].item()
                joint_change_val = loss_dict['joint_change_penalty'].item()
                max_joint_val = loss_dict['max_joint_penalty'].item()
                print(f"[GPU] Iter [{iteration_count[0]}] Loss: {loss_value:.6f} (physics: {physics_loss_val:.6f}, joint_sq: {joint_sq_val:.6f}, joint_change: {joint_change_val:.6f}, max_joint: {max_joint_val:.6f})")

            return loss

        opt_start = time.time()
        optimizer.step(closure)
        opt_end = time.time()

        # Results (GPU)
        with torch.no_grad():
            total_loss, loss_dict = physics.calculate_total_loss(
                waypoints_param, q0_start, q0_goal,
                joint_squared_weight=JOINT_SQUARED_WEIGHT,
                joint_change_weight=JOINT_CHANGE_WEIGHT,
                max_joint_weight=MAX_JOINT_WEIGHT,
                q_start_joint=q_start_joint,
                q_end_joint=q_goal_joint
            )
            physics_loss = loss_dict['physics_loss'].item()
            joint_squared_penalty = loss_dict['joint_squared_penalty'].item()
            joint_change_penalty = loss_dict['joint_change_penalty'].item()
            max_joint_penalty = loss_dict['max_joint_penalty'].item()
            total_loss_val = loss_dict['total_loss'].item()

            # Calculate actual quaternion error for display
            q_traj_temp, q_dot_traj_temp = physics.generate_trajectory(
                waypoints_param,
                q_start=q_start_joint,
                q_end=q_goal_joint
            )
            sim_out_temp = physics.simulate_single(q_traj_temp[0], q_dot_traj_temp[0], q0_start[0], q0_goal[0])
            q_final_temp = sim_out_temp[1]
            q1 = q_final_temp
            q2 = q0_goal[0]
            dot = torch.sum(q1 * q2).abs().clamp(-1.0, 1.0)
            angle_rad = 2.0 * torch.acos(dot)
            final_deg = angle_rad * 180.0 / math.pi

        print(f"\nInference Finished (CVAE warm start). Time: {inference_end - inference_start:.4f}s")
        print(f"Optimization Finished (LBFGS, GPU). Time: {opt_end - opt_start:.4f}s")
        print(f"Total Loss: {total_loss_val:.2e} (physics: {physics_loss:.2e}, joint_sq: {joint_squared_penalty:.2e}, joint_change: {joint_change_penalty:.2e}, max_joint: {max_joint_penalty:.2e})")
        print(f"Angle Error: {final_deg.item():.2e}°")
        print(f"Iterations: {len(loss_history)}")
        print(f"Final waypoints (on GPU): {waypoints_param}")

    # 3. 최종 궤적 생성 및 저장 (GPU)
    with torch.no_grad():
        # Generate trajectory on GPU
        q_traj, q_dot_traj = physics.generate_trajectory(
            waypoints_param,
            q_start=q_start_joint,
            q_end=q_goal_joint
        )
        q_traj_single = q_traj[0]
        q_dot_traj_single = q_dot_traj[0]
        
        # compute_orientation_traj function
        euler_traj = compute_orientation_traj(physics, q_traj_single, q_dot_traj_single, q0_start[0])

        # Target body orientation
        R_goal = quat_to_rot(q0_goal[0])
        target_euler = rot_to_euler(R_goal)

        # --------------------------------------------------------------
        # Compare final vs desired orientation (quat + Euler)
        # --------------------------------------------------------------
        final_euler = euler_traj[-1]              # [3] (yaw, pitch, roll)
        target_euler_vec = target_euler           # [3] (yaw, pitch, roll)

        # Euler는 비유일/래핑이 있으므로 출력은 [-pi, pi]로 래핑해서 비교
        final_euler_wrapped = wrap_to_pi(final_euler)
        target_euler_wrapped = wrap_to_pi(target_euler_vec)

        final_euler_deg = final_euler_wrapped * 180.0 / math.pi
        target_euler_deg = target_euler_wrapped * 180.0 / math.pi

        yaw_f, pitch_f, roll_f = final_euler_wrapped[0], final_euler_wrapped[1], final_euler_wrapped[2]
        q_final = euler_to_quaternion(
            roll_f.unsqueeze(0),
            pitch_f.unsqueeze(0),
            yaw_f.unsqueeze(0),
        )  # [1, 4]
        # 실제 자세 차이(최단 회전각)도 같이 출력: 2*acos(|<q1,q2>|)
        q1 = q_final[0]
        q2 = q0_goal[0]
        dot = torch.sum(q1 * q2).abs().clamp(-1.0, 1.0)
        quat_angle_err = 2.0 * torch.acos(dot) * 180.0 / math.pi

        # Also get q_final from simulate_single for comparison
        sim_out_viz = physics.simulate_single(q_traj_single, q_dot_traj_single, q0_start[0], q0_goal[0])
        q_final_from_sim_viz = sim_out_viz[1]
        q1_sim = q_final_from_sim_viz
        dot_sim = torch.sum(q1_sim * q2).abs().clamp(-1.0, 1.0)
        quat_angle_err_sim = 2.0 * torch.acos(dot_sim) * 180.0 / math.pi

        print("\n=== Orientation Check ===")
        print("Final Euler (deg)   [yaw, pitch, roll]:", final_euler_deg)
        print("Target Euler (deg)  [yaw, pitch, roll]:", target_euler_deg)
        print("Final quaternion (from Euler) :", q_final)
        print("Final quaternion (from simulate_single):", q_final_from_sim_viz)
        print("Target quaternion (q0_goal)   :", q0_goal)
        print(f"Quat angle error (from Euler)        : {quat_angle_err.item():.2e}°")
        print(f"Quat angle error (from simulate_single): {quat_angle_err_sim.item():.2e}°")

        # Plot title depends on whether optimization was performed
        # Use quat_angle_err_sim for angle display (available in both cases)
        # Get physics_loss for plot title from already computed loss_dict
        plot_title = f"CVAE Joint{'+LBFGS' if optimize_flag else ''} (Err: {physics_loss:.6f}, Angle: {quat_angle_err_sim.item():.2e}°)"
        plot_filename = "cvae_joint_lbfgs_traj_v1.png" if optimize_flag else "cvae_joint_sample_traj_v1.png"
        plot_trajectory(
            q_traj_single,
            q_dot_traj_single,
            euler_traj,
            plot_title,
            os.path.join(save_dir, plot_filename),
            TOTAL_TIME,
            target_euler=target_euler,
        )

        # ------------------------------------------------------------------
        # Save data for external (e.g., MATLAB) plotting as CSV files
        # ------------------------------------------------------------------
        dt = float(physics.dt)
        num_steps = q_traj_single.shape[0]
        t = np.linspace(0.0, TOTAL_TIME, num_steps)

        q_traj_np = q_traj_single.detach().cpu().numpy()
        q_dot_np = q_dot_traj_single.detach().cpu().numpy()
        euler_np = euler_traj.detach().cpu().numpy()
        waypoints_np = waypoints_param.detach().cpu().numpy()
        q0_start_np = q0_start.detach().cpu().numpy()
        q0_goal_np = q0_goal.detach().cpu().numpy()
        target_euler_np = target_euler.detach().cpu().numpy()
        q_start_joint_np = q_start_joint.detach().cpu().numpy()
        q_goal_joint_np = q_goal_joint.detach().cpu().numpy()

        # CSV 파일 저장 로직
        n_q = robot["n_q"]
        
        # 1) Joint position trajectory
        header_q = "t," + ",".join([f"J{i+1}" for i in range(n_q)])
        q_traj_mat = np.column_stack([t, q_traj_np])
        np.savetxt(
            os.path.join(save_dir, "q_traj.csv"),
            q_traj_mat,
            delimiter=",",
            header=header_q,
            comments="",
        )

        # 2) Joint velocity trajectory
        header_qdot = "t," + ",".join([f"dJ{i+1}" for i in range(n_q)])
        q_dot_mat = np.column_stack([t, q_dot_np])
        np.savetxt(
            os.path.join(save_dir, "q_dot_traj.csv"),
            q_dot_mat,
            delimiter=",",
            header=header_qdot,
            comments="",
        )

        # 3) Body orientation (Euler) and target orientation (rad)
        target_tile = np.tile(target_euler_np.reshape(1, 3), (num_steps, 1))
        body_mat = np.column_stack([t, euler_np, target_tile])
        header_body = "t,yaw,pitch,roll,yaw_target,pitch_target,roll_target"
        np.savetxt(
            os.path.join(save_dir, "body_orientation.csv"),
            body_mat,
            delimiter=",",
            header=header_body,
            comments="",
        )

        # 4) Waypoints (single row)
        header_wp = ",".join([f"W{i+1}" for i in range(waypoints_np.shape[1])])
        np.savetxt(
            os.path.join(save_dir, "waypoints.csv"),
            waypoints_np,
            delimiter=",",
            header=header_wp,
            comments="",
        )

        # 5) Start / goal quaternion
        np.savetxt(
            os.path.join(save_dir, "q0_start.csv"),
            q0_start_np,
            delimiter=",",
            header="qx,qy,qz,qw",
            comments="",
        )
        np.savetxt(
            os.path.join(save_dir, "q0_goal.csv"),
            q0_goal_np,
            delimiter=",",
            header="qx,qy,qz,qw",
            comments="",
        )
        
        # 6) Start / goal joint angles
        np.savetxt(
            os.path.join(save_dir, "q0_start_joint.csv"),
            q_start_joint_np,
            delimiter=",",
            header=",".join([f"J{i+1}" for i in range(n_q)]),
            comments="",
        )
        np.savetxt(
            os.path.join(save_dir, "q0_goal_joint.csv"),
            q_goal_joint_np,
            delimiter=",",
            header=",".join([f"J{i+1}" for i in range(n_q)]),
            comments="",
        )

        # 7) Meta info
        meta_path = os.path.join(save_dir, "meta.csv")
        with open(meta_path, "w") as f:
            f.write("dt,total_time\n")
            f.write(f"{dt},{TOTAL_TIME}\n")

        print(f"Saved CSV trajectory data to {save_dir}")


if __name__ == "__main__":
    main()

