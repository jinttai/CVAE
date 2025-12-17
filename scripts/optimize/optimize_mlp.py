import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import math

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# 프로젝트 내 모듈은 `src` 패키지를 통해 일관되게 import
from src.models.cvae import MLP
from src.training.physics_layer import PhysicsLayer   # Rmat 버전
from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart


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
    Args:
        batch_size: Number of quaternions to generate
        max_angle_deg: Maximum angle in degrees for each Euler angle (default: 10 degrees)
        device: Device to create tensors on
    Returns:
        quaternions: [batch_size, 4] tensor of quaternions (x, y, z, w)
    """
    max_angle_rad = math.radians(max_angle_deg)
    
    # Generate random Euler angles in [-max_angle_deg, max_angle_deg]
    # Using torch.rand to generate uniform distribution in [0, 1], then scale to [-max, max]
    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device) - max_angle_rad
    
    # Convert to quaternion
    quaternions = euler_to_quaternion(roll, pitch, yaw)
    
    return quaternions


# === Orientation & Trajectory Helpers ===
def quat_to_rot(q):
    """
    쿼터니언 q = [x, y, z, w] 를 회전행렬 R (3x3) 로 변환.
    """
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)]),
        torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)]),
        torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)])
    ])
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
    theta = torch.linalg.norm(wb) * dt

    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew(axis)
    I = torch.eye(3, device=device, dtype=dtype)
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


def compute_orientation_traj(physics, q_traj, q_dot_traj, q0_init):
    """
    PhysicsLayer에서 사용하는 Rmat 동역학과 동일하게 body orientation 궤적을 적분하여
    각 스텝의 Euler angle (yaw, pitch, roll)을 반환.

    Args:
        physics: PhysicsLayer 인스턴스
        q_traj: [num_steps, n_q]
        q_dot_traj: [num_steps, n_q]
        q0_init: [4]
    Returns:
        euler_traj: [num_steps, 3] (rad)
    """
    device = physics.device
    num_steps = physics.num_steps

    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)

    R_curr = quat_to_rot(q0_init)

    eulers = []
    for t in range(num_steps):
        qm = q_traj[t]
        qd = q_dot_traj[t]

        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)

        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]

        R_delta = rot_from_omega(wb, physics.dt)
        R_curr = R_curr @ R_delta

        eulers.append(rot_to_euler(R_curr))

    euler_traj = torch.stack(eulers, dim=0)
    return euler_traj


def plot_trajectory(q_traj, q_dot_traj, euler_traj, title, save_path, total_time, target_euler=None):
    """
    joint trajectory는 PhysicsLayer.generate_trajectory의 3차 스플라인 결과를 그대로 사용하고,
    body orientation 궤적은 Euler angle 로 함께 plot.
    """
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    euler_traj = euler_traj.detach().cpu().numpy()  # [T, 3], rad

    # Optional target Euler angle (single 3-vector, rad)
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
    axes[0].legend(loc="right", fontsize="small")

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
            axes[2].axhline(target_deg[i], linestyle="--", linewidth=1.5, label=f"Target {labels[i]}")
    axes[2].set_title("Body Orientation (Euler)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Angle [deg]")
    axes[2].grid(True)
    axes[2].legend(loc="right", fontsize="small")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def load_model(weights_path, input_dim, output_dim, device="cpu"):
    """
    MLP 모델을 로드하는 유틸 함수.
    Rmat 버전도 동일한 인터페이스를 사용한다.
    """
    model = MLP(input_dim, output_dim).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    # Rmat 버전에서는 구조 변경 가능성을 고려하여 strict=False 로딩
    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[Warning] strict=True 로드 실패, strict=False로 재시도합니다.\n  {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== NN-based Initialization + LBFGS Start on {device} ===")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=device)

    # 파라미터 (원본 프로젝트 구조와 동일하게 정리)
    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot["n_q"]
    TOTAL_TIME = 10.0  # 10초 trajectory

    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/opt_nn_lbfgs")
    os.makedirs(save_dir, exist_ok=True)

    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    # Fixed desired orientation: roll=15deg, pitch=15deg, yaw=-15deg (within angle limits)
    roll_deg, pitch_deg, yaw_deg = 15.0, 15.0, -15.0
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    q0_goal = euler_to_quaternion(
        torch.tensor([roll_rad], device=device),
        torch.tensor([pitch_rad], device=device),
        torch.tensor([yaw_rad], device=device),
    )
    condition = torch.cat([q0_start, q0_goal], dim=1)

    print("\n--- [Task 1] Fixed Goal Optimization with MLP Init (LBFGS) ---")

    # 1. MLP Inference (Warm Start)
    inference_start = time.time()

    # 학습 스크립트(train_mlp.py)에서 저장한 가중치 경로와 동일하게 맞춤
    mlp_weights_path = os.path.join(ROOT_DIR, "outputs/weights/mlp_debug/v3.pth")
    mlp_model = load_model(
        mlp_weights_path,
        COND_DIM,
        OUTPUT_DIM,
        device,
    )

    with torch.no_grad():
        # MLP는 deterministic이므로 직접 추론
        best_waypoints = mlp_model(condition)
        best_loss = physics.calculate_loss(best_waypoints, q0_start, q0_goal).item()

    inference_end = time.time()
    print(f"[MLP Init] Initial waypoints with loss {best_loss:.8f}")

    # =================================================================
    # 2. LBFGS Refinement를 위해 CPU로 전환 (장치 전환)
    # =================================================================
    refinement_device = "cpu"
    print(f"\n--- Switching Refinement to {refinement_device} ---")

    # (A) Physics Layer 및 Robot 데이터를 CPU로 이동/재생성
    robot_cpu, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"), verbose_flag=False, device=refinement_device)
    physics_cpu = PhysicsLayer(robot_cpu, NUM_WAYPOINTS, TOTAL_TIME, refinement_device)
    
    # (B) 최적화에 사용될 텐서들을 CUDA -> CPU로 이동
    waypoints_param = best_waypoints.detach().cpu().clone()
    q0_start_cpu = q0_start.cpu()
    q0_goal_cpu = q0_goal.cpu()

    waypoints_param.requires_grad = True  # CPU 텐서에 대해 gradient 설정
    
    print(f"Initial waypoints (on CPU): {waypoints_param}")

    # (C) LBFGS 최적화 (CPU 텐서 사용)
    optimizer = optim.LBFGS(
        [waypoints_param],
        max_iter=50,
        history_size=100,
        tolerance_grad=1e-6,
        tolerance_change=1e-6,
        line_search_fn="strong_wolfe"
    )

    loss_history = [best_loss]
    iteration_count = [0]

    def closure():
        optimizer.zero_grad()
        loss = physics_cpu.calculate_loss(waypoints_param, q0_start_cpu, q0_goal_cpu)
        loss.backward()
        loss_value = loss.item()
        loss_history.append(loss_value)
        iteration_count[0] += 1

        if iteration_count[0] <= 20 or iteration_count[0] % 10 == 0:
            print(f"[Iter] Iter [{iteration_count[0]}] Loss: {loss_value:.6f}")

        return loss

    opt_start = time.time()
    optimizer.step(closure)
    opt_end = time.time()

    # 결과 확인 (Euler 기반 loss)
    final_loss = physics_cpu.calculate_loss(waypoints_param, q0_start_cpu, q0_goal_cpu).item()
    final_deg = np.rad2deg(np.sqrt(final_loss)) if final_loss > 0 else 0.0

    print(f"Inference Finished (MLP warm start). Time: {inference_end - inference_start:.4f}s")
    print(f"Optimization Finished (LBFGS). Time: {opt_end - opt_start:.4f}s")
    print(f"Final Error: {final_loss:.10f} ({final_deg:.4f}°)")
    print(f"Iterations: {len(loss_history)}")
    print(f"Final waypoints: {waypoints_param}")

    with torch.no_grad():
        q_traj, q_dot_traj = physics_cpu.generate_trajectory(waypoints_param)
        q_traj_single = q_traj[0]
        q_dot_traj_single = q_dot_traj[0]
        euler_traj = compute_orientation_traj(physics_cpu, q_traj_single, q_dot_traj_single, q0_start_cpu[0])

        # Target body orientation in Euler angles (rad)
        R_goal = quat_to_rot(q0_goal_cpu[0])
        target_euler = rot_to_euler(R_goal)

        # --------------------------------------------------------------
        # Debug: compare final vs desired orientation (quat + Euler)
        # --------------------------------------------------------------
        final_euler = euler_traj[-1]                     # [3] (yaw, pitch, roll)
        target_euler_vec = target_euler                  # [3] (yaw, pitch, roll)

        # Convert to degrees for readability
        final_euler_deg = final_euler * 180.0 / math.pi
        target_euler_deg = target_euler_vec * 180.0 / math.pi

        # Reconstruct quaternion from final Euler to check Euler<->quat consistency.
        # NOTE: rot_to_euler returns [yaw, pitch, roll] (ZYX),
        #       while euler_to_quaternion expects (roll, pitch, yaw).
        yaw_f, pitch_f, roll_f = final_euler[0], final_euler[1], final_euler[2]
        q_final = euler_to_quaternion(
            roll_f.unsqueeze(0),
            pitch_f.unsqueeze(0),
            yaw_f.unsqueeze(0),
        )  # [1, 4]

        print("\n=== Orientation Check ===")
        print("Final Euler (rad)   [yaw, pitch, roll]:", final_euler)
        print("Target Euler (rad)  [yaw, pitch, roll]:", target_euler_vec)
        print("Final Euler (deg)   [yaw, pitch, roll]:", final_euler_deg)
        print("Target Euler (deg)  [yaw, pitch, roll]:", target_euler_deg)
        print("Final quaternion (from Euler) :", q_final)
        print("Target quaternion (q0_goal)   :", q0_goal_cpu)

        plot_trajectory(
            q_traj_single,
            q_dot_traj_single,
            euler_traj,
            f"MLP+LBFGS (Err: {final_loss:.6f})",
            os.path.join(save_dir, "mlp_lbfgs_traj.png"),
            TOTAL_TIME,
            target_euler=target_euler,
        )

        # ------------------------------------------------------------------
        # Save data for external (e.g., MATLAB) plotting as CSV files
        # ------------------------------------------------------------------
        dt = float(physics_cpu.dt)
        num_steps = q_traj_single.shape[0]
        t = np.linspace(0.0, TOTAL_TIME, num_steps)

        q_traj_np = q_traj_single.detach().cpu().numpy()       # [T, n_q]
        q_dot_np = q_dot_traj_single.detach().cpu().numpy()    # [T, n_q]
        euler_np = euler_traj.detach().cpu().numpy()           # [T, 3] (rad)
        waypoints_np = waypoints_param.detach().cpu().numpy()  # [1, W]
        q0_start_np = q0_start_cpu.detach().cpu().numpy()          # [1, 4]
        q0_goal_np = q0_goal_cpu.detach().cpu().numpy()            # [1, 4]
        target_euler_np = target_euler.detach().cpu().numpy()  # [3] (rad)

        # 1) Joint position trajectory: time + J1..Jn
        n_q = robot["n_q"]
        header_q = "t," + ",".join([f"J{i+1}" for i in range(n_q)])
        q_traj_mat = np.column_stack([t, q_traj_np])
        np.savetxt(
            os.path.join(save_dir, "q_traj_mlp.csv"),
            q_traj_mat,
            delimiter=",",
            header=header_q,
            comments="",
        )

        # 2) Joint velocity trajectory: time + dJ1..dJn
        header_qdot = "t," + ",".join([f"dJ{i+1}" for i in range(n_q)])
        q_dot_mat = np.column_stack([t, q_dot_np])
        np.savetxt(
            os.path.join(save_dir, "q_dot_traj_mlp.csv"),
            q_dot_mat,
            delimiter=",",
            header=header_qdot,
            comments="",
        )

        # 3) Body orientation (Euler) and target orientation (rad)
        #    Columns: t, yaw, pitch, roll, yaw_target, pitch_target, roll_target
        target_tile = np.tile(target_euler_np.reshape(1, 3), (num_steps, 1))
        body_mat = np.column_stack([t, euler_np, target_tile])
        header_body = "t,yaw,pitch,roll,yaw_target,pitch_target,roll_target"
        np.savetxt(
            os.path.join(save_dir, "body_orientation_mlp.csv"),
            body_mat,
            delimiter=",",
            header=header_body,
            comments="",
        )

        # 4) Waypoints (single row)
        header_wp = ",".join([f"W{i+1}" for i in range(waypoints_np.shape[1])])
        np.savetxt(
            os.path.join(save_dir, "waypoints_mlp.csv"),
            waypoints_np,
            delimiter=",",
            header=header_wp,
            comments="",
        )

        # 5) Start / goal quaternion (each as separate CSV)
        np.savetxt(
            os.path.join(save_dir, "q0_start_mlp.csv"),
            q0_start_np,
            delimiter=",",
            header="qx,qy,qz,qw",
            comments="",
        )
        np.savetxt(
            os.path.join(save_dir, "q0_goal_mlp.csv"),
            q0_goal_np,
            delimiter=",",
            header="qx,qy,qz,qw",
            comments="",
        )

        # 6) Meta info (dt, total_time)
        meta_path = os.path.join(save_dir, "meta_mlp.csv")
        with open(meta_path, "w") as f:
            f.write("dt,total_time\n")
            f.write(f"{dt},{TOTAL_TIME}\n")

        print(f"Saved CSV trajectory data to {save_dir}")


if __name__ == "__main__":
    main()

