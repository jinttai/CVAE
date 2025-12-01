import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from src.models.cvae import CVAE, MLP


def plot_trajectory(q_traj, q_dot_traj, title, save_path):
    q_traj = q_traj.detach().cpu().numpy()
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    # 1. Joint Positions
    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f'J{i+1}')
    axes[0].set_title(f'{title} - Joint Angles')
    axes[0].set_ylabel('Rad')
    axes[0].grid(True)
    axes[0].legend(loc='right', fontsize='small')
    
    # 2. Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f'J{i+1}')
    axes[1].set_title('Joint Velocities')
    axes[1].set_ylabel('Rad/s')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def load_cvae(device, robot, weights_path="weights/cvae_debug/v1.pth"):
    """학습된 CVAE 로드 (initial guess용)"""
    COND_DIM = 8
    NUM_WAYPOINTS = 4
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']
    LATENT_DIM = 8

    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, NUM_WAYPOINTS, OUTPUT_DIM


def load_mlp(device, robot, weights_path="weights/mlp_debug/v1.pth"):
    """학습된 MLP 로드 (initial guess용)"""
    COND_DIM = 8
    NUM_WAYPOINTS = 4
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']

    model = MLP(COND_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, NUM_WAYPOINTS, OUTPUT_DIM


def main():
    # 1. 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== NN-based Initialization + Gradient Optimization Start on {device} ===")
    
    # 로봇 로드
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # 어떤 네트워크를 사용할지 선택 ( 'cvae' 또는 'mlp' )
    # 간단히 코드 상단에서 문자열로 변경해서 사용
    model_type = 'cvae'  # 'cvae' 또는 'mlp'
    
    if model_type == 'cvae':
        nn_model, NUM_WAYPOINTS, OUTPUT_DIM = load_cvae(device, robot)
        LATENT_DIM = nn_model.latent_dim if hasattr(nn_model, "latent_dim") else 8
    elif model_type == 'mlp':
        nn_model, NUM_WAYPOINTS, OUTPUT_DIM = load_mlp(device, robot)
        LATENT_DIM = None
    else:
        raise ValueError("model_type must be 'cvae' or 'mlp'")

    TOTAL_TIME = 1.0  # 학습/평가 코드와 일치

    # 물리 엔진
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 결과 저장
    save_dir = f"results/opt_nn_{model_type}"
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 2. 최적화 대상 데이터 생성
    # ==========================================
    # (A) 고정된 목표 (Visual Check용)
    q0_start = torch.tensor([[0., 0., 0., 1.]], device=device)
    q0_goal = torch.tensor([[0., 0., 0.7071, 0.7071]], device=device)  # 90 deg Z
    
    print(f"\n--- [Task 1] Fixed Goal Optimization with {model_type.upper()} Init ---")
    
    # 3. NN을 이용한 initial guess 계산
    with torch.no_grad():
        condition = torch.cat([q0_start, q0_goal], dim=1)
        if model_type == 'cvae':
            # CVAE: z 샘플링 후 decode
            z = torch.randn(1, LATENT_DIM, device=device)
            init_waypoints = nn_model.decode(condition, z)
        else:
            # MLP: deterministic forward
            init_waypoints = nn_model(condition)
    
    # 4. 최적화 변수 (Waypoints) 초기화
    waypoints_param = init_waypoints.detach().clone().to(device)
    waypoints_param.requires_grad = True  # [중요] 미분 추적 켜기
    
    # 5. Optimizer 설정 (initial guess 이후는 optimize_direct와 동일)
    optimizer = optim.Adam([waypoints_param], lr=0.05)
    
    # 6. 최적화 루프 (Optimization Loop)
    iterations = 200  # 최대 반복 횟수
    loss_history = []
    stop_threshold = 1e-4  # 손실이 이 값 아래로 떨어지면 조기 종료

    # === 여기서부터 시간 측정: NN forward, 로봇 로드 등은 포함 X ===
    start_time = time.time()
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # 물리 엔진 시뮬레이션
        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        if (i + 1) % 20 == 0:
            print(f"Iter [{i+1}/{iterations}] Loss: {loss_value:.6f}")
        
        # 조기 종료 조건
        if loss_value < stop_threshold:
            print(f"Loss {loss_value:.6f} < {stop_threshold:.6f}. Early stopping at iter {i+1}.")
            break
            
    end_time = time.time()
    print(f"Optimization Finished (NN init: {model_type}). Time: {end_time - start_time:.4f}s")
    
    # 결과 시각화
    final_error = loss.item()
    final_deg = np.rad2deg(np.sqrt(final_error))  # L1 Loss 가정 시 sqrt 불필요, L2면 필요
    print(f"Final Error: {final_error:.6f} (approx {final_deg:.2f}°)")
    
    # 궤적 생성 및 저장
    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        plot_trajectory(
            q_traj[0],
            q_dot_traj[0],
            f"NN-{model_type.upper()} Init Opt (Err: {final_error:.4f})",
            os.path.join(save_dir, f"fixed_goal_traj_{model_type}.png"),
        )


if __name__ == "__main__":
    main()


