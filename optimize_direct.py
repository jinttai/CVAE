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

def main():
    # 1. 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Direct Optimization Start on {device} ===")
    
    # 로봇 로드
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # 파라미터
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']
    TOTAL_TIME = 1.0 # 학습/평가 코드와 일치시킬 것!
    
    # 물리 엔진
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 결과 저장
    save_dir = "results/direct_opt"
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 2. 최적화 대상 데이터 생성
    # ==========================================
    # (A) 고정된 목표 (Visual Check용)
    q0_start = torch.tensor([[0., 0., 0., 1.]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0., 0., 0.7071, 0.7071]], device=device, dtype=torch.float32) # 90 deg Z
    
    print("\n--- [Task 1] Fixed Goal Optimization ---")
    
    # 3. 최적화 변수 (Waypoints) 초기화
    # 신경망이 없으므로, Waypoint 텐서 자체가 최적화 대상(Parameter)입니다.
    # 초기값: 0 주변의 작은 랜덤 노이즈 (완전 0이면 Gradient가 0일 수 있음)
    # Shape: [Batch=1, Dim=24]
    waypoints_param = torch.zeros(1, OUTPUT_DIM, device=device)
    torch.nn.init.normal_(waypoints_param, mean=0.0, std=0.1)
    waypoints_param.requires_grad = True # [중요] 미분 추적 켜기
    
    # 4. Optimizer 설정
    # 변수 직접 최적화는 Adam이나 LBFGS가 좋음. 여기선 Adam 사용.
    # LR을 0.1 정도로 크게 잡습니다 (파라미터 공간 직접 탐색이므로)
    optimizer = optim.Adam([waypoints_param], lr=0.05)
    
    # 5. 최적화 루프 (Optimization Loop)
    start_time = time.time()
    iterations = 200  # 최대 반복 횟수
    
    loss_history = []
    stop_threshold = 1e-4  # 손실이 이 값 아래로 떨어지면 조기 종료
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # 물리 엔진 시뮬레이션
        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 0~20 iteration: 매번 출력, 21번 이후: 10번마다 출력
        if (i + 1) <= 20 or (i + 1) % 10 == 0:
            print(f"Iter [{i+1}/{iterations}] Loss: {loss_value:.6f}")
        
        # 조기 종료 조건
        if loss_value < stop_threshold:
            print(f"Loss {loss_value:.6f} < {stop_threshold:.6f}. Early stopping at iter {i+1}.")
            break
            
    end_time = time.time()
    print(f"Optimization Finished. Time: {end_time - start_time:.4f}s")
    
    # 결과 시각화
    final_error = loss.item()
    final_deg = np.rad2deg(np.sqrt(final_error)) # L1 Loss 가정 시 sqrt 불필요, L2면 필요
    print(f"Final Error: {final_error:.6f} (approx {final_deg:.2f}°)")
    
    # 궤적 생성 및 저장
    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        plot_trajectory(q_traj[0], q_dot_traj[0], 
                       f"Direct Opt (Err: {final_error:.4f})", 
                       os.path.join(save_dir, "fixed_goal_traj.png"))

if __name__ == "__main__":
    main()