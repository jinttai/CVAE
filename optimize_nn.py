import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np

# 프로젝트 루트 경로 설정 (Import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

from src.models.cvae import CVAE, MLP
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

def load_model(model_class, weights_path, input_dim, output_dim, latent_dim=None, device='cpu'):
    if model_class == CVAE:
        model = CVAE(input_dim, output_dim, latent_dim).to(device)
    else:
        model = MLP(input_dim, output_dim).to(device)
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
        
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def main():
    # 1. 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== NN-based Initialization + LBFGS Refinement Start on {device} ===")
    
    # 로봇 로드
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # 파라미터 (학습 코드와 일치)
    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']
    LATENT_DIM = 8
    TOTAL_TIME = 1.0 # 1초
    
    # 물리 엔진
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 결과 저장
    save_dir = "results/opt_nn_lbfgs"
    os.makedirs(save_dir, exist_ok=True)

    # 모델 가중치 경로
    cvae_path = "weights/cvae_debug/v2.pth"
    # mlp_path = "weights/mlp_debug/v1.pth"
    
    # ==========================================
    # [Task] Fixed Goal Optimization
    # ==========================================
    q0_start = torch.tensor([[0., 0., 0., 1.]], device=device, dtype=torch.float32)
    q0_goal = torch.tensor([[0., 0., 0.7071, 0.7071]], device=device, dtype=torch.float32) # 90 deg Z
    condition = torch.cat([q0_start, q0_goal], dim=1)
    
    print("\n--- [Task 1] Fixed Goal Optimization with CVAE Init (LBFGS) ---")
    
    # 2. CVAE Inference (Warm Start) - 추론 시간 측정
    inference_start = time.time()

    cvae = load_model(CVAE, cvae_path, COND_DIM, OUTPUT_DIM, LATENT_DIM, device)
    
    with torch.no_grad():
        # 10개 샘플링하여 가장 좋은 것 선택
        num_samples = 10
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1)
        
        candidates = cvae.decode(cond_batch, z)
        
        # 물리 엔진으로 평가 (vmap 사용)
        q_traj, q_dot_traj = physics.generate_trajectory(candidates)
        batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0,0,0,0))
        losses = batch_sim_fn(q_traj, q_dot_traj, q0_start.repeat(num_samples, 1), q0_goal.repeat(num_samples, 1))
        
        best_idx = torch.argmin(losses)
        best_waypoints = candidates[best_idx].unsqueeze(0).clone() # Best 1개 선택
        best_loss = losses[best_idx].item()

    inference_end = time.time()

    print(f"[CVAE Init] Selected best of {num_samples} samples with loss {best_loss:.8f}")
    
    # 3. 최적화 변수 설정 (Gradient 켜기)
    waypoints_param = best_waypoints.detach().clone()
    waypoints_param.requires_grad = True
    
    # [핵심] LBFGS Optimizer 사용
    # lr=1.0이 기본값이며, 뉴턴법 기반이라 보통 1.0을 씁니다.
    # max_iter: step() 한 번에 수행할 최대 반복 횟수
    optimizer = optim.LBFGS([waypoints_param], 
                            lr=1.0, 
                            max_iter=20, 
                            history_size=10, 
                            line_search_fn="strong_wolfe") # Line Search 필수

    # 4. 최적화 루프 (LBFGS는 closure 함수가 필요함)
    loss_history = [best_loss]
    iteration_count = [0]  # closure 내에서 iteration 추적용
    
    def closure():
        optimizer.zero_grad()
        loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal)
        loss.backward()
        loss_value = loss.item()
        loss_history.append(loss_value)
        iteration_count[0] += 1
        
        # 0~20 iteration: 매번 출력, 21번 이후: 10번마다 출력
        if iteration_count[0] <= 20 or iteration_count[0] % 10 == 0:
            print(f"Iter [{iteration_count[0]}] Loss: {loss_value:.6f}")
        
        return loss

    # LBFGS는 step() 한 번 호출에 내부적으로 여러 번 반복(iter)함
    opt_start = time.time()
    optimizer.step(closure)
    opt_end = time.time()
    
    # 결과 확인 및 시간 출력
    final_loss = physics.calculate_loss(waypoints_param, q0_start, q0_goal).item()
    final_deg = np.rad2deg(np.sqrt(final_loss)) if final_loss > 0 else 0.0 # sqrt for L2 loss assumption
    
    print(f"Inference Finished (CVAE warm start). Time: {inference_end - inference_start:.4f}s")
    print(f"Optimization Finished (LBFGS). Time: {opt_end - opt_start:.4f}s")
    print(f"Final Error: {final_loss:.10f}")
    print(f"Iterations: {len(loss_history)}")
    
    # 궤적 생성 및 저장
    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(waypoints_param)
        plot_trajectory(
            q_traj[0],
            q_dot_traj[0],
            f"CVAE+LBFGS Opt (Err: {final_loss:.6f})",
            os.path.join(save_dir, "cvae_lbfgs_traj.png"),
        )

if __name__ == "__main__":
    main()