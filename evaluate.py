import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

from src.models.cvae import CVAE, MLP
from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot

def load_model(model_class, weights_path, input_dim, output_dim, latent_dim=None, device='cpu'):
    """모델 로드 헬퍼 함수"""
    if model_class == CVAE:
        model = CVAE(input_dim, output_dim, latent_dim).to(device)
    else:
        model = MLP(input_dim, output_dim).to(device)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def plot_best_trajectories(q_trajs, losses, title, save_path):
    """상위 3개 궤적 시각화"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by loss
    sorted_idx = np.argsort(losses)
    
    colors = ['r', 'g', 'b'] # Top 1, 2, 3 colors
    
    for i in range(min(3, len(losses))):
        idx = sorted_idx[i]
        traj = q_trajs[idx] # [Steps, Joints]
        loss = losses[idx]
        
        # Plot Joint 1 only for clarity (or sum of joints)
        # Here we plot J1 as example
        ax.plot(traj[:, 0], color=colors[i], label=f'Rank {i+1} (Loss: {loss:.4f})', alpha=0.8, linewidth=2)
        
    ax.set_title(title)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Joint 1 Angle (Rad)')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    # 1. 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Evaluation Start on {device} ===")
    
    # 로봇 로드
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # 파라미터 (학습 때와 동일해야 함)
    COND_DIM = 8
    NUM_WAYPOINTS = 3
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']
    LATENT_DIM = 8
    TOTAL_TIME = 1.0 # 학습 때 설정한 시간
    
    # 물리 엔진
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 저장된 가중치 경로 (여기를 본인 경로로 수정하세요!)
    # 예시: 가장 최근에 저장된 파일 찾기
    cvae_weights_dir = "weights/cvae_debug"
    mlp_weights_dir = "weights/mlp_debug" # 만약 저장했다면
    
    # 가장 최근 파일 찾기 함수
    def get_latest_weight(dir_path):
        if not os.path.exists(dir_path): return None
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pth')]
        if not files: return None
        return max(files, key=os.path.getctime)

    cvae_path = "weights/cvae_debug/v2.pth" # get_latest_weight(cvae_weights_dir)
    mlp_path = "weights/mlp_debug/v2.pth" # get_latest_weight(mlp_weights_dir) 
    # mlp_path = None # MLP 웨이트가 없다면 None
    
    print(f"Loading CVAE from: {cvae_path}")
    
    # 2. 테스트 데이터 생성 (고정된 90도 회전 목표)
    q0_start = torch.tensor([[0., 0., 0., 1.]], device=device, dtype=torch.float32)
    goal_rpy = [np.pi/100, np.pi/100, np.pi/100]
    goal_quat = R.from_euler('xyz', goal_rpy).as_quat().reshape(1, 4)
    q0_goal = torch.tensor(goal_quat, device=device, dtype=torch.float32)
    condition = torch.cat([q0_start, q0_goal], dim=1)
    
    # 결과 저장 폴더
    save_dir = "results/evaluation"
    os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================
    # 3. CVAE 평가 (Best of 10)
    # ==========================================
    if cvae_path:
        cvae = load_model(CVAE, cvae_path, COND_DIM, OUTPUT_DIM, LATENT_DIM, device)
        
        print("\n--- CVAE Evaluation (10 Samples) ---")
        
        # 10개 샘플링
        num_samples = 100
        z = torch.randn(num_samples, LATENT_DIM, device=device, dtype=torch.float32)
        cond_batch = condition.repeat(num_samples, 1) # [10, 8]
        start_batch = q0_start.repeat(num_samples, 1)
        goal_batch = q0_goal.repeat(num_samples, 1)
        
        with torch.no_grad():
            waypoints = cvae.decode(cond_batch, z)
            
            # 물리 시뮬레이션 (vmap 활용) - RK4 기반 최종 오차 사용
            q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
            
            batch_sim_fn = torch.func.vmap(physics.simulate_single_rk4, in_dims=(0, 0, 0, 0))
            errors = batch_sim_fn(q_traj, q_dot_traj, start_batch, goal_batch)
            
        # 결과 정리
        errors_np = errors.cpu().numpy()
        q_traj_np = q_traj.cpu().numpy()
        
        # 정렬
        sorted_indices = np.argsort(errors_np)
        
        print(f"{'Rank':<5} | {'Error (Rad^2)':<15} | {'Error (Deg)':<15}")
        print("-" * 40)
        
        for i, idx in enumerate(sorted_indices):
            # Error는 angle_error^2 형태이므로 sqrt 씌워서 각도로 변환
            err_rad = np.sqrt(errors_np[idx])
            err_deg = np.rad2deg(err_rad)
            print(f"{i+1:<5} | {errors_np[idx]:.6f}        | {err_deg:.4f}°")
            
        # 상위 3개 궤적 시각화 저장
        plot_best_trajectories(q_traj_np, errors_np, "CVAE Top-3 Trajectories", os.path.join(save_dir, "cvae_top3.png"))

    # ==========================================
    # 4. MLP 평가 (Single Shot)
    # ==========================================
    if mlp_path:
        mlp = load_model(MLP, mlp_path, COND_DIM, OUTPUT_DIM, device=device)
        print("\n--- MLP Evaluation ---")
        
        with torch.no_grad():
            wp = mlp(condition)
            # MLP는 1개이므로 배치 차원 유지
            q_traj, q_dot_traj = physics.generate_trajectory(wp)
            error = physics.simulate_single_rk4(q_traj[0], q_dot_traj[0], q0_start[0], q0_goal[0])
            
        err_val = error.item()
        err_deg = np.rad2deg(np.sqrt(err_val))
        print(f"MLP Error: {err_val:.6f} ({err_deg:.4f}°)")
        
        # MLP는 하나만 그림
        plt.figure()
        plt.plot(q_traj[0].cpu().numpy()[:, 0], label='J1')
        plt.title(f'MLP Trajectory (Error: {err_val:.4f})')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, "mlp_traj.png"))
        print("Saved MLP plot.")

if __name__ == "__main__":
    main()