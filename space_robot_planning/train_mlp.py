import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys
import csv

# 프로젝트 루트 경로 설정 (Import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))


from torch.utils.tensorboard import SummaryWriter
from src.models.cvae import MLP
from src.training.physics_layer import PhysicsLayer
from dynamics.urdf2robot_torch import urdf2robot

# --- 시각화 헬퍼 함수 ---
def plot_trajectory(q_traj, q_dot_traj, epoch):
    """
    생성된 궤적을 Matplotlib 그림으로 변환하여 TensorBoard에 기록
    """
    q_traj = q_traj.detach().cpu().numpy() # [Steps, Joints]
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    # 1. Joint Positions
    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f'J{i+1}')
    axes[0].set_title(f'MLP Joint Angles (Epoch {epoch})')
    axes[0].set_ylabel('Rad')
    axes[0].grid(True)
    axes[0].legend(loc='right', fontsize='small')
    
    # 2. Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f'J{i+1}')
    axes[1].set_title('MLP Joint Velocities')
    axes[1].set_ylabel('Rad/s')
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # 1. 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== MLP (Baseline) Training Start on {device} ===")
    
    # 로봇 로드
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # TensorBoard Writer (MLP용 로그 디렉토리)
    writer = SummaryWriter(log_dir="runs/mlp_debug_v1")
    
    # ==========================================
    # [설정] CVAE 실험과 동일하게 유지
    # ==========================================
    COND_DIM = 8  # Start(4) + Goal(4)
    NUM_WAYPOINTS = 4
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']
    
    # 디버깅용 파라미터 (빠른 실행 확인용)
    BATCH_SIZE = 10000        # (기존 32 -> 2)
    TOTAL_TIME = 1.0      # (기존 5.0 -> 1.0)
    NUM_EPOCHS = 5000
    
    # 2. 모델 및 물리 엔진 준비
    # MLP는 Latent Dim이 필요 없음
    model = MLP(COND_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 시각화용 고정 테스트 셋
    fixed_start = torch.tensor([[0., 0., 0., 1.]], device=device) 
    fixed_goal = torch.tensor([[0., 0., 0.7071, 0.7071]], device=device) 
    fixed_cond = torch.cat([fixed_start, fixed_goal], dim=1)
    
    # 3. 학습 루프
    total_start_time = time.time()
    epoch_start_time = time.time()

    # 손실 기록용 리스트
    train_losses = []
    val_losses = []
    epoch_durations = []
    
    for epoch in range(NUM_EPOCHS):
        # --- Training Step ---
        # random start and goal quaternion
        q0_start = torch.randn(BATCH_SIZE, 4, device=device)
        q0_start /= torch.norm(q0_start, dim=1, keepdim=True)
        q0_goal = torch.randn(BATCH_SIZE, 4, device=device)
        q0_goal /= torch.norm(q0_goal, dim=1, keepdim=True)
        
        condition = torch.cat([q0_start, q0_goal], dim=1)
        
        optimizer.zero_grad()
        
        # MLP Forward (Deterministic)
        waypoints_pred = model(condition)
        
        # Physics Simulation & Loss
        loss = physics.calculate_loss(waypoints_pred, q0_start, q0_goal)
        
        loss.backward()
        optimizer.step()
        
        # --- Logging ---
        loss_value = loss.item()
        train_losses.append(loss_value)
        writer.add_scalar('Loss/train', loss_value, epoch)
        
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss.item():.6f} | Time: {epoch_duration:.2f}s")
        epoch_start_time = time.time()
        
        # --- Validation & Visualization (10 에폭마다) ---
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                # 고정된 목표에 대해 추론 (z 없음)
                wp_vis = model(fixed_cond)
                
                q_traj, q_dot_traj = physics.generate_trajectory(wp_vis)
                val_loss = physics.calculate_loss(wp_vis, fixed_start, fixed_goal)
                val_value = val_loss.item()
                val_losses.append((epoch + 1, val_value))

                fig = plot_trajectory(q_traj[0], q_dot_traj[0], epoch+1)
                writer.add_figure('Trajectory/Fixed_Goal', fig, epoch)
                plt.close(fig)
                
                print(f"   >>> Validation Loss: {val_value:.6f}")
            
    print(f"Training Finished. Total Time: {time.time()-total_start_time:.2f}s")

    # === 학습 곡선 저장 ===
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # CSV 저장을 위한 타임스탬프
    timestamp = time.time()

    if len(train_losses) > 0:
        epochs = list(range(1, len(train_losses) + 1))
        
        # CSV 파일 저장
        csv_dir = os.path.join(plots_dir, "mlp_training_curve")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_path = os.path.join(csv_dir, f"{timestamp}.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'train_loss', 'epoch_duration', 'val_loss'])
            
            # Validation loss를 딕셔너리로 변환 (빠른 조회를 위해)
            val_dict = {e: v for e, v in val_losses}
            
            for i, (epoch, train_loss, duration) in enumerate(zip(epochs, train_losses, epoch_durations)):
                val_loss = val_dict.get(epoch, '')
                writer.writerow([epoch, train_loss, duration, val_loss])
        
        print(f"Training data saved to: {csv_path}")
        
        # 플롯 저장
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")

        if len(val_losses) > 0:
            val_epochs = [e for (e, _) in val_losses]
            val_values = [v for (_, v) in val_losses]
            plt.plot(val_epochs, val_values, label="Val Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MLP Training Curve")
        plt.grid(True)
        plt.legend()
        save_dir = os.path.join(plots_dir, "mlp_training_curve")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{timestamp}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    # 모델 저장
    save_dir = os.path.join("weights/cvae_debug")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{time.time()}.pth")
    torch.save(model.state_dict(), save_path)
    writer.close()

if __name__ == "__main__":
    main()