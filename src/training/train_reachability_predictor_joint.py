"""
Reachability Predictor Training Script (Joint-based version)

데이터: outputs/data/reachable_set.pt (data_generation_joint.py에서 생성)
모델: ReachabilityPredictor (MLP)
학습 목표: (start_joint, goal_joint, query_quat) → reachability score

입력: start_joint(6) + goal_joint(6) + query_quat(4) = 16D
출력: reachability score (스칼라, 낮을수록 도달 쉬움)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import math
import argparse
import numpy as np
from tqdm import tqdm

# Add root directory to sys.path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.models.reachability_predictor import ReachabilityPredictor


def random_quaternion_batch(batch_size, device='cpu'):
    """
    SO(3)에서 균등 분포로 quaternion 샘플링 (batch 버전)
    
    Reference: "Uniform Random Rotations" - Shoemake, K. (1992)
    
    Args:
        batch_size: 배치 크기
        device: torch device
    
    Returns:
        q: [batch_size, 4] quaternion (x, y, z, w)
    """
    u = torch.rand(batch_size, 3, device=device)
    
    sqrt_1_u0 = torch.sqrt(1 - u[:, 0])
    sqrt_u0 = torch.sqrt(u[:, 0])
    two_pi_u1 = 2 * math.pi * u[:, 1]
    two_pi_u2 = 2 * math.pi * u[:, 2]
    
    qx = sqrt_1_u0 * torch.sin(two_pi_u1)
    qy = sqrt_1_u0 * torch.cos(two_pi_u1)
    qz = sqrt_u0 * torch.sin(two_pi_u2)
    qw = sqrt_u0 * torch.cos(two_pi_u2)
    
    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return q


def quaternion_distance(q1, q2):
    """
    두 quaternion 사이의 각도 거리 계산
    
    distance = 2 * arccos(|q1 · q2|), 범위 [0, π]
    
    Args:
        q1: [batch_size, 4] quaternion
        q2: [batch_size, 4] quaternion
    
    Returns:
        distance: [batch_size] 각도 거리 (radians)
    """
    # 내적의 절댓값 (q와 -q는 같은 회전)
    dot = torch.sum(q1 * q2, dim=-1).abs()
    # clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    # 각도 거리
    distance = 2.0 * torch.acos(dot)
    return distance


class ReachabilityDatasetJoint(Dataset):
    """
    Reachability 학습을 위한 Dataset (Joint-based version)
    
    데이터: start_joint, goal_joint, waypoints, q_final
    
    Positive 샘플 (짝수 idx):
        - 입력: (start_joint, goal_joint, q_final)
        - 라벨: 0.0
    
    Negative 샘플 (홀수 idx):
        - 입력: (start_joint, goal_joint, random_quaternion)
        - 라벨: quaternion_distance(random_quat, q_final)
    """
    
    def __init__(self, data_path, device='cpu'):
        """
        Args:
            data_path: reachable_set.pt 경로
            device: torch device
        """
        print(f"Loading data from: {data_path}")
        data = torch.load(data_path, map_location='cpu')
        
        self.start_joint = data['start_joint']  # [N, 6]
        self.goal_joint = data['goal_joint']    # [N, 6]
        self.q_final = data['q_final']          # [N, 4]
        
        self.num_samples = self.start_joint.shape[0]
        self.device = device
        
        print(f"Loaded {self.num_samples:,} samples")
        print(f"  start_joint: {self.start_joint.shape}")
        print(f"  goal_joint: {self.goal_joint.shape}")
        print(f"  q_final: {self.q_final.shape}")
    
    def __len__(self):
        # positive + negative = 2 * N
        return self.num_samples * 2
    
    def __getitem__(self, idx):
        """
        idx가 짝수: positive 샘플
        idx가 홀수: negative 샘플
        """
        is_positive = (idx % 2 == 0)
        data_idx = idx // 2
        
        start = self.start_joint[data_idx]
        goal = self.goal_joint[data_idx]
        q_true = self.q_final[data_idx]
        
        if is_positive:
            # Positive: 실제 도달한 quaternion
            query_quat = q_true
            label = torch.tensor(0.0)
        else:
            # Negative: 랜덤 quaternion
            query_quat = random_quaternion_batch(1, device='cpu').squeeze(0)
            label = quaternion_distance(query_quat.unsqueeze(0), q_true.unsqueeze(0)).squeeze(0)
        
        # 입력 concatenate
        x = torch.cat([start, goal, query_quat], dim=-1)  # [16]
        
        return x, label


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None):
    """한 epoch 학습"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (x, labels) in enumerate(pbar):
        x = x.to(device)
        labels = labels.to(device).unsqueeze(-1)  # [B] -> [B, 1]
        
        optimizer.zero_grad()
        
        pred = model(x)
        loss = criterion(pred, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # TensorBoard 로깅 (매 100 batch)
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/batch', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device).unsqueeze(-1)
            
            pred = model(x)
            loss = criterion(pred, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Reachability Predictor (Joint-based)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to reachable_set.pt (default: outputs/data/reachable_set.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size (default: 4096)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer dimension (default: 256)")
    parser.add_argument("--num-layers", type=int, default=5,
                        help="Number of hidden layers (default: 5)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader num_workers (default: 0 on Windows, 4 elsewhere)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Checkpoint save interval (default: 10 epochs)")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    args = parser.parse_args()
    
    # Windows에서는 num_workers>0 시 워커 크래시로 학습이 끊기는 경우가 있음 → 기본 0
    if args.num_workers is None:
        args.num_workers = 0 if os.name == "nt" else 4
    
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Reachability Predictor Training (Joint-based) on {device} ===")
    
    # 경로 설정
    if args.data_path is None:
        data_path = os.path.join(ROOT_DIR, "outputs/data/reachable_set.pt")
    else:
        data_path = args.data_path
    
    weights_dir = os.path.join(ROOT_DIR, "outputs/weights/reachability_predictor_joint")
    os.makedirs(weights_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if not args.no_tensorboard:
        log_dir = os.path.join(ROOT_DIR, "outputs/logs/reachability_predictor_joint")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")
    
    # Dataset & DataLoader
    dataset = ReachabilityDatasetJoint(data_path, device=device)
    
    # Train/Val split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val: {len(val_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Model: input_dim=16 (start_joint[6] + goal_joint[6] + query_quat[4])
    model = ReachabilityPredictor(
        input_dim=16,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"\nModel:")
    print(f"  Input dim: 16 (start_joint[6] + goal_joint[6] + query_quat[4])")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Optimizer & Scheduler & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    print(f"\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: Adam")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Loss: MSELoss")
    print()
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # TensorBoard 로깅
        if writer is not None:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)
        
        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(weights_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, save_path)
            print(f"  >>> New best model saved (val_loss: {val_loss:.6f})")
        
        # Checkpoint 저장 (매 save_interval epoch)
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, save_path)
            print(f"  >>> Checkpoint saved: {save_path}")
    
    # Final model 저장
    total_time = time.time() - start_time
    save_path = os.path.join(weights_dir, "final.pth")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
    }, save_path)
    
    print(f"\n=== Training Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Final model saved: {save_path}")
    
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
