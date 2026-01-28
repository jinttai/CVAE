"""
Reachability Predictor Training Script (Bin-based version)

전처리된 데이터(reachable_set_binned.pt)를 사용하여 학습
Negative 샘플의 label은 같은 bin 내 모든 q_final 중 최소 거리

입력: start_joint(6) + goal_joint(6) + query_quat(4) = 16D
출력: reachability score (0 ~ π, 낮을수록 도달 쉬움)
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


class ReachabilityPredictor(nn.Module):
    """
    MLP-based Reachability Predictor
    
    입력: start_joint(6) + goal_joint(6) + query_quat(4) = 16D
    출력: reachability score (스칼라, 0에 가까우면 도달 가능)
    """
    
    def __init__(self, input_dim=16, hidden_dim=256, num_layers=5):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (마지막은 activation 없음)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, start_joint, goal_joint, query_quat):
        """편의 함수"""
        if start_joint.dim() == 1:
            start_joint = start_joint.unsqueeze(0)
            goal_joint = goal_joint.unsqueeze(0)
            query_quat = query_quat.unsqueeze(0)
        
        x = torch.cat([start_joint, goal_joint, query_quat], dim=-1)
        return self.forward(x)


def random_quaternion():
    """SO(3) 균등 분포 랜덤 quaternion (x,y,z,w)"""
    u = torch.rand(3)
    q = torch.stack([
        torch.sqrt(1 - u[0]) * torch.sin(2 * math.pi * u[1]),
        torch.sqrt(1 - u[0]) * torch.cos(2 * math.pi * u[1]),
        torch.sqrt(u[0]) * torch.sin(2 * math.pi * u[2]),
        torch.sqrt(u[0]) * torch.cos(2 * math.pi * u[2]),
    ])
    return q


def compute_min_distance(query, qfinals):
    """
    query quaternion과 qfinals 중 가장 가까운 것과의 거리
    
    Args:
        query: [4] tensor (x,y,z,w)
        qfinals: [K, 4] tensor 또는 None
    
    Returns:
        min_distance: float (0 ~ π)
    """
    if qfinals is None or len(qfinals) == 0:
        return math.pi  # bin이 비어있으면 최대 거리
    
    # 배치 내적 계산
    dot = torch.abs(torch.sum(query.unsqueeze(0) * qfinals, dim=1))  # [K]
    dot = dot.clamp(-1, 1)
    distances = 2 * torch.acos(dot)  # [K]
    
    return distances.min().item()


class ReachabilityDataset(Dataset):
    """
    Reachability 학습을 위한 Dataset (Bin-based)
    
    전반부 (idx < N): Positive 샘플
        - query = q_final[idx]
        - label = 0.0
    
    후반부 (idx >= N): Negative 샘플
        - query = random_quaternion
        - label = min_distance(query, bin의 모든 q_finals)
    """
    
    def __init__(self, data_path):
        print(f"Loading preprocessed data from: {data_path}")
        data = torch.load(data_path, map_location='cpu')
        
        self.start_joint = data['start_joint']  # [N, 6]
        self.goal_joint = data['goal_joint']    # [N, 6]
        self.q_final = data['q_final']          # [N, 4]
        self.bin_indices = data['bin_indices']  # [N]
        self.bin_to_qfinals = data['bin_to_qfinals']  # {bin_idx: [K, 4]}
        
        self.num_samples = self.start_joint.shape[0]
        
        print(f"Loaded {self.num_samples:,} samples")
        print(f"  Non-empty bins: {len(self.bin_to_qfinals):,}")
    
    def __len__(self):
        return self.num_samples * 2  # positive + negative
    
    def __getitem__(self, idx):
        real_idx = idx % self.num_samples
        is_positive = idx < self.num_samples
        
        start = self.start_joint[real_idx]
        goal = self.goal_joint[real_idx]
        bin_idx = self.bin_indices[real_idx].item()
        
        if is_positive:
            # Positive: 실제 도달한 quaternion
            query = self.q_final[real_idx]
            label = torch.tensor(0.0)
        else:
            # Negative: 랜덤 quaternion, 같은 bin 내 최소 거리
            query = random_quaternion()
            qfinals_in_bin = self.bin_to_qfinals.get(bin_idx, None)
            min_dist = compute_min_distance(query, qfinals_in_bin)
            label = torch.tensor(min_dist, dtype=torch.float32)
        
        # 입력 concatenate
        x = torch.cat([start, goal, query], dim=-1)  # [16]
        
        return x, label


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None):
    """한 epoch 학습"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (x, labels) in enumerate(pbar):
        x = x.to(device)
        labels = labels.to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        
        pred = model(x)
        loss = criterion(pred, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/batch', loss.item(), global_step)
    
    return total_loss / num_batches


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
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Reachability Predictor (Bin-based)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to reachable_set_binned.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.01,
                        help="Validation split ratio (default: 0.01 = 1%)")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Reachability Predictor Training (Bin-based) on {device} ===")
    
    # 경로 설정
    if args.data_path is None:
        data_path = os.path.join(ROOT_DIR, "outputs/data/reachable_set_binned.pt")
    else:
        data_path = args.data_path
    
    weights_dir = os.path.join(ROOT_DIR, "outputs/weights/reachability_predictor")
    os.makedirs(weights_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if not args.no_tensorboard:
        log_dir = os.path.join(ROOT_DIR, "outputs/logs/reachability_predictor")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")
    
    # Dataset & DataLoader
    dataset = ReachabilityDataset(data_path)
    
    # Train/Val split
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
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
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    
    # Model
    model = ReachabilityPredictor(
        input_dim=16,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel:")
    print(f"  Input dim: 16")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Parameters: {num_params:,}")
    
    # Optimizer & Scheduler & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    print(f"\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    # Training
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)
        
        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(weights_dir, "best.pth"))
            print(f"  >>> New best model (val_loss: {val_loss:.6f})")
        
        # Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(weights_dir, f"epoch_{epoch+1}.pth"))
            print(f"  >>> Checkpoint saved")
    
    # Final save
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(weights_dir, "final.pth"))
    
    print(f"\n=== Training Complete ===")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.6f}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
