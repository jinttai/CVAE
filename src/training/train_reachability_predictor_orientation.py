"""
Reachability Predictor Training Script (Orientation-only version)

데이터: outputs/data/orientation_reachable_set.pt
- start_joint, goal_joint = 0으로 고정
- waypoint만 랜덤 생성
- q_final: 도달 가능한 orientation

모델 입력: query_quat(4) only
출력: reachability score (0 ~ π, 낮을수록 도달 쉬움)

목적: waypoint만 변할 때 어떤 orientation에 도달 가능한지 예측
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
from tqdm import tqdm

# Add root directory to sys.path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)


class OrientationReachabilityPredictor(nn.Module):
    """
    MLP-based Orientation Reachability Predictor
    
    입력: query_quat(4) only
    출력: reachability score (스칼라, 0에 가까우면 도달 가능)
    
    Note: start/goal joint가 0으로 고정되어 있으므로
          입력에서 joint 정보 없이 orientation만 사용
    """
    
    def __init__(self, input_dim=4, hidden_dim=256, num_layers=3):
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
    
    def predict(self, query_quat):
        """편의 함수"""
        if query_quat.dim() == 1:
            query_quat = query_quat.unsqueeze(0)
        return self.forward(query_quat)


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


def quaternion_distance(q1, q2):
    """
    두 quaternion 사이의 각도 거리 계산
    
    distance = 2 * arccos(|q1 · q2|), 범위 [0, π]
    """
    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    distance = 2.0 * torch.acos(dot)
    return distance


def compute_min_distance_to_all(query, all_qfinals):
    """
    query quaternion과 모든 q_final 중 가장 가까운 것과의 거리
    
    Args:
        query: [4] tensor
        all_qfinals: [N, 4] tensor (전체 도달 가능한 자세들)
    
    Returns:
        min_distance: float (0 ~ π)
    """
    dot = torch.abs(torch.sum(query.unsqueeze(0) * all_qfinals, dim=1))
    dot = dot.clamp(-1, 1)
    distances = 2 * torch.acos(dot)
    return distances.min().item()


class OrientationReachabilityDataset(Dataset):
    """
    Orientation Reachability 학습을 위한 Dataset
    
    데이터: waypoints, q_final (start/goal joint = 0 고정)
    
    전반부 (idx < N): Positive 샘플
        - query = q_final[idx]
        - label = 0.0
    
    후반부 (idx >= N): Negative 샘플
        - query = random_quaternion
        - label = min_distance(query, 전체 q_finals)
    """
    
    def __init__(self, data_path, use_all_for_min_dist=True):
        """
        Args:
            data_path: orientation_reachable_set.pt 경로
            use_all_for_min_dist: True면 전체 q_final과 비교, False면 해당 샘플만
        """
        print(f"Loading data from: {data_path}")
        data = torch.load(data_path, map_location='cpu')
        
        self.waypoints = data['waypoints']  # [N, 18]
        self.q_final = data['q_final']      # [N, 4]
        
        self.num_samples = self.q_final.shape[0]
        self.use_all_for_min_dist = use_all_for_min_dist
        
        print(f"Loaded {self.num_samples:,} samples")
        print(f"  waypoints: {self.waypoints.shape}")
        print(f"  q_final: {self.q_final.shape}")
        print(f"  use_all_for_min_dist: {use_all_for_min_dist}")
    
    def __len__(self):
        return self.num_samples * 2  # positive + negative
    
    def __getitem__(self, idx):
        real_idx = idx % self.num_samples
        is_positive = idx < self.num_samples
        
        if is_positive:
            # Positive: 실제 도달한 quaternion
            query = self.q_final[real_idx]
            label = torch.tensor(0.0)
        else:
            # Negative: 랜덤 quaternion
            query = random_quaternion()
            
            if self.use_all_for_min_dist:
                # 전체 q_final과의 최소 거리 (비용이 크지만 정확)
                min_dist = compute_min_distance_to_all(query, self.q_final)
            else:
                # 해당 샘플의 q_final과의 거리 (빠르지만 근사)
                q_true = self.q_final[real_idx]
                min_dist = quaternion_distance(
                    query.unsqueeze(0), q_true.unsqueeze(0)
                ).squeeze(0).item()
            
            label = torch.tensor(min_dist, dtype=torch.float32)
        
        return query, label


class OrientationReachabilityDatasetFast(Dataset):
    """
    빠른 버전: 전체 비교 대신 샘플별 비교만 수행
    
    Negative 샘플의 label = distance(random_quat, q_final[idx])
    """
    
    def __init__(self, data_path):
        print(f"Loading data from: {data_path}")
        data = torch.load(data_path, map_location='cpu')
        
        self.q_final = data['q_final']  # [N, 4]
        self.num_samples = self.q_final.shape[0]
        
        print(f"Loaded {self.num_samples:,} samples")
        print(f"  q_final: {self.q_final.shape}")
    
    def __len__(self):
        return self.num_samples * 2
    
    def __getitem__(self, idx):
        real_idx = idx % self.num_samples
        is_positive = idx < self.num_samples
        
        if is_positive:
            query = self.q_final[real_idx]
            label = torch.tensor(0.0)
        else:
            query = random_quaternion()
            q_true = self.q_final[real_idx]
            dist = quaternion_distance(
                query.unsqueeze(0), q_true.unsqueeze(0)
            ).squeeze(0)
            label = dist
        
        return query, label


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
    parser = argparse.ArgumentParser(description="Train Orientation Reachability Predictor")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to orientation_reachable_set.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.01)
    parser.add_argument("--fast-mode", action="store_true",
                        help="Use fast dataset (per-sample distance instead of min over all)")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    
    # Windows에서는 num_workers>0 시 워커 크래시로 학습이 끊기는 경우가 있음 → 기본 0
    if args.num_workers is None:
        args.num_workers = 0 if os.name == "nt" else 4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Orientation Reachability Predictor Training on {device} ===")
    
    # 경로 설정
    if args.data_path is None:
        data_path = os.path.join(ROOT_DIR, "outputs/data/orientation_reachable_set.pt")
    else:
        data_path = args.data_path
    
    weights_dir = os.path.join(ROOT_DIR, "outputs/weights/reachability_predictor_orientation")
    os.makedirs(weights_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if not args.no_tensorboard:
        log_dir = os.path.join(ROOT_DIR, "outputs/logs/reachability_predictor_orientation")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")
    
    # Dataset & DataLoader
    if args.fast_mode:
        print("Using FAST mode (per-sample distance)")
        dataset = OrientationReachabilityDatasetFast(data_path)
    else:
        print("Using ACCURATE mode (min distance to all q_finals)")
        dataset = OrientationReachabilityDataset(data_path, use_all_for_min_dist=False)
    
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
    model = OrientationReachabilityPredictor(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel:")
    print(f"  Input dim: 4 (quaternion only)")
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
    print(f"Weights saved to: {weights_dir}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
