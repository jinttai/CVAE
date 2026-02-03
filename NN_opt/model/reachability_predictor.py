"""
Reachability Predictor Model

입력: start_joint(6) + goal_joint(6) + query_quat(4) = 16D
출력: reachability score (스칼라, 낮을수록 도달 쉬움)
"""

import torch
import torch.nn as nn


class ReachabilityPredictor(nn.Module):
    """
    MLP-based Reachability Predictor
    
    주어진 (start_joint, goal_joint)에서 query_quat에 도달할 수 있는지 예측
    출력값이 낮을수록 도달 가능성이 높음 (0 = 정확히 도달 가능)
    """
    
    def __init__(self, input_dim=16, hidden_dim=256, num_layers=5):
        """
        Args:
            input_dim: 입력 차원 (default: 16 = 6+6+4)
            hidden_dim: 히든 레이어 차원 (default: 256)
            num_layers: 히든 레이어 수 (default: 5)
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (스칼라 출력)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, 16] - concatenated (start_joint, goal_joint, query_quat)
        
        Returns:
            score: [batch_size, 1] - reachability score
        """
        return self.network(x)
    
    def predict(self, start_joint, goal_joint, query_quat):
        """
        편의 함수: 개별 입력을 받아서 예측
        
        Args:
            start_joint: [batch_size, 6] or [6]
            goal_joint: [batch_size, 6] or [6]
            query_quat: [batch_size, 4] or [4]
        
        Returns:
            score: [batch_size, 1] or [1]
        """
        # 1D 입력을 2D로 변환
        if start_joint.dim() == 1:
            start_joint = start_joint.unsqueeze(0)
            goal_joint = goal_joint.unsqueeze(0)
            query_quat = query_quat.unsqueeze(0)
        
        x = torch.cat([start_joint, goal_joint, query_quat], dim=-1)
        return self.forward(x)
