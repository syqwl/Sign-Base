# Energy_models/interaction_energy.py
import torch
import torch.nn as nn

class InteractionEnergyFunction(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        # 【方案二】输入特征扩展：
        # 相对位置(3) + 相对速度(3) + 相对加速度(3) + 相对Jerk(3) = 12
        self.network = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, left_hand, right_hand):
        """
        left_hand, right_hand: (B, T, N, 3)
        """
        B, T, N, C = left_hand.shape
        
        # 计算手部中心
        l_center = left_hand.mean(dim=2) # (B, T, 3)
        r_center = right_hand.mean(dim=2) # (B, T, 3)
        
        # 相对位置
        rel_pos = l_center - r_center # (B, T, 3)
        
        # 相对速度
        rel_vel = torch.zeros_like(rel_pos)
        if T > 1:
            rel_vel[:, 1:, :] = rel_pos[:, 1:, :] - rel_pos[:, :-1, :]
            
        # 相对加速度
        rel_acc = torch.zeros_like(rel_vel)
        if T > 2:
            rel_acc[:, 2:, :] = rel_vel[:, 2:, :] - rel_vel[:, 1:-1, :]
        
        # 【方案二】相对 Jerk (加加速度)
        rel_jerk = torch.zeros_like(rel_acc)
        if T > 3:
            rel_jerk[:, 3:, :] = rel_acc[:, 3:, :] - rel_acc[:, 2:-1, :]
            
        # 拼接特征: [rel_pos, rel_vel, rel_acc, rel_jerk]
        features = torch.cat([rel_pos, rel_vel, rel_acc, rel_jerk], dim=-1) # (B, T, 12)

        # 展平 Batch 和 Time 维度
        B, T, _ = features.shape
        features_flat = features.view(B * T, -1)
        
        energy = self.network(features_flat)
        return energy.view(B, T, 1)