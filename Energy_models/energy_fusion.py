# Energy-models/energy_fusion.py
import torch
import torch.nn as nn

class SynergisticFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 可学习权重
        self.w_face = nn.Parameter(torch.tensor(1.0))
        self.w_left = nn.Parameter(torch.tensor(1.0))
        self.w_right = nn.Parameter(torch.tensor(1.0))
        self.w_interaction = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, face_energy, left_energy, right_energy, interaction_energy):
        """
        face_energy: (B, T, 128)
        left_energy: (B, T, 21)
        right_energy: (B, T, 21)
        interaction_energy: (B, T, 1)
        """
        # 对齐维度
        face_energy = face_energy.mean(dim=-1, keepdim=True)  # (B, T, 1)
        left_energy = left_energy.mean(dim=-1, keepdim=True)
        right_energy = right_energy.mean(dim=-1, keepdim=True)
        
        total_energy = (
            self.w_face * face_energy +
            self.w_left * left_energy +
            self.w_right * right_energy +
            self.w_interaction * interaction_energy
        )
        
        return total_energy  # (B, T, 1)