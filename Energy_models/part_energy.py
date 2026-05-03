# Energy-models/part_energy.py
import torch
import torch.nn as nn

class PartEnergyFunction(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, use_dynamics=True):
        """
        :param input_dim: 基础坐标维度 (3 for x,y,z)
        :param hidden_dim: 隐藏层维度
        :param use_dynamics: 是否启用动力学特征输入（速度、加速度）
        """
        super().__init__()
        self.use_dynamics = use_dynamics
        
        # 【方案一】如果启用动力学，输入维度变为:
        # 3 (pos) + 3 (vel) + 3 (acc) + 3 (jerk) = 12
        # Jerk (加加速度) 是衡量运动平滑性的关键指标
        actual_input_dim = input_dim * 4 if use_dynamics else input_dim
        
        self.network = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出每个关节点的能量值
            nn.Softplus()  # 确保能量为非负值，形成能量地形图
        )

    def forward(self, joints):
        """
        joints: (B, T, N, C) 
        如果 use_dynamics=True，我们期望输入已经是拼接好的 (B, T, N, C*3) [pos+vel+acc]
        但我们在 loss.py 中会手动拼接好再调用。
        
        【方案一增强】这里假设传入的是 (B, T, N, 9)，我们会额外计算 Jerk 并拼接
        """
        # joints shape: (B, T, N, C_in)
        B, T, N, C = joints.shape
        
        # 【方案一】如果输入是动力学特征 (C=9)，提取 Acc 并计算 Jerk
        if self.use_dynamics and C == 9:
            # 提取位置、速度、加速度
            pos = joints[:, :, :, :3]
            vel = joints[:, :, :, 3:6]
            acc = joints[:, :, :, 6:9]
            
            # 计算 Jerk (加加速度): d(Acc)/dt ≈ Acc_t - Acc_{t-1}
            jerk = torch.zeros_like(acc)
            if T > 1:
                jerk[:, 1:, :] = acc[:, 1:, :] - acc[:, :-1, :]
            
            # 拼接完整特征: [Pos, Vel, Acc, Jerk]
            features = torch.cat([joints, jerk], dim=-1)  # (B, T, N, 12)
        else:
            # 如果没有启用动力学或输入不是9维，直接使用原始输入
            features = joints
        
        # 展平 Batch 和 Time 维度，对每个关节点独立计算能量
        joints_flat = features.view(B * T * N, -1)
        
        # 通过网络计算能量
        energy = self.network(joints_flat)  # (B*T*N, 1)
        
        # 恢复形状为 (B, T, N)
        energy = energy.view(B, T, N)
        
        return energy


class GlobalEnergyEvaluator(nn.Module):
    """
    全局能量评估器：接收完整骨架序列，输出标量能量值
    整合了动力学合理性 + 骨骼刚性约束 + Jerk平滑性约束
    """
    def __init__(self, num_joints=178, hidden_dim=256, use_rigidity=True):
        super().__init__()
        self.use_rigidity = use_rigidity
        self.num_joints = num_joints
        
        # 【方案一核心】输入特征扩展：
        # 位置(534) + 速度(534) + 加速度(534) + Jerk(534) + 骨骼长度方差(1) = 2137
        # Jerk 特征的加入使能量模型对高频抖动极其敏感
        input_dim = 534 * 4 + (1 if use_rigidity else 0)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        # 时序建模：使用简单的注意力机制或平均池化
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 最终回归头：输出标量能量
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保能量非负
        )
    
    def _compute_kinematics(self, joints):
        """
        计算速度、加速度和Jerk
        joints: (B, T, 534)
        return: vel (B, T, 534), acc (B, T, 534), jerk (B, T, 534)
        """
        B, T, C = joints.shape
        
        # 速度
        vel = torch.zeros_like(joints)
        if T > 1:
            vel[:, 1:, :] = joints[:, 1:, :] - joints[:, :-1, :]
        
        # 加速度
        acc = torch.zeros_like(joints)
        if T > 2:
            acc[:, 2:, :] = vel[:, 2:, :] - vel[:, 1:-1, :]
        
        # 【方案一】Jerk (加加速度)
        jerk = torch.zeros_like(joints)
        if T > 3:
            jerk[:, 3:, :] = acc[:, 3:, :] - acc[:, 2:-1, :]
        
        return vel, acc, jerk
    
    def _compute_bone_rigidity_feature(self, joints):
        """
        计算骨骼刚性特征：所有骨骼长度的方差均值
        joints: (B, T, 534) -> reshape to (B, T, 178, 3)
        return: (B, 1) 标量特征
        """
        from helpers import getSkeletalModelStructure
        
        B, T, C = joints.shape
        joints_reshaped = joints.view(B, T, 178, 3)
        
        skeletons = getSkeletalModelStructure()
        bone_variances = []
        
        for skeleton in skeletons:
            joint_a = joints_reshaped[:, :, skeleton[0], :]  # (B, T, 3)
            joint_b = joints_reshaped[:, :, skeleton[1], :]  # (B, T, 3)
            
            # 计算每帧的骨骼长度
            bone_lengths = torch.norm(joint_a - joint_b, dim=-1)  # (B, T)
            
            # 计算该骨骼在所有时间步上的长度方差
            mean_length = bone_lengths.mean(dim=1, keepdim=True)  # (B, 1)
            variance = ((bone_lengths - mean_length) ** 2).mean(dim=1, keepdim=True)  # (B, 1)
            
            bone_variances.append(variance)
        
        # 对所有骨骼的方差取平均
        avg_variance = torch.stack(bone_variances, dim=1).mean(dim=1)  # (B,)
        avg_variance = avg_variance.unsqueeze(-1)  # (B, 1)
        
        return avg_variance
    
    def forward(self, joints):
        """
        joints: (B, T, 534) 展平的骨架坐标
        return: (B, 1) 标量能量值
        """
        B, T, C = joints.shape
        
        # 1. 计算运动学特征（包含Jerk）
        vel, acc, jerk = self._compute_kinematics(joints)  # (B, T, 534) each
        
        # 2. 拼接特征：[pos, vel, acc, jerk]
        features = torch.cat([joints, vel, acc, jerk], dim=-1)  # (B, T, 2136)
        
        # 3. 【新增】添加骨骼刚性特征
        if self.use_rigidity:
            rigidity_feat = self._compute_bone_rigidity_feature(joints)  # (B, 1)
            # 广播到所有时间步：(B, 1) -> (B, T, 1)
            rigidity_feat_expanded = rigidity_feat.view(B, 1, 1).expand(B, T, 1)  # (B, T, 1)
            features = torch.cat([features, rigidity_feat_expanded], dim=-1)  # (B, T, 2137)
        
        # 4. 编码特征
        encoded = self.encoder(features)  # (B, T, 128)
        
        # 5. 时序池化
        encoded_transposed = encoded.transpose(1, 2)  # (B, 128, T)
        pooled = self.temporal_pool(encoded_transposed).squeeze(-1)  # (B, 128)
        
        # 6. 回归能量标量
        energy = self.regression_head(pooled)  # (B, 1)
        
        return energy
