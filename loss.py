# coding: utf-8
import torch
import torch.nn as nn

from helpers import getSkeletalModelStructure
from Energy_models.part_energy import PartEnergyFunction, GlobalEnergyEvaluator
from Energy_models.interaction_energy import InteractionEnergyFunction
from Energy_models.energy_fusion import SynergisticFusion

class Loss(nn.Module):

    def __init__(self, cfg, target_pad=0.0):
        super(Loss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()
        self.bone_loss = cfg["training"]["bone_loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        if self.bone_loss == "l1":
            self.criterion_bone = nn.L1Loss()
        elif self.bone_loss == "mse":
            self.criterion_bone = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_bone = nn.MSELoss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

        # GT-aware Energy Configuration
        self.use_energy = cfg["training"].get("use_energy", False)
        self.energy_weight = cfg["training"].get("energy_weight", 0.1)
        
        # 【新增】骨骼刚性约束配置
        self.use_rigidity = cfg["training"].get("use_rigidity", True)
        self.rigidity_weight = cfg["training"].get("rigidity_weight", 1.0)
        
        if self.use_energy:
            # 【重构】使用统一的全局能量评估器，内部整合骨骼刚性约束
            # 注意：这里假设 GlobalEnergyEvaluator 能够处理输入维度。
            # 如果 _compute_kinematics 返回的是 (B, T, N, 9)，则 input_dim 应适配或在该类内部处理。
            # 根据参考方案，我们直接初始化它。
            self.global_energy_evaluator = GlobalEnergyEvaluator(
                num_joints=178, 
                hidden_dim=256, 
                use_rigidity=True  # 启用骨骼刚性约束
            )
            
            print("✅ Unified Energy Evaluator enabled! (Dynamics + Bone Rigidity)")
        
        if self.use_rigidity:
            print(f"✅ Bone Rigidity Constraint enabled! Weight: {self.rigidity_weight}")
        
    def _compute_kinematics(self, joints):
        """
        计算关节点的速度和加速度，并拼接成增强输入
        joints: (B, T, N, 3)
        return: (B, T, N, 9) -> [pos, vel, acc]
        """
        B, T, N, C = joints.shape
        
        # 1. 位置 (Position)
        pos = joints
        
        # 2. 速度 (Velocity): y_t - y_{t-1}
        # 使用 pad 保持维度一致，第一帧速度设为 0
        vel = torch.zeros_like(joints)
        if T > 1:
            vel[:, 1:, :] = joints[:, 1:, :] - joints[:, :-1, :]
        
        # 3. 加速度 (Acceleration): v_t - v_{t-1}
        acc = torch.zeros_like(joints)
        if T > 2:
            acc[:, 2:, :] = vel[:, 2:, :] - vel[:, 1:-1, :]
            
        # 拼接: (B, T, N, 9)
        kinematics_input = torch.cat([pos, vel, acc], dim=-1)
        return kinematics_input

    def forward(self, preds, targets):
        """
        preds: (B, T, C) where C is flattened joints (e.g., 534 = 178 * 3)
        targets: (B, T, C)
        """
        # 用于记录步数，方便定期打印和 Warm-up
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        # 1. Masking
        loss_mask = (targets != self.target_pad).float()
        
        # 2. Reshape to (B, T, 178, 3)
        B, T, _ = preds.shape
        preds_reshaped = preds.view(B, T, 178, 3)
        targets_reshaped = targets.view(B, T, 178, 3)
        
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask
        
        # Bone Loss calculation
        preds_masked_length, preds_masked_direct = get_length_direct(preds_masked)
        targets_masked_length, targets_masked_direct = get_length_direct(targets_masked)
        
        # Reconstruction Loss + Bone Loss (Always Positive)
        rec_loss = self.criterion(preds_masked, targets_masked)
        bone_loss_val = 0.1 * self.criterion_bone(preds_masked_direct, targets_masked_direct)
        loss = rec_loss + bone_loss_val

        # 3. Unified Energy Manifold Learning (EBM Ranking)
        if self.use_energy:
            # 【方案四】恢复 Warm-up 策略，避免初期梯度冲突
            warmup_steps = 1000
            current_weight = self.energy_weight * min(1.0, self.step_count / warmup_steps)

            # --- 【方案三】构造硬负样本 (Hard Negative Mining) ---
            # 混合使用两种噪声：全局变形 + 高频抖动
            # 这强迫 Energy Model 必须对"抖动"敏感
            
            # 1. 标准高斯噪声 (Global Distortion)
            noise_global = torch.randn_like(targets_reshaped) * 0.1
            
            # 2. 高频抖动噪声 (Jitter Noise) - 模拟视频中的抖动问题
            jitter_scale = 0.05  # 小幅度的独立帧间噪声
            noise_jitter = torch.randn_like(targets_reshaped) * jitter_scale
            
            # 3. 随机混合：50% 概率使用全局变形，50% 概率使用高频抖动
            mask_jitter = (torch.rand(B, 1, 1, 1, device=targets.device) > 0.5).float()
            perturbed_targets = targets_reshaped + (1 - mask_jitter) * noise_global + mask_jitter * noise_jitter
            
            # --- 使用统一的全局能量评估器 ---
            # GlobalEnergyEvaluator 期望接收 (B, T, 534) 格式的展平骨架
            targets_flat = targets_reshaped.view(B, T, -1)  # (B, T, 534)
            perturbed_flat = perturbed_targets.view(B, T, -1)  # (B, T, 534)
            
            t_total_e = self.global_energy_evaluator(targets_flat)  # (B, 1)
            p_total_e = self.global_energy_evaluator(perturbed_flat)  # (B, 1)
            
            # --- 【核心】EBM Ranking Loss ---
            # 目标: E(GT) < E(Perturbed) - margin
            # 由于 Perturbed 包含 Jitter，模型会学到：Jitter = High Energy
            margin = 0.5
            ebm_loss = torch.nn.functional.relu(t_total_e - p_total_e + margin).mean()
            
            loss += current_weight * ebm_loss
            
            # Debug logging
            if self.step_count % 100 == 0:
                print(f"\n[ENERGY DEBUG] Step {self.step_count}:")
                print(f"   - Rec Loss: {rec_loss.item():.4f} | Bone Loss: {bone_loss_val.item():.4f}")
                print(f"   - Pos Energy (GT): {t_total_e.mean().item():.4f} | Neg Energy (Perturbed): {p_total_e.mean().item():.4f}")
                print(f"   - EBM Gap: {(p_total_e.mean() - t_total_e.mean()).item():.4f}")
                print(f"   - Current Weight: {current_weight:.4f} | Weighted Energy Contribution: {current_weight * ebm_loss.item():.4f}")
                print(f"   - Total Loss: {loss.item():.4f}\n")

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

def get_length_direct(trg):
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 178, 3)
    trg_list = trg_reshaped.split(1, dim=2)
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]
    skeletons = getSkeletalModelStructure()

    length = []
    direct = []
    for skeleton in skeletons:
        result_length = Skeleton_length = torch.norm(trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]], p=2, dim=2, keepdim=True)
        result_direct = (trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]]) / (Skeleton_length+torch.finfo(Skeleton_length.dtype).tiny)
        direct.append(result_direct)
        length.append(result_length)
    lengths = torch.stack(length, dim=-1).squeeze()
    directs = torch.stack(direct, dim=2).view(trg.shape[0], trg.shape[1], -1)

    return lengths, directs
