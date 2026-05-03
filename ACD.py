# coding: utf-8
import math
import torch
import torch.nn.functional as F

from collections import namedtuple
from torch import nn
from ACD_Denoiser import ACD_Denoiser
from ID import ID

# 【新增】导入能量模块
try:
    from Energy_models.part_energy import PartEnergyFunction, GlobalEnergyEvaluator
    from Energy_models.interaction_energy import InteractionEnergyFunction
    from Energy_models.energy_fusion import SynergisticFusion
except ImportError:
    PartEnergyFunction = None
    GlobalEnergyEvaluator = None
    InteractionEnergyFunction = None
    SynergisticFusion = None

__all__ = ["ACD"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class ACD(nn.Module):

    def __init__(self, args, trg_vocab):
        super().__init__()

        timesteps = args["diffusion"].get('timesteps', 1000)
        sampling_timesteps = args["diffusion"].get('sampling_timesteps', 5)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args["diffusion"].get('scale', 1.0)
        self.box_renewal = True
        self.use_ensemble = True
        
        # 【新增】检索增强配置
        
        # 【调试】打印配置读取情况（仅第一次）
        if not hasattr(self, '_config_debug_printed'):
            diffusion_cfg = args.get("diffusion", {})
            print(f"🔍 ACD Initialization - Diffusion Config:")
            print(f"   - use_retrieval from config: {diffusion_cfg.get('use_retrieval', 'NOT FOUND')}")
            print(f"   - retrieval_alpha_max from config: {diffusion_cfg.get('retrieval_alpha_max', 'NOT FOUND')}")
            self._config_debug_printed = True
        
        self.use_retrieval = args.get("diffusion", {}).get("use_retrieval", False)
        self.retrieval_alpha_max = args.get("diffusion", {}).get("retrieval_alpha_max", 0.4)  # 最大引导强度 40%
        
        # 【新增】Energy-guided Sampling 配置
        self.use_energy_guidance = args.get("diffusion", {}).get("use_energy_guidance", False)
        self.energy_guidance_alpha = args.get("diffusion", {}).get("energy_guidance_alpha", 0.1)
        
        if self.use_energy_guidance:
            if GlobalEnergyEvaluator is None:
                raise ImportError("GlobalEnergyEvaluator not found. Please ensure Energy_models are available.")
            # 【重构】使用统一的全局能量评估器（与 loss.py 保持一致）
            self.global_energy_evaluator = GlobalEnergyEvaluator(
                num_joints=178, 
                hidden_dim=256, 
                use_rigidity=True  # 启用骨骼刚性约束
            )
            print(f"✅ Energy-guided Sampling enabled! Alpha: {self.energy_guidance_alpha}")
        
        # 【调试】打印最终值（仅第一次）
        if not hasattr(self, '_config_final_debug_printed'):
            print(f"   - Final self.use_retrieval: {self.use_retrieval}")
            print(f"   - Final self.retrieval_alpha_max: {self.retrieval_alpha_max}")
            self._config_final_debug_printed = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # Above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.ACD_Denoiser = ACD_Denoiser(num_layers=args["diffusion"].get('num_layers', 2),
                                         num_heads=args["diffusion"].get('num_heads', 4),
                                         hidden_size=args["diffusion"].get('hidden_size', 512),
                                         ff_size=args["diffusion"].get('ff_size', 512),
                                         dropout=args["diffusion"].get('dropout', 0.1),
                                         emb_dropout=args["diffusion"]["embeddings"].get('dropout', 0.1),
                                         vocab_size=len(trg_vocab),
                                         freeze=False,
                                         trg_size=args.get('trg_size', 150),
                                         decoder_trg_trg_=True)
        
        # 【关键修复】不再建立双向引用，改为在 forward 中直接管理指标
        # self.ACD_Denoiser.parent_model = self  <-- 删除此行以避免 RecursionError

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

    def compute_energy_gradient(self, x_start):
        """
        计算 x_start 的能量梯度 ∇E(x_start)
        使用统一的全局能量评估器（内部整合动力学 + 骨骼刚性约束）
        
        x_start: (B, T, C) where C = 534
        return: gradient with same shape as x_start
        """
        if not self.use_energy_guidance:
            return torch.zeros_like(x_start)

        B, T, C = x_start.shape
        
        # 【修复】在 eval 模式下跳过能量引导，避免梯度计算错误
        if not self.global_energy_evaluator.training:
            return torch.zeros_like(x_start)
        
        x_start.requires_grad_(True)
        
        # 【重构】直接使用全局能量评估器，它会自动处理运动学特征和骨骼刚性
        total_energy = self.global_energy_evaluator(x_start).sum()  # scalar
        
        # 计算梯度 ∇E
        grad = torch.autograd.grad(total_energy, x_start, create_graph=True)[0]
        
        # 清除梯度标记，避免内存泄漏
        x_start.detach_()
        
        return grad

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, encoder_output, t, src_mask, trg_mask, retrieval_cond=None):
        x_t = x / self.scale

        # 【新增调试】检查 Denoiser 输出的动态性
        if not hasattr(self, '_denoiser_dynamics_check'):
            if x.shape[1] > 1:
                input_diff = (x[:, 1:] - x[:, :-1]).abs().mean().item()
                print(f"[ACD.model_predictions DEBUG]")
                print(f"  Input x frame_diff: {input_diff:.6f}")
                
                if retrieval_cond is not None and retrieval_cond.shape[1] > 1:
                    ret_diff = (retrieval_cond[:, 1:] - retrieval_cond[:, :-1]).abs().mean().item()
                    print(f"  Retrieval_cond frame_diff: {ret_diff:.6f}")

            # 【关键修改】传入 retrieval_cond
            pred_pose = self.ACD_Denoiser(encoder_output=encoder_output,
                                          trg_embed=x_t,
                                          src_mask=src_mask,
                                          trg_mask=trg_mask,
                                          t=t,
                                          retrieval_cond=retrieval_cond)

            if pred_pose.shape[1] > 1:
                frame_diff = (pred_pose[:, 1:] - pred_pose[:, :-1]).abs().mean().item()
                print(f"  Output pred_pose frame_diff: {frame_diff:.6f}")
                if frame_diff < 0.01:
                    print(f"  ⚠️ CRITICAL: Denoiser output lost dynamics!")
                else:
                    print(f"  ✅ Dynamics preserved in Denoiser")
            
            self._denoiser_dynamics_check = True
            return ModelPrediction(
                self.predict_noise_from_start(x, t, pred_pose * self.scale), 
                pred_pose * self.scale
            )
        
        # 【关键修改】传入 retrieval_cond
        pred_pose = self.ACD_Denoiser(encoder_output=encoder_output,
                                      trg_embed=x_t,
                                      src_mask=src_mask,
                                      trg_mask=trg_mask,
                                      t=t,
                                      retrieval_cond=retrieval_cond)

        x_start = pred_pose
        x_start = x_start * self.scale
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def ddim_sample(self, encoder_output, input_3d, src_mask, trg_mask, retrieved_poses=None):
        device = encoder_output.device
        batch = encoder_output.shape[0]
        shape = (batch, input_3d.shape[1], input_3d.shape[2])
        
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # 【调试】打印采样配置
        if not hasattr(self, '_ddim_config_debug'):
            print(f"\n{'='*60}")
            print(f"[DDIM SAMPLE CONFIG]")
            print(f"  sampling_timesteps: {sampling_timesteps}")
            print(f"  time_pairs length: {len(time_pairs)}")
            print(f"  retrieved_poses is None: {retrieved_poses is None}")
            if retrieved_poses is not None:
                print(f"  retrieved_poses shape: {retrieved_poses.shape}")
                if retrieved_poses.shape[1] > 1:
                    frame_diff = (retrieved_poses[:, 1:] - retrieved_poses[:, :-1]).abs().mean().item()
                    print(f"  retrieved_poses frame_diff: {frame_diff:.6f}")
                    if frame_diff < 0.001:
                        print(f"  ⚠️ WARNING: Retrieved poses are STATIC!")
            print(f"{'='*60}\n")
            self._ddim_config_debug = True

        img = torch.randn(shape, device=device)
        
        # 【调试】打印初始噪声
        if not hasattr(self, '_ddim_init_debug'):
            print(f"[DDIM DEBUG] Initial noise stats - Max: {img.max().item():.4f}, Min: {img.min().item():.4f}, Std: {img.std().item():.4f}")
            self._ddim_init_debug = True

        preds_all = []
        for step_idx, (time, time_next) in enumerate(time_pairs):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            # 【关键】每一步都传入 retrieved_poses，由 Denoiser 内部的 Cross-Attention 处理
            preds = self.model_predictions(x=img, 
                                           encoder_output=encoder_output, 
                                           t=time_cond, 
                                           src_mask=src_mask, 
                                           trg_mask=trg_mask,
                                           retrieval_cond=retrieved_poses)
            
            pred_noise, x_start = preds.pred_noise.float(), preds.pred_x_start
            
            # 【调试】打印每一步的 x_start 动态性
            if step_idx == 0 or step_idx == len(time_pairs) - 1:
                if not hasattr(self, f'_ddim_step_{step_idx}_debug'):
                    print(f"[DDIM DEBUG] Step {step_idx}/{len(time_pairs)-1} (t={time}):")
                    print(f"  x_start - Max: {x_start.max().item():.4f}, Min: {x_start.min().item():.4f}, Mean: {x_start.mean().item():.4f}")
                    if x_start.shape[1] > 1:
                        frame_diff = (x_start[:, 1:] - x_start[:, :-1]).abs().mean().item()
                        print(f"  x_start frame_diff: {frame_diff:.6f}")
                        if frame_diff < 0.001:
                            print(f"  ⚠️ x_start is STATIC at this step!")
                    setattr(self, f'_ddim_step_{step_idx}_debug', True)
            
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return preds_all

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, encoder_output, input_3d, src_mask, trg_mask, is_train, retrieved_poses=None, global_step=0):
        """
        Args:
            retrieved_poses: (B, T, C) Aggregated retrieval result
            global_step: Current training step for curriculum learning
        """
        if not is_train:
            # 推理阶段：传入 retrieved_poses
            results = self.ddim_sample(encoder_output=encoder_output, 
                                       input_3d=input_3d, 
                                       src_mask=src_mask, 
                                       trg_mask=trg_mask,
                                       retrieved_poses=retrieved_poses)
            return results[self.sampling_timesteps-1]

        if is_train:
            # 【关键修改】训练时不再修改 x_t，而是使用标准扩散流程
            # 检索结果将作为 Condition 传入 Denoiser 内部
            x_poses, noises, t = self.prepare_targets(input_3d)
                
            x_poses = x_poses.float()
            
            # 将 retrieved_poses 传给 Denoiser（作为 Cross-Attention 的 Key/Value）
            pred_pose = self.ACD_Denoiser(encoder_output=encoder_output,
                                          trg_embed=x_poses,
                                          src_mask=src_mask,
                                          trg_mask=trg_mask,
                                          t=t,
                                          retrieval_cond=retrieved_poses) 
            
            # 【关键修复】从 Denoiser 读取计算出的相似度并同步到 ACD 实例
            if hasattr(self.ACD_Denoiser, 'current_batch_avg_sim'):
                self.current_batch_avg_sim = self.ACD_Denoiser.current_batch_avg_sim
            
            # 【Step 4.2: Motion Loss (Velocity Loss)】
            # 计算预测姿态和 GT 的速度（帧间差值）
            vel_pred = pred_pose[:, 1:] - pred_pose[:, :-1]
            vel_gt = input_3d[:, 1:] - input_3d[:, :-1]
            
            # 返回预测结果和速度损失项
            return pred_pose, vel_pred, vel_gt

    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(pose_3d.shape[0], pose_3d.shape[1], device='cuda')

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = x / self.scale

        return x, noise, t

    def prepare_targets(self, targets):
        diffused_poses = []
        noises = []
        ts = []
        for i in range(0,targets.shape[0]):
            targets_per_sample = targets[i]

            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)

    def compute_adaptive_alpha(self, t, global_step=0):
        """
        计算自适应引导强度 alpha_t
        策略：
        1. 课程学习：根据 global_step 动态调整最大强度
        2. 早期门控：仅在去噪早期 (t > 0.5T) 生效
        """
        # 1. 课程学习：全局步数衰减
        # Stage 1: 0 - 10000 步，alpha = 0 (纯 Diffusion 学习)
        # Stage 2: 10000 - 30000 步，alpha 线性增至 0.2
        # Stage 3: > 30000 步，alpha 维持在 0.05 (微弱参考)
        if global_step < 10000:
            current_max_alpha = 0.0
        elif global_step < 30000:
            # 线性插值: 0.0 -> 0.2
            progress = (global_step - 10000) / 20000.0
            current_max_alpha = 0.2 * progress
        else:
            current_max_alpha = 0.05
        
        # 2. 去噪时间步门控
        t_normalized = t.float() / self.num_timesteps
        
        # 仅在去噪早期 (t > 0.5T) 提供引导，后期完全由模型主导
        alpha_t = torch.where(
            t_normalized > 0.5,
            torch.full_like(t_normalized, current_max_alpha),
            torch.zeros_like(t_normalized)
        )
        
        return alpha_t  # Shape: (B,)

    def prepare_targets_with_retrieval(self, targets, retrieved_poses, global_step=0):
        """
        训练时：在标准扩散噪声基础上加入检索引导
        """
        B, T, C = targets.shape
        device = targets.device
        
        # 1. 采样时间步
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        
        # 2. 生成标准噪声
        noise = torch.randn_like(targets)
        
        # 3. 标准扩散部分
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, targets.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, targets.shape)
        
        x_start = targets * self.scale
        standard_diffused = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # 4. 计算自适应引导强度 (传入 global_step)
        alpha_t = self.compute_adaptive_alpha(t, global_step)  # (B,)
        alpha_t = alpha_t.view(B, 1, 1)  # 广播到 (B, 1, 1)
        
        # 5. 检索引导部分
        # 对齐 retrieved_poses 的时间维度
        ret_poses_aligned = retrieved_poses.clone()
        if ret_poses_aligned.shape[1] != T:
            if ret_poses_aligned.shape[1] > T:
                ret_poses_aligned = ret_poses_aligned[:, :T, :]
            else:
                pad_len = T - ret_poses_aligned.shape[1]
                # 【修复】使用最后一帧复制填充，而不是 zeros
                last_frame = ret_poses_aligned[:, -1:, :]
                padding = last_frame.repeat(1, pad_len, 1)
                ret_poses_aligned = torch.cat([ret_poses_aligned, padding], dim=1)
        
        # 【新增】检测无效样本（全零或接近全零）
        # 如果检索结果是无效的（例如 model.py 传入的 zero poses），则不施加引导
        is_valid = (ret_poses_aligned.abs().sum(dim=[1, 2]) > 1e-6).float().view(B, 1, 1)
        
        retrieval_bias = alpha_t * (ret_poses_aligned * self.scale - x_start)
        
        # 【关键】只对有有效检索结果的样本施加 bias
        retrieval_bias = retrieval_bias * is_valid
        
        # 6. 融合
        x_t = standard_diffused + retrieval_bias
        x_t = x_t / self.scale
        
        return x_t, noise, t
