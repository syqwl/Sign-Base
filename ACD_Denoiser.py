# coding: utf-8
import torch.nn as nn
import torch
import math
from torch import Tensor

import torch.nn.functional as F
from helpers import freeze_params, subsequent_mask
from transformer_layers import PositionalEncoding, TransformerDecoderLayer, MultiHeadedAttention

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ACD_Denoiser(nn.Module):

    def __init__(self,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 trg_size: int = 150,
                 decoder_trg_trg_: bool = True,
                 **kwargs):
        super(ACD_Denoiser, self).__init__()

        self.in_feature_size = trg_size #+ (trg_size // 3) * 4
        self.out_feature_size = trg_size

        self.pos_drop = nn.Dropout(p=emb_dropout)
        self.trg_embed = nn.Linear(self.in_feature_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=i) for i in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            freeze_params(self)
        
        # 【新增】检索条件的投影层和 Cross-Attention 模块
        self.retrieval_proj = nn.Linear(trg_size, hidden_size) if hidden_size != trg_size else nn.Identity()
        self.retrieval_cross_attn = MultiHeadedAttention(num_heads, hidden_size, dropout=dropout)
        self.retrieval_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self,
                t,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                retrieval_cond: Tensor = None,
                **kwargs):

        assert trg_mask is not None, "trg_mask required for Transformer"
        
        # 【关键修复】如果 retrieval_cond 存在但 Batch 维度不匹配，进行广播或丢弃
        if retrieval_cond is not None:
            if retrieval_cond.shape[0] != trg_embed.shape[0]:
                print(f"[WARNING] retrieval_cond batch size ({retrieval_cond.shape[0]}) mismatch with trg_embed ({trg_embed.shape[0]}). Setting to None.")
                retrieval_cond = None
        
        # 【调试】打印输入动态性
        if not hasattr(self, '_denoiser_input_debug'):
            if trg_embed.shape[1] > 1:
                input_diff = (trg_embed[:, 1:] - trg_embed[:, :-1]).abs().mean().item()
                print(f"[DENOISER INPUT DEBUG]")
                print(f"  trg_embed frame_diff: {input_diff:.6f}")
                if retrieval_cond is not None:
                    ret_diff = (retrieval_cond[:, 1:] - retrieval_cond[:, :-1]).abs().mean().item()
                    print(f"  retrieval_cond frame_diff: {ret_diff:.6f}")
            self._denoiser_input_debug = True
        
        # 1. Time Embedding
        if t.dim() > 1:
            t = t.squeeze(-1)
            
        time_embed = self.time_mlp(t) 
        time_embed = time_embed.unsqueeze(1) 
        time_embed = time_embed.repeat(1, encoder_output.shape[1], 1)
        
        condition = encoder_output + time_embed
        condition = self.pos_drop(condition)

        # 2. Target Embedding
        x = self.trg_embed(trg_embed) 

        # 3. Positional Encoding
        x = self.pe(x)
        x = self.emb_dropout(x)

        padding_mask = trg_mask
        # 【关键修复】强制确保 padding_mask 是二维的 (B, T)
        if padding_mask.dim() == 4:
            # 如果传进来的是 (B, 1, T, T)，取第一个头并压缩
            padding_mask = padding_mask[:, 0, 0, :] 
        elif padding_mask.dim() == 3:
            padding_mask = padding_mask[:, 0, :]
        
        sub_mask = subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        # 4. Transformer Layers with Retrieval Cross-Attention
        for i, layer in enumerate(self.layers):
            # Standard Decoder Layer (Self-Attn + Encoder-Decoder Attn)
            x_before = x.clone()
            x = layer(x=x, memory=condition,
                      src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)
            
            # 【调试】检查标准层后的动态性
            if i == 0 and not hasattr(self, '_after_standard_layer_check'):
                if x.shape[1] > 1:
                    diff_before = (x_before[:, 1:] - x_before[:, :-1]).abs().mean().item()
                    diff_after = (x[:, 1:] - x[:, :-1]).abs().mean().item()
                    print(f"[DENOISER LAYER 0 After Standard Layer]")
                    print(f"  Before: {diff_before:.6f} -> After: {diff_after:.6f}")
                self._after_standard_layer_check = True
            
            # 【最终统一】启用训练时使用的 Cross-Attention 逻辑
            if retrieval_cond is not None:
                ret_emb = self.retrieval_proj(retrieval_cond) # (B, T, H)
                # 叠加位置编码，确保模型能感知动作的时间先后顺序
                ret_emb = ret_emb + self.pe(ret_emb) 
                
                # 【调试】检查 padding_mask 形状
                if not hasattr(self, '_padding_mask_shape_check'):
                    print(f"[DEBUG] padding_mask shape: {padding_mask.shape}")
                    print(f"[DEBUG] x shape (Query): {x.shape}")
                    print(f"[DEBUG] ret_emb shape (Key/Value): {ret_emb.shape}")
                    self._padding_mask_shape_check = True

                # 【调试】记录 Cross-Attention 前的动态性
                if i == 0 and not hasattr(self, '_before_cross_attn_check'):
                    if x.shape[1] > 1:
                        diff_before_ca = (x[:, 1:] - x[:, :-1]).abs().mean().item()
                        print(f"[DENOISER LAYER 0 Before Cross-Attention]")
                        print(f"  frame_diff: {diff_before_ca:.6f}")
                    self._before_cross_attn_check = True

                # Cross-Attention: Query=x, Key=Value=ret_emb
                x_attended = self.retrieval_cross_attn(
                    k=ret_emb, 
                    v=ret_emb, 
                    q=x, 
                    mask=None, 
                    padding_mask=padding_mask
                )
                
                # 【新增】计算检索相似度 (Cosine Similarity between x and ret_emb)
                # 通过 kwargs 接收回调函数或状态容器，避免循环引用
                if i == 0:
                    with torch.no_grad():
                        # 归一化特征向量
                        x_norm = F.normalize(x, p=2, dim=-1)
                        ret_norm = F.normalize(ret_emb, p=2, dim=-1)
                        # 计算余弦相似度并取均值
                        sim = (x_norm * ret_norm).sum(dim=-1).mean()
                        # 映射到 0-1 范围 (sim 原始范围 -1 到 1)
                        sim_score = (sim + 1) / 2
                        
                        # 将相似度存入 denoiser 自身属性，由 ACD forward 统一读取
                        self.current_batch_avg_sim = sim_score.item()

                # 【核心优化】动态调整检索引导强度 (Curriculum Learning)
                decay_rate = 0.9
                decay_steps = 5000
                current_step = kwargs.get('global_step', 0)
                
                # 计算衰减因子
                decay_factor = decay_rate ** (current_step / decay_steps)
                # 初始 0.1，最低降至 0.02 (0.1 * 0.2)
                scale_value = 0.1 * max(0.2, decay_factor)
                
                x = x + scale_value * x_attended
                
                # 【调试】检查 Cross-Attention 后的动态性
                if i == 0 and not hasattr(self, '_after_cross_attn_check'):
                    if x.shape[1] > 1:
                        diff_after_ca = (x[:, 1:] - x[:, :-1]).abs().mean().item()
                        print(f"[DENOISER LAYER 0 After Cross-Attention]")
                        print(f"  frame_diff: {diff_after_ca:.6f}")
                        if diff_after_ca < 0.01:
                            print(f"  ⚠️ Cross-Attention destroyed dynamics!")
                    self._after_cross_attn_check = True

        # 5. Output
        # 【核心修复】恢复 LayerNorm 以稳定特征分布
        x = self.layer_norm(x)
        
        output = self.output_layer(x)
        
        # 【核心修复】使用 Clamp 限制输出范围，防止数值爆炸
        # 根据经验，手语骨架坐标通常在 [-2, 2] 或 [-5, 5] 之间
        output = torch.clamp(output, min=-5.0, max=5.0)

        return output

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)
