
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random

from torch import Tensor
from encoder import Encoder
from ACD import ACD
from batch import Batch
from embeddings import Embeddings
from vocabulary import Vocabulary
from initialization import initialize_model
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from retrieval import MotionRetriever, GlossRetriever

class Model(nn.Module):
    def __init__(self, cfg: dict, 
                 encoder: Encoder, 
                 ACD: ACD, 
                 src_embed: Embeddings, 
                 src_vocab: Vocabulary, 
                 trg_vocab: Vocabulary, 
                 in_trg_size: int, 
                 out_trg_size: int):
        """
        Create Sign-IDD

        :param encoder: encoder
        :param ACD: ACD
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]
        self.src_embed = src_embed
        self.encoder = encoder
        self.ACD = ACD
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size

        # 【统一】从 model.diffusion 中读取检索配置
        diffusion_cfg = cfg["model"].get("diffusion", {})
        self.use_retrieval = diffusion_cfg.get("use_retrieval", False)
        self.retrieval_type = diffusion_cfg.get("retrieval_type", "semantic")  # "semantic" 或 "gloss"
        
        # 【核心修复】读取 Velocity Loss 权重，解决动态性丢失问题
        self.velocity_loss_weight = diffusion_cfg.get("velocity_loss_weight", 0.2)
        
        self.retriever = None
        self.gloss_retriever = None
        
        if self.use_retrieval:
            try:
                if self.retrieval_type == "gloss":
                    # 【新增】基于 Gloss 的检索
                    gloss_db_path = diffusion_cfg.get("gloss_db_path", "./Data/phoenix_gloss2pose_results.pkl")
                    text2gloss_path = diffusion_cfg.get("text2gloss_path", None)
                    self.gloss_retriever = GlossRetriever(gloss_db_path, text2gloss_path)
                    print(f"✅ GlossRetriever initialized (type={self.retrieval_type})")
                else:
                    # 原有的基于语义向量的检索
                    embed_dim = cfg["model"]["encoder"]["hidden_size"]
                    top_k = diffusion_cfg.get("retrieval_top_k", 1)
                    self.retriever = MotionRetriever(embed_dim=embed_dim, top_k=top_k)
                    # 加载数据库
                    db_path = "./Data/retrieval_index.faiss"
                    skel_path = "./Data/retrieval_skeletons.npy"
                    if os.path.exists(db_path):
                        self.retriever.load_database(db_path, skel_path)
                    else:
                        print("⚠️ Retrieval DB not found. Disabling retrieval.")
                        self.use_retrieval = False
            except Exception as e:
                print(f"❌ Error loading retriever: {e}")
                self.use_retrieval = False

    def forward(self, is_train: bool, src: Tensor, trg_input: Tensor, src_mask: Tensor, src_lengths: Tensor, trg_mask: Tensor, gloss: list = None, sample_idx: int = None, global_step: int = 0):
        """
        :param gloss: List[str] 或 List[List[str]]，Batch 中每个样本的 Gloss 序列（训练时用 GT）
        :param sample_idx: int，当前样本在测试集中的索引（推理时用）
        :param global_step: int，当前训练步数（用于课程学习）
        """
        # 1. Encode
        encoder_output = self.encode(src=src, src_length=src_lengths, src_mask=src_mask)
        
        # 2. 检索增强逻辑
        retrieved_poses = None
        avg_similarity = 0.0
        
        # ==========================================
        # 【Step 1: 构造 Retrieval (核心模块)】
        # ==========================================
        retrieved_poses = None
        
        # 【核心原则】仅在训练阶段启用检索增强，推理阶段完全关闭以实现分布对齐
        if is_train and self.use_retrieval and self.gloss_retriever is not None:
            try:
                device = trg_input.device
                T_curr = trg_input.shape[1]
                batch_size = trg_input.shape[0]
                
                retrieved_poses_list = []
                success_count = 0
                
                # --- 训练阶段：严格模拟推理分布 (90% Text2Gloss + 10% GT) ---
                use_text2gloss = random.random() < 0.9
                
                for b_idx in range(batch_size):
                    ret_pose = None
                    
                    # 【全链路诊断】仅对 Batch 中第一个样本进行详细打印，避免刷屏
                    is_debug_sample = (b_idx == 0 and hasattr(self, 'steps') and self.steps % 100 == 0)
                    
                    # 1. 获取检索源 (优先使用 Text2Gloss)
                    if use_text2gloss and sample_idx is not None:
                        current_sample_idx = sample_idx + b_idx
                        if is_debug_sample:
                            print(f"\n[RETRIEVAL DIAGNOSTIC] Step {self.steps}, Sample ID: {current_sample_idx}")
                            print(f"  - Retriever Loaded: {self.gloss_retriever.text2gloss_results is not None}")
                            if self.gloss_retriever.text2gloss_results:
                                print(f"  - Total Samples in DB: {len(self.gloss_retriever.text2gloss_results)}")
                        
                        ret_pose = self.gloss_retriever.retrieve_from_text2gloss(
                            current_sample_idx, T_curr, device
                        )
                        
                        if is_debug_sample:
                            print(f"  - Source: Text2Gloss")
                            print(f"  - Retrieved Pose: {'Success' if ret_pose is not None else 'Failed (None)'}")
                            if ret_pose is not None:
                                diff = (ret_pose[1:] - ret_pose[:-1]).abs().mean().item()
                                print(f"  - Shape: {ret_pose.shape}, Frame Diff: {diff:.6f}")
                    else:
                        # 备选：使用 GT Gloss (仅占 10%)
                        gloss_seq = gloss[b_idx] if gloss is not None and b_idx < len(gloss) else None
                        if gloss_seq and isinstance(gloss_seq, str) and len(gloss_seq.strip()) > 0:
                            gloss_tokens = gloss_seq.split()
                            if is_debug_sample:
                                print(f"\n[RETRIEVAL DIAGNOSTIC] Step {self.steps}, Sample ID: GT_{b_idx}")
                                print(f"  - GT Gloss Tokens: {gloss_tokens[:5]}...")
                            
                            ret_pose = self.gloss_retriever.retrieve_from_gloss_sequence(
                                gloss_tokens, T_curr, device
                            )
                            
                            if is_debug_sample:
                                print(f"  - Source: GT Gloss")
                                print(f"  - Retrieved Pose: {'Success' if ret_pose is not None else 'Failed (None)'}")
                    
                    # 2. 质量评估与过滤 (Gating)
                    filter_reason = "None"
                    if ret_pose is not None:
                        frame_diff = (ret_pose[1:] - ret_pose[:-1]).abs().mean().item()
                        
                        # 【核心调整】降低阈值至 0.002，允许微动动作进入训练
                        # 之前的 0.01 过于严格，导致 Phoenix 数据集中的大部分参考动作被丢弃
                        if frame_diff < 0.002: 
                            filter_reason = f"Low Dynamics ({frame_diff:.6f})"
                            ret_pose = None
                        # 3. Retrieval Dropout (防止模型过度依赖检索)
                        elif random.random() < 0.3:
                            filter_reason = "Random Dropout (30%)"
                            ret_pose = None

                    if is_debug_sample:
                        if ret_pose is None:
                            print(f"  - Final Status: DROPPED due to {filter_reason}")
                        else:
                            print(f"  - Final Status: ACCEPTED")
                        print("-" * 50)

                    retrieved_poses_list.append(ret_pose)
                    if ret_pose is not None:
                        success_count += 1

                # 【全局强制对齐】
                while len(retrieved_poses_list) < batch_size:
                    retrieved_poses_list.append(None)

                # 堆叠处理
                valid_poses = []
                for p in retrieved_poses_list:
                    if p is not None and p.dim() == 2 and p.shape == (T_curr, self.in_trg_size):
                        valid_poses.append(p)
                    else:
                        valid_poses.append(torch.zeros(T_curr, self.in_trg_size, device=device))
                
                retrieved_poses = torch.stack(valid_poses, dim=0)
                
                # 如果全部被 Drop out，则设为 None
                if success_count == 0:
                    retrieved_poses = None
                    
            except Exception as e:
                print(f"❌ Retrieval construction failed: {e}")
                retrieved_poses = None

        # ==========================================
        # 【Step 2 & 3: Diffusion 输入与模型调用】
        # ==========================================
        diffusion_output = self.ACD(encoder_output=encoder_output,
                                    input_3d=trg_input,
                                    src_mask=src_mask,
                                    trg_mask=trg_mask,
                                    is_train=is_train,
                                    retrieved_poses=retrieved_poses,
                                    global_step=global_step)
        
        # 【深度诊断】检查 ACD 返回的相似度指标
        if hasattr(self.ACD, 'current_batch_avg_sim'):
            sim_score = self.ACD.current_batch_avg_sim
            if hasattr(self, 'steps') and self.steps % 100 == 0:
                print(f"[MODEL DEBUG] ACD returned Avg Sim: {sim_score:.6f}")
            self.current_batch_avg_sim = sim_score
        else:
            if hasattr(self, 'steps') and self.steps % 100 == 0:
                print(f"[MODEL DEBUG] ACD did NOT return avg_sim attribute!")
            self.current_batch_avg_sim = 0.0
            
        return diffusion_output

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor):

        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs
        """

        # Encode an embedded source
        encode_output = self.encoder(embed_src=self.src_embed(src), 
                                     src_length=src_length, 
                                     mask=src_mask)

        return encode_output
    
    def diffusion(self, is_train: bool, encoder_output: Tensor, src_mask: Tensor, trg_input: Tensor, trg_mask: Tensor):
        
        """
        diffusion the target sentence.

        :param src: param encoder_output: encoder states for attention computation
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param trg_mask: mask for target steps
        :return: diffusion outputs
        """

        diffusion_output = self.ACD(is_train=is_train,
                                    encoder_output=encoder_output,
                                    input_3d=trg_input,
                                    src_mask=src_mask, 
                                    trg_mask=trg_mask)
        
        return diffusion_output
    
    def get_loss_for_batch(self, is_train, batch: Batch, loss_function: nn.Module, global_step: int = 0, sample_idx: int = None) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        model_output = self.forward(src=batch.src,
                                    trg_input=batch.trg_input[:, :, :534],
                                    src_mask=batch.src_mask,
                                    src_lengths=batch.src_lengths,
                                    trg_mask=batch.trg_mask,
                                    is_train=is_train,
                                    gloss=batch.gloss,  # 【新增】传递 Gloss
                                    sample_idx=sample_idx,  # 【关键修复】传递 sample_idx
                                    global_step=global_step)  # 【新增】传递步数
        
        # 兼容 ACD 可能返回元组 (skel_out, vel_pred, vel_gt) 或单张量 skel_out 的情况
        if isinstance(model_output, tuple):
            skel_out = model_output[0]
        else:
            skel_out = model_output

        # 【Step 4.1: Diffusion Loss】
        batch_loss = loss_function(skel_out, batch.trg_input[:, :, :534])

        # 【Step 4.2: Velocity Loss (Motion Loss)】
        if is_train and hasattr(self, 'velocity_loss_weight') and self.velocity_loss_weight > 0:
            # 使用参考方案中的显式差分计算，不依赖 ACD 内部返回的速度项，更加稳健
            vel_pred = skel_out[:, 1:] - skel_out[:, :-1]
            vel_gt = batch.trg_input[:, 1:, :534] - batch.trg_input[:, :-1, :534]
            velocity_loss = F.mse_loss(vel_pred, vel_gt)
            batch_loss = batch_loss + self.velocity_loss_weight * velocity_loss
            
            # 记录速度损失到 TensorBoard（可选）
            # if hasattr(self, 'tb_writer'):
            #     self.tb_writer.add_scalar("train/velocity_loss", velocity_loss.item(), global_step)

        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

def build_model(cfg: dict, src_vocab: Vocabulary, trg_vocab: Vocabulary):

    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"]
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"]

    # Define source embedding
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)
    
    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.) # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
           "for transformer, emb_size must be hidden_size"
    
    # Transformer Encoder
    encoder = Encoder(**cfg["encoder"],
                      emb_size=src_embed.embedding_dim,
                      emb_dropout=enc_emb_dropout)
    
    # ACD
    diffusion = ACD(args=cfg, 
                    trg_vocab=trg_vocab)
    
    # Define the model
    model = Model(encoder=encoder,
                  ACD=diffusion,
                  src_embed=src_embed,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
