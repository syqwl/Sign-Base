# retrieval.py
import torch
import torch.nn.functional as F
import numpy as np
import faiss
import os
import pickle
from typing import List, Optional, Dict

class MotionRetriever:
    def __init__(self, embed_dim: int, top_k: int = 1):
        self.top_k = top_k
        self.index = None
        self.database_skeletons = None
        self.embed_dim = embed_dim

    def load_database(self, index_path: str, skeleton_path: str):
        """加载预构建的检索数据库"""
        if not os.path.exists(index_path) or not os.path.exists(skeleton_path):
            raise FileNotFoundError(f"Retrieval DB not found at {index_path} or {skeleton_path}")
        
        self.index = faiss.read_index(index_path)
        # 假设骨架数据保存为 (N, T, C)
        self.database_skeletons = np.load(skeleton_path)
        print(f"✅ Retrieval DB Loaded: Index={index_path}, Skels={skeleton_path}")

    def search(self, query_embedding: torch.Tensor) -> tuple:
        """
        检索 Top-K 相似样本
        :param query_embedding: (B, D) 归一化后的查询向量
        :return: retrieved_skeletons (B, K, T, C), distances (B, K)
        """
        query_np = query_embedding.cpu().numpy().astype(np.float32)
        
        # FAISS 搜索
        distances, indices = self.index.search(query_np, self.top_k)
        
        # 提取骨架
        # indices shape: (B, K)
        B, K = indices.shape
        # database_skeletons shape: (N, T, C)
        # 我们需要获取 (B, K, T, C)
        retrieved_skels = self.database_skeletons[indices] 
        
        return torch.from_numpy(retrieved_skels).float(), torch.from_numpy(distances).float()


class GlossRetriever:
    """
    基于 Gloss 的检索器（查表法）
    使用 phoenix_gloss2pose_results.pkl 数据库
    """
    def __init__(self, gloss_db_path: str, text2gloss_path: Optional[str] = None):
        """
        :param gloss_db_path: Gloss-Pose 数据库路径
        :param text2gloss_path: Text2Gloss 预测结果路径（用于推理阶段）
        """
        if not os.path.exists(gloss_db_path):
            raise FileNotFoundError(f"Gloss DB not found at {gloss_db_path}")
        
        with open(gloss_db_path, 'rb') as f:
            self.gloss_db = pickle.load(f)
        
        self.text2gloss_results = None
        if text2gloss_path and os.path.exists(text2gloss_path):
            with open(text2gloss_path, 'rb') as f:
                raw_data = pickle.load(f)
            
            # 【关键修复】处理不同的数据结构
            if isinstance(raw_data, dict):
                # 情况1: Text2Gloss 直接输出的格式（样本字典）
                # 检查是否包含 'gls_hyp' 字段（直接预测结果）
                first_key = list(raw_data.keys())[0] if raw_data else None
                if first_key and isinstance(raw_data[first_key], dict) and 'gls_hyp' in raw_data[first_key]:
                    self.text2gloss_results = raw_data
                    print(f"✅ Text2Gloss results loaded from {text2gloss_path} (direct prediction format, {len(self.text2gloss_results)} samples)")
                
                # 情况2: 顶层字典包含 'wer_list'
                elif 'wer_list' in raw_data:
                    wer_list = raw_data['wer_list']
                    
                    # 检查是否是列表结构（alignment_lst）
                    if isinstance(wer_list, dict) and 'alignment_lst' in wer_list:
                        # 将列表转换为字典格式 {sentence_0: {...}, sentence_1: {...}}
                        alignment_list = wer_list['alignment_lst']
                        self.text2gloss_results = {}
                        for idx, item in enumerate(alignment_list):
                            self.text2gloss_results[f"sentence_{idx}"] = item
                        print(f"✅ Text2Gloss results loaded from {text2gloss_path} (converted from alignment_lst, {len(self.text2gloss_results)} samples)")
                    else:
                        # 已经是字典格式
                        self.text2gloss_results = wer_list
                        print(f"✅ Text2Gloss results loaded from {text2gloss_path} (from wer_list)")
                else:
                    # 情况3: 顶层字典直接就是样本索引
                    self.text2gloss_results = raw_data
                    print(f"✅ Text2Gloss results loaded from {text2gloss_path}")
            else:
                print(f"⚠️ WARNING: Unexpected data type: {type(raw_data)}")
        
        print(f"✅ GlossRetriever initialized with {len(self.gloss_db)} unique glosses")
    
    def retrieve_from_gloss_sequence(
        self, 
        gloss_sequence: List[str], 
        target_length: int,
        device: torch.device = torch.device('cpu')
    ) -> Optional[torch.Tensor]:
        """
        根据 Gloss 序列检索并拼接动作片段
        :param gloss_sequence: List[str]，如 ["I", "LOVE", "YOU"]
        :param target_length: int，目标动作序列长度
        :param device: 目标设备
        :return: Tensor (target_length, 534) 或 None
        """
        if not gloss_sequence or len(gloss_sequence) == 0:
            return None
        
        pose_segments = []
        for gls in gloss_sequence:
            gls_upper = gls.upper()  # 确保大写
            if gls_upper in self.gloss_db:
                # 取该 Gloss 的第一个参考片段
                segment = self.gloss_db[gls_upper][0]
                if isinstance(segment, np.ndarray):
                    pose_segments.append(segment)
                elif isinstance(segment, torch.Tensor):
                    pose_segments.append(segment.cpu().numpy())
        
        if len(pose_segments) == 0:
            return None
        
        # 拼接所有片段
        concatenated = np.concatenate(pose_segments, axis=0)  # (T_concat, J, 3) or (T_concat, J*3)
        
        # 处理形状：如果是 (T, J, 3)，展平为 (T, J*3)
        if concatenated.ndim == 3:
            concatenated = concatenated.reshape(concatenated.shape[0], -1)
        
        # 时间步对齐（截断或填充到 target_length）
        if concatenated.shape[0] > target_length:
            # 均匀采样截断
            indices = np.linspace(0, concatenated.shape[0] - 1, target_length, dtype=int)
            concatenated = concatenated[indices]
        elif concatenated.shape[0] < target_length:
            # 【修复】使用循环播放而非重复最后一帧，保持动态性
            if concatenated.shape[0] > 0:
                original_length = concatenated.shape[0]
                # 通过循环拼接达到目标长度
                repeat_times = (target_length + original_length - 1) // original_length  # 向上取整
                repeated = np.tile(concatenated, (repeat_times, 1))  # (T_repeat, C)
                concatenated = repeated[:target_length]  # 截断到目标长度
            else:
                # 如果没有任何有效片段，返回 None
                return None
        
        return torch.from_numpy(concatenated).float().to(device)
    
    def retrieve_from_text2gloss(
        self, 
        sample_idx: int, 
        target_length: int,
        device: torch.device = torch.device('cpu')
    ) -> Optional[torch.Tensor]:
        """
        从 Text2Gloss 预测结果中检索（用于推理阶段）
        :param sample_idx: 样本索引
        :param target_length: 目标动作序列长度
        :param device: 目标设备
        :return: Tensor (target_length, 534) 或 None
        """
        if self.text2gloss_results is None:
            # 【调试】打印警告（仅第一次）
            if not hasattr(self, '_text2gloss_none_warning_printed'):
                print(f"⚠️ WARNING: text2gloss_results is None! Cannot use retrieval in inference.")
                self._text2gloss_none_warning_printed = True
            return None
        
        # 获取预测的 Gloss 序列
        # 根据实际数据结构调整 Key 的访问方式
        sample_key = f"sentence_{sample_idx}"
        
        # 【深度诊断】打印查找过程 (仅前几个样本)
        debug_print = sample_idx < 2
        
        if debug_print:
            print(f"[RETRIEVAL LOOKUP] Looking for key: '{sample_key}' for sample_idx={sample_idx}")
            
        found_data = None
        used_key = None

        if sample_key in self.text2gloss_results:
            found_data = self.text2gloss_results[sample_key]
            used_key = sample_key
        elif sample_idx in self.text2gloss_results:
            found_data = self.text2gloss_results[sample_idx]
            used_key = sample_idx
            if debug_print:
                print(f"  - Fallback: Found using integer key {sample_idx}")
        else:
            if debug_print:
                print(f"  - ERROR: Key '{sample_key}' and idx {sample_idx} not found in text2gloss_results!")
                if self.text2gloss_results:
                     print(f"  - Available keys (first 5): {list(self.text2gloss_results.keys())[:5]}")
            return None

        if not isinstance(found_data, dict):
            if debug_print:
                print(f"  - ERROR: Data for key '{used_key}' is not a dict: {type(found_data)}")
            return None

        gloss_sequence = []
        
        # 【优先级1】使用 gls_hyp（Text2Gloss 直接输出的预测字符串）
        if 'gls_hyp' in found_data:
            raw_str = found_data['gls_hyp']
            gloss_sequence = raw_str.split()
            if debug_print:
                preview = raw_str[:50] + "..." if len(raw_str) > 50 else raw_str
                print(f"  - Found gls_hyp: '{preview}' -> Tokens count: {len(gloss_sequence)}")
                if gloss_sequence:
                    print(f"    First 5 tokens: {gloss_sequence[:5]}")
        
        # 【优先级2】使用 align_hyp_lst（但需要过滤占位符）
        elif 'align_hyp_lst' in found_data:
            raw_gloss = found_data['align_hyp_lst']
            # 【关键修复】过滤掉占位符（如 "*****"）和空字符串
            gloss_sequence = [g for g in raw_gloss if g and not all(c == '*' for c in g)]
            if debug_print:
                print(f"  - Found align_hyp_lst: Raw Len={len(raw_gloss)}, Filtered Len={len(gloss_sequence)}")
                if gloss_sequence:
                    print(f"    First 5 tokens: {gloss_sequence[:5]}")
        else:
            if debug_print:
                print(f"  - ERROR: No valid gloss field (gls_hyp/align_hyp_lst) found in sample data. Keys: {list(found_data.keys())}")
            return None

        if not gloss_sequence:
            if debug_print:
                print(f"  - WARNING: Extracted gloss_sequence is empty after filtering!")
            return None
            
        # 调用底层查表逻辑
        result = self.retrieve_from_gloss_sequence(gloss_sequence, target_length, device)
        
        if debug_print:
            if result is not None:
                # 计算平均帧间差，验证动作是否有变化（非全零或全静止）
                if result.shape[0] > 1:
                    diff = (result[1:] - result[:-1]).abs().mean().item()
                    print(f"  - FINAL RESULT: Shape={result.shape}, Mean Frame Diff={diff:.6f}")
                else:
                    print(f"  - FINAL RESULT: Shape={result.shape} (Single frame)")
            else:
                print(f"  - FINAL RESULT: None (Failed to retrieve poses from DB for glosses)")
        
        return result
