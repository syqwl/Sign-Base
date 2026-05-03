import torch
import numpy as np
import faiss
import os
import pickle
from helpers import load_config
from data import load_data, make_data_iter
from model import build_model
from batch import Batch

def ensure_1d_array(x, expected_size):
    if isinstance(x, torch.Tensor):
        arr = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.array(x)
    
    if arr.ndim == 0:
        arr = np.full((expected_size,), arr.item() if hasattr(arr, 'item') else arr)
    elif arr.ndim > 1:
        arr = arr.flatten()
        
    if len(arr) != expected_size:
        if len(arr) > expected_size:
            arr = arr[:expected_size]
        else:
            last_val = arr[-1] if len(arr) > 0 else 0
            padding = np.full((expected_size - len(arr),), last_val)
            arr = np.concatenate([arr, padding])
    return arr

def build_retrieval_database(cfg_file):
    print("🚀 Starting to build retrieval database...")
    
    cfg = load_config(cfg_file)
    train_data, _, src_vocab, trg_vocab = load_data(cfg=cfg)
    
    model = build_model(cfg=cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.eval()
    
    if cfg["training"]["use_cuda"]:
        model.cuda()

    all_embeddings = []
    all_skeletons_raw = [] # 临时存储变长骨架
    all_lengths = []
    
    print("📊 Extracting features from training data...")
    
    train_iter = make_data_iter(train_data, batch_size=64, train=False, shuffle=False)
    
    with torch.no_grad():
        for i, torch_batch in enumerate(train_iter):
            batch = Batch(torch_batch=torch_batch, pad_index=model.pad_index, model=model)
            
            # 1. Text Embedding
            encoder_output = model.encode(src=batch.src, 
                              src_length=batch.src_lengths, 
                              src_mask=batch.src_mask)

            # 【优化】使用 src_mask 进行加权平均
            # src_mask shape: (B, 1, T). 1 for valid, 0 for padding.
            mask = batch.src_mask.squeeze(1).float() # (B, T)

            # 将 padding 部分置为 0
            masked_emb = encoder_output * mask.unsqueeze(-1) # (B, T, H)

            # 求和并除以有效长度 (防止除以0，加一个极小值 eps)
            sum_emb = masked_emb.sum(dim=1) # (B, H)
            len_emb = mask.sum(dim=1, keepdim=True) # (B, 1)
            sentence_emb = (sum_emb / (len_emb + 1e-9)).cpu().numpy()
            
            # 2. Skeletons & Lengths
            skeletons_padded = batch.trg_input[:, :, :534].cpu().numpy()
            bsz = skeletons_padded.shape[0]
            
            if hasattr(batch, 'trg_lengths') and batch.trg_lengths is not None:
                raw_lengths = batch.trg_lengths
            else:
                raw_lengths = batch.src_lengths
            
            lengths = ensure_1d_array(raw_lengths, bsz)

            for j in range(bsz):
                true_len = int(lengths[j])
                if true_len <= 0: true_len = 1
                if true_len > skeletons_padded.shape[1]: true_len = skeletons_padded.shape[1]
                
                skel_valid = skeletons_padded[j, :true_len, :]
                all_embeddings.append(sentence_emb[j])
                all_skeletons_raw.append(skel_valid)
                all_lengths.append(true_len)

            if i % 10 == 0:
                print(f"   Processed {i} batches...")

    if not all_embeddings:
        print("❌ No data processed!")
        return

    final_embeddings = np.array(all_embeddings).astype(np.float32)
    final_lengths = np.array(all_lengths)
    
    # 【关键】将所有变长骨架 Padding 到全局最大长度，形成 (N, T_max, 534)
    max_T = int(np.max(final_lengths))
    N = len(all_skeletons_raw)
    C = 534
    
    print(f"🔧 Padding skeletons to global max length: {max_T}")
    final_skeletons = np.zeros((N, max_T, C), dtype=np.float32)
    
    for idx, skel in enumerate(all_skeletons_raw):
        t_len = skel.shape[0]
        final_skeletons[idx, :t_len, :] = skel

    print(f"✅ Final Shapes: Emb={final_embeddings.shape}, Skel={final_skeletons.shape}, Len={final_lengths.shape}")

    # 3. Build FAISS Index
    print("🔨 Building FAISS index...")
    d = final_embeddings.shape[1]
    faiss.normalize_L2(final_embeddings)
    index = faiss.IndexFlatIP(d) 
    index.add(final_embeddings)
    
    # 4. Save
    save_dir = "./Data"
    os.makedirs(save_dir, exist_ok=True)
    
    faiss.write_index(index, os.path.join(save_dir, "retrieval_index.faiss"))
    np.save(os.path.join(save_dir, "retrieval_embeddings.npy"), final_embeddings)
    np.save(os.path.join(save_dir, "retrieval_skeletons.npy"), final_skeletons) # 统一命名，方便 model.py 加载
    np.save(os.path.join(save_dir, "retrieval_lengths.npy"), final_lengths)
    
    print(f"💾 Database saved to {save_dir}")

if __name__ == "__main__":
    build_retrieval_database("./Configs/Sign-Base.yaml")