import numpy as np
import faiss

# 加载索引
index = faiss.read_index("./Data/retrieval_index.faiss")
# 获取前 5 个向量
vecs = index.reconstruct_n(0, 5)
# 计算它们的 L2 范数
norms = np.linalg.norm(vecs, axis=1)
print("Database Vector Norms (should be 1.0):", norms)