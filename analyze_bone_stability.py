"""
骨骼长度稳定性分析工具
用于检测生成结果中手部关节点的尺寸变化
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers import getSkeletalModelStructure


def analyze_bone_stability(prediction_data, gt_data=None, sample_id=None):
    """
    分析骨骼长度的稳定性
    
    Args:
        prediction_data: dict or tensor, 预测的骨架数据
        gt_data: dict or tensor, GT 骨架数据（可选）
        sample_id: str, 样本 ID（如果数据是字典格式）
    """
    # 提取预测数据
    if isinstance(prediction_data, dict):
        if sample_id:
            pred_sample = prediction_data[sample_id]
            if isinstance(pred_sample, dict):
                pred_joints = pred_sample.get("poses_3d", pred_sample)
            else:
                pred_joints = pred_sample
        else:
            # 取第一个样本
            first_key = list(prediction_data.keys())[0]
            pred_sample = prediction_data[first_key]
            pred_joints = pred_sample.get("poses_3d", pred_sample) if isinstance(pred_sample, dict) else pred_sample
    else:
        pred_joints = prediction_data
    
    # 确保形状为 (T, 178, 3)
    if pred_joints.dim() == 2:
        pred_joints = pred_joints.view(-1, 178, 3)
    
    T, N, C = pred_joints.shape
    print(f"📊 分析样本: {sample_id or 'Unknown'}")
    print(f"   - 时间步数: {T}")
    print(f"   - 关节点数: {N}")
    
    # 获取骨骼连接关系
    skeletons = getSkeletalModelStructure()
    
    # 重点关注手部骨骼（假设索引 0-49 是身体+手）
    hand_skeletons = [s for s in skeletons if s[0] < 50 and s[1] < 50]
    print(f"   - 手部骨骼数量: {len(hand_skeletons)}")
    
    # 计算每根骨骼在所有时间步上的长度
    bone_lengths_over_time = {}
    bone_stats = {}
    
    for idx, skeleton in enumerate(hand_skeletons[:10]):  # 只分析前10根手部骨骼
        joint_a = pred_joints[:, skeleton[0], :]  # (T, 3)
        joint_b = pred_joints[:, skeleton[1], :]  # (T, 3)
        
        # 计算每帧的骨骼长度
        lengths = torch.norm(joint_a - joint_b, dim=-1).cpu().numpy()  # (T,)
        
        bone_lengths_over_time[f"Bone_{idx}"] = lengths
        
        # 统计信息
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        cv = std_len / (mean_len + 1e-6)  # 变异系数
        
        bone_stats[f"Bone_{idx}"] = {
            "mean": mean_len,
            "std": std_len,
            "cv": cv,
            "min": np.min(lengths),
            "max": np.max(lengths)
        }
    
    # 打印统计结果
    print("\n📏 手部骨骼长度统计:")
    print("-" * 60)
    print(f"{'骨骼ID':<10} {'平均长度':<12} {'标准差':<12} {'变异系数':<12} {'范围':<15}")
    print("-" * 60)
    
    for bone_id, stats in bone_stats.items():
        range_str = f"[{stats['min']:.3f}, {stats['max']:.3f}]"
        print(f"{bone_id:<10} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
              f"{stats['cv']:<12.4f} {range_str:<15}")
    
    # 计算整体稳定性指标
    all_cvs = [stats["cv"] for stats in bone_stats.values()]
    avg_cv = np.mean(all_cvs)
    max_cv = np.max(all_cvs)
    
    print("\n" + "=" * 60)
    print(f"🎯 整体稳定性指标:")
    print(f"   - 平均变异系数 (Avg CV): {avg_cv:.4f}")
    print(f"   - 最大变异系数 (Max CV): {max_cv:.4f}")
    
    if avg_cv < 0.01:
        print("   ✅ 优秀！骨骼长度非常稳定")
    elif avg_cv < 0.05:
        print("   ⚠️  良好，但仍有改进空间")
    else:
        print("   ❌ 较差，骨骼长度波动明显，建议增强刚性约束")
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (bone_id, lengths) in enumerate(bone_lengths_over_time.items()):
        if idx < 10:
            ax = axes[idx]
            time_steps = np.arange(len(lengths))
            ax.plot(time_steps, lengths, 'b-', linewidth=1.5, label='Bone Length')
            ax.axhline(y=np.mean(lengths), color='r', linestyle='--', alpha=0.5, label=f'Mean={np.mean(lengths):.3f}')
            ax.set_title(f"{bone_id}\nCV={bone_stats[bone_id]['cv']:.4f}", fontsize=10)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Length")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("bone_stability_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 可视化结果已保存至: bone_stability_analysis.png")
    plt.close()
    
    return bone_stats, avg_cv


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python analyze_bone_stability.py <prediction.pt> [sample_id]")
        print("示例: python analyze_bone_stability.py ./Data/prediction.pt sample_001")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    sample_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"🔍 加载预测数据: {pred_file}")
    prediction_data = torch.load(pred_file, map_location='cpu')
    
    gt_file = "./Data/test.pt"
    try:
        print(f"🔍 加载 GT 数据: {gt_file}")
        gt_data = torch.load(gt_file, map_location='cpu')
    except:
        gt_data = None
        print("⚠️  未找到 GT 数据，仅分析预测结果")
    
    analyze_bone_stability(prediction_data, gt_data, sample_id)
