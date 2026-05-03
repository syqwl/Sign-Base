import os
import sys
import cv2
import math
import torch
import numpy as np
from dtw import dtw
from constants import PAD_TOKEN

# =========================
# 178 -> 50 (工程稳定版)
# =========================

def map_178_to_50(joints):
    def safe(i):
        return joints[i] if i < joints.shape[0] else np.zeros(3)

    out = []

    # =================
    # body (0~7)
    # =================
    body_ids = [0,1,2,3,4,5,6,7]
    # 0：右肩膀，1：右手肘，2：右手腕；
    # 3：左肩膀，4：左手肘，5：左手腕；
    for i in body_ids:
        out.append(safe(i))

    # =================
    # left hand (8~28)
    # =================
    left_map = [
        8, 9,10,11,12,
        13,14,15,16,
        17,18,19,20,
        21,22,23,24,
        25,26,27,28
    ]
    for i in left_map:
        out.append(safe(i))

    # =================
    # right hand (29~49)
    # =================
    right_map = [
        29,30,31,32,33,
        34,35,36,37,
        38,39,40,41,
        42,43,44,45,
        46,47,48,49
    ]
    for i in right_map:
        out.append(safe(i))

    return np.stack(out)[:50]

# =========================
# skeleton structure
# =========================

def getSkeletalModelStructure():
    return (
        (0, 1, 0),
        (1, 2, 1),
        (3, 4, 2),
        (4, 5, 3),

        # (1, 8, 4),
        (8, 9, 5), (8, 13, 9), (8, 17, 13), (8, 21, 17), (8, 25, 21),

        (9, 10, 6), (10, 11, 7), (11, 12, 8),
        (13, 14, 10), (14, 15, 11), (15, 16, 12),
        (17, 18, 14), (18, 19, 15), (19, 20, 16),
        (21, 22, 18), (22, 23, 19), (23, 24, 20),
        (25, 26, 22), (26, 27, 23), (27, 28, 24),

        # (3, 29, 4),
        (29, 30, 5), (29, 34, 9), (29, 38, 13),(29, 42, 17), (29, 46, 21),

        (30, 31, 6), (31, 32, 7), (32, 33, 8),
        (34, 35, 10), (35, 36, 11), (36, 37, 12),
        (38, 39, 14), (39, 40, 15), (40, 41, 16),
        (42, 43, 18), (43, 44, 19), (44, 45, 20),
        (46, 47, 22), (47, 48, 23), (48, 49, 24),
    )


# =========================
# drawing
# =========================

def draw_line(im, joint1, joint2, c=(0,0,255), width=2):
    if joint1 is None or joint2 is None:
        return

    if joint1[0] < -50 or joint2[0] < -50:
        return

    center = (int((joint1[0]+joint2[0])/2), int((joint1[1]+joint2[1])/2))
    length = int(math.sqrt((joint1[0]-joint2[0])**2 + (joint1[1]-joint2[1])**2)/2)
    angle = math.degrees(math.atan2(joint1[0]-joint2[0], joint1[1]-joint2[1]))

    cv2.ellipse(im, center, (width, length), -angle, 0, 360, c, -1)


def draw_frame_2D(frame, joints_178):
    offset = np.array([350, 250])

    # =========================
    # 1️⃣ 画骨架（前50）
    # =========================
    joints_50 = map_178_to_50(joints_178)
    joints_50 = joints_50[:, :2]

    skeleton = np.array(getSkeletalModelStructure())

    joints_50 = joints_50 * 800
    joints_50 = joints_50 + offset

    for i in range(len(skeleton)):
        a, b, _ = skeleton[i]
        draw_line(frame, joints_50[a], joints_50[b])

    # =========================
    # 2️⃣ 画面部（50~177）
    # =========================
    face = joints_178[50:178, :2]

    face = face * 800
    face = face + offset

    for (x, y) in face:
        if x < -50 or y < -50:
            continue
        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 0), -1)

# =========================
# video pipeline
# =========================

def plot_video(joints_data, out_dir, sequence_ID="sample"):
    os.makedirs(out_dir, exist_ok=True)
    video_file = os.path.join(out_dir, f"{sequence_ID}.mp4")

    video = cv2.VideoWriter(video_file,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            25,
                            (650, 650))

    for frame in joints_data:

        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        # remove extra dim if exists
        if frame.shape[-1] > 3:
            frame = frame[:, :3]

        # ensure shape
        try:
            frame = frame.reshape(178, 3)
        except:
            continue

        # ===== key step =====
        frame_50 = map_178_to_50(frame)
        frame_2d = frame_50[:, :2]

        canvas = np.ones((650,650,3), np.uint8) * 255
        draw_frame_2D(canvas, frame)

        video.write(canvas)

    video.release()
    print("Saved:", video_file)


def plot_video_comparison(pred_data, gt_data, out_dir, sequence_ID="sample"):
    """
    并排可视化 Pred 和 GT
    pred_data/gt_data: list of frames or (T, 178, 3) tensor/array
    """
    os.makedirs(out_dir, exist_ok=True)
    video_file = os.path.join(out_dir, f"{sequence_ID}_comparison.mp4")

    # 统一转换为列表格式
    if isinstance(pred_data, (torch.Tensor, np.ndarray)):
        pred_list = [pred_data[i] for i in range(pred_data.shape[0])]
    else:
        pred_list = pred_data

    if isinstance(gt_data, (torch.Tensor, np.ndarray)):
        gt_list = [gt_data[i] for i in range(gt_data.shape[0])]
    else:
        gt_list = gt_data

    # 对齐到最长的序列
    max_len = max(len(pred_list), len(gt_list))
    
    # 如果长度不一致，短的那个重复最后一帧
    if len(pred_list) < max_len:
        last_frame = pred_list[-1]
        pred_list = pred_list + [last_frame.copy() for _ in range(max_len - len(pred_list))]
    
    if len(gt_list) < max_len:
        last_frame = gt_list[-1]
        gt_list = gt_list + [last_frame.copy() for _ in range(max_len - len(gt_list))]
    
    # 宽度翻倍：左边 Pred，右边 GT
    width, height = 650, 650
    video = cv2.VideoWriter(video_file,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            25,
                            (width * 2, height))

    for i in range(max_len):
        p_frame = pred_list[i]
        g_frame = gt_list[i]

        if isinstance(p_frame, torch.Tensor):
            p_frame = p_frame.cpu().numpy()
        if isinstance(g_frame, torch.Tensor):
            g_frame = g_frame.cpu().numpy()

        # 处理维度
        if p_frame.shape[-1] > 3: p_frame = p_frame[:, :3]
        if g_frame.shape[-1] > 3: g_frame = g_frame[:, :3]

        try:
            p_frame = p_frame.reshape(178, 3)
            g_frame = g_frame.reshape(178, 3)
        except:
            continue

        # 绘制左侧 Pred
        canvas_p = np.ones((height, width, 3), np.uint8) * 255
        draw_frame_2D(canvas_p, p_frame)
        
        # 绘制右侧 GT
        canvas_g = np.ones((height, width, 3), np.uint8) * 255
        draw_frame_2D(canvas_g, g_frame)

        # 添加文字标签
        cv2.putText(canvas_p, "Generated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(canvas_g, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        # 拼接
        combined = np.hstack((canvas_p, canvas_g))
        video.write(combined)

    video.release()
    print("Saved:", video_file)


def draw_debug(joints, save_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))

    for i in range(30):
        x, y = joints[i][0], joints[i][1]

        plt.scatter(x, y)
        plt.text(x, y, str(i), fontsize=14)

    plt.axis("equal")
    plt.grid(True)

    # 保存图片（强烈建议）
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()


# =========================
# 对比可视化主函数
# =========================

def visualize_comparison(gt_path, pred_path, out_dir):
    """
    对比可视化 GT 和预测结果（全部样本）
    
    Args:
        gt_path: GT 数据路径（如 ./Data/test.pt）
        pred_path: 预测数据路径（如 ./prediction.pt）
        out_dir: 输出目录
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading GT data from: {gt_path}")
    gt_data = torch.load(gt_path, map_location="cpu")
    
    print(f"Loading prediction data from: {pred_path}")
    pred_data = torch.load(pred_path, map_location="cpu")
    
    # 确保都是字典格式
    if not isinstance(gt_data, dict):
        print("❌ GT data is not a dictionary.")
        return
    
    if not isinstance(pred_data, dict):
        print("❌ Prediction data is not a dictionary.")
        return
    
    # 按GT文件的原始顺序处理，只保留两者都有的样本
    common_keys = [k for k in gt_data.keys() if k in pred_data]
    
    if len(common_keys) == 0:
        print("❌ No matching sample IDs found!")
        return
    
    print(f"Processing {len(common_keys)} samples...")
    
    for sample_id in common_keys:
        # 提取 GT 数据（test.pt 中是嵌套字典，需要取 poses_3d）
        gt_sample = gt_data[sample_id]
        if isinstance(gt_sample, dict):
            # test.pt 格式：{"name": ..., "text": ..., "gloss": ..., "poses_3d": tensor}
            if "poses_3d" in gt_sample:
                gt_sample = gt_sample["poses_3d"]
            else:
                print(f"⚠️  Warning: {sample_id} has no 'poses_3d' key, skipping.")
                continue
        
        # 提取预测数据（通常是直接张量）
        pred_sample = pred_data[sample_id]
        
        # 转换为 NumPy 数组
        if isinstance(gt_sample, torch.Tensor):
            gt_sample = gt_sample.cpu().numpy()
        if isinstance(pred_sample, torch.Tensor):
            pred_sample = pred_sample.cpu().numpy()
        
        # 转换为帧列表（保留所有帧）
        gt_frames = [gt_sample[i] for i in range(gt_sample.shape[0])]
        pred_frames = [pred_sample[i] for i in range(pred_sample.shape[0])]
        
        # 调用 plot_video_comparison 进行对比可视化
        try:
            plot_video_comparison(
                pred_data=pred_frames,
                gt_data=gt_frames,
                out_dir=out_dir,
                sequence_ID=sample_id.replace("/", "_")
            )
        except Exception as e:
            print(f"Error: {sample_id} - {e}")
    
    print(f"Done! Saved to: {out_dir}")


# =========================
# main
# =========================

if __name__ == "__main__":
    
    # 判断是单文件可视化还是对比可视化
    if len(sys.argv) >= 4:
        # 对比可视化模式：gt_path pred_path out_dir
        gt_path = sys.argv[1]
        pred_path = sys.argv[2]
        out_dir = sys.argv[3]
        
        visualize_comparison(gt_path, pred_path, out_dir)
    
    elif len(sys.argv) == 3:
        # 单文件可视化模式：pt_path out_dir
        pt_path = sys.argv[1]
        out_dir = sys.argv[2]

        data = torch.load(pt_path, map_location="cpu")

        if isinstance(data, dict):
            for k,v in data.items():
                seq = [v[i] for i in range(v.shape[0])]
                plot_video(seq, out_dir, k.replace("/", "_"))

        else:
            seq = [data[i] for i in range(data.shape[0])]
            plot_video(seq, out_dir)
    
    else:
        print("Usage:")
        print("  Single file visualization:")
        print("    python visualization.py <pt_file> <output_dir>")
        print("  Comparison visualization:")
        print("    python visualization.py <gt_pt> <pred_pt> <output_dir>")
