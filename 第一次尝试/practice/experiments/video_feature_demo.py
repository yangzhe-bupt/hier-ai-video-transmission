# ======================================================================
# 文件：video_feature_demo.py
# 位置：practice/experiments
# 功能：视频预处理、固定单GOP训练、检查点保存加载及重建评测的演示入口。
# 类与函数汇总：
#   - read_frames：从指定帧开始，顺序读取视频。
#   - center_crop_frames：对每帧做中心裁剪，并校验裁剪内容。
#   - frames_to_gop：把裁剪帧转为 RGB 浮点张量并组装成 GOP [1, T, 3, H, W]。
#   - load_video_as_gop：读取、裁剪视频片段并组装GOP。
#   - train_single_gop：在固定GOP上训练轻量编解码器。
#   - save_checkpoint：保存编解码器参数和训练损失。
#   - save_evaluation_results：将逐片段评测结果保存为CSV。
#   - show_reconstruction：并排比较原始帧与重建帧。
#   - check_model_interface：检查编码器提取特征的接口与形状。
#   - train_main：主流程：读取视频 -> 提取特征验证接口 -> 训练 -> 保存 -> 检查加载。
#   - eval_main：使用同一组已训练权重，评测训练片段和新片段
# 运行：python D:/learn/pytorch/practice/run.py experiments/video_feature_demo.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import cv2
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from semantic_encoder import GOPEncoder, encode_gop_for_inference
from semantic_decoder import GOPDecoder
from check_checkpoint import check_checkpoint
from check_checkpoint import load_models

# ==================== 1. 数据处理模块 ====================
def read_frames(video_path, num_frames, start_frame=0):
    """
    从指定帧开始，顺序读取视频。

    参数:
        video_path: 视频路径。
        num_frames: 读取帧数，必须为正整数。
        start_frame: 起始帧编号，从0开始。

    返回:
        frames: BGR图像列表。
    """

    if num_frames <= 0:
        raise ValueError("num_frames必须大于0")

    if start_frame < 0:
        raise ValueError("start_frame不能为负数")

    cap = cv2.VideoCapture(video_path)

    # 先检查打开状态，再尝试读取
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("视频打开失败，请检查路径")

    frames = []

    # finally确保读取失败时也会释放资源
    try:
        for _ in range(start_frame):
            ok, _ = cap.read()

            if not ok:
                raise RuntimeError("无法到达指定起始帧")

        for i in range(num_frames):
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"需要{num_frames}帧，但只读取到{len(frames)}帧"
                )

            frames.append(frame)
            #print(f"帧编号{start_frame + i}:{frame.shape}")

    finally:
        cap.release()

    return frames


def center_crop_frames(frames, crop_size):
    """对每帧做中心裁剪，并校验裁剪内容。"""
    cropped_frames = []
    for i, frame in enumerate(frames):
        height, width, channels = frame.shape
        if height < crop_size or width < crop_size:
            raise ValueError("原图尺寸小于裁剪尺寸")

        top = (height - crop_size) // 2
        left = (width - crop_size) // 2

        cropped = frame[top:top + crop_size, left:left + crop_size, :]
        cropped_frames.append(cropped)

    # 校验
    for i in range(len(frames)):
        height, width, _ = frames[i].shape
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        expected = frames[i][top:top + crop_size, left:left + crop_size, :]

        assert torch.equal(
            torch.from_numpy(cropped_frames[i]),
            torch.from_numpy(expected)
        ), f"第{i + 1}帧裁剪内容不对应"

        #print(f'第{i+1}帧裁剪后：{cropped_frames[i].shape}')

    #print("裁剪内容对应检查通过")
    return cropped_frames


def frames_to_gop(cropped_frames):
    """把裁剪帧转为 RGB 浮点张量并组装成 GOP [1, T, 3, H, W]。"""
    tensor_frames = []
    for frame in cropped_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1) # HWC -> CHW
        tensor_frames.append(tensor)

    gop = torch.stack(tensor_frames, dim=0).unsqueeze(0) # [T,C,H,W] -> [1,T,C,H,W]
    #print("GOP形状：", gop.shape, "| 数据类型：", gop.dtype, "| 范围：", gop.min().item(), "-", gop.max().item())
    return gop


def load_video_as_gop(video_path,crop_size,num_frames,start_frame):
    """
    读取、裁剪视频片段并组装GOP。

    参数:
        video_path: 视频路径。
        crop_size: 中心裁剪边长。
        num_frames: GOP帧数。
        start_frame: 起始帧编号，从0开始。

    返回:
        gop: [1,T,3,crop_size,crop_size]。
    """

    frames = read_frames(
        video_path,
        num_frames=num_frames,
        start_frame=start_frame
    )

    cropped_frames = center_crop_frames(
        frames,
        crop_size=crop_size
    )

    return frames_to_gop(cropped_frames)


# ==================== 2. 模型训练与保存模块 ====================

def train_single_gop(gop, steps, learning_rate):
    """在固定GOP上训练轻量编解码器。"""
    torch.manual_seed(24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gop = gop.to(device)

    encoder = GOPEncoder(embed_dim=24).to(device)
    decoder = GOPDecoder(feature_channels=24).to(device)
    encoder.train()
    decoder.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        features = encoder(gop)
        reconstructed = decoder(features)

        loss = criterion(reconstructed, gop)
        if not torch.isfinite(loss).item():
            raise RuntimeError("损失出现NaN或无穷值")

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step == 0 or (step + 1) % 20 == 0:
            print(f"步骤 {step + 1}/{steps}，MSE={loss.item():.6f}")

    print("训练设备：", device)
    return encoder, decoder, losses


def save_checkpoint(encoder, decoder, losses, save_path):
    """保存编解码器参数和训练损失。"""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_version": "lightweight_cnn_v1",
        "feature_channels": 24,
        "layout": "BTCHW",
        "training_scope": "single_gop",
        "steps": len(losses),
        "encoder": {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()},
        "decoder": {name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()},
        "losses": losses
    }
    torch.save(checkpoint, path)
    print("检查点已保存：", path.resolve())

# ============ 保存评测记录 ============

def save_evaluation_results(records, save_path):
    """
    将逐片段评测结果保存为CSV。

    参数:
        records: 字典列表，每个字典记录一个片段。
        save_path: CSV保存路径。

    返回:
        无。
    """

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig：方便Excel识别中文
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "start_frame",
                "num_frames",
                "used_for_training",
                "mse",
                "checkpoint"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    print("评测记录已保存：", path.resolve())

# ==================== 3. 可视化与测试模块 ====================

def show_reconstruction(encoder, decoder, gop):
    """并排比较原始帧与重建帧。"""
    device = next(encoder.parameters()).device
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        inputs = gop.to(device)
        reconstructed = decoder(encoder(inputs))

    mse = (reconstructed - inputs).square().mean().item()
    originals = inputs[0].cpu()
    outputs = reconstructed[0].cpu()
    T = originals.shape[0]

    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6), squeeze=False)
    for t in range(T):
        axes[0, t].imshow(originals[t].permute(1, 2, 0))
        axes[1, t].imshow(outputs[t].permute(1, 2, 0))
        axes[0, t].set_title(f"Original {t + 1}")
        axes[1, t].set_title(f"Reconstructed {t + 1}")
        axes[0, t].axis("off")
        axes[1, t].axis("off")

    plt.tight_layout()
    print(f"更新后模型的固定GOP MSE：{mse:.6f}")
    plt.show()

    # 将指标交给调用方，方便保存
    return mse

def check_model_interface(gop):
    """检查编码器提取特征的接口与形状。"""
    torch.manual_seed(24)
    model = GOPEncoder(embed_dim=24)
    gop_features = encode_gop_for_inference(model, gop)

    assert gop_features.shape == (1, 4, 24, 32, 32), f"特征形状错误: {gop_features.shape}"
    assert torch.isfinite(gop_features).all().item(), "特征存在 NaN 或 Inf"
    print("真实视频特征接口检查通过")


# ==================== 4. 主流程 ====================

def train_main():
    """主流程：读取视频 -> 提取特征验证接口 -> 训练 -> 保存 -> 检查加载。"""
    video_path = r"D:\learn\pytorch\data\毕洋洋.mp4"
    checkpoint_path = "outputs/single_gop_5000steps_v1.pt"

    # 1. 准备数据
    gop = load_video_as_gop(video_path, crop_size=128, num_frames=4)

    # 2. 验证模型接口形状
    check_model_interface(gop)

    # 3. 训练模型
    encoder, decoder, losses = train_single_gop(gop, steps=5000, learning_rate=1e-3)

    # 4. 保存检查点
    save_checkpoint(encoder, decoder, losses, save_path=checkpoint_path)

    # 5. 检查加载一致性（确保保存的文件能正确读回并输出）
    check_checkpoint(encoder, decoder, gop, checkpoint_path=checkpoint_path)

    # 6. (可选) 可视化重建效果
    # show_reconstruction(encoder, decoder, gop)

def eval_main():
    """
    使用同一组已训练权重，评测训练片段和新片段

    不更新模型参数，不保存或覆盖检查点
    """

    video_path = r"D:\learn\pytorch\data\毕洋洋.mp4"

    checkpoint_path = (
        r"D:\learn\pytorch\outputs"
        r"\single_gop_5000steps_v1.pt"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    encoder, decoder = load_models(
        checkpoint_path,
        device
    )

    records = []

    for start_frame in [0, 100]:
        gop = load_video_as_gop(
            video_path,
            crop_size=128,
            num_frames=4,
            start_frame=start_frame
        )

        mse = show_reconstruction(encoder, decoder, gop)

        records.append({
            "start_frame": start_frame,
            "num_frames": 4,
            "used_for_training": start_frame == 0,
            "mse": mse,
            "checkpoint": checkpoint_path
        })

    save_evaluation_results(
        records,
        save_path="outputs/evaluation_single_gop_v1.csv"
    )
    
    for start_frame in [0, 100]:
        print(
            f"\n评测帧编号："
            f"{start_frame}～{start_frame + 3}"
        )

        gop = load_video_as_gop(
            video_path,
            crop_size=128,
            num_frames=4,
            start_frame=start_frame
        )

        show_reconstruction(encoder, decoder, gop)

if __name__ == "__main__":
    #train_main()
    eval_main()