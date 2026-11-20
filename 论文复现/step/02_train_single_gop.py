"""第二步：在一个固定的合成视频 GOP 上训练编解码器。"""

from pathlib import Path
import sys

import torch

import matplotlib.pyplot as plt
# 文件位于 Conquering/step 中。
# 向上两级得到 Conquering，随后定位已有模型文件。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models" / "swin"))

from swin_encoder import SwinGOPEncoder
from semantic_decoder import GOPDecoder

def make_video():
    """
    创建一个固定的合成视频。

    返回：
        video：浮点张量，形状 [1, 4, 3, 32, 32]。
        像素值位于 [0, 1]。

    内容：
        深色背景，红色方块在四帧中逐渐向右移动。
    """

    # B=1、T=4、C=3、H=W=32。
    # 用 0.1 初始化，让背景呈深灰色。
    video = torch.full((1,4,3,32,32),0.1)

    for frame in range(4):
        # 四帧中，方块左边界依次为 4、7、10、13。
        left = 4 + frame * 3

        # 第三个维度是颜色通道，0 表示红色通道。
        # 将红色通道的矩形区域设为 0.9。
        # 绿色、蓝色仍为背景值 0.1，因此显示为红色方块。
        video[0,frame,0,10:22,left:left+10] = 0.9
        #把第 0 个视频、第 0 帧、时间点 10:22（或第 10 到 22 帧的范围）、左侧区域（左边距加 10 像素的位置）的数值设为 0.9。

    return video

def show_reconstruction(video, before, after):
    """
    对比四帧视频的重建效果。

    三个输入的形状均为 [1, 4, 3, 32, 32]。
    三行分别展示原图、训练前重建、训练后重建。
    四列分别对应视频的四帧。
    """
    figure, axes = plt.subplots(3, 4, figsize=(10, 7))

    rows = [
        ("Original", video),
        ("Before training", before),
        ("After training", after),
    ]

    for row_index, (label, frames) in enumerate(rows):
        for frame_index in range(4):
            # 取第 0 个样本的第 frame_index 帧。
            # 形状从 [B,T,C,H,W] 变成 [C,H,W]。
            frame = frames[0, frame_index]

            # detach：从计算图分离，便于转换为 NumPy。
            # cpu：将数据放到 CPU，兼容以后在 GPU 上训练。
            # permute：Matplotlib 需要 [H,W,C] 的图像布局。
            image = frame.detach().cpu().permute(1, 2, 0).numpy()

            axis = axes[row_index, frame_index]
            axis.imshow(image.clip(0, 1))
            axis.set_xticks([])
            axis.set_yticks([])

            if row_index == 0:
                axis.set_title(f"Frame {frame_index}")

            if frame_index == 0:
                axis.set_ylabel(label)

    figure.tight_layout()

    # 保存到项目 outputs 下，方便回看和比较。
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "step02_reconstruction.png"

    figure.savefig(image_path, dpi=150)
    print(f"重建对比图已保存到：{image_path}")

    plt.show()
    plt.close(figure)

def main():
    # 固定初始化，方便比较多次运行的结果。
    torch.manual_seed(42)

    # 小型 CPU 实验限制为两个线程，减少线程调度开销。
    torch.set_num_threads(2)

    video = make_video()

    encoder = SwinGOPEncoder(
        embed_dim = 24,
        num_heads = 3,
        window_size = 4,
    )
    decoder = GOPDecoder(feature_channels=24)

    # 同时把编码器和解码器的参数交给优化器。
    # 两个网络共同学习如何编码和恢复这个视频。
    parameters = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.Adam(parameters,lr=0.001)

    encoder.train()
    decoder.train()

    # 记录训练前的误差，作为比较起点。
    # 此处只测量结果，无须记录梯度。
    with torch.no_grad():
        initial_output = decoder(encoder(video))
        initial_mse = (initial_output - video).square().mean().item()

    print(f"训练前 MSE：{initial_mse:.6f}")

    # 每次循环都使用同一个 GOP，共执行 200 次参数更新。
    for step in range(1,201):
        # 1. 清除上一次迭代的梯度。
        optimizer.zero_grad(set_to_none=True)

        # 2. 前向传播。
        # 编码：[1,4,3,32,32] → [1,4,24,8,8]
        # 解码：[1,4,24,8,8] → [1,4,3,32,32]
        features = encoder(video)
        reconstructed = decoder(features)

        # 3. 计算重建误差。
        # 对全部帧、通道和像素的平方误差取平均。
        loss = (reconstructed - video).square().mean()

        # 4. 反向传播，计算模型参数的梯度。
        loss.backward()

        # 5. 根据梯度更新编码器和解码器的参数。
        optimizer.step()

        # 此处 loss 是本次更新之前算出的误差。
        if step == 1 or step % 20 == 0:
            print(
                f"第 {step:3d} 次更新，"
                f"更新前 MSE：{loss.item():.6f}"
            )

    # 完成全部更新后，测量最终模型的重建误差。
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        final_output = decoder(encoder(video))
        final_mse = (final_output - video).square().mean().item()

    print(f"训练后 MSE：{final_mse:.6f}")

     # initial_output 是训练前重建，final_output 是训练后重建。
    show_reconstruction(video, initial_output, final_output)
    
    # 创建检查点保存目录。
    checkpoint_dir = PROJECT_ROOT / "outputs" / "step02"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # state_dict 保存模型中可学习参数等状态。
    # 加载时，需要先用相同配置重新创建编码器和解码器。
    checkpoint = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),

        # 记录结构参数，避免加载时使用了不同的模型尺寸。
        "config": {
            "embed_dim": 24,
            "num_heads": 3,
            "window_size": 4,
        },

        # 标明权重是在什么输入上训练的。
        "input_kind": "synthetic_moving_square",
        "seed": 42,
        "steps": 200,
        "final_mse": final_mse,
    }

    # 同名文件再次保存时会被覆盖。
    checkpoint_path = checkpoint_dir / "autoencoder.pt"
    torch.save(checkpoint, checkpoint_path)

    print(f"模型权重已保存到：{checkpoint_path}")
    
if __name__ == "__main__":
    main()