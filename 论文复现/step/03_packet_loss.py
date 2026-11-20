"""第三步：加载已有权重，观察特征丢包对重建的影响。"""

from pathlib import Path
import importlib
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 定位已有模型和丢包模拟函数。
sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models" / "swin"))
sys.path.insert(0, str(PROJECT_ROOT / "packet"))

from swin_encoder import SwinGOPEncoder
from semantic_decoder import GOPDecoder
from packet_loss import simulate_packet_loss

# 复用第二步的 make_video，保证测试输入与训练输入一致。
# 文件名以数字开头，使用 importlib 按模块名导入。
# 第二步的训练入口有 __main__ 判断，导入时只会定义函数。
step02 = importlib.import_module("02_train_single_gop")
make_video = step02.make_video

def load_models():
    """根据检查点里的配置创建模型，再加载训练好的参数。"""

    checkpoint_path = (
        PROJECT_ROOT / "outputs" / "step02" / "autoencoder.pt"
    )

    # map_location="cpu"：本轮统一在 CPU 上推理。
    # weights_only=True 适用于本项目保存的参数和基础类型字典。
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    encoder = SwinGOPEncoder(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
    )
    decoder = GOPDecoder(feature_channels=config["embed_dim"])

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    # 切换为评估模式。
    encoder.eval()
    decoder.eval()

    return encoder, decoder

def main():
    torch.set_num_threads(2)

    encoder, decoder = load_models()
    video = make_video()

    # 第一行展示原始四帧。
    rows = [("Original", video)]

    # p 是每个包独立丢失的概率。
    loss_rates = [0.0, 0.25, 0.5, 0.9]

    # 推理阶段关闭梯度记录。
    with torch.no_grad():
        features = encoder(video)

        # [1,4,24,8,8] 共包含 6144 个特征数值。
        # 本实验每包包含 16 个数值，因此共有 384 个包。
        # 注意，这里的包长单位是“元素数”。
        packet_size = 16
        print("特征形状：", tuple(features.shape))
        print("总包数：", features.numel() // packet_size)

        for rate_index, p in enumerate(loss_rates):
            # 固定每组实验的随机种子，便于重复观察。
            torch.manual_seed(42 + rate_index)

            received, missing_mask, dropped_packets = (
                simulate_packet_loss(
                    features,
                    p=p,
                    packet_size=packet_size,
                )
            )

            # received：收到的特征，丢失位置已被置零。
            # missing_mask：元素级 bool 掩码，True 表示缺失。
            # dropped_packets：包级 bool 记录，True 表示该包丢失。

            # p=0 时，特征应完整保留。
            if p == 0.0:
                assert torch.equal(received, features)

            # 当前解码器只接收特征。
            # mask 保留下来供后续恢复模块使用。
            reconstructed = decoder(received)

            mse = (reconstructed - video).square().mean().item()

            # 实际丢包率由本次随机抽样决定，未必恰好等于 p。
            actual_loss = dropped_packets.float().mean().item()

            print(
                f"设定丢包概率={p:.0%}，"
                f"实际丢包率={actual_loss:.1%}，"
                f"MSE={mse:.6f}"
            )

            rows.append((f"p={p:.0%}", reconstructed))

    # 五行分别展示原图和四种丢包条件，四列对应四帧。
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(10, 2 * len(rows))
    )

    for row_index, (label, frames) in enumerate(rows):
        for frame_index in range(4):
            image = (
                frames[0, frame_index]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )

            axis = axes[row_index, frame_index]
            axis.imshow(image.clip(0, 1))
            axis.set_xticks([])
            axis.set_yticks([])

            if row_index == 0:
                axis.set_title(f"Frame {frame_index}")

            if frame_index == 0:
                axis.set_ylabel(label)

    figure.tight_layout()

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "step03_packet_loss.png"

    figure.savefig(image_path, dpi=150)
    print(f"丢包对比图已保存到：{image_path}")

    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()