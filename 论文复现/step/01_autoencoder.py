"""第一步：理解视频编码和解码的张量形状。"""

from pathlib import Path
import sys

import torch

# 获取当前脚本所在的目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 将两个源码目录加入搜索路径，使 Python 能找到这些模块。
sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models" / "swin"))

from swin_encoder import SwinGOPEncoder
from semantic_decoder import GOPDecoder

def main():
    torch.manual_seed(42)

    # 创建编码器。
    # embed_dim=24：编码特征有 24 个通道。
    # num_heads=3：注意力划分为 3 个头。
    # window_size=4：在 4×4 的局部窗口内计算注意力。
    encoder = SwinGOPEncoder(
        embed_dim=24,
        num_heads=3,
        window_size=4,
    )

    decoder = GOPDecoder(feature_channels = 24)

    # 构造 [B,T,C,H,W] 格式的随机视频。
    # torch.rand 生成 [0,1) 范围内的浮点数，
    # 用于模拟归一化后的 RGB 像素。
    video = torch.rand(1,4,3,32,32)

    features = encoder(video)

    reconstructed = decoder(features)

    print("原始视频形状：", tuple(video.shape))
    print("编码特征形状：", tuple(features.shape))
    print("重建视频形状：", tuple(reconstructed.shape))

    # 检查形状是否符合约定。
    assert features.shape == (1, 4, 24, 8, 8)
    assert reconstructed.shape == video.shape

    # MSE：每个重建像素与原像素之差的平方，再取平均。
    loss = (reconstructed - video).square().mean()

    loss.backward()
    
    # 查看 PatchEmbedding 卷积权重收到的梯度。
    # 有限且非零的梯度说明损失能够传回编码器这一层。
    gradient = encoder.patch_embed.conv.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()

    print("重建 MSE：", loss.item())
    print("PatchEmbedding 平均梯度绝对值：",
          gradient.abs().mean().item())


if __name__ == "__main__":
    main()