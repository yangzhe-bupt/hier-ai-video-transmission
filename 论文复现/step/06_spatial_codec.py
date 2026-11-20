"""Step06：逐步构建空间编解码器，首先检查 Patch Merging。"""

import torch
import torch.nn as nn

from pathlib import Path
import sys

# 定位已有的 PatchEmbedding 和 SwinBlockPair。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "practice" / "models" / "swin"))

from semantic_encoder import PatchEmbedding
from swin_block_pair import SwinBlockPair

import importlib

# 复用第五步的视频读取与裁剪函数。
step05 = importlib.import_module("05_video_dataset")

class PatchMerging(nn.Module):
    """
    合并相邻的 2×2 特征位置，并映射通道数。

    输入：
        [N, in_channels, H, W]

    输出：
        [N, out_channels, H/2, W/2]

    本轮采用 LayerNorm + Linear 的常见 Swin 实现。
    四个位置的拼接顺序需要在后续模型中保持一致。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels

        # 每个合并位置由四个原始位置组成，因此有 4C 个特征。
        # LayerNorm 对每个位置最后一维的 4C 个数做归一化。
        self.norm = nn.LayerNorm(4 * in_channels)

        # 将拼接后的 4C 维表示映射到目标通道数。
        # 例如第一层是 96 → 48，第二层是 192 → 72。
        self.reduction = nn.Linear(
            4*in_channels,
            out_channels,
            bias=False,
        )

    def forward(self,x):
        if x.ndim != 4:
            raise ValueError("输入必须为 [N,C,H,W]")

        _, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError("输入通道数与模块配置不一致")

        # 本轮先要求尺寸为偶数。
        # 后续处理任意分辨率时，再统一设计边界填充。
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Patch Merging 要求高和宽为偶数")

        # [N,C,H,W] → [N,H,W,C]
        x = x.permute(0, 2, 3, 1)

        # 从每个 2×2 区域取出四个位置。
        # 每个张量的形状均为 [N,H/2,W/2,C]
        top_left = x[:, 0::2, 0::2, :]
        bottom_left = x[:, 1::2, 0::2, :]
        top_right = x[:, 0::2, 1::2, :]
        bottom_right = x[:, 1::2, 1::2, :]

        # 拼接到通道维：[N,H/2,W/2,4C]。
        x = torch.cat(
            [top_left, bottom_left, top_right, bottom_right],
            dim=-1,
        )

        x = self.norm(x)
        x = self.reduction(x)

        # 恢复项目采用的 NCHW 布局。
        # contiguous 让输出在内存中连续，方便后续操作。
        return x.permute(0, 3, 1, 2).contiguous()

class SpatialEncoder(nn.Module):
    """
    三阶段空间编码器的接口验证版本。

    输入：
        frames：[N,3,H,W]，N 可以表示 B*T。

    输出：
        features：[N,72,H/8,W/8]。

    本轮使用64×64输入，以满足现有移位窗口实现的尺寸要求。
    """

    def __init__(self):
        super().__init__()

        # Stage1：2×2 Patch Embedding，高宽各缩小一半。
        # RGB 的3个通道映射为24个特征通道。
        self.patch_embed = PatchEmbedding(embed_dim=24)

        self.stage1 = SwinBlockPair(
            dim=24,
            num_heads=3,
            window_size=4,
        )

        # Stage2：合并空间位置，再提取当前尺度上的特征。
        self.merge1 = PatchMerging(
            in_channels=24,
            out_channels=48,
        )

        self.stage2 = SwinBlockPair(
            dim=48,
            num_heads=3,
            window_size=4,
        )

        # Stage3：再次降低分辨率，将通道调整到72。
        self.merge2 = PatchMerging(
            in_channels=48,
            out_channels=72,
        )

        self.stage3 = SwinBlockPair(
            dim=72,
            num_heads=3,
            window_size=4,
        )

    def forward(self,frames):
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("输入必须为 [N,3,H,W]")

        # 输入64×64时，Stage1输出32×32。
        x = self.patch_embed(frames)
        x = self.stage1(x)

        # Stage2输出：[N,48,16,16]。
        x = self.merge1(x)
        x = self.stage2(x)

        # Stage3输出：[N,72,8,8]。
        x = self.merge2(x)
        x = self.stage3(x)

        return x

class PatchExpansion(nn.Module):
    """
    将特征图的高、宽各扩大2倍，并调整通道数。

    输入：
        [N, in_channels, H, W]

    输出：
        [N, out_channels, 2H, 2W]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels

        # 每个输入位置要生成四个输出位置。
        # 每个输出位置有 out_channels 个通道，
        # 因此先生成 4*out_channels 个数。
        self.expansion = nn.Linear(
            in_channels,
            4*out_channels,
            bias=False,
        )

        # 将4组通道重新排列成2×2的空间位置。
        self.shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self,x):
        if x.ndim != 4:
            raise ValueError("输入必须为 [N,C,H,W]")

        if x.shape[1] != self.in_channels:
            raise ValueError("输入通道数与模块配置不一致")

        # [N,C,H,W] → [N,H,W,C]
        x = x.permute(0, 2, 3, 1)

        # [N,H,W,C] → [N,H,W,4*Cout]
        x = self.expansion(x)

        # [N,H,W,4*Cout] → [N,4*Cout,H,W]
        x = x.permute(0, 3, 1, 2).contiguous()

        # 重排后得到 [N,Cout,2H,2W]。
        return self.shuffle(x)

class SpatialDecoder(nn.Module):
    """
    将三阶段空间编码特征恢复为 RGB 图像。

    输入：
        [N,72,H/8,W/8]

    输出：
        [N,3,H,W]，数值位于 [0,1]。
    """
    def __init__(self):
        super().__init__()

        # 先处理最深层的72通道特征。
        self.stage3 = SwinBlockPair(
            dim=72,
            num_heads=3,
            window_size=4,
        )
        self.expand3 = PatchExpansion(
            in_channels=72,
            out_channels=48,
        )

        # 恢复到中间尺度后，继续提取和整合空间信息。
        self.stage2 = SwinBlockPair(
            dim=48,
            num_heads=3,
            window_size=4,
        )
        self.expand2 = PatchExpansion(
            in_channels=48,
            out_channels=24,
        )

        # 最后一级恢复到原始图像分辨率，并输出RGB三通道。
        self.stage1 = SwinBlockPair(
            dim=24,
            num_heads=3,
            window_size=4,
        )
        self.expand1 = PatchExpansion(
            in_channels=24,
            out_channels=3,
        )

        # 将输出限制到 [0,1]，与归一化后的图像范围一致。
        self.output_activation = nn.Sigmoid()

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != 72:
            raise ValueError("输入必须为 [N,72,H,W]")

        x = self.stage3(features)
        x = self.expand3(x)

        x = self.stage2(x)
        x = self.expand2(x)

        x = self.stage1(x)
        x = self.expand1(x)

        return self.output_activation(x)

def check_spatial_codec():
    """检查完整空间编解码器的尺寸、数值和参数梯度。"""
    encoder = SpatialEncoder()
    decoder = SpatialDecoder()

    # 两张64×64图像，经编码得到8×8特征。
    images = torch.rand(2, 3, 64, 64)

    features = encoder(images)
    reconstructed = decoder(features)

    print("编解码检查输入：", tuple(images.shape))
    print("编解码检查输出：", tuple(reconstructed.shape))

    assert reconstructed.shape == images.shape
    assert torch.isfinite(reconstructed).all()
    assert reconstructed.min() >= 0
    assert reconstructed.max() <= 1

    # 从重建误差反向传播，检查编码器和解码器参数。
    loss = (reconstructed - images).square().mean()
    loss.backward()

    for model_name, model in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if parameter.grad is None:
                raise RuntimeError(
                    f"{model_name}.{parameter_name} 没有梯度"
                )

            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"{model_name}.{parameter_name} 梯度存在异常"
                )

    print("完整空间编解码器梯度检查通过")

def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    video_path = Path(
        r"D:\学习与项目\学习代码\pytorch\data\videos"
        r"\train\big_buck_bunny_trailer_720p.mov"
    )

    # 复用 Step05：读取连续四帧。
    frames = step05.read_frames(
        video_path,
        start_frame=100,
        num_frames=4,
    )

    # 对四帧使用同一个裁剪位置，得到 RGB、[0,1] 的 GOP。
    # gop 形状：[1,4,3,256,256]。
    gop, crop_position = step05.frames_to_gop(
        frames,
        crop_size=256,
        seed=42,
    )

    encoder = SpatialEncoder()
    encoder.eval()

    # 读取原始维度，后面恢复视频结构时使用。
    batch, time, channels, height, width = gop.shape

    # 二维空间编码器接收 NCHW。
    # 合并 B、T，将所有帧作为同一批图像处理。
    frame_batch = gop.reshape(
        batch * time,
        channels,
        height,
        width,
    )

    # 本轮只检查真实视频的前向计算，关闭梯度以节省内存。
    with torch.no_grad():
        spatial_features = encoder(frame_batch)

    # 从实际输出读取通道和空间尺寸。
    _, feature_channels, feature_height, feature_width = (
        spatial_features.shape
    )

    # 恢复批次与时间维度，保留四帧各自的空间特征。
    video_features = spatial_features.reshape(
        batch,
        time,
        feature_channels,
        feature_height,
        feature_width,
    )

    print("共同裁剪位置：", crop_position)
    print("原始 GOP：", tuple(gop.shape))
    print("合并后的帧批次：", tuple(frame_batch.shape))
    print("逐帧空间特征：", tuple(spatial_features.shape))
    print("恢复视频维度后：", tuple(video_features.shape))

    assert video_features.shape == (1, 4, 72, 32, 32)
    assert torch.isfinite(video_features).all()

        # 单独检查上采样组件，先使用小型随机特征。
    # requires_grad=True 便于验证梯度能否通过这些操作。
    test_features = torch.randn(
        2, 72, 8, 8,
        requires_grad=True,
    )

    expand1 = PatchExpansion(in_channels=72, out_channels=48)
    expand2 = PatchExpansion(in_channels=48, out_channels=24)
    expand3 = PatchExpansion(in_channels=24, out_channels=3)

    x1 = expand1(test_features)
    x2 = expand2(x1)
    x3 = expand3(x2)

    print("第一次上采样：", tuple(x1.shape))
    print("第二次上采样：", tuple(x2.shape))
    print("第三次上采样：", tuple(x3.shape))

    assert x1.shape == (2, 48, 16, 16)
    assert x2.shape == (2, 24, 32, 32)
    assert x3.shape == (2, 3, 64, 64)

    loss = x3.square().mean()
    loss.backward()

    assert test_features.grad is not None
    assert torch.isfinite(test_features.grad).all()

    print("PatchExpansion 梯度检查通过")

        # 小尺寸输入验证完整编解码链路的反向传播。
    check_spatial_codec()

    # 再把前面得到的真实视频空间特征交给新解码器。
    decoder = SpatialDecoder()
    decoder.eval()

    with torch.no_grad():
        # spatial_features 是前面编码得到的：
        # [B*T,72,32,32]。
        reconstructed_frames = decoder(spatial_features)

        # 恢复为视频布局 [B,T,3,H,W]。
        reconstructed_gop = reconstructed_frames.reshape(
            batch,
            time,
            3,
            height,
            width,
        )

    print("真实 GOP 重建形状：", tuple(reconstructed_gop.shape))

    assert reconstructed_gop.shape == gop.shape
    assert torch.isfinite(reconstructed_gop).all()

if __name__ == "__main__":
    main()

        