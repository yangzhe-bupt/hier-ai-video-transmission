"""Step08：可学习的公共特征提取与初步残差分解。"""

import torch
import torch.nn as nn

import importlib

from pathlib import Path

step05 = importlib.import_module("05_video_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 复用已经检查过的空间尺寸变换和3D Swin块。
step06 = importlib.import_module("06_spatial_codec")
step07 = importlib.import_module("07_spatiotemporal_codec")

PatchMerging = step06.PatchMerging
PatchExpansion = step06.PatchExpansion
SwinBlock3D = step07.SwinBlock3D

class CommonFeatureExtractor(nn.Module):
    """
    从整个GOP的时空特征中提取公共特征。

    输入：
        [B,T,C,H,W]，T必须等于配置的gop_size。

    输出：
        [B,1,C,H,W]。

    Conv3d接收BCTHW，内部需要进行布局转换。
    """

    def __init__(self, channels=120, gop_size=4):
        super().__init__()

        self.channels = channels
        self.gop_size = gop_size

        # 时间卷积核覆盖整个GOP，时间步长也为GOP长度。
        # 时间维从4压到1。
        # 空间方向使用3×3卷积并补零，保持高、宽不变。
        self.temporal_compression = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(gop_size, 3, 3),
            stride=(gop_size, 1, 1),
            padding=(0, 1, 1),
        )

        self.activation = nn.GELU()

        # 在每个位置上混合通道信息，不改变时空尺寸。
        self.channel_mixing = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
        )

        # 本轮采用两层3×3×3卷积构成残差分支。
        # padding=1使时间、高、宽尺寸保持不变。
        self.refinement = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features):
        if features.ndim != 5:
            raise ValueError("输入必须为 [B,T,C,H,W]")

        if features.shape[1] != self.gop_size:
            raise ValueError("输入帧数必须等于配置的GOP长度")

        if features.shape[2] != self.channels:
            raise ValueError("输入通道数与配置不一致")

        # BTCHW → BCTHW。
        x = features.permute(0, 2, 1, 3, 4).contiguous()

        # [B,C,4,H,W] → [B,C,1,H,W]。
        x = self.temporal_compression(x)
        x = self.activation(x)

        x = self.channel_mixing(x)

        # 残差分支学习对当前公共表示的修正。
        x = x + self.refinement(x)

        # BCTHW → BTCHW。
        return x.permute(0, 2, 1, 3, 4).contiguous()

class IndividualFeatureExtractor(nn.Module):
    """
    压缩初步个体特征。

    输入：[B,4,120,8,8]
    输出：[B,4,144,4,4]
    """

    def __init__(self):
        super().__init__()

        self.merge = PatchMerging(
            in_channels=120,
            out_channels=144,
        )

        self.block = SwinBlock3D(
            dim=144,
            num_heads=3,
            window_size=(4, 4, 4),
            shift_size=(0, 0, 0),
        )

    def forward(self, individual, return_aux=False):
        if individual.ndim != 5 or individual.shape[2] != 120:
            raise ValueError("输入必须为 [B,T,120,H,W]")

        batch, time, channels, height, width = individual.shape

        # 暂时合并B、T，分别对各帧做空间降采样。
        x = individual.reshape(
            batch * time, channels, height, width
        )
        x = self.merge(x)

        _, new_channels, new_height, new_width = x.shape

        # 恢复时间维，让3D Swin处理帧间关系。
        x = x.reshape(
            batch, time, new_channels, new_height, new_width
        )

        # 默认仍返回特征；完整模型可显式收集该MoE块的辅助损失。
        return self.block(x, return_aux=return_aux)

class IndividualUpsample(nn.Module):
    """
    将压缩个体特征恢复为能够与公共特征相加的尺寸。

    输入：[B,4,144,4,4]
    输出：[B,4,120,8,8]
    """

    def __init__(self):
        super().__init__()

        # 与当前解码实现一致，先处理特征，再扩大空间尺寸。
        self.block = SwinBlock3D(
            dim=144,
            num_heads=3,
            window_size=(4, 4, 4),
            shift_size=(0, 0, 0),
        )

        self.expansion = PatchExpansion(
            in_channels=144,
            out_channels=120,
        )

    def forward(self, compressed, return_aux=False):
        if compressed.ndim != 5 or compressed.shape[2] != 144:
            raise ValueError("输入必须为 [B,T,144,H,W]")

        batch, time, channels, height, width = compressed.shape

        if return_aux:
            x, auxiliary_loss = self.block(compressed, return_aux=True)
        else:
            x = self.block(compressed)

        # PatchExpansion按帧处理，因此暂时合并B、T。
        x = x.reshape(
            batch * time, channels, height, width
        )
        x = self.expansion(x)

        _, new_channels, new_height, new_width = x.shape

        output = x.reshape(
            batch, time, new_channels, new_height, new_width
        )
        if return_aux:
            return output, auxiliary_loss
        return output

def check_individual_compression():
    features = torch.randn(
        1, 4, 120, 8, 8,
        requires_grad=True,
    )

    common_extractor = CommonFeatureExtractor(
        channels=120,
        gop_size=4,
    )
    individual_extractor = IndividualFeatureExtractor()
    individual_upsample = IndividualUpsample()

    common = common_extractor(features)
    individual = features - common

    compressed = individual_extractor(individual)
    recovered = individual_upsample(compressed)

    # common的时间维为1，自动广播到4帧。
    restored = common + recovered

    print("初步个体特征：", tuple(individual.shape))
    print("压缩个体特征：", tuple(compressed.shape))
    print("上采样个体特征：", tuple(recovered.shape))
    print("重组时空特征：", tuple(restored.shape))

    assert compressed.shape == (1, 4, 144, 4, 4)
    assert recovered.shape == features.shape
    assert restored.shape == features.shape
    assert torch.isfinite(restored).all()

    # 压缩与上采样包含可学习变换，不再要求逐元素恢复原值。
    # 用特征重建误差检查整个分解、压缩、恢复链路。
    loss = (restored - features).square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    for model in (
        common_extractor,
        individual_extractor,
        individual_upsample,
    ):
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"{name} 没有梯度")

            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"{name} 梯度存在异常")

    # 比较待表示的元素数量，暂不涉及量化和协议开销。
    print("原始特征元素数：", features.numel())
    print(
        "两分支特征元素总数：",
        common.numel() + compressed.numel(),
    )
    print("个体特征压缩与上采样梯度检查通过")

def check_full_decomposition_path():
    """将公共/个体分支接入真实视频的完整前向链路。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)

    video_path = Path(
        r"D:\学习与项目\学习代码\pytorch\data\videos"
        r"\train\big_buck_bunny_trailer_720p.mov"
    )

    frames = step05.read_frames(
        video_path,
        start_frame=100,
        num_frames=4,
    )
    gop, _ = step05.frames_to_gop(
        frames,
        crop_size=256,
        seed=42,
    )

    # 创建完整链路需要的各个模块。
    spatial_encoder = step06.SpatialEncoder()
    temporal_encoder = step07.SpatiotemporalEncoder()

    common_extractor = CommonFeatureExtractor(
        channels=120,
        gop_size=4,
    )
    individual_extractor = IndividualFeatureExtractor()
    individual_upsample = IndividualUpsample()

    temporal_decoder = step07.SpatiotemporalDecoder()
    spatial_decoder = step06.SpatialDecoder()

    models = (
        spatial_encoder,
        temporal_encoder,
        common_extractor,
        individual_extractor,
        individual_upsample,
        temporal_decoder,
        spatial_decoder,
    )
    for model in models:
        model.eval()

    batch, time, channels, height, width = gop.shape

    with torch.no_grad():
        # ---------- 空间与时空编码 ----------

        frame_batch = gop.reshape(
            batch * time, channels, height, width
        )
        encoded_frames = spatial_encoder(frame_batch)

        _, cs, hs, ws = encoded_frames.shape

        spatial_features = encoded_frames.reshape(
            batch, time, cs, hs, ws
        )
        temporal_features = temporal_encoder(spatial_features)

        # ---------- 发送端：公共/个体分解 ----------

        common = common_extractor(temporal_features)
        individual = temporal_features - common
        compressed = individual_extractor(individual)

        # 当前直接传递两分支特征。
        # 后续会在这里加入量化、交织和丢包通道。
        received_common = common
        received_individual = compressed

        # ---------- 接收端：分支重组 ----------

        recovered_individual = individual_upsample(
            received_individual
        )

        # 公共分支沿时间维自动广播。
        restored_temporal = received_common + recovered_individual

        assert restored_temporal.shape == temporal_features.shape

        # ---------- 时空与空间解码 ----------

        decoded_features = temporal_decoder(restored_temporal)

        assert decoded_features.shape == spatial_features.shape

        decoder_input = decoded_features.reshape(
            batch * time, cs, hs, ws
        )
        decoded_frames = spatial_decoder(decoder_input)

        reconstructed = decoded_frames.reshape(
            batch, time, 3, height, width
        )

        mse = (reconstructed - gop).square().mean().item()

    print("真实 GOP：", tuple(gop.shape))
    print("时空编码结果：", tuple(temporal_features.shape))
    print("公共分支：", tuple(common.shape))
    print("压缩个体分支：", tuple(compressed.shape))
    print("重组时空特征：", tuple(restored_temporal.shape))
    print("最终重建 GOP：", tuple(reconstructed.shape))

    for tensor in (
        temporal_features,
        common,
        compressed,
        restored_temporal,
        reconstructed,
    ):
        assert torch.isfinite(tensor).all()

    assert reconstructed.shape == gop.shape
    assert reconstructed.min() >= 0
    assert reconstructed.max() <= 1

    print("随机初始化完整链路 MSE：", mse)
    print("双分支完整视频路径检查通过")

def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    # 模拟时空编码器的输出。
    features = torch.randn(
        1, 4, 120, 8, 8,
        requires_grad=True,
    )

    extractor = CommonFeatureExtractor(
        channels=120,
        gop_size=4,
    )

    common = extractor(features)

    # 时间维为1的common会自动广播到4帧。
    # 每帧分别减去同一个公共特征。
    individual = features - common

    print("输入时空特征：", tuple(features.shape))
    print("公共特征：", tuple(common.shape))
    print("初步个体特征：", tuple(individual.shape))

    assert common.shape == (1, 1, 120, 8, 8)
    assert individual.shape == features.shape

    # 在尚未压缩、量化或丢包时，
    # “公共特征 + 初步个体特征”应恢复输入。
    restored = common + individual
    torch.testing.assert_close(restored, features)

    # 用两个分支构造检查用损失，确认提取器有梯度。
    # 这是程序检查用的目标，正式训练将使用论文损失。
    loss = common.square().mean() + individual.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    for name, parameter in extractor.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{name} 没有梯度")

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{name} 梯度存在异常")

    print("公共与初步个体特征重组检查通过")
    print("公共特征提取器梯度检查通过")
    check_individual_compression()
    check_full_decomposition_path()

if __name__ == "__main__":
    main()