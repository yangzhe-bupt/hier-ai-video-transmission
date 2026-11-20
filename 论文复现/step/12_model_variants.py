"""仅改变图7公共残差支路的受控版本；不宣称完整论文复现。"""
import importlib
import torch
from torch import nn

base = importlib.import_module("12_train_stage1")


class Figure7CommonExtractor(nn.Module):
    """保留原时间压缩和通道混合，残差支路改为2D卷积及两处GELU。

    输入BTCHW → 时间压缩为1 → 去掉单例T维 → 2D残差 → 恢复BTCHW。
    原Conv3d在T=1、padding=1时仅中间时间切片有效。复制随机初始化的
    中心切片，使两版本初始有效卷积权重相同，减少无关初始化差异。
    这不是加载已训练权重；新实验仍从随机初始化开始。
    """
    def __init__(self, original):
        super().__init__()
        self.channels = original.channels
        self.gop_size = original.gop_size
        self.temporal_compression = original.temporal_compression
        self.activation = original.activation
        self.channel_mixing = original.channel_mixing
        # 构造新层时恢复全局RNG，避免改变后续量化噪声序列。
        with torch.random.fork_rng(devices=[]):
            self.refinement = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                nn.GELU(),
            )
        with torch.no_grad():
            for index in (0, 2):
                source = original.refinement[index]
                self.refinement[index].weight.copy_(source.weight[:, :, 1])
                self.refinement[index].bias.copy_(source.bias)

    def forward(self, features):
        if features.ndim != 5 or features.shape[1:3] != (self.gop_size, self.channels):
            raise ValueError("需要[B,gop_size,channels,H,W]")
        x = features.permute(0, 2, 1, 3, 4).contiguous()
        x = self.channel_mixing(self.activation(self.temporal_compression(x)))
        if x.shape[2] != 1:
            raise ValueError("公共分支时间维必须压缩为1")
        spatial = x.squeeze(2)
        refined = spatial + self.refinement(spatial)
        return refined.unsqueeze(1)  # [B,1,C,H,W]


class Stage6BlockPair(nn.Module):
    """Stage6两个独立块；单空间窗口时第二块使用非移位注意力。

    当前模型要求正方形空间窗口且输入尺寸为256的倍数。若只有一个轴
    大于窗口，现有掩码不支持单轴移位，明确报错而不悄悄改变规则。
    两块辅助损失取均值，外层再按真实块数加权。
    """
    def __init__(self, first):
        super().__init__()
        self.first = first
        self.second = base.step07.SwinBlock3D(
            dim=144, num_heads=3, window_size=(4,4,4), shift_size=(0,0,0))
        base.step11.replace_mlp_with_moe(self.second, num_experts=4, top_k=2)

    def forward(self, features, return_aux=False):
        h, w = features.shape[-2:]
        if (h > 4) != (w > 4):
            raise ValueError("Stage6当前不支持单轴多窗口，需扩展掩码后使用")
        # second的shift_size只含运行时几何配置，不参与state_dict。
        # 单窗口的4×4输入不移位；8×8及以上启用空间移位。
        self.second.shift_size = (0,2,2) if h > 4 and w > 4 else (0,0,0)
        if return_aux:
            x, a = self.first(features, return_aux=True)
            x, b = self.second(x, return_aux=True)
            return x, (a+b)/2
        return self.second(self.first(features))


def build_model(variant="legacy"):
    if variant not in ("legacy", "common_fig7", "common_fig7_stage6_pair"):
        raise ValueError(f"未知模型版本：{variant}")
    model = base.Stage1Model()
    if variant in ("common_fig7", "common_fig7_stage6_pair"):
        model.common_extractor = Figure7CommonExtractor(model.common_extractor)
    if variant == "common_fig7_stage6_pair":
        model.individual_extractor.block = Stage6BlockPair(model.individual_extractor.block)
        model.individual_upsample.block = Stage6BlockPair(model.individual_upsample.block)
        # Stage1Model.forward会按此数量加权各模块的辅助损失均值。
        model.moe_counts = (4,2,2,4)
    return model
