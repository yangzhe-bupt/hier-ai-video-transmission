"""分离额外降采样、3D注意力与MoE的固定8GOP拟合对照。
复用31的相同数据/步数/优化器；默认只检查数据，--train才训练。
结果仍写入outputs/step31_fixed_gop_ablation的新时间戳目录，以config中的模型名区分。
不做码率公平比较，不将结构差异直接归因为单个算子缺陷。
"""
import importlib
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'step'), str(ROOT / 'tools')]
runner = importlib.import_module('31_fixed_gop_ablation')
original_build = runner.variants.build_model
base = runner.variants.base


class TemporalFactor(nn.Module):
    def __init__(self, use_attention):
        super().__init__()
        # 与原空间/时空对照相同顺序初始化空间编码器和解码器。
        self.spatial_encoder = base.step06.SpatialEncoder()
        self.spatial_decoder = base.step06.SpatialDecoder()
        self.temporal_encoder = base.step07.SpatiotemporalEncoder()
        self.temporal_decoder = base.step07.SpatiotemporalDecoder()
        self.use_attention = use_attention
        if not use_attention:
            # 真正移除注意力参数，仅保留原Patch Merging与Expansion。
            for stage in (self.temporal_encoder.stage4, self.temporal_encoder.stage5,
                          self.temporal_decoder.decode_stage5, self.temporal_decoder.decode_stage4):
                stage.blocks = nn.Sequential()

    @staticmethod
    def spatial_apply(module, x):
        b, t = x.shape[:2]
        y = module(x.flatten(0,1))
        return y.reshape(b, t, *y.shape[1:])

    def forward(self, video):
        x = self.spatial_apply(self.spatial_encoder, video)
        if self.use_attention:
            x = self.temporal_encoder(x)
            x = self.temporal_decoder(x)
        else:
            x = self.spatial_apply(self.temporal_encoder.stage4.merge, x)
            x = self.spatial_apply(self.temporal_encoder.stage5.merge, x)
            x = self.spatial_apply(self.temporal_decoder.decode_stage5.expansion, x)
            x = self.spatial_apply(self.temporal_decoder.decode_stage4.expansion, x)
        output = self.spatial_apply(self.spatial_decoder, x)
        return output, output.new_zeros(()), {'common': 0., 'individual': 0.}


def factory(name):
    if name == 'merge_only':
        return TemporalFactor(False)
    if name == 'dense_3d_swin':
        return TemporalFactor(True)
    return original_build(name)


if __name__ == '__main__':
    runner.NAMES = ('spatial_only', 'merge_only', 'dense_3d_swin', 'spatiotemporal_only')
    runner.variants.build_model = factory
    runner.main()
