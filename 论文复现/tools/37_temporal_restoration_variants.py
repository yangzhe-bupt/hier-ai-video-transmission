"""文献启发的时空重建结构对照；保持时空瓶颈元素数相同。
仅用于固定8GOP诊断，尚未接入原双分支量化通信模型。
"""
import importlib
from pathlib import Path
import sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'step')]
study=importlib.import_module('33_temporal_factor_ablation')
runner=study.runner
base=study.base


class ResidualGroup(nn.Module):
    """借鉴SwinIR的块组残差与卷积；3D扩展和零初始化是本项目选择。
    仅在各编码/解码阶段内部连接，不跨通信通道绕过传输。
    """
    def __init__(self,blocks,channels):
        super().__init__()
        self.blocks=blocks
        self.conv=nn.Conv2d(channels,channels,3,padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self,x):
        y=self.blocks(x)
        b,t=y.shape[:2]
        correction=self.conv(y.flatten(0,1)).reshape_as(y)
        return x+correction


class HighResolutionBottleneck(nn.Module):
    """少一次降采样、改为窄通道：120*8*8 == 30*16*16。
    这是待验证的容量分配假设，不声称来自原论文或保证高频无损。
    时间维均4，不额外传输任何高分辨率跳连。
    """
    def __init__(self):
        super().__init__()
        self.spatial_encoder=base.step06.SpatialEncoder()
        self.spatial_decoder=base.step06.SpatialDecoder()
        self.stage4=base.step07.SpatiotemporalStage(72,96)
        self.channel_encode=nn.Conv2d(96,30,1)
        self.narrow_encode=self.pair(30)
        self.narrow_decode=self.pair(30)
        self.channel_decode=nn.Conv2d(30,96,1)
        self.decode4=base.step07.SpatiotemporalDecoderStage(96,72)

    @staticmethod
    def pair(channels):
        return nn.Sequential(base.step07.SwinBlock3D(dim=channels,num_heads=3,shift_size=(0,0,0)),
                             base.step07.SwinBlock3D(dim=channels,num_heads=3,shift_size=(0,2,2)))

    def encode(self,video):
        apply=study.TemporalFactor.spatial_apply
        x=apply(self.spatial_encoder,video)
        x=self.stage4(x)
        return self.narrow_encode(apply(self.channel_encode,x))

    def forward(self,video):
        apply=study.TemporalFactor.spatial_apply
        latent=self.encode(video)
        x=self.narrow_decode(latent)
        x=apply(self.channel_decode,x)
        x=self.decode4(x)
        out=apply(self.spatial_decoder,x)
        return out,out.new_zeros(()),{}


def factory(name):
    if name=='higher_resolution_same_elements':
        return HighResolutionBottleneck()
    if name in ('dense_3d_swin','dense_residual_groups'):
        model=study.TemporalFactor(True)
        if name=='dense_residual_groups':
            for stage,channels in [(model.temporal_encoder.stage4,96),
                                   (model.temporal_encoder.stage5,120),
                                   (model.temporal_decoder.decode_stage5,120),
                                   (model.temporal_decoder.decode_stage4,96)]:
                stage.blocks=ResidualGroup(stage.blocks,channels)
        return model
    raise ValueError(name)


if __name__=='__main__':
    runner.NAMES=('dense_3d_swin','dense_residual_groups','higher_resolution_same_elements')
    runner.OUTPUT_FOLDER='step37_temporal_restoration'
    runner.EXTRA_CONFIG=dict(study='temporal_restoration_hypotheses',
        reference_bottleneck=[4,120,8,8],alternative_bottleneck=[4,30,16,16],
        bottleneck_elements=30720,quantization=False,moe=False,
        zero_initialized_residual_convolution='implementation choice, not SwinIR exact replica',
        references=['https://github.com/SwinTransformer/Video-Swin-Transformer',
                    'https://arxiv.org/abs/2108.10257'],
        note='equal latent elements only; parameter count and compute differ; not original full dual-branch CBR or true bitrate')
    runner.variants.build_model=factory
    runner.main()
