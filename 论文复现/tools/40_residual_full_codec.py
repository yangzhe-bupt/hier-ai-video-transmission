"""固定8GOP完整模型逐项恢复。新模型从头训练，不覆盖旧权重。
阶段内残差不跨传输通道；aux按原始Swin块数汇总，不按包装器数计数。
"""
import importlib
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'step')]
study = importlib.import_module('37_temporal_restoration_variants')
runner = study.runner
original_build = runner.variants.build_model
NAMES = ('residual_branches_dense_float', 'residual_branches_moe_float',
         'residual_branches_moe_3bit')


class AuxiliaryResidualGroup(study.ResidualGroup):
    # 原stage逐块迭代以提取aux；本对象作为一个组执行，但len保留原块数。
    def __iter__(self):
        return iter((self,))

    def __len__(self):
        return len(self.blocks)

    def forward(self, x, return_aux=False):
        if not return_aux:
            return super().forward(x)
        y = x
        terms = []
        for block in self.blocks:
            y, aux = block(y, return_aux=True)
            terms.append(aux)
        correction = self.conv(y.flatten(0, 1)).reshape_as(y)
        return x + correction, torch.stack(terms).sum()


def factory(name):
    if name not in NAMES:
        raise ValueError(name)
    # 三组先构造完全相同的完整MoE骨干，量化器无参数。
    model = original_build('common_fig7')
    if name == NAMES[0]:
        # 在全部10个3D Swin块中恢复普通MLP，保留注意力和分支结构。
        for block in model.modules():
            if isinstance(block, study.base.step07.SwinBlock3D):
                dim = block.dim
                block.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(),
                                          nn.Linear(dim*4, dim))
    for stage, channels in ((model.temporal_encoder.stage4, 96),
                            (model.temporal_encoder.stage5, 120),
                            (model.temporal_decoder.decode_stage5, 120),
                            (model.temporal_decoder.decode_stage4, 96)):
        stage.blocks = AuxiliaryResidualGroup(stage.blocks, channels)
    if name != NAMES[2]:
        model.common_quantizer = nn.Identity()
        model.individual_quantizer = nn.Identity()
    return model


def checks():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    group = AuxiliaryResidualGroup(study.HighResolutionBottleneck.pair(96), 96)
    # 非零卷积下验证默认/aux路径一致，避免零初始化掩盖错误。
    nn.init.normal_(group.conv.weight, std=.001)
    x = torch.randn(1, 4, 96, 8, 8)
    a = group(x)
    b, aux = group(x, return_aux=True)
    torch.testing.assert_close(a, b)
    assert len(group) == 2 and len(list(group)) == 1 and aux.item() == 0
    signatures = []
    for name in NAMES:
        torch.manual_seed(42)
        model = factory(name).cuda()
        signatures.append(runner.compare.diag.model_sha(model))
        x = torch.rand(1, 4, 3, 256, 256, device='cuda')
        model.train()
        y, aux, _ = model(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert torch.isfinite(aux) and (aux.item() == 0 if name == NAMES[0] else aux.item() > 0)
        ((y-x).square().mean() + .01*aux).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        model.eval()
        with torch.no_grad():
            a = model(x)[0]
            b = model(x)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        print(f'{name}: 前向/反向/确定性评估通过；aux={aux.item():.6f}', flush=True)
        del model, x, y, aux, grads, a, b
        torch.cuda.empty_cache()
    assert signatures[1] == signatures[2], 'MoE浮点/量化初始参数必须完全一致'
    print('接口检查通过；浮点/量化初始权重一致；optimizer更新0次', flush=True)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        checks()
    else:
        runner.NAMES = NAMES
        runner.OUTPUT_FOLDER = 'step40_residual_full_codec'
        runner.EXTRA_CONFIG = dict(study='residual_full_codec_progressive_ablation',
            initialization='fresh seed42; no weights transferred from fixed-GOP best',
            residual_stages=[96,120,120,96], packet_loss=0,
            quantizer='original 3bit [-1,1]: training clamp+noise, evaluation clamp+round',
            note='fixed8 training fitting only; branch latent size differs from temporal-only; no generalization claim')
        runner.variants.build_model = factory
        runner.main()
