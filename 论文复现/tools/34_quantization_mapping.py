"""量化前有界激活映射对照；不改旧量化器，不加载/覆盖旧检查点。

四组：原浮点、原3bit、tanh浮点、tanh3bit。
tanh浮点用于区分映射自身影响与舍入影响。默认检查，--train才训练。
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
UniformQuantizer = runner.variants.base.step09.UniformQuantizer


class TanhQuantizer(UniformQuantizer):
    """先tanh映射，再执行原尺度的加噪/舍入；不是在硬clamp后才tanh。

u=tanh(x)，z=(u+1)*7/2，评估q=round(z)，输出2*q/7-1。
训练仍等价于u+U(-1/7,1/7)，不再对原始x硬裁剪。
没有新增参数、侧信息或传输元素；解码器学习映射后的特征，不做atanh。
保留对称8级不含零的设定，作为本次固定控制变量，不声称已符合作者实现。
"""
    def __init__(self, quantized=True):
        super().__init__(bits=3, min_value=-1., max_value=1.)
        self.quantized = quantized

    def forward(self, features):
        mapped = torch.tanh(features)
        if not self.quantized:
            return mapped
        # 明确在整数码值坐标内加U(-.5,.5)或round；两阶段共用同一缩放。
        code_position = (mapped + 1) * ((self.num_levels - 1) / 2)
        if self.training:
            codes = code_position + torch.rand_like(code_position) - .5
        else:
            codes = torch.round(code_position)
        return 2 * codes / (self.num_levels - 1) - 1


def factory(name):
    if name in ('tanh_float', 'tanh_3bit'):
        model = original_build('common_fig7')
        model.common_quantizer = TanhQuantizer(quantized=name == 'tanh_3bit')
        model.individual_quantizer = TanhQuantizer(quantized=name == 'tanh_3bit')
        # Stage1旧接口的clip统计仍针对原始特征；在本变体中重命名，防止
        # 将它误读为映射后饱和率。拟合runner只使用输出与aux，不用该统计。
        original_forward = model.forward
        def mapped_forward(video):
            y, aux, raw_stats = original_forward(video)
            stats = {f'raw_before_mapping_outside_range_{k}': v for k,v in raw_stats.items()}
            return y, aux, stats
        model.forward = mapped_forward
        return model
    return original_build(name)


def self_check():
    # 在原始特征已超出范围时，tanh映射仍可给出非零梯度。
    x = torch.tensor([-2., -.5, 0., .5, 2.], requires_grad=True)
    q = TanhQuantizer().train()
    q(x).sum().backward()
    expected = 1 - x.detach().tanh().square()
    assert torch.allclose(x.grad, expected, atol=1e-6)
    assert x.grad[0] > 0 and x.grad[-1] > 0
    hard_x = x.detach().clone().requires_grad_(True)
    UniformQuantizer(3,-1.,1.).train()(hard_x).sum().backward()
    assert hard_x.grad[0] == 0 and hard_x.grad[-1] == 0
    q.eval()
    source = torch.linspace(-10,10,1001)
    actual = q(source)
    expected = torch.round((source.tanh()+1)*3.5) / 3.5 - 1
    assert torch.allclose(actual, expected, atol=2e-7)
    assert actual.unique().numel() == 8 and actual.min() >= -1 and actual.max() <= 1
    assert torch.equal(q(source), q(source))
    # 浮点与量化、激活映射三种变体必须从完全相同的参数开始。
    initial_hash = None
    for name in ('branches_float', 'common_fig7', 'tanh_float', 'tanh_3bit'):
        torch.manual_seed(42)
        model = factory(name)
        h = runner.compare.diag.model_sha(model)
        if initial_hash is None:
            initial_hash = h
        assert h == initial_hash, name
        del model
    print('PASS: mapping gradients, 8-level rounding, deterministic eval, identical initial weights; optimizer_updates=0', flush=True)


if __name__ == '__main__':
    self_check()
    runner.NAMES = ('branches_float', 'common_fig7', 'tanh_float', 'tanh_3bit')
    runner.OUTPUT_FOLDER = 'step34_quantization_mapping'
    runner.EXTRA_CONFIG = {
        'study': 'pre_quantization_activation_mapping', 'bits': 3, 'range': [-1,1],
        'mapping': 'tanh, fixed temperature 1',
        'no_additional_parameters_or_transmitted_elements': True,
        'quantized_train': 'tanh(x) + U(-1/7,1/7)',
        'quantized_eval': '2*round(7*(tanh(x)+1)/2)/7-1',
        'control': 'tanh_float isolates mapping without noise/rounding',
        'note': 'not author-confirmed quantization; fixed training subset; 2000 steps may not converge'}
    runner.variants.build_model = factory
    runner.main()
