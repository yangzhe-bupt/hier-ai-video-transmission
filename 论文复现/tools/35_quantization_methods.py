"""有文献依据的3bit传输特征量化候选，不量化网络权重。
参考与改动见量化方法依据_20261123.txt；默认只检查，--train才训练。
"""
import importlib
import math
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'step')]
runner = importlib.import_module('31_fixed_gop_ablation')
base_build = runner.variants.build_model
original_evaluate = runner.evaluate


def round_ste(x):
    return x + (x.round() - x).detach()


def grad_scale(x, factor):
    return x.detach() + (x - x.detach()) * factor


def pack_3bit(codes):
    """低位先行；shape/元素数量及量化参数由两端约定，不塞进码值负载。"""
    codes = codes.detach().cpu().flatten().long()
    if codes.numel() and (codes.min() < 0 or codes.max() > 7):
        raise ValueError('码值必须在0..7')
    bits = ((codes[:, None] >> torch.arange(3)) & 1).flatten()
    padding = (-bits.numel()) % 8
    bits = torch.nn.functional.pad(bits, (0,padding))
    return (bits.reshape(-1,8) * (1 << torch.arange(8))).sum(1).byte()


def unpack_3bit(payload, count):
    bits = ((payload.cpu().long()[:,None] >> torch.arange(8)) & 1).flatten()[:count*3]
    return (bits.reshape(-1,3) * (1 << torch.arange(3))).sum(1)


class Bounded3Bit(nn.Module):
    """DoReFa式[0,1]激活量化。sigmoid是本项目选择，非作者指定。
    u=sigmoid(x), code=round(7u), decoder_input=code/7。
    noise组在码值坐标加U(-.5,.5)；STE组训练真实舍入、反传用恒等近似。
    """
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, x):
        z = torch.sigmoid(x) * 7
        if not self.training:
            q = z.round()
        elif self.estimator == 'noise':
            q = z + torch.rand_like(z) - .5
        else:
            q = round_ste(z)
        return q / 7

    def codes(self, x):
        return (x.sigmoid() * 7).round().long()

    def decode(self, codes):
        return codes.float() / 7


class LSQSigned3Bit(nn.Module):
    """LSQ式有符号激活适配：signed codes -4..3，含精确零。
    q=round(clamp(x/s,-4,3)); decoder=q*s；bitcode=q+4。
    s每个公共/个体分支各一个，随模型保存并由两端共享，不逐帧传scale。
    首个训练微批按2*mean(abs(x))/sqrt(3)初始化，梯度按1/sqrt(N*3)缩放。
    原文分类网络的优化器/预训练设置并未照搬，不能称原LSQ实验复现。
    """
    def __init__(self):
        super().__init__()
        self.step = nn.Parameter(torch.tensor(2/7))
        self.register_buffer('initialized', torch.tensor(False))

    def positive_step(self):
        return self.step.abs().clamp_min(1e-6)

    def forward(self, x):
        if self.training and not bool(self.initialized):
            with torch.no_grad():
                self.step.copy_((2*x.abs().mean()/math.sqrt(3)).clamp_min(1e-4))
                self.initialized.fill_(True)
        s = self.positive_step()
        if self.training:
            s = grad_scale(s, 1/math.sqrt(x.numel()*3))
        z = (x/s).clamp(-4,3)
        q = round_ste(z) if self.training else z.round()
        return q*s

    def codes(self, x):
        return (x/self.positive_step()).clamp(-4,3).round().long()+4

    def decode(self, codes):
        return (codes.float()-4)*self.positive_step()


def factory(name):
    model = base_build('common_fig7')
    if name == 'common_fig7':
        return model
    kind = {'bounded_noise': lambda: Bounded3Bit('noise'),
            'bounded_ste': lambda: Bounded3Bit('ste'),
            'lsq_signed': LSQSigned3Bit}[name]
    model.common_quantizer, model.individual_quantizer = kind(), kind()
    original_forward = model.forward
    def forward(video):
        y, aux, raw = original_forward(video)
        # 原Stage1统计的是raw是否超出[-1,1]，不是新量化器的饱和率。
        return y, aux, {f'legacy_raw_range_{k}': v for k,v in raw.items()}
    model.forward = forward
    return model


def evaluate_with_codes(model, videos, device, batch):
    stats = {}
    handles = []
    for name, q in [('common',model.common_quantizer),('individual',model.individual_quantizer)]:
        stats[name] = {'counts': torch.zeros(8,dtype=torch.long), 'elements': 0}
        def capture(module, inputs, name=name):
            x = inputs[0]
            if hasattr(module,'codes'):
                codes = module.codes(x)
            else:
                codes = ((x.clamp(module.min_value,module.max_value)-module.min_value)/module.step_size).round().long()
            stats[name]['counts'] += torch.bincount(codes.flatten(),minlength=8).cpu()
            stats[name]['elements'] += codes.numel()
        handles.append(q.register_forward_pre_hook(capture))
    try:
        result = original_evaluate(model,videos,device,batch)
    finally:
        for handle in handles:
            handle.remove()
    result['quantizers'] = {}
    for name, q in [('common',model.common_quantizer),('individual',model.individual_quantizer)]:
        result['quantizers'][name] = {'code_fraction': (stats[name]['counts']/stats[name]['elements']).tolist()}
        if isinstance(q,LSQSigned3Bit):
            result['quantizers'][name].update(step=q.positive_step().item(),
                range=[-4*q.positive_step().item(),3*q.positive_step().item()])
    return result


def self_check():
    torch.manual_seed(42)
    for q in (Bounded3Bit('noise'), Bounded3Bit('ste'), LSQSigned3Bit()):
        x = torch.tensor([-10.,-2.,-.3,0.,.3,2.,10.],requires_grad=True)
        q.train()
        y = q(x)
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        if isinstance(q, LSQSigned3Bit):
            assert q.step.grad is not None and torch.isfinite(q.step.grad)
        q.eval()
        expected = q(x)
        codes = q.codes(x)
        payload = pack_3bit(codes)
        decoded_codes = unpack_3bit(payload, x.numel()).to(x.device)
        assert torch.equal(codes, decoded_codes)
        assert torch.allclose(q.decode(decoded_codes), expected, atol=1e-6)
        assert len(payload) == math.ceil(x.numel()*3/8)
        if isinstance(q, LSQSigned3Bit):
            assert q(torch.zeros(1)).item() == 0
    # 原始训练模型主干必须完全相同；允许LSQ多出两份步长与初始化标记。
    reference = None
    for name in ('common_fig7','bounded_noise','bounded_ste','lsq_signed'):
        torch.manual_seed(42)
        model = factory(name)
        state = {k:v.detach().clone() for k,v in model.state_dict().items()
                 if not k.startswith(('common_quantizer.','individual_quantizer.'))}
        if reference is None:
            reference = state
        assert state.keys() == reference.keys() and all(torch.equal(v,reference[k]) for k,v in state.items())
        del model
    print('PASS: finite gradients, LSQ step gradient, 3bit pack/decode, exact zero (LSQ), identical backbone initialization', flush=True)


if __name__ == '__main__':
    self_check()
    runner.NAMES = ('common_fig7','bounded_noise','bounded_ste','lsq_signed')
    runner.OUTPUT_FOLDER = 'step35_quantization_methods'
    runner.EXTRA_CONFIG = dict(study='literature_informed_quantizer_adaptations', bits=3,
        models={'common_fig7':'old clamp [-1,1], noise training, 8-level eval',
                'bounded_noise':'sigmoid to [0,1]; code 0..7; code noise; decode /7',
                'bounded_ste':'same bounded mapping; actual rounding forward, STE backward',
                'lsq_signed':'signed -4..3, learned step per branch, STE and LSQ gradient scaling'},
        learned_scales='two FP32 parameters shared offline through matching sender/receiver model versions',
        bitrate='3 bits/element plus tail padding; packet headers and model distribution costs not benchmarked',
        calibration='LSQ first training microbatch only; no validation calibration',
        references=['https://arxiv.org/abs/1611.01704','https://arxiv.org/abs/1606.06160','https://arxiv.org/abs/1902.08153'],
        note='fixed training subset adaptation study, not paper replication or held-out best; may not converge at 2000 updates')
    runner.variants.build_model = factory
    runner.evaluate = evaluate_with_codes
    runner.main()
