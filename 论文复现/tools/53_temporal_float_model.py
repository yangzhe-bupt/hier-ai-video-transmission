"""不分支时空浮点模型与训练可视化辅助；原模型代码保持不变。"""
import importlib
from pathlib import Path
import sys
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'step')]
codec = importlib.import_module('40_residual_full_codec')
NAME = 'residual_temporal_moe_float'


class TemporalFloat(nn.Module):
    def __init__(self):
        super().__init__()
        # 先按完整浮点组的同一初始化顺序创建，再仅保留四个主模块。
        full = codec.factory('residual_branches_moe_float')
        for name in ('spatial_encoder','temporal_encoder','temporal_decoder','spatial_decoder'):
            setattr(self,name,getattr(full,name))

    def forward(self,video):
        if video.ndim != 5 or video.shape[1:3] != (4,3):
            raise ValueError('输入须为[B,4,3,H,W]')
        b,t,c,h,w = video.shape
        x = self.spatial_encoder(video.flatten(0,1))
        x = x.reshape(b,t,*x.shape[1:])
        x,a = self.temporal_encoder(x,return_aux=True)
        x,d = self.temporal_decoder(x,return_aux=True)
        y = self.spatial_decoder(x.flatten(0,1)).reshape(b,t,c,h,w)
        return y,(a+d)/2,{}


def build(name):
    if name != NAME: raise ValueError(name)
    return TemporalFloat()


def plot_history(out,history):
    fig,axes = plt.subplots(2,2,figsize=(10,7))
    val = [r for r in history if 'val_rounded' in r]
    axes[0,0].plot([r['epoch'] for r in history],[r['train_noise_mse'] for r in history])
    axes[0,0].set(title='Train reconstruction MSE',yscale='log')
    axes[0,1].plot([r['epoch'] for r in val],[r['val_rounded']['psnr_db'] for r in val])
    axes[0,1].set(title='Validation PSNR (dB)')
    axes[1,0].plot([r['epoch'] for r in val],[r['val_rounded']['mse'] for r in val])
    axes[1,0].set(title='Validation MSE',yscale='log')
    axes[1,1].plot([r['epoch'] for r in history],[r['train_aux'] for r in history])
    axes[1,1].set(title='MoE auxiliary')
    for ax in axes.flat: ax.set_xlabel('Epoch'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'curves.png',dpi=150); plt.close(fig)


@torch.no_grad()
def preview(model,dataset,device,out,epoch):
    model.eval()
    sample = dataset[0]
    x = sample['video'].unsqueeze(0).to(device)
    y = model(x)[0]
    if y.shape != x.shape or not torch.isfinite(y).all(): raise RuntimeError('重建图数值异常')
    error = (y-x).abs().mean(2)[0].cpu()
    fig,axes = plt.subplots(3,4,figsize=(12,9))
    for t in range(4):
        for i,frames in enumerate((x[0].cpu(),y[0].cpu())):
            axes[i,t].imshow(frames[t].permute(1,2,0).clamp(0,1).numpy())
            axes[i,t].set_title(f'{"Original" if i==0 else "Reconstruction"} / {int(sample["start"])+t}')
            axes[i,t].axis('off')
        axes[2,t].imshow(error[t].numpy(),cmap='magma',vmin=0,vmax=.15)
        axes[2,t].set_title('Mean absolute RGB error (0..0.15)');axes[2,t].axis('off')
    directory = out/'reconstructions';directory.mkdir(exist_ok=True)
    fig.suptitle(f'{sample["sequence"]} / epoch {epoch} / same fixed center GOP')
    fig.tight_layout();fig.savefig(directory/f'epoch_{epoch:04d}.png',dpi=120);plt.close(fig)


if __name__ == '__main__':
    torch.set_num_threads(2);torch.manual_seed(42)
    m = build(NAME)
    x = torch.rand(1,4,3,256,256)
    m.train();y,a,_ = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all() and torch.isfinite(a)
    ((y-x).square().mean()+.01*a).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert not hasattr(m,'common_extractor') and not hasattr(m,'common_quantizer')
    print(f'CPU forward/backward passed; parameters={sum(p.numel() for p in m.parameters())}; optimizer updates=0')
