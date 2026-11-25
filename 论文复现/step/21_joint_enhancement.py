"""接收端解码器与残差增强联合微调，不改变发送特征或3-bit量化。

复用源实验的uint8 GOP缓存，不再复制视频/缓存，也不重复保存基础权重。
保存增强器、三个接收端模块和Adam；原量化best.pt和缓存必须保留。
与Step20纯增强对照分开保存；默认MSE+0.05边缘MSE，非论文原配置。
所有样本来自同一训练视频，本实验不构成泛化验证。
"""
import argparse
import csv
import copy
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import zipfile

import cv2
import numpy as np
import torch
from torch import nn

variants=importlib.import_module('12_model_variants')
base=variants.base
utils=importlib.import_module('12_train_single_gop')
ROOT=Path(__file__).resolve().parent.parent
DECODER_PREFIXES=('individual_upsample.','temporal_decoder.','spatial_decoder.')


def restore_decoder(model,state):
    """只允许完整恢复指定接收端模块，禁止覆盖发送端或漏载权重。"""
    expected={k for k in model.state_dict() if k.startswith(DECODER_PREFIXES)}
    if set(state)!=expected:raise ValueError('解码器检查点名称缺失或越界')
    merged=model.state_dict();merged.update(state)
    model.load_state_dict(merged,strict=True)


class ResidualEnhancer(nn.Module):
    """逐帧同尺寸增强；末层零初始化，使初始输出等于原重建。

    输入只有接收端解码图像。原始视频只作为训练目标，不送入增强器。
    首版不做跨帧融合，因此不声称能解决时间闪烁。
    """
    def __init__(self,width=24,blocks=4):
        super().__init__()
        self.head=nn.Conv2d(3,width,3,padding=1)
        self.blocks=nn.ModuleList(nn.Sequential(nn.Conv2d(width,width,3,padding=1),
            nn.GELU(),nn.Conv2d(width,width,3,padding=1)) for _ in range(blocks))
        self.tail=nn.Conv2d(width,3,3,padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self,x):
        b,t,c,h,w=x.shape
        flat=x.reshape(b*t,c,h,w)
        z=self.head(flat)
        for block in self.blocks:z=z+0.1*block(z)
        return (flat+self.tail(z)).clamp(0,1).reshape_as(x)


def edge_mse(pred,target,mask):
    """相邻有效像素差的MSE；排除补边边界和尾部补齐帧。"""
    loss=pred.new_zeros(())
    for axis in (-1,-2):
        a=[slice(None)]*5;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        a,b=tuple(a),tuple(b)
        loss=loss+base.masked_mse(pred[a]-pred[b],target[a]-target[b],mask[a]&mask[b])
    return loss/2


def psnr(mse):return -10*math.log10(max(mse,1e-12))


def preview(path,original,baseline,enhanced,mask,start):
    positions=mask[0,0,0].nonzero()
    top,left=positions.min(0).values.tolist();bottom,right=(positions.max(0).values+1).tolist()
    fig,axes=base.plt.subplots(3,4,figsize=(12,8))
    for row,(name,video) in enumerate((('Original',original),('Quantized baseline',baseline),('Enhanced',enhanced))):
        for t in range(4):
            axes[row,t].imshow(video[0,t,:,top:bottom,left:right].detach().cpu().permute(1,2,0).numpy().clip(0,1))
            axes[row,t].set_title(f'{name} / {start+t}');axes[row,t].axis('off')
    fig.suptitle('Same-video fit / fixed quantized sender / joint decoder enhancement')
    fig.tight_layout();fig.savefig(path,dpi=140);base.plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True,help='量化common_fig7基础模型best.pt')
    p.add_argument('--resume',type=Path,help='本脚本的last.pt；epochs为累计增强轮数')
    p.add_argument('--init-enhancer',type=Path,help='Step20最佳增强器；只加载权重，新建联合优化器')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--eval-every',type=int,default=1)
    p.add_argument('--width',type=int,default=24)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--decoder-lr',type=float,default=1e-5)
    p.add_argument('--edge-weight',type=float,default=0.05)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--export-video',action='store_true',help='结束后从本次最佳增强权重导出完整缓存视频')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/step21_joint_enhancement')
    args=p.parse_args()
    if args.epochs<1 or args.eval_every<1 or args.width<1 or args.blocks<1:p.error('轮数、宽度和块数必须为正')
    if not math.isfinite(args.lr) or args.lr<=0 or not math.isfinite(args.edge_weight) or args.edge_weight<0:p.error('损失系数或学习率错误')
    if not math.isfinite(args.decoder_lr) or args.decoder_lr<=0:p.error('解码器学习率错误')
    if bool(args.init_enhancer)==bool(args.resume):p.error('init-enhancer和resume必须且只能指定一个')
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    checkpoint=args.checkpoint.resolve();source_dir=checkpoint.parent
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    source_config=source['config'];starts=source_config['gop_starts']
    if source_config['model_variant']!='common_fig7' or source_config.get('quantization_bits')!=3:
        p.error('基础模型须为已启用3-bit量化的common_fig7')
    total_frames=source_config['decoded_frames']
    if starts!=list(range(0,total_frames,4)):p.error('此脚本要求源实验覆盖整段视频含尾帧')
    frozen=variants.build_model('common_fig7').to(args.device)
    frozen.load_state_dict(source['model'],strict=True)
    frozen.eval().requires_grad_(False)
    initial_state={k:v.cpu().clone() for k,v in frozen.state_dict().items()}
    # 独立的原始量化模型只负责对照；训练不得改变该基线。
    reference=copy.deepcopy(frozen)
    for name,parameter in frozen.named_parameters():
        parameter.requires_grad_(name.startswith(DECODER_PREFIXES))
    decoder_parameters=[q for q in frozen.parameters() if q.requires_grad]
    source_hash=utils.sha256(checkpoint)
    mask=torch.load(source_dir/'valid_mask.pt',map_location=args.device,weights_only=False)
    enhancer=ResidualEnhancer(args.width,args.blocks).to(args.device)
    init_hash=None;starting_metrics=None
    if args.init_enhancer:
        init=torch.load(args.init_enhancer,map_location='cpu',weights_only=False)
        if 'decoder' in init:p.error('初始化源须为Step20纯增强检查点；联合实验用resume恢复')
        for key,value in (('source_checkpoint_sha256',source_hash),('width',args.width),('blocks',args.blocks)):
            if init['config'][key]!=value:p.error(f'增强初始化配置不一致：{key}')
        enhancer.load_state_dict(init['enhancer'],strict=True)
        starting_metrics=init['metrics'];init_hash=utils.sha256(args.init_enhancer)
    optimizer=torch.optim.Adam([{'params':list(enhancer.parameters()),'lr':args.lr},
                                {'params':decoder_parameters,'lr':args.decoder_lr}])
    trainable_parameters=list(enhancer.parameters())+decoder_parameters
    epoch_start=0;updates=0
    if args.resume:
        prior=torch.load(args.resume,map_location='cpu',weights_only=False)
        for key,value in (('source_checkpoint_sha256',source_hash),('width',args.width),('blocks',args.blocks),
                          ('lr',args.lr),('decoder_lr',args.decoder_lr),('edge_weight',args.edge_weight),('seed',args.seed)):
            if prior['config'][key]!=value:p.error(f'续训配置不同：{key}')
        enhancer.load_state_dict(prior['enhancer'],strict=True)
        restore_decoder(frozen,prior['decoder']);optimizer.load_state_dict(prior['optimizer'])
        starting_metrics=prior['metrics']
        epoch_start=prior['epoch'];updates=prior['updates']
        torch.set_rng_state(prior['torch_rng']);random.setstate(prior['python_rng'])
        if args.device=='cuda':
            if prior['cuda_rng'] is None or len(prior['cuda_rng'])!=torch.cuda.device_count():p.error('续训可见GPU配置不同')
            torch.cuda.set_rng_state_all(prior['cuda_rng'])
        if args.epochs<=epoch_start:p.error('epochs须大于已完成轮数')
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(source_checkpoint=str(checkpoint),source_checkpoint_sha256=source_hash,
        source_epoch=source['epoch'],source_video_sha256=source_config['video_sha256'],
        source_config=source_config,enhancer_parameters=sum(q.numel() for q in enhancer.parameters()),
        decoder_parameters=sum(q.numel() for q in decoder_parameters),decoder_modules=list(DECODER_PREFIXES),
        init_enhancer_sha256=init_hash,
        run_start_psnr_db=starting_metrics['psnr_db'],
        purpose='same-video fit; no packet loss; frozen sender, joint decoder and enhancer',
        transmission_unchanged=True,quantization='frozen eval 3-bit rounding',
        cache_directory=str(source_dir/'gops'))
    utils.write_json(out/'config.json',config)
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    def load(start):
        x=torch.from_numpy(np.load(source_dir/'gops'/f'{start:06d}.npy')).float().div_(255).unsqueeze(0).to(args.device)
        valid=mask.expand(1,4,1,*x.shape[-2:]).clone()
        valid[:,min(4,total_frames-start):]=False
        return x,valid
    def assert_frozen():
        if frozen.training or reference.training:raise RuntimeError('量化模型须保持eval舍入')
        for name,q in frozen.named_parameters():
            if q.requires_grad!=name.startswith(DECODER_PREFIXES):raise RuntimeError('训练范围异常')
            if not q.requires_grad and q.grad is not None:raise RuntimeError('冻结发送端出现梯度')
        for k,v in frozen.state_dict().items():
            if not k.startswith(DECODER_PREFIXES) and not torch.equal(v.cpu(),initial_state[k]):
                raise RuntimeError('发送端参数/缓冲发生改变')
        if any(q.grad is not None or q.requires_grad for q in reference.parameters()):raise RuntimeError('原始参考模型出现梯度')
        if any(not torch.equal(v.cpu(),initial_state[k]) for k,v in reference.state_dict().items()):
            raise RuntimeError('原始参考模型改变')
    def save(name,epoch,metrics):
        temp=out/(name+'.tmp')
        torch.save(dict(enhancer=enhancer.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,
            decoder={k:v for k,v in frozen.state_dict().items() if k.startswith(DECODER_PREFIXES)},
            config=config,metrics=metrics,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None),temp)
        temp.replace(out/name)
    baseline_rows={};history=[];best=float('inf');best_epoch=epoch_start
    started=time.perf_counter()
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    print(f'联合实验目录：{out}；基础权重epoch={source["epoch"]}；发送端与3-bit量化冻结',flush=True)
    print(f'增强参数={config["enhancer_parameters"]}；接收端可训练参数={config["decoder_parameters"]}；轮数{epoch_start}→{args.epochs}',flush=True)
    print(f'学习率：增强={args.lr}，解码器={args.decoder_lr}；边缘损失权重={args.edge_weight}',flush=True)
    def evaluate(epoch):
        enhancer.eval();rows=[]
        with torch.no_grad():
            for start in starts:
                x,valid=load(start);before=reference(x)[0];decoded=frozen(x)[0];after=enhancer(decoded)
                baseline=base.masked_mse(before,x,valid).item()
                if start in baseline_rows and abs(baseline-baseline_rows[start])>1e-7:raise RuntimeError('冻结基线指标漂移')
                baseline_rows[start]=baseline
                error=base.masked_mse(after,x,valid).item()
                if not math.isfinite(error):raise RuntimeError('非有限评估MSE')
                rows.append(dict(epoch=epoch,start_frame=start,actual_frames=min(4,total_frames-start),
                    baseline_mse=baseline,baseline_psnr_db=psnr(baseline),mse=error,psnr_db=psnr(error),
                    decoder_mse=base.masked_mse(decoded,x,valid).item(),
                    baseline_edge_mse=edge_mse(before,x,valid).item(),edge_mse=edge_mse(after,x,valid).item()))
                if start in show:preview(out/f'epoch_{epoch:04d}_frame_{start:06d}.png',x,before,after,valid,start)
        pooled=sum(r['mse']*r['actual_frames'] for r in rows)/total_frames
        original=sum(r['baseline_mse']*r['actual_frames'] for r in rows)/total_frames
        if abs(original-source['metrics']['mean_mse'])>1e-6:raise RuntimeError('基础量化权重的指标未复现')
        if epoch==epoch_start and abs(pooled-starting_metrics['mse'])>1e-6:
            raise RuntimeError('初始化/续训的增强起点指标未复现')
        summary=dict(epoch=epoch,updates=updates,mse=pooled,psnr_db=psnr(pooled),baseline_psnr_db=psnr(original),
            delta_psnr_db=psnr(pooled)-psnr(original),worst_gop_psnr_db=min(r['psnr_db'] for r in rows),
            delta_from_run_start_db=psnr(pooled)-starting_metrics['psnr_db'],
            decoder_only_psnr_db=psnr(sum(r['decoder_mse']*r['actual_frames'] for r in rows)/total_frames),
            passed25=sum(r['psnr_db']>=25 for r in rows),passed27=sum(r['psnr_db']>=27 for r in rows),
            passed30=sum(r['psnr_db']>=30 for r in rows),total_gops=len(rows),
            improved_gops=sum(r['mse']<r['baseline_mse'] for r in rows),
            edge_mse=sum(r['edge_mse']*r['actual_frames'] for r in rows)/total_frames,
            elapsed_seconds=time.perf_counter()-started)
        return summary,rows
    # 确定性展示首/中/尾，以及原基线CSV中最差两段。
    show={starts[0],starts[len(starts)//2],starts[-1]}
    if (source_dir/'per_gop.csv').exists():
        with (source_dir/'per_gop.csv').open(encoding='utf-8') as f:
            scored=[r for r in csv.DictReader(f) if int(r['epoch'])==source['epoch']]
        show.update(int(r['start_frame']) for r in sorted(scored,key=lambda r:float(r['eval_psnr_db']))[:2])
    for epoch in range(epoch_start,args.epochs+1):
        if epoch>epoch_start:
            enhancer.train();order=starts.copy();random.shuffle(order)
            for start in order:
                x,valid=load(start)
                optimizer.zero_grad(set_to_none=True)
                # 发送端参数无需梯度；接收端保持可微。eval确保量化是真实舍入。
                before=frozen(x)[0]
                after=enhancer(before);loss=base.masked_mse(after,x,valid)
                if args.edge_weight:loss=loss+args.edge_weight*edge_mse(after,x,valid)
                if not torch.isfinite(loss):raise RuntimeError('非有限训练loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable_parameters,1.,error_if_nonfinite=True)
                optimizer.step();updates+=1
                if updates%100==0:print(f'epoch={epoch}/{args.epochs} updates={updates} loss={loss.item():.6f}',flush=True)
        if epoch!=epoch_start and epoch%args.eval_every and epoch!=args.epochs:continue
        assert_frozen();summary,rows=evaluate(epoch);history.append(summary)
        with (out/'per_gop.csv').open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            if epoch==epoch_start:writer.writeheader()
            writer.writerows(rows)
        if summary['mse']<best:best=summary['mse'];best_epoch=epoch;save('best.pt',epoch,summary)
        save('last.pt',epoch,summary)
        utils.write_json(out/'history.json',history)
        utils.write_json(out/'result.json',dict(status='completed' if epoch==args.epochs else 'running',
            final=summary,best_epoch=best_epoch,best_mse=best,frozen_sender_unchanged=True,
            reference_baseline_unchanged=True,
            source_checkpoint_unchanged=utils.sha256(checkpoint)==source_hash,
            optimizer_updates=updates,enhancer_parameters=config['enhancer_parameters'],
            decoder_parameters=config['decoder_parameters'],
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else None))
        fig,axes=base.plt.subplots(1,2,figsize=(10,4))
        axes[0].plot([r['epoch'] for r in history],[r['psnr_db'] for r in history],'o-',label='Enhanced')
        axes[0].axhline(summary['baseline_psnr_db'],linestyle='--',label='Frozen quantized baseline')
        axes[1].plot([r['epoch'] for r in history],[100*r['passed25']/len(starts) for r in history],'o-',label='GOP >=25 dB (%)')
        for ax in axes:ax.set_xlabel('Enhancement epoch');ax.grid(alpha=.25);ax.legend()
        axes[0].set_ylabel('Pooled PSNR (dB)');axes[1].set_ylabel('GOP pass rate (%)')
        fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);base.plt.close(fig)
        print(f'epoch={epoch} PSNR={summary["psnr_db"]:.3f} 增益={summary["delta_psnr_db"]:+.3f} dB 最差={summary["worst_gop_psnr_db"]:.3f} 达标25={summary["passed25"]}/{len(starts)}',flush=True)
    if args.export_video:
        best_saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        enhancer.load_state_dict(best_saved['enhancer']);enhancer.eval()
        restore_decoder(frozen,best_saved['decoder'])
        height,width=source_config['preprocessing']['resized_hw']
        top,_,left,_=source_config['preprocessing']['padding_tblr']
        path=out/'enhanced_best.mp4';fps=source_config['fps']
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened():raise RuntimeError('视频写入器无法打开')
        count=0;squared=0.
        try:
            with torch.no_grad():
                for start in starts:
                    x,valid=load(start);after=enhancer(frozen(x)[0])
                    actual=min(4,total_frames-start)
                    squared+=base.masked_mse(after,x,valid).item()*actual
                    for frame in after[0,:actual,:,top:top+height,left:left+width]:
                        rgb=frame.permute(1,2,0).cpu().numpy()
                        writer.write(cv2.cvtColor(np.rint(rgb*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2BGR));count+=1
        finally:writer.release()
        cap=cv2.VideoCapture(str(path));readback=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame.shape[:2]!=(height,width):raise RuntimeError('导出尺寸异常')
                readback+=1
        finally:cap.release()
        if count!=total_frames or readback!=count:raise RuntimeError('导出帧数不一致')
        if abs(squared/total_frames-best_saved['metrics']['mse'])>1e-6:raise RuntimeError('最佳增强指标未复现')
        utils.write_json(out/'video_result.json',dict(checkpoint_epoch=best_saved['epoch'],output_frames=count,
            psnr_db=psnr(squared/total_frames),metrics='float before MP4 encoding',video_readback_verified=True,
            duration_seconds=count/fps,output_fps=fps,audio=False))
    assert_frozen()
    if utils.sha256(checkpoint)!=source_hash:raise RuntimeError('源检查点文件改变')
    (out/'summary.txt').write_text('个体上采样、时空解码、空间解码和增强器联合训练；发送端冻结，3-bit舍入与传输特征不变。\n'
        '相对Step20同时改变训练范围和边缘监督；总收益不能单独归因于边缘损失，需后续消融。\n'
        '同视频训练集拟合，不是泛化或论文复现原结构。增强器逐帧处理，需人工检查视频闪烁与细节。\n'
        '指标排除补边/重复尾帧，视频指标在MP4编码前计算；导出不含音频。\n'
        f'基础权重：{checkpoint}\n最佳增强轮数：{best_epoch}\n源缓存和基础权重必须保留以便恢复。\n',encoding='utf-8')
    with zipfile.ZipFile(out/'brief.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('result.json','curves.png','config.json','summary.txt','video_result.json'):
            path=out/name
            if path.exists():z.write(path,path.name)
    print(f'增强训练记录：{out}；小结果包：{out/"brief.zip"}',flush=True)


if __name__=='__main__':main()
