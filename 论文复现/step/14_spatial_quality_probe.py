"""空间编解码单GOP拟合诊断：运行会训练，默认1000次更新。

固定239–242帧，沿用完整画面预处理，只有逐帧空间编码/解码和MSE。
不含时空压缩、双分支、量化或通道，预算与完整模型不同。
下半有效区域为固定细节观察区，不是自动人体检测。
"""
import argparse
import csv
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch import nn

base=importlib.import_module('12_train_stage1')
utils=importlib.import_module('12_train_single_gop')
ROOT=Path(__file__).resolve().parent.parent
plt=base.plt


class SpatialOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=base.step06.SpatialEncoder()
        self.decoder=base.step06.SpatialDecoder()

    def forward(self,video):
        b,t,c,h,w=video.shape
        # 合并B和T仅为了逐帧计算，不进行时间混合。
        x=video.reshape(b*t,c,h,w)
        return self.decoder(self.encoder(x)).reshape(b,t,c,h,w)


@torch.no_grad()
def evaluate(model,video,mask,roi):
    model.eval()
    prediction=model(video)
    mse=base.masked_mse(prediction,video,mask).item()
    roi_mse=base.masked_mse(prediction,video,roi).item()
    if not math.isfinite(mse): raise RuntimeError('非有限评估MSE')
    return prediction.cpu(),dict(mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),
                               detail_mse=roi_mse,detail_psnr_db=-10*math.log10(max(roi_mse,1e-12)))


def save_preview(path,video,before,after,region,start,step,label,initial_step=0):
    coords=region[0,0,0].cpu().nonzero()
    top,left=coords.min(0).values.tolist();bottom,right=(coords.max(0).values+1).tolist()
    fig,axes=plt.subplots(3,4,figsize=(11,8))
    for row,(name,tensor) in enumerate((('Original',video.cpu()),(f'Start step {initial_step}',before),(f'Step {step}',after))):
        for t in range(4):
            axes[row,t].imshow(tensor[0,t,:,top:bottom,left:right].permute(1,2,0).numpy().clip(0,1))
            axes[row,t].set_title(f'{name} / frame {start+t}');axes[row,t].axis('off')
    fig.suptitle(f'Spatial-only diagnostic / {label} / same training GOP')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,default=ROOT/'data/videos/new_year.mp4')
    p.add_argument('--start-frame',type=int,default=239)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--steps',type=int,default=1000,help='累计目标更新次数')
    p.add_argument('--resume',type=Path,help='自己的空间诊断last.pt；恢复模型、Adam和随机状态')
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/step14_spatial_probe')
    args=p.parse_args()
    if args.steps<1 or args.eval_every<1 or not math.isfinite(args.lr) or args.lr<=0: p.error('步数、间隔、学习率必须为正')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    frames=base.step05.read_frames(args.video,args.start_frame,4)
    video,mask,info=base.step05.frames_to_full_gop(frames,256)
    # 训练前固定细节区域，避免看完结果再选择效果好的位置。
    roi=mask.clone();positions=mask[0,0,0].nonzero()
    top,left=positions.min(0).values.tolist();bottom,right=(positions.max(0).values+1).tolist()
    roi[...,:top+(bottom-top)//2,:]=False
    video,mask,roi=[x.to(args.device) for x in (video,mask,roi)]
    model=SpatialOnly().to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    start_step=0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        old=saved['config']
        for key,current in (('video_sha256',utils.sha256(args.video)),('start_frame',args.start_frame),('lr',args.lr),('seed',args.seed)):
            if old[key]!=current: p.error(f'续训配置不同：{key}')
        start_step=int(saved['step'])
        if args.steps<=start_step: p.error('--steps必须大于检查点累计步数')
        saved_input=torch.load(args.resume.parent/'input_gop.pt',map_location='cpu',weights_only=False)
        for key,current in (('video',video),('valid_mask',mask),('detail_mask',roi)):
            if not torch.equal(saved_input[key],current.cpu()): p.error(f'输入或区域与原实验不同：{key}')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if 'python_rng' in saved: random.setstate(saved['python_rng'])
        if 'numpy_rng' in saved: np.random.set_state(saved['numpy_rng'])
        # 老版未保存Python/NumPy随机状态；本训练循环不使用它们。
        if args.device=='cuda' and saved['cuda_rng'] is not None:
            if len(saved['cuda_rng'])!=torch.cuda.device_count(): p.error('可见GPU数量与检查点不同')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        saved_metrics=saved['metrics']
        del saved,saved_input
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(video_sha256=utils.sha256(args.video),preprocessing=info,
        start_step=start_step,parent_checkpoint_sha256=utils.sha256(args.resume) if args.resume else None,
        parameters=sum(p.numel() for p in model.parameters()),loss='valid-area MSE only',
        detail_region='lower half of valid image, fixed before training',
        detail_tlbr=[top+(bottom-top)//2,left,bottom,right],optimizer='Adam',grad_clip=1.,
        latent_shape=[1,4,72,32,32],note='spatial-only, no quantization; not equal bitrate comparison')
    utils.write_json(out/'config.json',config)
    if args.resume:
        parent=out/'parent_records';parent.mkdir()
        for name in ('config.json','result.json','metrics.csv','summary.txt'):
            source=args.resume.parent/name
            if source.exists(): shutil.copy2(source,parent/name)
    utils.write_json(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None))
    torch.save(dict(video=video.cpu(),valid_mask=mask.cpu(),detail_mask=roi.cpu()),out/'input_gop.pt')
    for folder in ('step','practice','Biran','packet'):
        for source in (ROOT/folder).rglob('*.py'):
            target=out/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (out/'architecture.txt').write_text(str(model),encoding='utf-8')
    # 若本地从历史快照导入预处理，归档实际执行的版本而非同名旧文件。
    shutil.copy2(Path(base.step05.__file__),out/'source/step/05_video_dataset.py')
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();before,initial=evaluate(model,video,mask,roi)
    if args.resume and abs(initial['mse']-saved_metrics['mse'])>1e-6:
        raise RuntimeError('恢复后起点MSE未复现，停止更新')
    best=initial['mse'];best_step=start_step
    def checkpoint(name,step,metrics):
        temporary=out/(name+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            config=config,metrics=metrics,torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None),temporary)
        temporary.replace(out/name)
    checkpoint('best.pt',start_step,initial)
    print(f'实验目录：{out}；累计更新{start_step} → {args.steps}；参数={config["parameters"]}',flush=True)
    print(f'起点MSE={initial["mse"]:.8f}，PSNR={initial["psnr_db"]:.3f} dB',flush=True)
    rows=[dict(step=start_step,**initial)]
    fields=['step','train_mse_before_update','grad_norm','mse','psnr_db','detail_mse','detail_psnr_db']
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerow(rows[0]);stream.flush()
        for step in range(start_step+1,args.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            prediction=model(video)
            loss=base.masked_mse(prediction,video,mask)
            if not torch.isfinite(loss): raise RuntimeError('非有限损失')
            loss.backward()
            if step==start_step+1:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            row=dict(step=step,train_mse_before_update=loss.item(),grad_norm=norm.item())
            del prediction,loss
            if step==start_step+1 or step%args.eval_every==0 or step==args.steps:
                after,metrics=evaluate(model,video,mask,roi);row.update(metrics)
                if metrics['mse']<best: best,best_step=metrics['mse'],step;checkpoint('best.pt',step,metrics)
                checkpoint('last.pt',step,metrics)
                save_preview(out/f'full_{step:04d}.png',video,before,after,mask,args.start_frame,step,'valid area',start_step)
                save_preview(out/f'detail_{step:04d}.png',video,before,after,roi,args.start_frame,step,'fixed lower-half detail crop',start_step)
                print(f'更新{step}次 | MSE={metrics["mse"]:.6f} PSNR={metrics["psnr_db"]:.3f} dB | 下半区域PSNR={metrics["detail_psnr_db"]:.3f} dB',flush=True)
            writer.writerow(row);stream.flush();rows.append(row)
    selected=[r for r in rows if 'mse' in r]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for key,label in (('psnr_db','Valid image'),('detail_psnr_db','Fixed lower half')):
        axes[1].plot([r['step'] for r in selected],[r[key] for r in selected],'o-',label=label)
    axes[0].plot([r['step'] for r in selected],[r['mse'] for r in selected],'o-')
    axes[0].set_ylabel('Valid MSE');axes[1].set_ylabel('PSNR (dB)');axes[1].legend()
    for ax in axes: ax.set_xlabel('Updates');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    report=dict(status='completed',optimizer_updates=args.steps,initial=initial,final=metrics,
        start_step=start_step,updates_this_run=args.steps-start_step,
        best_mse=best,best_step=best_step,elapsed_seconds=time.perf_counter()-started,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else None)
    utils.write_json(out/'result.json',report)
    (out/'summary.txt').write_text('空间编解码器固定GOP拟合诊断。仅MSE训练，单样本非泛化结论。\n'
        '有效画面及预先固定的下半区域分别记录；下半区域不是人体检测结果。\n'
        '无量化、时空编码、分支分解或通道；特征预算明显大于完整模型，不是同码率性能对照。\n'
        '完整模型旧单GOP实验约28.458 dB，只用于定位信息损失路径，不能将差值全部归因于单一模块。\n'
        f'本次累计更新{start_step}到{args.steps}，新增{args.steps-start_step}次；曲线和best范围为本次运行含起点，旧记录在parent_records。\n'
        '使用--resume恢复Adam及随机状态，--steps为累计目标。续训图中间行是本次恢复起点。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));print(f'记录保存完成：{out}')


if __name__=='__main__': main()
