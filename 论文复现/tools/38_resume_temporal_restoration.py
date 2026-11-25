"""原普通3D Swin与残差组续训；保留旧实验，新目录写入，不重置Adam/RNG。"""
import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'step')]
study=importlib.import_module('37_temporal_restoration_variants')
runner=study.runner
NAMES=('dense_3d_swin','dense_residual_groups')


def save(path,payload):
    temp=path.with_suffix('.pt.tmp')
    torch.save(payload,temp)
    temp.replace(path)


def plot(out,histories):
    fig,ax=plt.subplots(figsize=(8,5))
    for name,rows in histories.items():
        ax.plot([r['step'] for r in rows],[r['psnr_db'] for r in rows],label=name)
    ax.set(xlabel='Cumulative Adam updates per model',ylabel='Fixed training GOP PSNR (dB)')
    ax.axvline(2000,color='gray',linestyle='--',alpha=.5)
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/'curves.png',dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,default=ROOT/'outputs/step37_temporal_restoration/20261123_221626_731514')
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--train',action='store_true')
    args=p.parse_args()
    if args.steps<=2000: p.error('累计目标steps必须大于2000')
    torch.set_num_threads(2)
    runner.compare.data.cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # 模型核心代码及候选定义必须与原训练快照相同。
    paths=list((args.source_run/'source/step').glob('*.py'))
    paths += [args.source_run/'source/tools'/name for name in
              ('31_fixed_gop_ablation.py','33_temporal_factor_ablation.py','37_temporal_restoration_variants.py')]
    for path in paths:
        relative=path.relative_to(args.source_run/'source')
        if path.read_bytes()!=(ROOT/relative).read_bytes():
            raise RuntimeError(f'代码不匹配: {relative}')
    videos,manifest=runner.fixed_data(Path('D:/Datasets/REDS'),42)
    histories={}
    for name in NAMES:
        checkpoint=torch.load(args.source_run/name/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['variant']!=name or checkpoint['step']!=2000 or checkpoint['manifest']!=manifest:
            raise RuntimeError(f'{name}: 检查点/数据不匹配')
        histories[name]=checkpoint['history']
        if any(checkpoint['config'][k]!=v for k,v in dict(batch_size=4,seed=42,lr=1e-4,aux_weight=.01,gradient_clip=1.).items()):
            raise RuntimeError('原训练配置不一致')
        del checkpoint
    if not args.train:
        print(json.dumps(dict(status='ready_no_training',optimizer_updates=0,
            source=str(args.source_run),target=args.steps,models=NAMES,sample_shape=list(videos.shape)),ensure_ascii=False))
        return
    out=ROOT/'outputs/step38_temporal_resume'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    runner.compare.write_json(out/'manifest.json',manifest)
    runner.compare.write_json(out/'config.json',dict(source_run=str(args.source_run),start_step=2000,
        target_step=args.steps,variants=NAMES,optimizer_and_rng_restored=True,
        original_experiment_unchanged=True,quantization=False,moe=False))
    for folder in ('step','packet','tools'):
        for source in (ROOT/folder).glob('*.py'):
            dest=out/'source'/folder/source.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
    print(f'续训目录：{out}；每组2000→{args.steps}，不改原实验',flush=True)
    summaries={}
    for name in NAMES:
        directory=out/name
        directory.mkdir()
        old=torch.load(args.source_run/name/'last.pt',map_location='cpu',weights_only=False)
        model=study.factory(name).cuda()
        model.load_state_dict(old['model'],strict=True)
        opt=torch.optim.Adam(model.parameters(),lr=old['config']['lr'])
        opt.load_state_dict(old['optimizer'])
        metrics=runner.evaluate(model,videos,'cuda',4)
        if abs(metrics['psnr_db']-old['history'][-1]['psnr_db'])>.002:
            raise RuntimeError('起点PSNR未复现')
        print(f'{name}: 起点PSNR复现={metrics["psnr_db"]:.6f}',flush=True)
        generator=torch.Generator()
        generator.set_state(old['sample_rng'])
        torch.set_rng_state(old['rng_cpu'])
        torch.cuda.set_rng_state_all(old['rng_cuda'])
        history=histories[name]
        prior_elapsed=history[-1]['elapsed_seconds']
        best_row=min(history[1:],key=lambda r:r['mse'])
        best,best_step=best_row['mse'],best_row['step']
        config=dict(old['config'],steps_per_model=args.steps,resume_from=str(args.source_run/name/'last.pt'))
        initial_best=torch.load(args.source_run/name/'best.pt',map_location='cpu',weights_only=False)
        save(directory/'best.pt',initial_best)
        del initial_best,old
        started=time.perf_counter()
        for step in range(2001,args.steps+1):
            model.train()
            x=videos[torch.randperm(8,generator=generator)[:4]].cuda()
            opt.zero_grad(set_to_none=True)
            y,aux,_=model(x)
            loss=(y-x).square().mean()+.01*aux
            if not bool(torch.isfinite(loss)): raise RuntimeError('非有限loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm)): raise RuntimeError('非有限梯度')
            opt.step()
            if step%200==0 or step==args.steps:
                metrics=runner.evaluate(model,videos,'cuda',4)
                history.append(dict(step=step,elapsed_seconds=prior_elapsed+time.perf_counter()-started,**metrics))
                payload=dict(variant=name,step=step,model=model.state_dict(),optimizer=opt.state_dict(),
                    manifest=manifest,config=config,history=history,sample_rng=generator.get_state(),
                    rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
                save(directory/'last.pt',payload)
                if metrics['mse']<best:
                    best,best_step=metrics['mse'],step
                    save(directory/'best.pt',payload)
                    runner.preview(model,videos,'cuda',manifest,directory,step)
                runner.compare.write_json(directory/'history.json',history)
                plot(out,histories)
                summaries[name]=dict(step=step,best_step=best_step,best_psnr_db=runner.compare.diag.psnr(best),
                    current_psnr_db=metrics['psnr_db'],new_optimizer_updates=step-2000,
                    parameters=sum(p.numel() for p in model.parameters()))
                runner.compare.write_json(out/'result.json',dict(status='completed' if name==NAMES[-1] and step==args.steps else 'running',models=summaries))
                print(f'{name} step={step}/{args.steps} PSNR={metrics["psnr_db"]:.3f}',flush=True)
        del payload,model,opt,x,y,aux,loss
        torch.cuda.empty_cache()
    print(f'续训对照完成：{out}',flush=True)


if __name__=='__main__': main()
