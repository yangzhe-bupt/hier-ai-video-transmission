"""补完Step33中断的最后一组，不重训已完成的三组；显式运行才更新新模型。"""
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'step')]
runner = importlib.import_module('31_fixed_gop_ablation')


def main():
    run = ROOT/'outputs/step31_fixed_gop_ablation/20261123_191701_209886'
    directory = run/'spatiotemporal_only'
    saved = torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    config = saved['config']
    if saved['variant'] != 'spatiotemporal_only' or saved['step'] >= config['steps_per_model']:
        raise RuntimeError('不是待补完的Step33检查点，停止')
    torch.set_num_threads(2)
    runner.compare.data.cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for p in (run/'source/step').glob('*.py'):
        if p.read_bytes() != (ROOT/'step'/p.name).read_bytes():
            raise RuntimeError(f'核心代码变化: {p.name}')
    videos, manifest = runner.fixed_data(Path('D:/Datasets/REDS'),config['seed'])
    if manifest != saved['manifest']:
        raise RuntimeError('固定数据不一致')
    model = runner.variants.build_model(saved['variant']).cuda()
    model.load_state_dict(saved['model'],strict=True)
    opt = torch.optim.Adam(model.parameters(),lr=config['lr'])
    opt.load_state_dict(saved['optimizer'])
    generator = torch.Generator()
    generator.set_state(saved['sample_rng'])
    torch.set_rng_state(saved['rng_cpu'])
    torch.cuda.set_rng_state_all(saved['rng_cuda'])
    history = saved['history']
    prior_elapsed = history[-1]['elapsed_seconds']
    best_row = min(history[1:],key=lambda r:r['mse'])
    best, best_step = best_row['mse'], best_row['step']
    runner.compare.write_json(run/'resume_record.json',dict(start_step=saved['step'],
        target=config['steps_per_model'],started=datetime.now().isoformat(),
        reason='interrupted final group; prior completed groups retained',core_source_and_data_verified=True))
    print(f'从{saved["step"]}步补训至{config["steps_per_model"]}步，仅spatiotemporal_only；其余三组保留',flush=True)
    started = time.perf_counter()
    for step in range(saved['step']+1,config['steps_per_model']+1):
        model.train()
        indices = torch.randperm(8,generator=generator)[:config['batch_size']]
        x = videos[indices].cuda()
        opt.zero_grad(set_to_none=True)
        y,aux,_ = model(x)
        loss = (y-x).square().mean()+config['aux_weight']*aux
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('loss非有限')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('梯度非有限')
        opt.step()
        if step%200 == 0 or step == config['steps_per_model']:
            metrics = runner.evaluate(model,videos,'cuda',config['batch_size'])
            history.append(dict(step=step,elapsed_seconds=prior_elapsed+time.perf_counter()-started,**metrics))
            payload = dict(variant=saved['variant'],step=step,model=model.state_dict(),
                optimizer=opt.state_dict(),manifest=manifest,config=config,history=history,
                sample_rng=generator.get_state(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
            torch.save(payload,directory/'last.pt.tmp')
            (directory/'last.pt.tmp').replace(directory/'last.pt')
            if metrics['mse']<best:
                best,best_step = metrics['mse'],step
                torch.save(payload,directory/'best.pt.tmp')
                (directory/'best.pt.tmp').replace(directory/'best.pt')
                runner.preview(model,videos,'cuda',manifest,directory,step)
            runner.compare.write_json(directory/'history.json',history)
            print(f'spatiotemporal_only step={step} PSNR={metrics["psnr_db"]:.3f}',flush=True)
    result['models']['spatiotemporal_only']=dict(best_step=best_step,best_psnr_db=runner.compare.diag.psnr(best),
        parameters=sum(p.numel() for p in model.parameters()),optimizer_updates=step,
        elapsed_seconds=history[-1]['elapsed_seconds'])
    result['status']='completed'
    result['resumed_from_step']=saved['step']
    runner.compare.write_json(run/'result.json',result)
    fig,ax=plt.subplots(figsize=(8,5))
    for name in config['variants']:
        rows=json.loads((run/name/'history.json').read_text(encoding='utf-8'))
        ax.plot([r['step'] for r in rows],[r['psnr_db'] for r in rows],label=name)
    ax.set(xlabel='Adam updates per model',ylabel='Fixed training GOP PSNR (dB)')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(run/'curves.png',dpi=150); plt.close(fig)
    print(f'已补完：{run}',flush=True)


if __name__=='__main__':
    main()
