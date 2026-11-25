"""固定8 GOP的四阶段拟合诊断。默认只检查数据，--train才训练。
同一步数/批量/样本顺序；不同模型传输维数、参数量和算力不同，非码率公平对比。
"""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'step')]
compare = importlib.import_module('30_compare_reds_train_val')
variants = compare.diag.variants
NAMES = ('spatial_only', 'spatiotemporal_only', 'branches_float', 'common_fig7')
OUTPUT_FOLDER = 'step31_fixed_gop_ablation'
EXTRA_CONFIG = {}


def fixed_data(root, seed):
    ds = compare.CenterDataset(root / 'train_sharp.zip', 'train')
    generator = torch.Generator().manual_seed(seed)
    others = [n for n in ds.sequence_names if n not in ('train/031', 'train/116')]
    names = ['train/031', 'train/116'] + [others[i] for i in torch.randperm(len(others), generator=generator)[:6]]
    manifest, videos = [], []
    for index, sequence in enumerate(names):
        start = (12, 32)[index] if index < 2 else int(torch.randint(25, (), generator=generator)) * 4
        row_index = next(i for i, r in enumerate(ds.rows) if r['sequence'] == sequence and r['start'] == start)
        a, b = ds[row_index], ds[row_index]
        if not torch.equal(a['video'], b['video']):
            raise RuntimeError('固定裁剪不一致')
        videos.append(a['video'])
        manifest.append(dict(sequence=sequence, start=start, crop_top=a['crop_top'], crop_left=a['crop_left'],
                             tensor_sha256=hashlib.sha256(a['video'].numpy().tobytes()).hexdigest()))
    ds.close()
    return torch.stack(videos), manifest


@torch.no_grad()
def evaluate(model, videos, device, batch):
    model.eval()
    errors = []
    for i in range(0, len(videos), batch):
        x = videos[i:i + batch].to(device)
        y = model(x)[0]
        errors.extend((y - x).square().double().flatten(1).mean(1).cpu().tolist())
    if not all(compare.diag.math.isfinite(e) for e in errors):
        raise RuntimeError('评估非有限')
    mse = sum(errors) / len(errors)
    return dict(mse=mse, psnr_db=compare.diag.psnr(mse),
                per_gop_psnr=[compare.diag.psnr(e) for e in errors])


@torch.no_grad()
def preview(model, videos, device, manifest, out, step):
    for index in (0, 1):
        x = videos[index:index + 1].to(device)
        y = model(x)[0].cpu()[0]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for r, images in enumerate((videos[index], y)):
            for t in range(4):
                axes[r, t].imshow(images[t].permute(1, 2, 0).clamp(0, 1).numpy())
                axes[r, t].axis('off')
        fig.suptitle(f'{manifest[index]["sequence"]} / step {step} / original above, reconstruction below')
        fig.tight_layout()
        fig.savefig(out / f'best_gop{index}.png', dpi=120)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('D:/Datasets/REDS'))
    p.add_argument('--train', action='store_true')
    p.add_argument('--steps', type=int, default=2000, help='每个模型Adam更新次数')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.batch_size > 8:
        p.error('参数必须为正，batch不能超过8')
    torch.set_num_threads(2)
    compare.data.cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    videos, manifest = fixed_data(args.data_root, args.seed)
    if not args.train:
        print(json.dumps(dict(status='checked_no_training', optimizer_updates=0,
                              shape=list(videos.shape), manifest=manifest), ensure_ascii=False, indent=2))
        return
    print('开始四模型拟合对照；会更新新模型参数，不修改第60轮模型。', flush=True)
    out = ROOT / 'outputs' / OUTPUT_FOLDER / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    compare.write_json(out / 'manifest.json', manifest)
    config = dict(steps_per_model=args.steps, batch_size=args.batch_size, seed=args.seed,
                  lr=1e-4, gradient_clip=1., aux_weight=.01, variants=NAMES,
                  fixed_center_crop=True, same_sample_order=True,
                  note='training subset fitting, not generalization or rate-matched comparison; 2000 steps may not converge')
    config.update(EXTRA_CONFIG)
    compare.write_json(out / 'config.json', config)
    for source in list((ROOT / 'step').glob('*.py')) + list((ROOT / 'packet').glob('*.py')) + list((ROOT / 'tools').glob('*.py')):
        dest = out / 'source' / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    all_history, summaries = {}, {}
    branch_initial_hash = None
    for name in NAMES:
        torch.manual_seed(args.seed)
        model = variants.build_model(name).to(args.device)
        initial_hash = compare.diag.model_sha(model)
        if name == 'branches_float':
            branch_initial_hash = initial_hash
        if name == 'common_fig7' and branch_initial_hash is not None and initial_hash != branch_initial_hash:
            raise RuntimeError('浮点与量化双分支初始参数不一致')
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        directory = out / name
        directory.mkdir()
        generator = torch.Generator().manual_seed(args.seed + 100)
        initial = evaluate(model, videos, args.device, args.batch_size)
        history = [dict(step=0, **initial)]
        best, best_step = float('inf'), 0
        start = time.perf_counter()
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randperm(8, generator=generator)[:args.batch_size]
            x = videos[indices].to(args.device)
            opt.zero_grad(set_to_none=True)
            y, aux, _ = model(x)
            mse = (y - x).square().mean()
            loss = mse + (0 if name == 'spatial_only' else .01) * aux
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('训练loss非有限')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('训练梯度非有限')
            opt.step()
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, videos, args.device, args.batch_size)
                history.append(dict(step=step, elapsed_seconds=time.perf_counter() - start, **metrics))
                payload = dict(variant=name, step=step, model=model.state_dict(), optimizer=opt.state_dict(),
                               manifest=manifest, config=config, history=history,
                               sample_rng=generator.get_state(), rng_cpu=torch.get_rng_state(),
                               rng_cuda=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [])
                torch.save(payload, directory / 'last.pt.tmp')
                (directory / 'last.pt.tmp').replace(directory / 'last.pt')
                if metrics['mse'] < best:
                    best, best_step = metrics['mse'], step
                    torch.save(payload, directory / 'best.pt.tmp')
                    (directory / 'best.pt.tmp').replace(directory / 'best.pt')
                    preview(model, videos, args.device, manifest, directory, step)
                compare.write_json(directory / 'history.json', history)
                print(f'{name} step={step}/{args.steps} PSNR={metrics["psnr_db"]:.3f} dB', flush=True)
        summaries[name] = dict(best_step=best_step, best_psnr_db=compare.diag.psnr(best),
                               parameters=sum(p.numel() for p in model.parameters()),
                               optimizer_updates=args.steps, elapsed_seconds=time.perf_counter() - start)
        all_history[name] = history
        compare.write_json(out / 'result.json', dict(status='completed' if name == NAMES[-1] else 'running',
                                                    models=summaries, manifest=manifest, config=config))
        fig, ax = plt.subplots(figsize=(8, 5))
        for title, records in all_history.items():
            ax.plot([r['step'] for r in records], [r['psnr_db'] for r in records], label=title)
        ax.set(xlabel='Adam updates per model', ylabel='Fixed training GOP PSNR (dB)')
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'curves.png', dpi=150)
        plt.close(fig)
        del payload, opt, model, x, y, aux, loss, mse
        torch.cuda.empty_cache()
    print(f'对照完成：{out}', flush=True)


if __name__ == '__main__':
    main()
