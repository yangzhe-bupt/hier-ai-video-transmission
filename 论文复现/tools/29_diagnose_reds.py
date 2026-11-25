"""只读REDS质量诊断；三个量化模式，无训练或检查点写入。
安装到项目tools目录，避免改变Stage I用于续训校验的step源码清单。
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'step'))
data = importlib.import_module('27_reds_dataset')
variants = importlib.import_module('12_model_variants')
MODES = ('rounded', 'clip_only', 'float_identity')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def model_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def psnr(mse):
    return -10 * math.log10(max(mse, 1e-12))


@contextmanager
def quant_mode(model, mode):
    qs = [model.common_quantizer, model.individual_quantizer]
    old = [q.forward for q in qs]
    try:
        for q in qs:
            if mode == 'clip_only':
                q.forward = types.MethodType(lambda self, x: x.clamp(self.min_value, self.max_value), q)
            elif mode == 'float_identity':
                q.forward = types.MethodType(lambda self, x: x, q)
            elif mode != 'rounded':
                raise ValueError(mode)
        yield
    finally:
        for q, forward in zip(qs, old):
            q.forward = forward


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='仅加载自己训练的可信检查点')
    p.add_argument('--data-root', type=Path, default=Path('D:/Datasets/REDS'))
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--all', action='store_true', help='评估全部750个验证GOP；默认每段只抽取第48帧GOP')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size必须为正')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_hash = sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    # 按检查点记录核对既有模型与数据代码，不修改训练源码来绕过续训校验。
    for name, expected in checkpoint['source_sha256'].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f'本地源码与检查点不符: {name}，请使用本次实验source快照')
    ds = data.REDSZipDataset(args.data_root / 'val_sharp.zip', 'val',
                             checkpoint['config']['val_sequences'])
    if ds.description() != checkpoint['data']['val']:
        raise RuntimeError('验证数据描述与检查点不一致')
    indices = list(range(len(ds))) if args.all else [i for i, r in enumerate(ds.rows) if r['start'] == 48]
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, num_workers=0)
    model = variants.build_model(checkpoint['config']['model_variant']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    original_model_hash = model_sha(model)
    del checkpoint['model']
    rows = []
    clipping = defaultdict(lambda: [0, 0])
    def hook(name):
        def capture(module, inputs):
            x = inputs[0]
            clipping[name][0] += int(((x < module.min_value) | (x > module.max_value)).sum())
            clipping[name][1] += x.numel()
        return capture
    offset = 0
    for batch in loader:
        video = batch['video'].to(args.device)
        for mode in MODES:
            handles = []
            if mode == 'rounded':
                handles = [model.common_quantizer.register_forward_pre_hook(hook('common')),
                           model.individual_quantizer.register_forward_pre_hook(hook('individual'))]
            try:
                with quant_mode(model, mode):
                    prediction, _, _ = model(video)
                    errors = (prediction - video).square().double().flatten(1).mean(1).cpu().tolist()
                for k, error in enumerate(errors):
                    rows.append(dict(mode=mode, dataset_index=indices[offset + k],
                                     sequence=batch['sequence'][k], start=int(batch['start'][k]),
                                     mse=error, psnr_db=psnr(error)))
            finally:
                for handle in handles:
                    handle.remove()
        offset += len(video)
        if offset % 100 == 0 or offset == len(indices):
            print(f'已诊断 {offset}/{len(indices)} GOP（每组比较三种模式）', flush=True)
    out = ROOT / 'outputs/step29_reds_diagnosis' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    summary = {}
    for mode in MODES:
        selected = [r for r in rows if r['mode'] == mode]
        mse = sum(r['mse'] for r in selected) / len(selected)
        summary[mode] = dict(mse=mse, psnr_db=psnr(mse), gops=len(selected),
                             passed25=sum(r['psnr_db'] >= 25 for r in selected))
        print(f"{mode}: PSNR={summary[mode]['psnr_db']:.3f} dB", flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['mode'], row['sequence']].append(row['mse'])
    per_sequence = [dict(mode=mode, sequence=seq, gops=len(errors),
                         mse=sum(errors) / len(errors), psnr_db=psnr(sum(errors) / len(errors)))
                    for (mode, seq), errors in sorted(grouped.items())]
    write_csv(out / 'per_gop.csv', rows)
    write_csv(out / 'per_sequence.csv', per_sequence)
    ranked = sorted([r for r in rows if r['mode'] == 'rounded'], key=lambda r: r['psnr_db'])
    # 选出最差、中位、最好GOP再读取和渲染，不缓存整个验证集。
    for label, row in [('worst', ranked[0]), ('median', ranked[len(ranked) // 2]), ('best', ranked[-1])]:
        sample = ds[row['dataset_index']]['video'].unsqueeze(0).to(args.device)
        pictures = [sample[0].cpu()]
        for mode in MODES:
            with quant_mode(model, mode):
                pictures.append(model(sample)[0][0].cpu())
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for r, (title, images) in enumerate(zip(('Original',) + MODES, pictures)):
            for t in range(4):
                axes[r, t].imshow(images[t].permute(1, 2, 0).clamp(0, 1).numpy())
                axes[r, t].set_title(f'{title} / frame {row["start"] + t}')
                axes[r, t].axis('off')
        fig.suptitle(f'{label}: {row["sequence"]} / rounded {row["psnr_db"]:.3f} dB')
        fig.tight_layout()
        fig.savefig(out / f'{label}.png', dpi=120)
        plt.close(fig)
    hist = checkpoint['history']
    fig, ax = plt.subplots(figsize=(8, 4))
    points = [(r['epoch'], r['val_rounded']['psnr_db']) for r in hist if 'val_rounded' in r]
    ax.plot([r[0] for r in points], [r[1] for r in points], marker='.')
    ax.set(xlabel='Epoch', ylabel='Validation rounded PSNR (dB)')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / 'curves.png', dpi=150)
    plt.close(fig)
    recorded = next((r['val_rounded']['psnr_db'] for r in reversed(hist)
                     if r['epoch'] == checkpoint['epoch'] and 'val_rounded' in r), None)
    delta = summary['rounded']['psnr_db'] - recorded if args.all and recorded is not None else None
    result = dict(status='completed', optimizer_updates=0, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_epoch=checkpoint['epoch'], scope='all_validation' if args.all else 'one_GOP_per_video',
                  metrics=summary, clip_fraction={k: a / n for k, (a, n) in clipping.items()},
                  full_validation_delta_from_training_db=delta,
                  weights_unchanged=model_sha(model) == original_model_hash,
                  checkpoint_unchanged=sha(args.checkpoint) == checkpoint_hash,
                  note='float modes are inference perturbations, not separately trained ablations; fixed 256 center crops')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    ds.close()
    if not result['weights_unchanged'] or not result['checkpoint_unchanged']:
        raise RuntimeError('只读检查失败')
    if delta is not None and abs(delta) > .02:
        raise RuntimeError(f'正常量化指标未复现：差异{delta:.6f}dB，保留诊断文件待检查')
    print(f'诊断目录：{out}；参数更新0次', flush=True)


if __name__ == '__main__':
    main()
