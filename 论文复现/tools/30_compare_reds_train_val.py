"""同口径评估训练/验证集：固定中心裁剪+eval舍入，无参数更新。
保存到tools目录，不改动训练源码或检查点。
"""
import argparse
from collections import defaultdict
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'step'))
sys.path.insert(0, str(ROOT / 'tools'))
data = importlib.import_module('27_reds_dataset')
diag = importlib.import_module('29_diagnose_reds')


class CenterDataset(data.REDSZipDataset):
    def __getitem__(self, index):
        # 仅改变本次对象的裁剪分支；ZIP路径/序列ID/官方划分不变。
        # 单进程加载，finally恢复split，既有数据与训练代码均不修改。
        original = self.split
        try:
            self.split = 'center_evaluation'
            return super().__getitem__(index)
        finally:
            self.split = original


def write_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('D:/Datasets/REDS'))
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size必须为正')
    torch.set_num_threads(2)
    data.cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original_checkpoint_hash = diag.sha(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    for rel, expected in saved['source_sha256'].items():
        if diag.sha(ROOT / rel) != expected:
            raise RuntimeError(f'训练源代码发生变化: {rel}')
    model = diag.variants.build_model(saved['config']['model_variant']).to(args.device).eval()
    model.load_state_dict(saved['model'], strict=True)
    del saved['model']
    model.requires_grad_(False)
    before = diag.model_sha(model)
    out = ROOT / 'outputs/step30_train_val_comparison' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    print(f'诊断目录：{out}；无训练、无参数更新', flush=True)
    results, all_rows, sequence_rows = {}, [], []
    started = time.perf_counter()
    for split in ('train', 'val'):
        ds = CenterDataset(args.data_root / f'{split}_sharp.zip', split,
                           saved['config'][f'{split}_sequences'])
        if ds.description() != saved['data'][split]:
            raise RuntimeError(f'{split}原数据描述不匹配')
        a, b = ds[0], ds[0]
        if not torch.equal(a['video'], b['video']):
            raise RuntimeError('中心裁剪不可重复')
        del a, b
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, shuffle=False)
        rows = []
        clipping = defaultdict(lambda: [0, 0])
        def capture(name):
            def hook(q, inputs):
                x = inputs[0]
                clipping[name][0] += int(((x < q.min_value) | (x > q.max_value)).sum())
                clipping[name][1] += x.numel()
            return hook
        handles = [model.common_quantizer.register_forward_pre_hook(capture('common')),
                   model.individual_quantizer.register_forward_pre_hook(capture('individual'))]
        offset = 0
        try:
            for batch in loader:
                x = batch['video'].to(args.device)
                pred = model(x)[0]
                errors = (pred - x).square().double().flatten(1).mean(1).cpu().tolist()
                if not bool(torch.isfinite(pred).all()):
                    raise RuntimeError('重建包含非有限值')
                for k, mse in enumerate(errors):
                    rows.append(dict(split=split, dataset_index=offset + k,
                                     sequence=batch['sequence'][k], start=int(batch['start'][k]),
                                     mse=mse, psnr_db=diag.psnr(mse)))
                offset += len(x)
                if offset % 100 == 0 or offset == len(ds):
                    print(f'{split}: {offset}/{len(ds)} GOP', flush=True)
        finally:
            for h in handles:
                h.remove()
        mse = sum(r['mse'] for r in rows) / len(rows)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['sequence']].append(row['mse'])
        for sequence, errors in sorted(grouped.items()):
            sm = sum(errors) / len(errors)
            sequence_rows.append(dict(split=split, sequence=sequence, gops=len(errors),
                                      mse=sm, psnr_db=diag.psnr(sm)))
        results[split] = dict(mse=mse, psnr_db=diag.psnr(mse), gops=len(rows),
                              sequences=len(grouped), passed25=sum(r['psnr_db'] >= 25 for r in rows),
                              clip_fraction={k: n / total for k, (n, total) in clipping.items()})
        print(f'{split}: PSNR={results[split]["psnr_db"]:.3f} dB', flush=True)
        ranked = sorted(rows, key=lambda r: r['psnr_db'])
        for label, row in [('worst', ranked[0]), ('median', ranked[len(ranked) // 2]), ('best', ranked[-1])]:
            x = ds[row['dataset_index']]['video'].unsqueeze(0).to(args.device)
            y = model(x)[0].cpu()[0]
            x = x.cpu()[0]
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for j, images in enumerate((x, y)):
                for t in range(4):
                    axes[j, t].imshow(images[t].permute(1, 2, 0).clamp(0, 1).numpy())
                    axes[j, t].set_title(f'{"Original" if j == 0 else "Rounded"} / {row["start"] + t}')
                    axes[j, t].axis('off')
            fig.suptitle(f'{split} {label}: {row["sequence"]} / {row["psnr_db"]:.3f} dB')
            fig.tight_layout()
            fig.savefig(out / f'{split}_{label}.png', dpi=120)
            plt.close(fig)
        all_rows.extend(rows)
        ds.close()
        write_json(out / 'progress.json', dict(completed_splits=results, optimizer_updates=0))
    diag.write_csv(out / 'per_gop.csv', all_rows)
    diag.write_csv(out / 'per_sequence.csv', sequence_rows)
    history_row = next(r for r in reversed(saved['history']) if r['epoch'] == saved['epoch'])
    delta = results['val']['psnr_db'] - history_row['val_rounded']['psnr_db']
    result = dict(status='completed', checkpoint_epoch=saved['epoch'], optimizer_updates=0,
                  metrics=results, train_minus_val_db=results['train']['psnr_db'] - results['val']['psnr_db'],
                  val_reproduction_delta_db=delta, crop='fixed_center_256_RGB',
                  quantization='eval_clamp_and_round', packet_loss=0, enhancement=False,
                  weights_unchanged=before == diag.model_sha(model),
                  checkpoint_unchanged=original_checkpoint_hash == diag.sha(args.checkpoint),
                  elapsed_seconds=time.perf_counter() - started,
                  interpretation='train/val source content differs; gap alone cannot uniquely identify underfitting or overfitting')
    write_json(out / 'result.json', result)
    if abs(delta) > .02 or not result['weights_unchanged'] or not result['checkpoint_unchanged']:
        raise RuntimeError('复现/只读检查失败，请检查result.json')
    print(f'同口径训练-验证差距：{result["train_minus_val_db"]:+.3f} dB', flush=True)
    print(f'诊断完成：{out}；参数更新0次', flush=True)


if __name__ == '__main__':
    main()
