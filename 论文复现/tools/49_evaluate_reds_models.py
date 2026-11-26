"""统一只读评估空间/完整浮点/3bit/独立postadd模型，默认CPU不干扰训练GPU。"""
import argparse
from collections import defaultdict
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'step')]
models = importlib.import_module('48_common_postadd_variant')
compare = importlib.import_module('30_compare_reds_train_val')
diag = compare.diag


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, action='append', required=True)
    p.add_argument('--split', choices=('train','val','both'), default='val')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--sample-every', type=int, default=1, help='1全量；25每视频抽1个GOP，抽样结果不可冒充全量')
    args = p.parse_args()
    if min(args.batch_size, args.sample_every) < 1:
        p.error('批量与间隔必须为正')
    torch.set_num_threads(2)
    compare.data.cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT/'outputs/step49_unified_evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    summary = []
    print(f'统一评估目录：{out}', flush=True)
    for number, checkpoint in enumerate(args.checkpoint):
        checksum = diag.sha(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for rel, expected in saved['source_sha256'].items():
            path = ROOT/rel
            if not path.is_file() or diag.sha(path) != expected:
                raise RuntimeError(f'训练源码不一致：{rel}')
        name = saved['config']['model_variant']
        if name == 'residual_temporal_moe_float':
            model = importlib.import_module('53_temporal_float_model').build(name)
        else:
            model = models.build_model(name)
        model = model.to(args.device).eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        before = diag.model_sha(model)
        directory = out/f'{number:02d}_{name}'
        directory.mkdir()
        rows, metrics, reproduction = [], {}, None
        data_root = args.data_root or Path(saved['config']['data_root'])
        for split in (('train','val') if args.split == 'both' else (args.split,)):
            ds = compare.CenterDataset(data_root/f'{split}_sharp.zip', split, saved['config'][f'{split}_sequences'])
            actual, expected = ds.description(), dict(saved['data'][split])
            # 只允许显式迁移路径，规模/成员清单等字段仍须一致。
            actual.pop('archive'); expected.pop('archive')
            if actual != expected:
                raise RuntimeError('数据划分或压缩包描述不一致')
            indices = list(range(0, len(ds), args.sample_every))
            if not torch.equal(ds[0]['video'], ds[0]['video']):
                raise RuntimeError('中心裁剪不可重复')
            loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, num_workers=0)
            selected, offset = [], 0
            for batch in loader:
                x = batch['video'].to(args.device)
                y = model(x)[0]
                if y.shape != x.shape or not torch.isfinite(y).all():
                    raise RuntimeError('重建形状/数值异常')
                errors = (y-x).square().double().flatten(1).mean(1).cpu().tolist()
                for k, error in enumerate(errors):
                    selected.append(dict(split=split, dataset_index=indices[offset+k], sequence=batch['sequence'][k],
                                         start=int(batch['start'][k]), mse=error, psnr_db=diag.psnr(error)))
                offset += len(x)
                if offset % 100 == 0 or offset == len(indices):
                    print(f'{name} {split}: {offset}/{len(indices)} GOP', flush=True)
            mse = sum(r['mse'] for r in selected)/len(selected)
            metrics[split] = dict(mse=mse, psnr_db=diag.psnr(mse), gops=len(indices), total_gops=len(ds),
                                  scope='full' if args.sample_every == 1 else 'subset',
                                  passed25=sum(r['psnr_db'] >= 25 for r in selected))
            if split == 'val' and args.sample_every == 1:
                recorded = next((r['val_rounded']['psnr_db'] for r in reversed(saved['history'])
                                 if r['epoch'] == saved['epoch'] and 'val_rounded' in r), None)
                if recorded is not None:
                    reproduction = metrics[split]['psnr_db'] - recorded
            ranked = sorted(selected, key=lambda r:r['psnr_db'])
            for label, row in (('worst', ranked[0]), ('median',ranked[len(ranked)//2]), ('best',ranked[-1])):
                x = ds[row['dataset_index']]['video'].unsqueeze(0).to(args.device)
                y = model(x)[0].cpu()[0]
                fig, axes = compare.plt.subplots(2,4,figsize=(12,6))
                for i, images in enumerate((x.cpu()[0],y)):
                    for t in range(4):
                        axes[i,t].imshow(images[t].permute(1,2,0).clamp(0,1).numpy())
                        axes[i,t].set_title(f'{"Original" if i==0 else "Reconstruction"} / {row["start"]+t}')
                        axes[i,t].axis('off')
                fig.suptitle(f'{name} | {split} {label} | {row["sequence"]} | {row["psnr_db"]:.3f} dB')
                fig.tight_layout(); fig.savefig(directory/f'{split}_{label}.png',dpi=110); compare.plt.close(fig)
            rows.extend(selected)
            ds.close()
        diag.write_csv(directory/'per_gop.csv',rows)
        grouped = defaultdict(list)
        for r in rows:
            grouped[r['split'],r['sequence']].append(r['mse'])
        diag.write_csv(directory/'per_sequence.csv',[dict(split=s,sequence=v,gops=len(e),mse=sum(e)/len(e),psnr_db=diag.psnr(sum(e)/len(e)))
                                                   for (s,v),e in sorted(grouped.items())])
        quantized = name.endswith('3bit') or name.endswith('3bit_postadd') or name.startswith('common_fig7')
        record = dict(checkpoint=str(checkpoint.resolve()), epoch=saved['epoch'], variant=name, metrics=metrics,
                      quantization='clamp/round 3bit' if quantized else 'identity float',
                      sample_every=args.sample_every, crop='fixed_center_256_RGB', optimizer_updates=0,
                      val_reproduction_delta_db=reproduction, weights_unchanged=before==diag.model_sha(model),
                      checkpoint_unchanged=checksum==diag.sha(checkpoint), device=args.device,
                      note='Different latent sizes/quantization: diagnostic comparison, not rate-matched; old weights must not be relabeled postadd.')
        compare.write_json(directory/'result.json',record)
        if not record['weights_unchanged'] or not record['checkpoint_unchanged'] or (reproduction is not None and abs(reproduction) > .02):
            raise RuntimeError('只读或全量验证复现检查失败')
        summary.append(record)
        compare.write_json(out/'result.json',dict(status='completed' if number+1==len(args.checkpoint) else 'running',models=summary))
        del model, saved
    print(f'评估完成：{out}；参数更新0次',flush=True)


if __name__ == '__main__':
    main()
