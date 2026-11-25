"""Windows本地REDS Stage I入口。默认只检查；显式--train才训练。

复用已有模型：发送端/接收端全模型训练、量化训练噪声、验证真实舍入。
无丢包、无增强、无边缘/时间损失。REDS替代数据实验，不宣称严格论文复现。
"""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
import cv2
import torch
from torch.utils.data import DataLoader

variants = importlib.import_module('12_model_variants')
data = importlib.import_module('27_reds_dataset')
ROOT = Path(__file__).resolve().parent.parent


def source_manifest():
    paths = sorted((ROOT / 'step').glob('*.py')) + sorted((ROOT / 'packet').glob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_error = count = 0
    for batch in loader:
        video = batch['video'].to(device)
        prediction, _, _ = model(video)
        # 每批误差用FP64求和，再按全部RGB像素数计算 pooled PSNR。
        total_error += (prediction - video).square().double().sum().item()
        count += video.numel()
    mse = total_error / count
    if not math.isfinite(mse):
        raise RuntimeError('验证出现非有限MSE')
    return dict(mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-12)),
                gops=len(loader.dataset), pixels=count, quantization='clamp and round')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('D:/Datasets/REDS'))
    p.add_argument('--train', action='store_true', help='显式授权本次进程更新参数；不带则只检查')
    p.add_argument('--model-variant', choices=('common_fig7', 'common_fig7_stage6_pair'), default='common_fig7')
    p.add_argument('--epochs', type=int, default=30, help='累计目标epoch；resume从上一轮继续')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accumulation-steps', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--aux-weight', type=float, default=.01, help='当前实验系数，不声称论文给出该值')
    p.add_argument('--workers', type=int, default=0, help='Windows先使用0；确认稳定后可增大')
    p.add_argument('--train-sequences', type=int, default=0, help='0使用全部240段；正整数为排序小子集')
    p.add_argument('--val-sequences', type=int, default=0, help='0使用全部30段；正整数为排序小子集')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--resume', type=Path, help='仅本脚本生成的可信last.pt，恢复Adam/RNG/epoch')
    p.add_argument('--output-root', type=Path, default=ROOT / 'outputs/step28_reds_stage1')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.eval_every) < 1:
        p.error('轮数、批量、累计数、评估间隔必须为正')
    if min(args.workers, args.train_sequences, args.val_sequences) < 0:
        p.error('workers及序列上限不能为负')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        p.error('lr/aux-weight不合法')
    if args.resume and not args.train:
        p.error('resume仅用于显式--train；默认检查不会加载检查点')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA不可用，请检查Python环境')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_data = data.REDSZipDataset(args.data_root / 'train_sharp.zip', 'train', args.train_sequences)
    val_data = data.REDSZipDataset(args.data_root / 'val_sharp.zip', 'val', args.val_sequences)
    if set(train_data.sequence_names) & set(val_data.sequence_names):
        raise RuntimeError('数据划分交叉')
    descriptions = {'train': train_data.description(), 'val': val_data.description()}
    # 只检查数据：每组真实解码4帧，验证中心裁剪可重复；不构造优化器。
    a, b, c = train_data[0], val_data[0], val_data[0]
    if a['video'].shape != (4, 3, 256, 256) or not torch.equal(b['video'], c['video']):
        raise RuntimeError('数据形状或确定性验证失败')
    check = dict(status='data_checked_no_training', optimizer_updates=0,
                 torch=torch.__version__, opencv=cv2.__version__,
                 cuda_available=torch.cuda.is_available(),
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 data=descriptions, sample_shape=list(a['video'].shape),
                 sample_range=[a['video'].min().item(), a['video'].max().item()],
                 validation_crop_repeatable=True, source_sha256=source_manifest())
    if not args.train:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        train_data.close()
        val_data.close()
        return

    print('开始正式Stage I训练：会更新全模型参数；无丢包、无增强、无时间损失。', flush=True)
    device = torch.device(args.device)
    model = variants.build_model(args.model_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(effective_batch_gops=args.batch_size * args.accumulation_steps,
                  precision='FP32', training_dataset='REDS substituted for Vimeo90K',
                  cbr_dimension_ratio=16896 / (4 * 3 * 256 * 256),
                  enhancement=False, temporal_loss=False, packet_loss=0,
                  auxiliary_reduction='mean per microbatch, sample weighted across accumulation',
                  final_partial_accumulation='normalized by actual GOP count',
                  checkpoint_trust='load only self-generated local checkpoints')
    start_epoch = updates = 0
    best_mse = float('inf')
    history = []
    if args.resume:
        # 检查点包含优化器/RNG，须使用本项目自己生成的可信文件。
        old = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ('model_variant', 'batch_size', 'accumulation_steps', 'lr', 'aux_weight', 'seed', 'workers'):
            if old['config'][key] != config[key]:
                raise RuntimeError(f'resume配置不一致: {key}')
        if old['data'] != descriptions:
            raise RuntimeError('resume数据路径/划分发生变化')
        if old['source_sha256'] != source_manifest():
            raise RuntimeError('resume源代码有变化，请保留原版本或另开实验')
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        start_epoch, updates = old['epoch'], old['updates']
        best_mse, history = old['best_mse'], old['history']
        torch.set_rng_state(old['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(old['rng_cuda'])
    if args.epochs <= start_epoch:
        raise ValueError('目标epochs必须大于已保存epoch')
    out = args.output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    for rel in source_manifest():
        destination = out / 'source' / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    write_json(out / 'config.json', config)
    write_json(out / 'data_check.json', check)
    write_json(out / 'data_manifest.json', {'description': descriptions,
                                          'train': train_data.rows, 'val': val_data.rows})
    # 全包仅读取，不复制；没有uint8/float缓存副本。
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(args.seed))
    val_data.close()
    train_data.close()  # Windows子进程不继承已打开的ZipFile句柄。
    started = time.perf_counter()
    initial = evaluate(model, val_loader, device)
    write_json(out / 'initial.json', initial)
    # 新目录的best只比较本次起点及后续评估，保证即使续训没提高也有best。
    # 历史run的best.pt原样保留，不把旧run历史最小值误报为本次结果。
    best_mse = initial['mse']
    initial_payload = dict(format='reds_stage1_v1', epoch=start_epoch, updates=updates,
                           model=model.state_dict(), optimizer=optimizer.state_dict(),
                           config=config, data=descriptions, source_sha256=source_manifest(),
                           best_mse=best_mse, history=history, rng_cpu=torch.get_rng_state(),
                           rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
    save_checkpoint(out / 'best.pt', initial_payload)
    del initial_payload
    print(f'实验目录：{out}\n训练{len(train_data)} GOP，验证{len(val_data)} GOP；'
          f'物理batch={args.batch_size}，累计={args.accumulation_steps}', flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # 每轮独立洗牌种子使续训从完整epoch边界可重现。
        sampler_generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            generator=sampler_generator, num_workers=args.workers,
                            pin_memory=device.type == 'cuda')
        model.train()
        epoch_mse_sum = epoch_aux_sum = 0.
        seen = 0
        for index, batch in enumerate(loader):
            if index % args.accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
                remaining = len(train_data) - index * args.batch_size
                group_gops = min(args.batch_size * args.accumulation_steps, remaining)
            video = batch['video'].to(device, non_blocking=True)
            pred, aux, _ = model(video)
            mse = (pred - video).square().mean()
            loss = mse + args.aux_weight * aux
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('训练loss非有限，停止；未覆盖上一轮检查点')
            (loss * len(video) / group_gops).backward()
            epoch_mse_sum += mse.item() * len(video)
            epoch_aux_sum += aux.item() * len(video)
            seen += len(video)
            if (index + 1) % args.accumulation_steps == 0 or index + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError('梯度非有限，停止')
                optimizer.step()
                updates += 1
            if (index + 1) % 25 == 0 or index + 1 == len(loader):
                print(f'epoch={epoch}/{args.epochs} GOP={seen}/{len(train_data)} updates={updates}', flush=True)
        row = dict(epoch=epoch, updates=updates, train_noise_mse=epoch_mse_sum / seen,
                   train_aux=epoch_aux_sum / seen, elapsed_seconds=time.perf_counter() - started)
        improved = False
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row['val_rounded'] = evaluate(model, val_loader, device)
            improved = row['val_rounded']['mse'] < best_mse
            if improved:
                best_mse = row['val_rounded']['mse']
            print(f"epoch={epoch} 验证舍入PSNR={row['val_rounded']['psnr_db']:.3f} dB", flush=True)
        history.append(row)
        payload = dict(format='reds_stage1_v1', epoch=epoch, updates=updates,
                       model=model.state_dict(), optimizer=optimizer.state_dict(),
                       config=config, data=descriptions, source_sha256=source_manifest(),
                       best_mse=best_mse, history=history, rng_cpu=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        save_checkpoint(out / 'last.pt', payload)
        if improved:
            save_checkpoint(out / 'best.pt', payload)
        write_json(out / 'history.json', history)
        write_json(out / 'result.json', dict(status='completed' if epoch == args.epochs else 'running',
                   final=row, best_val_mse=best_mse, optimizer_updates=updates,
                   best_scope='current_run_including_initial_or_resume_start'))
    train_data.close()
    val_data.close()
    print(f'训练记录保存完成：{out}', flush=True)


if __name__ == '__main__':
    main()
