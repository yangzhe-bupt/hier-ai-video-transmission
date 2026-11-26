"""恢复46中的一组，完成后继续其余组；不带--train只校验。
不修改41/46源文件，保持原检查点源码校验和确定性裁剪规则。
"""
import argparse
import hashlib
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
NAMES = ('spatial_only', 'residual_branches_moe_float', 'residual_branches_moe_3bit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--data-root', type=Path, default=Path('D:/Datasets/REDS'))
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    variant = saved['config']['model_variant']
    epoch = saved['epoch']
    if variant not in NAMES or args.epochs < epoch:
        parser.error('模型版本或目标轮数不匹配')
    for rel, expected in saved['source_sha256'].items():
        path = ROOT/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'原源码不一致: {rel}')
    if args.data_root.resolve() != Path(saved['config']['data_root']).resolve():
        raise RuntimeError('续训数据目录变化')
    for split in ('train','val'):
        if not (args.data_root/f'{split}_sharp.zip').is_file():
            raise FileNotFoundError(split)
    if not all(not t.is_floating_point() or torch.isfinite(t).all().item() for t in saved['model'].values()):
        raise RuntimeError('模型存在非有限值')
    start = NAMES.index(variant) + int(epoch == args.epochs)
    remaining = NAMES[start:]
    print(f'校验通过：{variant}，已保存epoch={epoch}，updates={saved["updates"]}；'
          f'目标各{args.epochs}轮；执行顺序={remaining}', flush=True)
    del saved
    if not args.train:
        print('只读校验完成；参数更新0次', flush=True)
        return
    if not remaining:
        print('全部目标已完成，无需训练', flush=True)
        return
    source = ROOT/'tools/46_reds_multivideo_ablation.py'
    text = source.read_text(encoding='utf-8-sig')

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('46源码入口结构变化，停止')
        text = text.replace(old, new, 1)

    replace('    names = (args.only,) if args.only else NAMES', '    names = remaining_variants')
    replace("        if args.train:\n            sys.argv.append('--train')",
            "        if name == resume_variant:\n            sys.argv += ['--resume', resume_path]\n        if args.train:\n            sys.argv.append('--train')")
    replace("        print(f'===== {name}：目标{args.epochs}轮；从头初始化 =====', flush=True)",
            "        print(f'===== {name}：目标{args.epochs}轮；' + ('恢复Adam/RNG' if name == resume_variant else '从头初始化') + ' =====', flush=True)")
    sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'step')]
    sys.argv = [str(source), '--train', '--epochs', str(args.epochs), '--data-root', str(args.data_root)]
    exec(compile(text, str(source), 'exec'), dict(__name__='__main__', __file__=str(source),
         remaining_variants=remaining, resume_variant=variant, resume_path=str(args.checkpoint.resolve())))


if __name__ == '__main__':
    main()
