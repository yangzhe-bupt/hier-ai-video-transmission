"""复用41依次训练三组；不带--train仅检查。默认各30轮，全量REDS。
保持数据顺序、逐轮裁剪一致，保留原训练脚本与旧检查点。
"""
import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'step')]
codec40 = importlib.import_module('40_residual_full_codec')
data = importlib.import_module('27_reds_dataset')
BASE_DATA = data.REDSZipDataset
NAMES = ('spatial_only', 'residual_branches_moe_float', 'residual_branches_moe_3bit')


class DeterministicCropDataset(BASE_DATA):
    current_epoch = 0
    crop_seed = 42

    def __getitem__(self, index):
        if self.split != 'train':
            return super().__getitem__(index)
        # 数据增强使用隔离RNG，不受网络初始化、路由或量化噪声影响。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.crop_seed + 1000003*self.current_epoch + index)
            return super().__getitem__(index)


def build(name):
    if name == 'spatial_only':
        return codec40.original_build(name)
    return codec40.factory(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--data-root', type=Path, default=Path('D:/Datasets/REDS'))
    parser.add_argument('--only', choices=NAMES)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs必须为正')
    data.REDSZipDataset = DeterministicCropDataset
    source = ROOT/'tools/41_train_reds_residual.py'
    text = source.read_text(encoding='utf-8-sig')

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError(f'41源码结构变化，停止: {old[:65]}')
        text = text.replace(old, new, 1)

    replace("codec = importlib.import_module('40_residual_full_codec')", 'codec = codec_adapter')
    replace('variants = codec.runner.variants', 'variants = codec40.runner.variants')
    replace("choices=('residual_branches_moe_3bit',)", f'choices={NAMES!r}')
    replace("    for epoch in range(start_epoch + 1, args.epochs + 1):",
            "    for epoch in range(start_epoch + 1, args.epochs + 1):\n        train_data.current_epoch = epoch")
    replace('            if (index + 1) % 25 == 0 or index + 1 == len(loader):',
            '            if seen % 2000 == 0 or index + 1 == len(loader):')
    replace("quantization='original 3bit clamp and round'", "quantization=quantization_label")
    replace("residual_groups=True, initialization=", "residual_groups=args.model_variant != 'spatial_only', initialization=")
    replace("comparison_reference='step28 epoch60 original architecture; historical control',",
            "comparison_reference='same-data same-updates three-model ablation', deterministic_crop_seed=42,")
    replace('cbr_dimension_ratio=16896 / (4 * 3 * 256 * 256)',
            "cbr_dimension_ratio=(.375 if args.model_variant == 'spatial_only' else 16896 / (4 * 3 * 256 * 256))")
    replace("    paths += [ROOT/'tools'/name for name in", "    paths += [ROOT/'tools/46_reds_multivideo_ablation.py']\n    paths += [ROOT/'tools'/name for name in")
    names = (args.only,) if args.only else NAMES
    for name in names:
        print(f'===== {name}：目标{args.epochs}轮；从头初始化 =====', flush=True)
        sys.argv = [str(source), '--model-variant', name, '--epochs', str(args.epochs),
                    '--data-root', str(args.data_root), '--output-root',
                    str(ROOT/'outputs/step46_reds_multivideo_ablation'/name)]
        if args.train:
            sys.argv.append('--train')
        exec(compile(text, str(source), 'exec'), dict(__name__='__main__', __file__=str(source),
            codec_adapter=SimpleNamespace(factory=build), codec40=codec40,
            quantization_label='clamp and round 3bit' if name.endswith('3bit') else 'identity; no clipping/noise/rounding'))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
