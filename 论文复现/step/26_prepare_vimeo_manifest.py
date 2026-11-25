"""Vimeo90K七帧数据清单检查；不下载、不训练、不复制图片。

按第一层目录保守分组划分train/val，官方test完全隔离。
第一层目录仅是数据集目录分组，不声称它一定对应原始视频ID。
支持完整数据和显式--allow-subset的已解压小子集。
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct


def read_list(path):
    rows = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        value = line.strip()
        if not value:
            continue
        parts = PurePosixPath(value).parts
        if (len(parts) != 2 or '\\' in value
                or any(not p.isdigit() for p in parts)):
            raise ValueError(f'非法序列ID: {value!r}')
        rows.append(value)
    if len(set(rows)) != len(rows):
        raise ValueError(f'清单有重复ID: {path}')
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(seed, text):
    return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()


def inspect_clip(root, clip):
    """检查PNG头尺寸；不等价于完整解码/CRC检查，训练加载时仍须解码。"""
    for index in range(1, 8):
        path = root / 'sequences' / clip / f'im{index}.png'
        if not path.is_file():
            return 'missing_frame'
        with path.open('rb') as stream:
            header = stream.read(24)
        if (len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n'
                or header[12:16] != b'IHDR'):
            return 'invalid_png_header'
        if struct.unpack('>II', header[16:24]) != (448, 256):
            return 'unexpected_dimensions'
    return None


def prepare(args):
    root = args.root.resolve()
    train_list = root / 'sep_trainlist.txt'
    test_list = root / 'sep_testlist.txt'
    train = read_list(train_list)
    test = read_list(test_list)
    if not train or not test:
        raise ValueError('官方train/test清单均不能为空')
    if set(train) & set(test):
        raise ValueError('官方train/test序列有交集，停止')
    # 只按完整官方清单分配分组，再检查本地图片是否存在。
    # 因此添加解压图片不会改变已有序列的train/val归属。
    groups = sorted({v.split('/')[0] for v in train},
                    key=lambda v: rank(args.seed, v))
    if len(groups) < 2:
        raise ValueError('至少需要两个目录分组才能隔离train/val')
    count = max(1, min(len(groups) - 1, round(len(groups) * args.val_fraction)))
    val_groups = set(groups[:count])
    candidates = {
        'train': [v for v in train if v.split('/')[0] not in val_groups],
        'val': [v for v in train if v.split('/')[0] in val_groups],
        'test': test,
    }
    selected, missing = {}, {}
    for split, rows in candidates.items():
        good, absent = [], []
        for clip in rows:
            error = inspect_clip(root, clip)
            if error:
                if error != 'missing_frame':
                    raise ValueError(f'{clip}: {error}')
                absent.append(clip)
            else:
                good.append(clip)
        selected[split] = sorted(good, key=lambda v: rank(args.seed, v))
        missing[split] = absent
    if any(missing.values()) and not args.allow_subset:
        raise ValueError('本地缺少序列。完整数据需补齐；小子集需显式--allow-subset')
    if not selected['train'] or not selected['val']:
        raise ValueError('已解压数据的train/val至少各有一个序列')
    if args.train_limit:
        selected['train'] = selected['train'][:args.train_limit]
    if args.val_limit:
        selected['val'] = selected['val'][:args.val_limit]
    output = args.output.resolve()
    # 防止覆盖前次实验的划分证据。
    if output.exists():
        raise ValueError('输出目录已存在，请指定新的目录')
    report = {
        'status': 'manifest_checked_no_training',
        'dataset_root': str(root), 'seed': args.seed,
        'validation_fraction_of_groups': args.val_fraction,
        'group_policy': 'first_directory_conservative_not_verified_source_video_id',
        'official_list_sha256': {'train': digest(train_list), 'test': digest(test_list)},
        'official_counts': {'train': len(train), 'test': len(test)},
        'available_counts': {k: len(candidates[k]) - len(missing[k]) for k in candidates},
        'missing_counts': {k: len(v) for k, v in missing.items()},
        'selected_counts': {k: len(v) for k, v in selected.items()},
        'subset_experiment': any(missing.values()) or bool(args.train_limit or args.val_limit),
        'train_val_group_overlap': 0,
        'official_test_sequence_overlap_with_train_val': 0,
        'image_check': 'PNG_header_and_dimensions_only_not_full_decode',
        'gop_size': 4,
        'training_policy': 'one_contiguous_window_start_1_to_4_random_per_clip',
        'validation_policy': 'fixed_im1_to_im4_center_crop_256',
        'shared_crop_for_all_gop_frames': True,
        'pixel_normalization': 'RGB_float_0_to_1',
        'extra_augmentation': False,
        'temporal_loss': False,
        'enhancement': False,
    }
    output.mkdir(parents=True)
    for split, rows in selected.items():
        target = output / f'{split}.txt'
        target.write_text(''.join(v + '\n' for v in rows), encoding='utf-8')
    report['manifest_sha256'] = {k: digest(output / f'{k}.txt') for k in selected}
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--allow-subset', action='store_true')
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error('--val-fraction须在0和1之间')
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error('limit不能小于0')
    prepare(args)


if __name__ == '__main__':
    main()
