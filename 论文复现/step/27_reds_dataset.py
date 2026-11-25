"""REDS Sharp四帧数据加载：直接读ZIP，不解压、不生成重复缓存。

train_sharp与val_sharp使用作者的原始视频级划分。每段100帧，按文件名
排序组成不重叠GOP4；训练同组随机256裁剪，验证固定中心裁剪。
不将不同视频拼为相邻GOP，不加入增强模块/时间损失。
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class REDSZipDataset(Dataset):
    def __init__(self, archive, split, sequence_limit=0, crop_size=256):
        self.archive = Path(archive).resolve()
        if split not in ('train', 'val'):
            raise ValueError('split只能是train或val')
        if sequence_limit < 0 or crop_size != 256:
            raise ValueError('sequence_limit不能为负，当前模型裁剪尺寸固定256')
        self.split, self.crop_size = split, crop_size
        self._zip = None
        self.sequence_limit = sequence_limit
        groups = defaultdict(list)
        prefix = f'{split}/{split}_sharp/'
        with zipfile.ZipFile(self.archive) as stream:
            for member in stream.infolist():
                if not member.filename.startswith(prefix) or not member.filename.endswith('.png'):
                    continue
                p = PurePosixPath(member.filename)
                if len(p.parts) != 4 or '..' in p.parts or not p.stem.isdigit():
                    raise ValueError(f'非法图片路径: {p}')
                groups[p.parent.name].append(member.filename)
        expected = 240 if split == 'train' else 30
        if len(groups) != expected:
            raise ValueError(f'{split}预期{expected}段，实际{len(groups)}段；请使用原始Sharp包')
        self.rows = []
        names = sorted(groups)
        if sequence_limit:
            # 仅作显式小规模实验，首批排序序列不是完整数据集的随机代表。
            names = names[:sequence_limit]
        for sequence in sorted(groups):
            members = sorted(groups[sequence])
            if len(members) != 100 or len(set(members)) != 100:
                raise ValueError(f'{split}/{sequence}: 帧数不等于100或存在重复')
            if [int(PurePosixPath(m).stem) for m in members] != list(range(100)):
                raise ValueError(f'{split}/{sequence}: 帧编号不连续')
            if sequence in names:
                for start in range(0, 100, 4):
                    self.rows.append(dict(sequence=f'{split}/{sequence}', start=start,
                                          members=members[start:start + 4]))
        self.sequence_names = [f'{split}/{name}' for name in names]
        self.manifest_sha256 = hashlib.sha256(
            json.dumps(self.rows, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zip'] = None
        return state

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __getitem__(self, index):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive)
        row = self.rows[index]
        frames = []
        top = left = shape = None
        for member in row['members']:
            # ZipFile.read验证成员CRC；cv2完整解码后再裁剪，不缩放原图。
            raw = np.frombuffer(self._zip.read(member), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f'图片解码失败: {member}')
            h, w = frame.shape[:2]
            if min(h, w) < self.crop_size:
                raise ValueError(f'源图片太小: {member}, {frame.shape}')
            if shape is None:
                shape = frame.shape
                if self.split == 'train':
                    top = int(torch.randint(h - self.crop_size + 1, ()).item())
                    left = int(torch.randint(w - self.crop_size + 1, ()).item())
                else:
                    top, left = (h - self.crop_size) // 2, (w - self.crop_size) // 2
            if frame.shape != shape:
                raise ValueError(f'同一GOP图片尺寸不一致: {member}')
            crop = frame[top:top + self.crop_size, left:left + self.crop_size]
            frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        return {'video': video.float().div_(255), 'sequence': row['sequence'],
                'start': row['start'], 'crop_top': top, 'crop_left': left}

    def description(self):
        return dict(archive=str(self.archive), archive_bytes=self.archive.stat().st_size,
                    split=self.split, sequences=len(self.sequence_names), gops=len(self),
                    selected_sequence_ids=self.sequence_names, manifest_sha256=self.manifest_sha256,
                    sequence_limit=self.sequence_limit, gop_size=4, crop_size=256,
                    crop='shared_random' if self.split == 'train' else 'fixed_center',
                    normalization='RGB float32 [0,1]', source='REDS Sharp official split',
                    subset_experiment=bool(self.sequence_limit))
