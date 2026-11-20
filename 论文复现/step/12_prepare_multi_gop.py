"""Step12数据准备：单视频、按时间划分的多GOP学习数据，不执行训练。

先顺序解码一次，将完整画面缩放补边后缓存为uint8，减少重复视频解码。
按时间段划分后再组成互不重叠的四帧GOP，禁止随机拆分相邻GOP。
同视频验证只能用于流程检查，不能代表跨视频泛化性能。
"""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

step05 = importlib.import_module("05_video_dataset")
ROOT = Path(__file__).resolve().parent.parent


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class CachedGOPDataset(Dataset):
    """每个样本返回[T,C,H,W]，DataLoader再增加批次维B。

    缓存使用内存映射，不必把全部float32视频装入内存。
    有效区域mask为[1,1,H,W]；组成批次后为[B,1,1,H,W]。
    """
    def __init__(self, data_dir, split="train"):
        if split not in ("train", "val"):
            raise ValueError("split必须为train或val")
        self.data_dir = Path(data_dir)
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.records = self.manifest["splits"][split]
        self.frames = None
        self.mask = torch.from_numpy(np.load(self.data_dir / "valid_mask.npy").copy()).bool()

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        # 多进程DataLoader各自打开映射，避免序列化整个缓存。
        state = self.__dict__.copy()
        state["frames"] = None
        return state

    def __getitem__(self, index):
        if self.frames is None:
            self.frames = np.load(self.data_dir / "frames_uint8.npy", mmap_mode="r")
        record = self.records[index]
        # 复制当前四帧，再转float32；不会向只读映射写入数据。
        frames = np.array(self.frames[record["start"]:record["stop"]], copy=True)
        video = torch.from_numpy(frames).float().div_(255)
        return {"video": video, "valid_mask": self.mask.clone(), "start_frame": record["start"]}


def save_preview(directory, split):
    dataset = CachedGOPDataset(directory, split)
    indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
    fig, axes = plt.subplots(len(indices), 4, figsize=(9, 3 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        item = dataset[index]
        for t in range(4):
            axes[row, t].imshow(item["video"][t].permute(1, 2, 0).numpy())
            axes[row, t].set_title(f"{split} GOP {index} / frame {item['start_frame'] + t}")
            axes[row, t].axis("off")
    fig.tight_layout()
    fig.savefig(directory / f"preview_{split}.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/videos/new_year.mp4")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练时间段占整段视频的比例")
    parser.add_argument("--gap-frames", type=int, default=16, help="训练时间段后丢弃的隔离帧数")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/multi_gop")
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or args.gap_frames < 4:
        parser.error("训练比例必须在(0,1)内，隔离区至少4帧")
    if not args.video.is_file():
        parser.error("视频不存在")
    torch.set_num_threads(2)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cached = []
    first_info = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gop, mask, info = step05.frames_to_full_gop([frame], 256)
            if first_info is None:
                first_info = info
            elif first_info != info:
                raise RuntimeError("视频尺寸中途发生变化，不能共享同一个有效区域mask")
            # 预处理仅包含uint8图像的resize和补边；转回uint8可无损保存该结果。
            cached.append(gop[0, 0].mul(255).round().byte().numpy())
    finally:
        cap.release()
    count = len(cached)
    if reported_count > 0 and count != reported_count:
        raise RuntimeError(f"元数据帧数{reported_count}与实际解码{count}不一致，请先检查视频")
    # 训练段终点对齐四帧；范围统一使用左闭右开[start,stop)。
    train_stop = (int(count * args.train_ratio) // 4) * 4
    val_start = train_stop + args.gap_frames
    splits = {
        "train": [{"start": s, "stop": s + 4} for s in range(0, train_stop, 4)],
        "val": [{"start": s, "stop": s + 4} for s in range(val_start, count - 3, 4)],
    }
    if not splits["train"] or not splits["val"]:
        raise RuntimeError("视频太短，无法在当前划分参数下构成训练和验证GOP")
    train_ids = {i for r in splits["train"] for i in range(r["start"], r["stop"])}
    val_ids = {i for r in splits["val"] for i in range(r["start"], r["stop"])}
    assert train_ids.isdisjoint(val_ids)
    assert min(val_ids) - max(train_ids) - 1 >= args.gap_frames
    output = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "frames_uint8.npy", np.stack(cached))
    np.save(output / "valid_mask.npy", mask[0].numpy())
    manifest = {
        "schema_version": 1, "purpose": "single_video_pipeline_validation",
        "video_path": str(args.video.resolve()), "video_sha256": file_hash(args.video),
        "decoded_frames": count, "reported_frames": reported_count, "fps": fps,
        "preprocessing": first_info, "canvas_size": 256, "gop_length": 4,
        "train_ratio": args.train_ratio, "gap_frames": args.gap_frames,
        "gap_range_start_stop": [train_stop, val_start],
        "unused_tail_start_stop": [splits["val"][-1]["stop"], count],
        "splits": splits, "cache_sha256": file_hash(output / "frames_uint8.npy"),
        "cache_dtype": "uint8", "cache_layout": "TCHW",
        "torch": torch.__version__, "opencv": cv2.__version__,
        "note": "同一视频的时间段验证；非跨视频测试。后续模型须随机初始化，不能加载单GOP拟合权重。",
    }
    save_json(output / "manifest.json", manifest)
    for source in (Path(__file__), Path(step05.__file__)):
        shutil.copy2(source, output / source.name)
    for split in splits:
        save_preview(output, split)
    # 检查DataLoader实际堆叠结果，避免Dataset重复添加B维。
    batch = next(iter(DataLoader(CachedGOPDataset(output), batch_size=2, shuffle=False)))
    assert batch["video"].shape[1:] == (4, 3, 256, 256)
    assert batch["valid_mask"].shape[1:] == (1, 1, 256, 256)
    assert batch["video"].dtype == torch.float32
    assert batch["valid_mask"].dtype == torch.bool
    # 与原始视频四帧预处理直接对比，确认缓存未改变像素。
    raw = step05.read_frames(args.video, 0, 4)
    direct, _, _ = step05.frames_to_full_gop(raw, 256)
    assert torch.equal(batch["video"][0], direct[0])
    report = {"status": "checked", "decoded_frames": count,
              "train_gops": len(splits["train"]), "val_gops": len(splits["val"]),
              "gap_frames": args.gap_frames, "batch_shape": list(batch["video"].shape),
              "mask_shape": list(batch["valid_mask"].shape),
              "valid_ratio": mask.float().mean().item(),
              "cache_mib": (output / "frames_uint8.npy").stat().st_size / 1024**2,
              "frame_overlap": 0, "cache_matches_direct_preprocessing": True}
    save_json(output / "checks.json", report)
    (output / "summary.md").write_text(
        f"# 多GOP数据准备\n\n解码{count}帧，训练{len(splits['train'])}个GOP，验证{len(splits['val'])}个GOP。\n"
        f"训练帧0–{train_stop-1}；隔离帧{train_stop}–{val_start-1}；"
        f"验证帧{val_start}–{splits['val'][-1]['stop']-1}。每个GOP连续4帧，不重叠。\n\n"
        "保持全画面等比缩放及补边，损失只统计有效区域。尚未进行训练。\n"
        "时间隔离只能减少相邻帧泄漏，不能消除同场景相关性；不代表跨视频泛化性能。\n"
        "后续随机初始化模型，不加载此前单GOP拟合权重。\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"数据目录：{output}")
    print("多GOP划分、缓存及DataLoader检查通过；未启动训练")


if __name__ == "__main__":
    main()
