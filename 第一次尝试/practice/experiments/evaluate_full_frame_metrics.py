# ======================================================================
# 文件：evaluate_full_frame_metrics.py
# 功能：对齐源视频及两种传输结果，记录全画面客观画质与时间误差。
# 函数汇总：
#   - ssim_rgb：计算 11×11 高斯窗单尺度 SSIM。
#   - evaluate：逐帧比较源视频和重建、保留冻结标签并汇总。
# 输入：全画面回环目录中的 summary.json、逐 GOP 日志和两段 MP4。
# 输出：逐帧指标、汇总 CSV/JSON，写入独立评测目录。
# ======================================================================

import argparse
import csv
import json
import math
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from train_packet_aware_decoder import write_csv
from accept_biran_swin import file_sha256


def ssim_rgb(reference, decoded):
    """用 11×11、标准差 1.5 的高斯窗计算逐通道 SSIM，再平均有效区域。"""
    mr = cv2.GaussianBlur(reference, (11, 11), 1.5)
    md = cv2.GaussianBlur(decoded, (11, 11), 1.5)
    a, b, c = mr * mr, md * md, mr * md
    vr = np.maximum(0, cv2.GaussianBlur(reference * reference, (11, 11), 1.5) - a)
    vd = np.maximum(0, cv2.GaussianBlur(decoded * decoded, (11, 11), 1.5) - b)
    cov = cv2.GaussianBlur(reference * decoded, (11, 11), 1.5) - c
    ssim = ((2 * c + 0.01 ** 2) * (2 * cov + 0.03 ** 2)
            / ((a + b + 0.01 ** 2) * (vr + vd + 0.03 ** 2)))
    return float(np.clip(ssim[5:-5, 5:-5], -1, 1).mean(dtype=np.float64))


def psnr(mse):
    """像素范围为 [0,1]，零误差用空 PSNR 和完美帧计数表示。"""
    return -10 * math.log10(mse) if mse > 0 else None


def evaluate(run_dir, out):
    """按同一播放帧序比较，画质统计包含冻结后的输出。"""
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    source = Path(meta["input"])
    sources = {case: run_dir / result["mp4"] for case, result in meta["results"].items()}
    with (run_dir / "gop_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        statuses = {(item["case"], int(item["gop"])): item["status"] for item in csv.DictReader(stream)}
    rows = []
    with ExitStack() as stack:
        readers = {}
        for case, path in {"reference": source, **sources}.items():
            reader = cv2.VideoCapture(str(path))
            stack.callback(reader.release)
            if not reader.isOpened() or not math.isclose(reader.get(cv2.CAP_PROP_FPS), meta["fps"], abs_tol=0.001):
                raise RuntimeError(f"无法读取 {path} 或帧率不同")
            readers[case] = reader

        previous, prior_ref = {}, None
        for idx in range(meta["frames"]):
            current = {}
            for case, reader in readers.items():
                ok, bgr = reader.read()
                if not ok or list(bgr.shape[1::-1]) != meta["output_width_height"]:
                    raise RuntimeError(f"{case} 第 {idx} 帧缺失或尺寸不一致")
                current[case] = bgr.astype(np.float32) / 255
            ref = current.pop("reference")
            for case, frame in current.items():
                error = frame - ref
                mse = float(np.mean(error * error, dtype=np.float64))
                status = statuses[case, idx // 4]
                temporal = None
                if prior_ref is not None:
                    delta = (frame - previous[case]) - (ref - prior_ref)
                    temporal = float(np.mean(delta * delta, dtype=np.float64))
                rows.append({
                    "case": case, "frame": idx, "seconds": idx / meta["fps"], "status": status,
                    "frozen": status.startswith("missing"), "mse_rgb": mse, "psnr_rgb_db": psnr(mse),
                    "ssim_rgb": ssim_rgb(ref, frame), "temporal_diff_mse_rgb": temporal,
                })
            previous, prior_ref = current, ref
            if (idx + 1) % 300 == 0:
                print(f"已评测 {idx + 1}/{meta['frames']} 帧", flush=True)
        if any(reader.read()[0] for reader in readers.values()):
            raise RuntimeError("存在评测范围以外的帧")

    summaries = []
    for case, transport in meta["results"].items():
        group = [row for row in rows if row["case"] == case]
        frozen = longest = 0
        for row in group:
            frozen = frozen + 1 if row["frozen"] else 0
            longest = max(frozen, longest)
        count = sum(row["frozen"] for row in group)
        if count != transport["frozen_frames"]:
            raise RuntimeError("冻结帧计数与传输日志不一致")
        mse = float(np.mean([row["mse_rgb"] for row in group]))
        scores = [row["psnr_rgb_db"] for row in group if row["psnr_rgb_db"] is not None]
        average_bytes = transport["mean_accounted_bytes_per_gop"] * meta["gops"]
        summaries.append({
            "case": case, "frames": len(group), "decoded_gops": transport["decoded_gops"],
            "realized_injected_loss": transport["injected"] / transport["offered_datagrams"],
            "sent_budget_mbps": average_bytes * 8 * meta["fps"] / (len(group) * 1_000_000),
            "frozen_frames": count, "longest_freeze_s": longest / meta["fps"],
            "mse_rgb": mse, "global_psnr_rgb_db": psnr(mse),
            "mean_finite_frame_psnr_rgb_db": float(np.mean(scores)) if scores else None,
            "perfect_frames": len(group) - len(scores),
            "mean_ssim_rgb": float(np.mean([row["ssim_rgb"] for row in group])),
            "mean_temporal_diff_mse_rgb": float(np.mean([row["temporal_diff_mse_rgb"] for row in group[1:]])),
            "lpips": None,
        })
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "per_frame.csv", rows)
    write_csv(out / "summary.csv", summaries)
    metadata = {
        "frame_count": meta["frames"], "resolution": meta["output_width_height"], "fps": meta["fps"],
        "metric_domain": "decoded original and reconstructed MP4 frames, RGB normalized to [0,1]",
        "source_sha256": file_sha256(source),
        "reference": str(source.resolve()), "outputs": {case: str(path.resolve()) for case, path in sources.items()},
        "output_sha256": {case: file_sha256(path) for case, path in sources.items()},
        "lpips_status": "pending_offline_pretrained_backbone",
        "results": summaries,
    }
    (out / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description="全画面重建逐帧指标")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    evaluate(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
