"""冻结第I阶段模型，对比舍入量化、仅裁剪、浮点直通；不训练。

两条分支同时切换。后两种条件没有3-bit码率约束，只用于诊断，
不是同码率通信性能比较，也不能替代重新训练的消融实验。
"""
import argparse
import csv
from datetime import datetime
import importlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

multi = importlib.import_module("12_train_multi_gop")
base, data, utils, plt = multi.base, multi.data, multi.utils, multi.plt
ROOT = Path(__file__).resolve().parent.parent
MODES = ("rounded", "clip_only", "float_identity")


class DiagnosticQuantizer(torch.nn.Module):
    """保留原量化器作为基线；诊断不使用train()加噪行为。"""
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.mode = "rounded"

    def forward(self, features):
        if self.mode == "rounded":
            return self.original(features)
        if self.mode == "clip_only":
            return features.clamp(self.original.min_value, self.original.max_value)
        if self.mode == "float_identity":
            return features
        raise ValueError(self.mode)


def set_mode(model, mode):
    model.common_quantizer.mode = mode
    model.individual_quantizer.mode = mode
    model.eval()


@torch.no_grad()
def compare_images(model, dataset, device, output, split):
    """每个固定样本一张图：原图和三种模式，均显示四帧有效区域。"""
    for index in sorted(set([0, len(dataset)//2, len(dataset)-1])):
        item = dataset[index]
        video = item["video"].unsqueeze(0).to(device)
        images = [("Original", video.cpu())]
        for mode in MODES:
            set_mode(model, mode)
            images.append((mode, model(video)[0].cpu()))
        positions = item["valid_mask"][0, 0].nonzero()
        top, left = positions.min(0).values.tolist()
        bottom, right = (positions.max(0).values + 1).tolist()
        fig, axes = plt.subplots(4, 4, figsize=(10, 12))
        for row, (label, tensor) in enumerate(images):
            for t in range(4):
                axes[row,t].imshow(tensor[0,t,:,top:bottom,left:right].permute(1,2,0).numpy().clip(0,1))
                axes[row,t].set_title(f"{label} / frame {item['start_frame']+t}", fontsize=9)
                axes[row,t].axis("off")
        fig.suptitle(f"{split}: fixed checkpoint, diagnostic only (not equal bitrate)")
        fig.tight_layout()
        fig.savefig(output / f"{split}_frame_{item['start_frame']:04d}.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="每个集合仅取前2个GOP，不能当作完整诊断结果")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/step12_quant_diagnostics")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA不可用")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    # 检查点包含优化器与Python对象，只加载自己产生的可信文件。
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variants = importlib.import_module("12_model_variants")
    model = variants.build_model(saved["config"].get("model_variant", "legacy")).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    original_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    datasets = {split: data.CachedGOPDataset(args.data_dir, split) for split in ("train", "val")}
    manifest = datasets["train"].manifest
    # 本地/服务器路径可以不同，但视频、帧划分、预处理及实际mask必须一致。
    old_manifest = json.loads((args.checkpoint.parent / "data_manifest.json").read_text(encoding="utf-8"))
    for key in ("video_sha256", "cache_sha256", "preprocessing", "splits"):
        if manifest[key] != old_manifest[key]:
            raise ValueError(f"数据与训练记录不一致：{key}")
    if utils.sha256(args.data_dir / "frames_uint8.npy") != manifest["cache_sha256"]:
        raise ValueError("实际视频缓存与记录哈希不同")
    if not np.array_equal(np.load(args.data_dir / "valid_mask.npy"), np.load(args.checkpoint.parent / "valid_mask.npy")):
        raise ValueError("mask与训练时不一致")
    if args.smoke:
        datasets = {k: Subset(v, range(min(2,len(v)))) for k,v in datasets.items()}
    loaders = {k: DataLoader(v, batch_size=1, shuffle=False, num_workers=0) for k,v in datasets.items()}
    # 在同一个输入上确认包装前后rounded输出一致，避免诊断本身改变基线。
    sample = datasets["train"][0]["video"].unsqueeze(0).to(device)
    with torch.no_grad():
        reference = model(sample)[0]
    originals = (model.common_quantizer, model.individual_quantizer)
    model.common_quantizer = DiagnosticQuantizer(originals[0])
    model.individual_quantizer = DiagnosticQuantizer(originals[1])
    set_mode(model, "rounded")
    with torch.no_grad():
        torch.testing.assert_close(model(sample)[0], reference)
    del reference, sample
    output = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = utils.sha256(args.checkpoint)
    config = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
              "checkpoint_epoch": saved["epoch"], "data_dir": str(args.data_dir),
              "model_variant": saved["config"].get("model_variant", "legacy"),
              "modes": MODES, "smoke": args.smoke, "optimizer_updates": 0,
              "train_gops": len(datasets["train"]), "val_gops": len(datasets["val"]),
              "python": platform.python_version(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": str(device),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "quantization_bits": originals[0].bits,
              "range": [originals[0].min_value, originals[0].max_value]}
    utils.write_json(output / "config.json", config)
    shutil.copy2(args.data_dir / "manifest.json", output / "data_manifest.json")
    # 诊断脚本及模型依赖均归档，原训练配置也单独保留。
    utils.write_json(output / "training_config.json", saved["config"])
    expected_metrics = saved["metrics"]
    del saved
    hashes = {}
    for folder in ("step", "practice", "Biran", "packet"):
        for source in (ROOT / folder).rglob("*.py"):
            relative = source.relative_to(ROOT)
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = utils.sha256(source)
    utils.write_json(output / "source_manifest.json", hashes)
    print(f"诊断目录：{output}；参数更新0次", flush=True)
    started = time.perf_counter()
    rows, baseline = [], {}
    for mode in MODES:
        set_mode(model, mode)
        for split, loader in loaders.items():
            metrics = multi.evaluate(model, loader, device)
            if mode == "rounded":
                baseline[split] = metrics
            row = {"mode": mode, "split": split, **metrics,
                   "delta_mse_vs_rounded": metrics["mse"]-baseline[split]["mse"],
                   "delta_psnr_db_vs_rounded": metrics["psnr_db"]-baseline[split]["psnr_db"]}
            rows.append(row)
            print(f"{mode:15s} {split:5s} MSE={metrics['mse']:.8f} PSNR={metrics['psnr_db']:.3f} dB", flush=True)
            with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
    reproduction = {split: {"saved_mse": expected_metrics[f"{split}_rounded_mse"],
                            "current_mse": baseline[split]["mse"],
                            "difference": baseline[split]["mse"]-expected_metrics[f"{split}_rounded_mse"]}
                    for split in datasets}
    utils.write_json(output / "baseline_check.json", {"comparable": not args.smoke, "metrics": reproduction})
    for split, dataset in datasets.items():
        compare_images(model, dataset, device, output, split)
    # 恢复原模块后逐项核对参数与buffer，确认评估没有改变模型。
    model.common_quantizer, model.individual_quantizer = originals
    assert all(torch.equal(v.cpu(), original_state[k]) for k,v in model.state_dict().items())
    assert utils.sha256(args.checkpoint) == checkpoint_hash
    utils.write_json(output / "result.json", {"status": "completed", "smoke": args.smoke,
        "optimizer_updates": 0, "weights_unchanged": True, "checkpoint_unchanged": True,
        "elapsed_seconds": time.perf_counter()-started, "metrics": rows})
    lines = ["# 冻结模型量化诊断", "", f"检查点轮数：{config['checkpoint_epoch']}；短检查：{args.smoke}；参数更新0次。", "",
             "|模式|集合|MSE|PSNR(dB)|相对基线PSNR变化|", "|---|---|---:|---:|---:|"]
    lines += [f"|{r['mode']}|{r['split']}|{r['mse']:.8f}|{r['psnr_db']:.3f}|{r['delta_psnr_db_vs_rounded']:+.3f}|" for r in rows]
    lines += ["", "两分支同时切换：rounded=裁剪后舍入；clip_only=仅裁剪；float_identity=原特征直通。",
              "三种模式全程eval，不加训练噪声。后两者不是相同比特率的通信方案；冻结模型对输入分布的适应会影响结果，不能等同重新训练消融。",
              "mask仅排除新增补边，原视频内部黑边仍参与统计。common_clip/individual_clip始终指量化前特征超出原范围的比例，并非直通模式执行了裁剪。",
              "指标在全部有效RGB像素上汇总MSE后计算PSNR。首、中、末GOP固定展示。baseline_check.json记录与原检查点指标的差异；短检查不可与全量指标直接比较。"]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"诊断完成，权重未改变：{output}", flush=True)


if __name__ == "__main__":
    main()
