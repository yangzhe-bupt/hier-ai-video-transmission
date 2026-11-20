"""多GOP无丢包训练：随机初始化，同视频时间段验证，不代表跨视频测试。

训练加噪量化，每轮结束对训练集及验证集使用舍入量化评估。
--smoke只运行各集合前两个GOP，专用于程序检查，不产出正式实验结论。
"""
import argparse
import csv
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

base = importlib.import_module("12_train_stage1")
data = importlib.import_module("12_prepare_multi_gop")
utils = importlib.import_module("12_train_single_gop")
plt = base.plt
ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def evaluate(model, loader, device):
    """按有效像素数汇总MSE，再换算PSNR；不是各GOP的PSNR简单平均。"""
    model.eval()
    squared_sum, pixel_count, examples = 0.0, 0, 0
    totals = dict(aux=0.0, common_clip=0.0, individual_clip=0.0)
    for batch in loader:
        video, mask = batch["video"].to(device), batch["valid_mask"].to(device)
        prediction, aux, saturation = model(video)
        error = torch.where(mask, (prediction - video).square(), 0.0)
        squared_sum += error.double().sum().item()
        pixel_count += mask.expand_as(video).sum().item()
        n = len(video)
        examples += n
        totals["aux"] += aux.item() * n
        totals["common_clip"] += saturation["common"] * n
        totals["individual_clip"] += saturation["individual"] * n
    mse = squared_sum / pixel_count
    if not math.isfinite(mse):
        raise RuntimeError("评估出现非有限MSE")
    return {"mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12)),
            **{k: v / examples for k, v in totals.items()}}


@torch.no_grad()
def preview(model, dataset, device, path, epoch, split="Validation"):
    """固定展示指定集合首、中、末GOP，避免每轮挑选最好看的样本。"""
    model.eval()
    indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
    fig, axes = plt.subplots(2 * len(indices), 4, figsize=(10, 5 * len(indices)), squeeze=False)
    for j, index in enumerate(indices):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        prediction, _, _ = model(video)
        positions = sample["valid_mask"][0, 0].nonzero()
        top, left = positions.min(0).values.tolist()
        bottom, right = (positions.max(0).values + 1).tolist()
        for row, tensor in enumerate((video, prediction)):
            for t in range(4):
                image = tensor[0, t, :, top:bottom, left:right].cpu().permute(1, 2, 0).numpy()
                ax = axes[2*j+row, t]
                ax.imshow(image.clip(0, 1))
                ax.set_title(f"{'Original' if row == 0 else 'Rounded'} / frame {sample['start_frame']+t}")
                ax.axis("off")
    fig.suptitle(f"{split} GOPs / epoch {epoch} / valid region only")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=10, help="累计目标轮数，不是新增轮数")
    parser.add_argument("--resume", type=Path, help="自己的多GOP检查点last.pt；同时恢复Adam和洗牌状态")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-variant", choices=("legacy", "common_fig7", "common_fig7_stage6_pair"), default="legacy")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/step12_multi_runs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("轮数、批次大小和学习率必须为正")
    if not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        parser.error("辅助权重必须为非负有限数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA不可用")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    train = data.CachedGOPDataset(args.data_dir, "train")
    val = data.CachedGOPDataset(args.data_dir, "val")
    manifest = train.manifest
    if utils.sha256(args.data_dir / "frames_uint8.npy") != manifest["cache_sha256"]:
        raise RuntimeError("视频缓存哈希与清单不符")
    train_frames = {f for r in train.records for f in range(r["start"], r["stop"])}
    val_frames = {f for r in val.records for f in range(r["start"], r["stop"])}
    if not train_frames or not val_frames or train_frames & val_frames:
        raise RuntimeError("数据划分为空或存在帧重叠")
    if args.smoke:
        train, val = Subset(train, range(min(2, len(train)))), Subset(val, range(min(2, len(val))))
    # 单独的采样随机源避免预览/评估改变训练集洗牌顺序。
    generator = torch.Generator().manual_seed(args.seed)
    training = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    train_eval = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed))
    validation = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed))
    # 本实验明确从随机初始化开始，不加载单GOP的拟合权重。
    variants = importlib.import_module("12_model_variants")
    model = variants.build_model(args.model_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, updates = 0, 0
    if args.resume:
        # 仅加载自己产生的检查点；其中包含Python/NumPy随机状态。
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        old = saved["config"]
        if old.get("model_variant", "legacy") != args.model_variant:
            parser.error("检查点模型版本不一致；结构对照必须随机初始化，不能跨版本续训")
        checks = {"batch_size": args.batch_size, "lr": args.lr,
                  "aux_weight": args.aux_weight, "seed": args.seed, "smoke": args.smoke,
                  "manifest_sha256": utils.sha256(args.data_dir / "manifest.json")}
        for key, value in checks.items():
            if old[key] != value:
                parser.error(f"续训配置不一致：{key}，原值={old[key]}，当前={value}")
        # 原实验已经保存mask副本，比较内容以防缓存目录中的mask被替换。
        if not np.array_equal(np.load(args.resume.parent / "valid_mask.npy"),
                              np.load(args.data_dir / "valid_mask.npy")):
            parser.error("有效区域mask与原实验不一致")
        start_epoch, updates = int(saved["epoch"]), int(saved["updates"])
        if args.epochs <= start_epoch:
            parser.error("--epochs必须大于检查点已完成轮数")
        if updates != start_epoch * len(training):
            parser.error("检查点更新次数与完整epoch不一致")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            if len(saved["cuda_rng"]) != torch.cuda.device_count():
                parser.error("续训时可见GPU数量必须与保存时一致")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    start_updates = updates
    run = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(initialization="resume" if args.resume else "random", train_gops=len(train), val_gops=len(val),
                  start_epoch=start_epoch, start_updates=start_updates,
                  parent_checkpoint_sha256=utils.sha256(args.resume) if args.resume else None,
                  packet_loss=0, gradient_clip_max_norm=1.0, quantization_bits=3,
                  variant_initialization="matched legacy effective center weights; unchanged global RNG",
                  quantization_range=[-1, 1], experts=4, top_k=2,
                  parameters=sum(p.numel() for p in model.parameters()),
                  manifest_sha256=utils.sha256(args.data_dir / "manifest.json"),
                  metric="MSE weighted by valid RGB pixels; PSNR from pooled MSE, peak=1")
    utils.write_json(run / "config.json", config)
    if args.resume:
        parent = run / "parent_records"
        parent.mkdir()
        for name in ("config.json", "result.json", "epochs.csv", "updates.csv", "summary.md"):
            source = args.resume.parent / name
            if source.exists():
                shutil.copy2(source, parent / name)
    shutil.copy2(args.data_dir / "manifest.json", run / "data_manifest.json")
    np.save(run / "valid_mask.npy", np.load(args.data_dir / "valid_mask.npy"))
    utils.write_json(run / "environment.json", {"python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {d.metadata["Name"]: d.version for d in utils.importlib.metadata.distributions() if d.metadata["Name"]}})
    (run / "architecture.txt").write_text(str(model), encoding="utf-8")
    source_hashes = {}
    for folder in ("step", "practice", "Biran", "packet"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            relative = source.relative_to(ROOT)
            destination = run / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_hashes[str(relative)] = utils.sha256(source)
    utils.write_json(run / "source_manifest.json", source_hashes)
    print(f"实验目录：{run}\n训练GOP={len(train)}，验证GOP={len(val)}；初始化={config['initialization']}", flush=True)
    print(f"累计轮数：{start_epoch} → {args.epochs}；起点更新次数={updates}", flush=True)
    print(f"模型版本：{args.model_variant}；参数量：{config['parameters']}", flush=True)
    # best限定为本次运行（包含恢复起点），历史最佳仍保留在原实验目录。
    best, best_epoch = float("inf"), start_epoch
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(name, epoch, metrics):
        temp = run / (name + ".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "updates": updates, "config": config, "metrics": metrics,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, temp)
        temp.replace(run / name)

    history = []
    with (run / "epochs.csv").open("w", newline="", encoding="utf-8") as ef, \
         (run / "updates.csv").open("w", newline="", encoding="utf-8") as uf:
        update_fields = ["epoch", "update", "start_frames", "train_mse_before_update", "aux", "total", "grad_norm", "common_clip", "individual_clip"]
        uw = csv.DictWriter(uf, fieldnames=update_fields)
        uw.writeheader()
        ew = None
        # 起始轮只评估恢复状态，不重复训练该轮；之后完整遍历训练集。
        for epoch in range(start_epoch, args.epochs + 1):
            if epoch > start_epoch:
                model.train()
                for batch in training:
                    video, mask = batch["video"].to(device), batch["valid_mask"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction, aux, saturation = model(video)
                    mse = base.masked_mse(prediction, video, mask)
                    loss = mse + args.aux_weight * aux
                    if not torch.isfinite(loss):
                        raise RuntimeError("训练损失非有限")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    uw.writerow(dict(epoch=epoch, update=updates, start_frames=json.dumps(batch["start_frame"].tolist()),
                        train_mse_before_update=mse.item(), aux=aux.item(), total=loss.item(), grad_norm=norm.item(),
                        common_clip=saturation["common"], individual_clip=saturation["individual"]))
                    uf.flush()
                    del prediction, aux, mse, loss
                    if updates % 25 == 0:
                        print(f"epoch={epoch}/{args.epochs}，已更新{updates}次", flush=True)
            tr = evaluate(model, train_eval, device)
            va = evaluate(model, validation, device)
            row = {"epoch": epoch, "updates": updates, "elapsed_seconds": time.perf_counter()-started,
                   **{"train_rounded_"+k: v for k,v in tr.items()}, **{"val_rounded_"+k: v for k,v in va.items()}}
            if ew is None:
                ew = csv.DictWriter(ef, fieldnames=list(row))
                ew.writeheader()
            ew.writerow(row)
            ef.flush()
            history.append(row)
            if va["mse"] < best:
                best, best_epoch = va["mse"], epoch
                save_checkpoint("best.pt", epoch, row)
            save_checkpoint("last.pt", epoch, row)
            preview(model, val, device, run / f"val_epoch_{epoch:03d}.png", epoch)
            preview(model, train, device, run / f"train_epoch_{epoch:03d}.png", epoch, "Training")
            print(f"epoch={epoch} | 训练舍入MSE={tr['mse']:.6f} | 验证舍入MSE={va['mse']:.6f} | 验证PSNR={va['psnr_db']:.2f} dB", flush=True)
            # 每轮更新图和摘要，即使长任务中断也保留已完成的进度。
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for split in ("train", "val"):
                for ax, metric in zip(axes, ("mse", "psnr_db")):
                    ax.plot([r["epoch"] for r in history], [r[f"{split}_rounded_{metric}"] for r in history], "o-", label=split)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Rounded " + metric)
                    ax.legend()
                    ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(run / "curves.png", dpi=140)
            plt.close(fig)
            report = {"status": "completed" if epoch == args.epochs else "running", "smoke": args.smoke,
                      "epoch": epoch, "optimizer_updates": updates, "initial": history[0], "final": row,
                      "start_epoch": start_epoch, "updates_this_run": updates-start_updates,
                      "best_scope": "current_run_including_resume_start",
                      "best_val_mse": best, "best_epoch": best_epoch,
                      "elapsed_seconds": time.perf_counter()-started,
                      "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else None,
                      "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2 if device.type == "cuda" else None}
            utils.write_json(run / "result.json", report)
            (run / "summary.md").write_text(
                f"# 多GOP无丢包学习实验\n\n短检查模式：{args.smoke}；初始化={config['initialization']}；已完成{epoch}轮、{updates}次更新。\n"
                f"本次起点第{start_epoch}轮，新增{updates-start_updates}次更新。曲线和最佳模型范围为本次运行（含恢复起点）。\n"
                f"模型版本：{args.model_variant}；参数量：{config['parameters']}。common_fig7仅更改公共残差为Conv2d-GELU-Conv2d-GELU；初始权重取原随机3D卷积中心切片，其余参数和随机状态保持一致。\n"
                "common_fig7_stage6_pair在common_fig7上为个体提取/上采样各加一个独立MoE块；Stage6单窗口不移位，多窗口时移位。MoE辅助损失按12块平均。N按块对解释仍属实现假设。\n"
                f"训练{len(train)}个GOP；验证{len(val)}个GOP。最佳验证MSE={best:.8f}，出现在第{best_epoch}轮。\n\n"
                "每轮对训练与验证集合均使用舍入量化，以所有有效RGB像素汇总MSE，再换算PSNR。"
                "updates.csv记录加噪量化训练时更新前指标；epochs.csv记录轮末舍入评估。\n"
                "验证集参与模型选择，不能作为独立测试集。同一视频的时间隔离不消除场景相关性。\n"
                "预处理采用全画面缩放补边；量化范围、MoE配置等为学习配置，尚非论文完整复现。"
                "本实验不包含丢包、恢复网络、MS-SSIM或实际比特率评估。\n"
                "mask仅排除预处理新增补边；原视频内部的黑边仍参与指标计算，保持与基线一致。\n"
                "每轮保存固定训练/验证样本首、中、末GOP对比图。\n"
                "检查点保存Adam、采样器及随机状态；--resume恢复，--epochs指定累计目标轮数。\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"记录保存完成：{run}", flush=True)


if __name__ == "__main__":
    main()
