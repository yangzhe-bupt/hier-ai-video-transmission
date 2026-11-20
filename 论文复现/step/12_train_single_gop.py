"""Step12：无丢包、固定四帧的拟合实验；不是独立测试集评估。

运行即更新参数。默认200步，Adam学习率1e-4，MoE权重0.01。
这些是学习实验设置，不宣称与论文完整配置一致。
"""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
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

base = importlib.import_module("12_train_stage1")
plt = base.plt
ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    """分块读取视频/代码，记录内容身份，不复制整段视频。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preview(path, video, before, after, start_frame, step, mask, initial_step):
    # 展示时统一遮黑无效补边；不修改参与训练的张量或指标。
    valid = mask.detach().cpu()
    tensors = [torch.where(valid, x.detach().cpu(), 0.0) for x in (video, before, after)]
    _preview(path, tensors, start_frame, step, initial_step)
    # 单独输出有效区域放大图，不对画面拉伸，便于观察人物和网格纹理。
    positions = valid[0, 0, 0].nonzero()
    top, left = positions.min(dim=0).values.tolist()
    bottom, right = (positions.max(dim=0).values + 1).tolist()
    cropped = [x[..., top:bottom, left:right] for x in tensors]
    _preview(path.with_name(path.stem + "_valid.png"), cropped, start_frame, step, initial_step)
    # 与空间诊断相同：固定使用有效画面的下半区域观察细节。
    detail = [x[..., top + (bottom-top)//2:bottom, left:right] for x in tensors]
    _preview(path.with_name(path.stem + "_detail.png"), detail, start_frame, step, initial_step)


def _preview(path, tensors, start_frame, step, initial_step):
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for row, (label, tensor) in enumerate((
        ("Original", tensors[0]), (f"Start step {initial_step} / rounded", tensors[1]),
        (f"Step {step} / rounded", tensors[2]),
    )):
        for t in range(4):
            axes[row, t].imshow(tensor[0, t].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1))
            axes[row, t].set_title(f"{label}\nFrame {start_frame + t}")
            axes[row, t].axis("off")
    fig.suptitle("Single training GOP, no packet loss; black padding excluded from metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, video, mask):
    """评估必须用eval：量化器真正舍入，且不构建梯度图。"""
    model.eval()
    prediction, auxiliary, saturation = model(video)
    mse = base.masked_mse(prediction, video, mask).item()
    if not math.isfinite(mse):
        raise RuntimeError("评估MSE非有限值")
    # RGB浮点值范围[0,1]，峰值为1；整个GOP有效像素统一计算MSE。
    psnr = -10 * math.log10(max(mse, 1e-12))
    # 该区域仅用于评估，不增加训练损失；与空间模型保持相同定义。
    detail_mask = mask.clone()
    positions = mask[0,0,0].nonzero()
    top = positions[:,0].min().item()
    bottom = positions[:,0].max().item()+1
    detail_mask[..., :top+(bottom-top)//2, :] = False
    detail_mse = base.masked_mse(prediction, video, detail_mask).item()
    return prediction.detach().cpu(), {
        "eval_mse": mse, "eval_psnr_db": psnr,
        "detail_mse": detail_mse, "detail_psnr_db": -10*math.log10(max(detail_mse,1e-12)),
        "eval_aux": auxiliary.item(),
        "eval_common_clip": saturation["common"],
        "eval_individual_clip": saturation["individual"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/videos/new_year.mp4")
    parser.add_argument("--start-frame", type=int, default=239)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=200, help="累计目标更新数，续训时不是新增步数")
    parser.add_argument("--resume", type=Path, help="从自己保存的last.pt恢复模型、Adam及随机状态")
    parser.add_argument("--model-variant", choices=("legacy", "common_fig7", "common_fig7_stage6_pair"), default="legacy")
    parser.add_argument("--reference-gop", type=Path, help="可选：空间实验input_gop.pt，检查实际输入及mask完全一致")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/step12_runs")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("步数、评估间隔和学习率必须为正")
    if not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        parser.error("辅助损失权重必须为非负有限数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA不可用")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    # 固定种子便于对比，但CUDA稀疏聚合等操作不保证逐位确定性。
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    frames = base.step05.read_frames(args.video, args.start_frame, 4)
    video, mask, preprocessing = base.step05.frames_to_full_gop(frames, 256)
    video, mask = video.to(device), mask.to(device)
    if args.reference_gop:
        reference = torch.load(args.reference_gop, map_location='cpu', weights_only=False)
        if not torch.equal(reference['video'], video.cpu()) or not torch.equal(reference['valid_mask'], mask.cpu()):
            parser.error('与空间实验的输入或mask不一致')
        del reference
    variants = importlib.import_module('12_model_variants')
    model = variants.build_model(args.model_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    video_hash = sha256(args.video)
    if args.resume:
        # 检查点含numpy/Python随机状态，需加载完整对象；仅用于自己的检查点。
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        old = saved["config"]
        if old.get('model_variant','legacy') != args.model_variant:
            parser.error('模型版本不同，不能跨版本续训')
        for key, current in (("video_sha256", video_hash), ("start_frame", args.start_frame),
                             ("lr", args.lr), ("aux_weight", args.aux_weight), ("seed", args.seed)):
            if old[key] != current:
                parser.error(f"续训配置不一致：{key}，原值={old[key]}，当前={current}")
        start_step = int(saved["step"])
        if args.steps <= start_step:
            parser.error("--steps必须大于检查点累计更新数")
        # 对比实际输入而不只是文件名，避免换帧或预处理后仍声称连续训练。
        saved_input = torch.load(args.resume.parent / "input_gop.pt", map_location="cpu", weights_only=False)
        if not torch.equal(saved_input["video"], video.cpu()) or not torch.equal(saved_input["valid_mask"], mask.cpu()):
            parser.error("当前GOP或有效区域与原实验不一致")
        model.load_state_dict(saved["model"], strict=True)
        # PyTorch会将参数对应的Adam状态迁移到参数所在设备。
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            if len(saved["cuda_rng"]) != torch.cuda.device_count():
                parser.error("续训时可见GPU数量必须与保存检查点时一致")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved, saved_input

    # 微秒时间戳及exist_ok=False防止无意覆盖以前的实验。
    run = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"experiment": "single_training_GOP_overfit", "packet_loss": 0,
                   "preprocessing": preprocessing, "valid_ratio": mask.float().mean().item(),
                   "video_sha256": video_hash, "optimizer": "Adam",
                   "detail_region": "lower half of valid image, same as spatial probe",
                   "start_step": start_step,
                   "parent_checkpoint_sha256": sha256(args.resume) if args.resume else None,
                   "gradient_clip_max_norm": 1.0, "quantization_bits": 3,
                   "quantization_range": [-1, 1], "experts": 4, "top_k": 2,
                   "moe_counts": model.moe_counts, "parameters": sum(p.numel() for p in model.parameters()),
                   "metric": "MSE over all valid RGB pixels in the same training GOP; PSNR peak=1"})
    write_json(run / "config.json", config)
    if args.resume:
        # 保留上一阶段记录，形成实验溯源链；不复制大权重及代码的重复副本。
        parent_dir = run / "parent_records"
        parent_dir.mkdir()
        for name in ("config.json", "result.json", "metrics.csv", "summary.md"):
            source = args.resume.parent / name
            if source.exists():
                shutil.copy2(source, parent_dir / name)
    (run / "architecture.txt").write_text(str(model), encoding="utf-8")
    write_json(run / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        "bitwise_deterministic": False,
    })
    # 保存实际依赖目录中的Python源文件，排除模型输出和大视频。
    manifest = {}
    for folder in ("step", "practice", "Biran", "packet"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            relative = source.relative_to(ROOT)
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest[str(relative)] = sha256(source)
    shutil.copy2(Path(base.step05.__file__), run / 'source/step/05_video_dataset.py')
    manifest['step/05_video_dataset.py'] = sha256(Path(base.step05.__file__))
    write_json(run / "source_manifest.json", manifest)
    # 连同此次输入保存，避免之后帧解码或预处理变更使结果难以追溯。
    torch.save({"video": video.cpu(), "valid_mask": mask.cpu()}, run / "input_gop.pt")
    previous_check = ROOT / "outputs/step12_check/backward_report.json"
    if previous_check.exists():
        shutil.copy2(previous_check, run / "prior_backward_check.json")

    def checkpoint(name, step, metrics):
        """保留模型、Adam状态和随机状态，可通过--resume继续训练。"""
        temporary = run / (name + ".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "metrics": metrics, "config": config,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, temporary)
        temporary.replace(run / name)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    before, initial = evaluate(model, video, mask)
    best = initial["eval_mse"]
    best_step = start_step
    checkpoint("best.pt", start_step, initial)
    preview(run / f"reconstruction_{start_step:04d}.png", video, before, before, args.start_frame, start_step, mask, start_step)
    print(f"实验目录：{run}", flush=True)
    print(f"模型版本：{args.model_variant}；参数量：{config['parameters']}；输入对照检查={bool(args.reference_gop)}", flush=True)
    print(f"训练前舍入量化 MSE={best:.6f}，PSNR={initial['eval_psnr_db']:.2f} dB", flush=True)
    print(f"累计更新：{start_step} → {args.steps}；本次新增{args.steps - start_step}次", flush=True)

    fields = ["step", "elapsed_seconds", "train_mse_before_update", "train_aux_before_update",
              "train_total_before_update", "grad_norm_before_clip", "train_common_clip",
              "train_individual_clip", *initial.keys()]
    rows = []
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        row = {"step": start_step, "elapsed_seconds": time.perf_counter() - started, **initial}
        writer.writerow(row)
        stream.flush()
        rows.append(row)
        for step in range(start_step + 1, args.steps + 1):
            model.train()  # 恢复加噪量化；evaluate调用会切换为eval。
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary, saturation = model(video)
            mse = base.masked_mse(prediction, video, mask)
            loss = mse + args.aux_weight * auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError(f"第{step}步损失非有限，已停止更新")
            loss.backward()
            # 抑制偶发的大梯度，同时记录裁剪前范数；非有限梯度直接报错。
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "train_mse_before_update": mse.item(),
                   "train_aux_before_update": auxiliary.item(), "train_total_before_update": loss.item(),
                   "grad_norm_before_clip": norm.item(),
                   "train_common_clip": saturation["common"],
                   "train_individual_clip": saturation["individual"]}
            del prediction, auxiliary, mse, loss
            if step == start_step + 1 or step % args.eval_every == 0 or step == args.steps:
                after, metrics = evaluate(model, video, mask)
                row.update(metrics)
                if metrics["eval_mse"] < best:
                    best, best_step = metrics["eval_mse"], step
                    checkpoint("best.pt", step, metrics)
                checkpoint("last.pt", step, metrics)
                preview(run / f"reconstruction_{step:04d}.png", video, before, after, args.start_frame, step, mask, start_step)
                print(f"更新{step:4d}次 | 训练MSE(更新前)={row['train_mse_before_update']:.6f} | "
                      f"舍入MSE(更新后)={metrics['eval_mse']:.6f} | PSNR={metrics['eval_psnr_db']:.2f} dB", flush=True)
            row["elapsed_seconds"] = time.perf_counter() - started
            writer.writerow(row)
            stream.flush()  # SSH中断时，已经完成的指标仍留在磁盘。
            rows.append(row)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    train_rows, eval_rows = rows[1:], [r for r in rows if "eval_mse" in r]
    axes[0].plot([r["step"] for r in train_rows], [r["train_mse_before_update"] for r in train_rows], label="Train noise / before update")
    axes[0].plot([r["step"] for r in eval_rows], [r["eval_mse"] for r in eval_rows], "o-", label="Same GOP rounded / after update")
    axes[0].set_ylabel("Valid-region MSE")
    axes[1].plot([r["step"] for r in train_rows], [r["train_aux_before_update"] for r in train_rows], label="MoE auxiliary")
    axes[2].plot([r["step"] for r in eval_rows], [r["eval_psnr_db"] for r in eval_rows], "o-", label="Rounded PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=150)
    plt.close(fig)
    report = {"status": "completed", "optimizer_updates": args.steps, "initial": initial,
              "start_step": start_step, "updates_this_run": args.steps - start_step,
              "final": metrics, "best_eval_mse": best, "best_step": best_step,
              "elapsed_seconds": time.perf_counter() - started,
              "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
              "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else None}
    write_json(run / "result.json", report)
    (run / "summary.md").write_text(
        f"# 固定GOP无丢包拟合实验\n\n目的：检验完整编解码链路是否可学习。\n\n"
        f"- 累计更新次数：{args.steps}\n- 本次起点：{start_step}\n- 本次新增更新：{args.steps - start_step}\n- 本次起点舍入MSE：{initial['eval_mse']:.8f}\n"
        f"- 最后舍入MSE：{metrics['eval_mse']:.8f}\n- 最佳舍入MSE：{best:.8f}（第{best_step}次更新）\n"
        f"- 最后PSNR：{metrics['eval_psnr_db']:.3f} dB\n\n"
        f"模型版本：{args.model_variant}。下半有效区域用于detail指标，不额外参与训练损失。\n"
        "所有评估均为同一个训练GOP，不代表泛化效果或论文复现指标。仅新增补边不计入MSE/PSNR。\n"
        "使用完整画面等比缩放补边，而非论文训练随机裁剪；固定3-bit [-1,1]量化范围、"
        "MoE设置及损失权重为学习配置，C=96对应位置及通信预算等细节尚未确认。\n"
        "本实验无丢包，不测试恢复网络；未测MS-SSIM、真实比特率或通信性能。\n"
        "best.pt仅在本次运行（含恢复起点）内按训练GOP舍入MSE选择，不是独立验证集最优模型。"
        "last.pt保留最近评估点，可通过--resume续训；--steps指定累计目标更新数。\n"
        "重建图统一遮黑无效补边，_valid.png为有效区域放大图。续训图中间一行为本次恢复起点。\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"训练与记录保存完成：{run}", flush=True)


if __name__ == "__main__":
    main()
