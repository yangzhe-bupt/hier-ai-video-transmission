"""Step12准备：无丢包完整模型的前向/反向检查，不执行参数更新。

这是当前学习配置：4专家、top-2、窗口(4,4,4)、固定[-1,1]量化。
论文精确复现仍需对齐块数、路由、量化范围和传输通道等参数。
完整画面letterbox与有效区域损失是本次视频学习实验的预处理选择。
"""
import argparse
import importlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

step05 = importlib.import_module("05_video_dataset")
step06 = importlib.import_module("06_spatial_codec")
step07 = importlib.import_module("07_spatiotemporal_codec")
step08 = importlib.import_module("08_feature_decomposition")
step09 = importlib.import_module("09_quantization_channel")
step11 = importlib.import_module("11_moe_swin")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def masked_mse(prediction, target, valid_mask):
    """只统计有效画面；按全部有效帧、通道、像素的数量取平均。"""
    if prediction.shape != target.shape:
        raise ValueError("预测和目标形状不同")
    if valid_mask.dtype != torch.bool:
        raise TypeError("有效区域mask必须为bool")
    weights = valid_mask.expand_as(target).to(target.dtype)
    count = weights.sum()
    if count.item() == 0:
        raise ValueError("有效画面不能为空")
    # 使用where显式选取有效位置，避免补边进入损失与分母。
    squared_error = (prediction - target).square()
    selected = torch.where(weights.bool(), squared_error, 0.0)
    return selected.sum() / count


class Stage1Model(nn.Module):
    """无丢包视频重建模型：两分支量化后直接送达接收端。

    无丢包时，交织/逆交织的组合为恒等映射，恢复模块的mask为零。
    当前检查省略这些恒等操作，阶段II再加入实际擦除和恢复。
    """
    def __init__(self):
        super().__init__()
        self.spatial_encoder = step06.SpatialEncoder()
        self.temporal_encoder = step07.SpatiotemporalEncoder()
        self.common_extractor = step08.CommonFeatureExtractor(120, 4)
        self.individual_extractor = step08.IndividualFeatureExtractor()
        self.individual_upsample = step08.IndividualUpsample()
        self.temporal_decoder = step07.SpatiotemporalDecoder()
        self.spatial_decoder = step06.SpatialDecoder()
        self.common_quantizer = step09.UniformQuantizer(3, -1.0, 1.0)
        self.individual_quantizer = step09.UniformQuantizer(3, -1.0, 1.0)

        # 在构造函数内完成替换，后续优化器才会持有全部专家参数。
        # 两个主时空模块各4块，个体提取/上采样各1块，共10块。
        self.moe_counts = tuple(
            step11.replace_mlp_with_moe(module, num_experts=4, top_k=2)
            for module in (
                self.temporal_encoder, self.individual_extractor,
                self.individual_upsample, self.temporal_decoder,
            )
        )
        if self.moe_counts != (4, 1, 1, 4):
            raise ValueError(f"MoE块数量与当前配置不符：{self.moe_counts}")

    def forward(self, video):
        if video.ndim != 5 or video.shape[1:3] != (4, 3):
            raise ValueError("视频必须为[B,4,3,H,W]")
        batch, frames, channels, height, width = video.shape
        # 当前固定窗口及个体分支多次下采样，先支持256的正整数倍。
        if height < 256 or width < 256 or height % 256 or width % 256:
            raise ValueError("当前模型要求高、宽均为256的正整数倍")

        # 1. 逐帧空间编码，再恢复时间维。
        spatial = self.spatial_encoder(video.reshape(batch * frames, channels, height, width))
        _, cs, hs, ws = spatial.shape
        spatial = spatial.reshape(batch, frames, cs, hs, ws)

        # 2. 时空编码及公共/个体分解。
        temporal, enc_aux = self.temporal_encoder(spatial, return_aux=True)
        common = self.common_extractor(temporal)
        individual, extract_aux = self.individual_extractor(temporal - common, return_aux=True)

        # 3. 量化器随整个模型切换train/eval：训练加噪，评估取整。
        common_q = self.common_quantizer(common)
        individual_q = self.individual_quantizer(individual)

        # 4. 无丢包接收端：个体上采样、相加、时空与空间解码。
        expanded, upsample_aux = self.individual_upsample(individual_q, return_aux=True)
        decoded, dec_aux = self.temporal_decoder(common_q + expanded, return_aux=True)
        decoded = decoded.reshape(batch * frames, cs, hs, ws)
        reconstructed = self.spatial_decoder(decoded)
        reconstructed = reconstructed.reshape(batch, frames, 3, height, width)

        # 两个主模块返回块均值；个体模块各只有1块。
        # 按实际块数加权，得到全部10块的平均辅助损失。
        auxiliary_terms = (enc_aux, extract_aux, upsample_aux, dec_aux)
        auxiliary_loss = sum(
            term * count for term, count in zip(auxiliary_terms, self.moe_counts)
        ) / sum(self.moe_counts)

        # 只为日志统计裁剪比例，避免误把损失变成无梯度数值。
        with torch.no_grad():
            saturation = {
                "common": ((common < -1) | (common > 1)).float().mean().item(),
                "individual": ((individual < -1) | (individual > 1)).float().mean().item(),
            }
        return reconstructed, auxiliary_loss, saturation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "data/videos/new_year.mp4")
    parser.add_argument("--start-frame", type=int, default=239)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backward", action="store_true", help="检查反向传播；不执行optimizer.step")
    parser.add_argument("--aux-weight", type=float, default=0.01, help="检查用系数，待核实论文设定")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA不可用")
    if args.aux_weight < 0:
        parser.error("辅助损失系数不能为负")

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    frames = step05.read_frames(args.video, args.start_frame, 4)
    video, valid_mask, info = step05.frames_to_full_gop(frames, 256)
    video, valid_mask = video.to(device), valid_mask.to(device)
    model = Stage1Model().to(device)
    model.train(args.backward)

    # 验证补边损失：只修改补边，不应改变有效区域MSE。
    with torch.no_grad():
        pad_only = torch.where(valid_mask, video, torch.ones_like(video))
        assert masked_mse(pad_only, video, valid_mask).item() == 0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = torch.enable_grad() if args.backward else torch.no_grad()
    with context:
        prediction, auxiliary_loss, saturation = model(video)
        reconstruction_loss = masked_mse(prediction, video, valid_mask)
        total_loss = reconstruction_loss + args.aux_weight * auxiliary_loss
        assert prediction.shape == video.shape and torch.isfinite(prediction).all()
        assert torch.isfinite(total_loss)

        if args.backward:
            # 辅助项单独传到全部10个路由器，检查汇总没有断图。
            routers = [m for m in model.modules() if isinstance(m, step11.TopKRouter)]
            assert len(routers) == 10
            auxiliary_grads = torch.autograd.grad(
                auxiliary_loss, [m.gate.weight for m in routers], retain_graph=True
            )
            assert all(torch.isfinite(g).all() for g in auxiliary_grads)
            del auxiliary_grads
            total_loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    # 未被选中的专家可能没有任务梯度。
                    if ".experts." in name:
                        continue
                    raise RuntimeError(f"{name}没有梯度")
                assert torch.isfinite(parameter.grad).all(), name

    if device.type == "cuda":
        torch.cuda.synchronize()
    report = {
        "device": str(device), "torch": torch.__version__,
        "mode": "train_noise_backward" if args.backward else "eval_rounding_forward",
        "moe_counts": model.moe_counts,
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": list(video.shape), "output_shape": list(prediction.shape),
        "valid_mse": reconstruction_loss.item(), "auxiliary_loss": auxiliary_loss.item(),
        "total_loss": total_loss.item(), "saturation": saturation,
        "elapsed_seconds": time.perf_counter() - started, "preprocessing": info,
        "optimizer_updates": 0,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else None,
    }
    output_dir = PROJECT_ROOT / "outputs/step12_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "backward" if args.backward else "forward"
    (output_dir / f"{suffix}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 4, figsize=(10, 6))
    for row, tensor in enumerate((video, prediction)):
        for frame in range(4):
            image = tensor[0, frame].detach().cpu().permute(1, 2, 0).numpy()
            axes[row, frame].imshow(image.clip(0, 1))
            axes[row, frame].axis("off")
            axes[row, frame].set_title(f"{'Input' if row == 0 else 'Untrained'} {args.start_frame + frame}")
    figure.tight_layout()
    figure.savefig(output_dir / f"{suffix}_preview.png", dpi=120)
    plt.close(figure)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("完整模型反向检查通过（参数更新0次）" if args.backward else "完整模型前向检查通过")


if __name__ == "__main__":
    main()
