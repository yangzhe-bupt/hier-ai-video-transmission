# ======================================================================
# 文件：evaluate_packet_loss_baseline.py
# 位置：practice/experiments
# 功能：在同一组 int8 包上比较无丢失与丢包后的置零重建质量。
# 函数汇总：
#   - evaluate_gop：编码、分包、回放、审计和解码，返回指标及包事件。
#   - summarize_run：统计包丢失率、GOP 失败率和配对重建误差。
#   - main：读取配置、加载模型、遍历指定片段并保存 CSV 和 JSON。
# 输入：视频、固定模型权重、GOP 起始帧、丢失概率及随机种子。
# 输出：gop_metrics.csv、events.json、summary.json。
# 运行：python practice/run.py experiments/evaluate_packet_loss_baseline.py
# 可选：--recover-residual 在同一接收数据上追加时间零和恢复对照。
# 可选：--compare-joint-recovery 对公共缺块 GOP 评测四种接收端策略。
# 可选：--recover-public-spatial 比较公共分支空间补全与置零。
# 可选：--allow-public-zero-fill 追加公共分支缺失时的置零降级对照。
# ======================================================================
"""在相同 int8 数据包上比较无丢失和丢包置零重建。"""

import argparse
import csv
import json
from pathlib import Path
from dataclasses import replace

import torch

from accept_biran_swin import file_sha256, mse, restore_quantized
from hier_video.transmission.feature_decomposition import FeatureParts, PublicResidualDecomposer
from hier_video.transmission.feature_mapping import build_group_manifest
from hier_video.transmission.network_replay import (
    FeaturePacket,
    LinkSample,
    replay_feature_packet_events,
)
from hier_video.transmission.packetization import build_feature_packets, unpack_feature_packets
from hier_video.transmission.quantization import quantize_symmetric
from hier_video.transmission.reception_audit import audit_quantized_reception
from recover_individual_residual import recover_individual_residual
from recover_public_spatial import recover_public_spatial
from train_swin_video import load_swin_models
from video_feature_demo import load_video_as_gop


@torch.inference_mode()
def evaluate_gop(encoder, decoder, gop, gop_id, start_frame, seed, loss,
                 recover_residual=False, allow_public_zero_fill=False,
                 recover_public=False, compare_joint=False):
    """
    对同一组编码数据分别模拟无丢失和指定丢失率的传输。

    参数：
        encoder、decoder：已加载且处于评测模式的模型。
        gop：原始 RGB 张量 [1,T,3,H,W]，与模型处在同一设备。
        gop_id、start_frame：实验中的 GOP 编号及源视频起始帧。
        seed：可重复的包级丢失种子。
        loss：模拟链路的目标丢失概率，取值为 [0,1]。
        recover_residual：在丢包条件下追加残差恢复与置零的配对比较。
        compare_joint：对可降级解码的公共缺块 GOP 做四组配对消融。
        recover_public：在公共置零样本上追加空间均值补全。
        allow_public_zero_fill：量化参数齐全时，测量公共缺块的置零降级质量。
    返回：
        指标字典和两条链路的事件记录。失败时丢包 MSE 留空。
    """
    device = gop.device
    # 同一 GOP 只编码、量化和分包一次，确保两种条件的发送数据相同。
    decomposer = PublicResidualDecomposer()
    parts = decomposer.encode(encoder(gop))
    public = quantize_symmetric(parts.public)
    individual = quantize_symmetric(parts.individual)
    manifest = build_group_manifest(public.tensor.shape, individual.tensor.shape)
    packets = build_feature_packets(
        public.tensor, individual.tensor, manifest,
        stream_id="acceptance-int8", sequence_id=gop_id, gop_id=gop_id,
        gop_start_frame=start_frame,
        quantization_by_branch={"public": public.metadata, "individual": individual.metadata},
    )
    wire = [packet.serialize() for packet in packets]
    # 复用验收的包 ID 和轨迹 ID，便于核对同一种子下的事件。
    replay_packets = tuple(
        FeaturePacket(f"{gop_id}:{packet.header.branch}:{packet.header.group_id}",
                      f"{gop_id}:{packet.header.branch}", len(raw))
        for packet, raw in zip(packets, wire)
    )
    packet_ids = tuple(packet.packet_id for packet in replay_packets)
    events = {}
    measurements = {}
    recovered_mse = None
    missing_elements = None
    public_fallback_mse = None
    # 联合消融关注公共分支不完整、量化参数仍可用的样本。
    recover_public = recover_public or compare_joint
    public_spatial_mse = None
    joint_mse = None
    residual_only_mse = None
    fallback_individual_missing = None
    allow_public_zero_fill = allow_public_zero_fill or recover_public
    missing_public_elements = None
    for label, fraction in (("no_loss", 0.0), ("packet_loss", loss)):
        report = replay_feature_packet_events(
            replay_packets, packet_ids, trace_id=f"acceptance-gop-{gop_id}",
            seed=seed, decision_time_ms=0, deadline_ms=1000,
            budget_bytes=sum(map(len, wire)), link=LinkSample(100_000_000, fraction),
            feature_interface_version=manifest.interface_version,
        )
        events[label] = report
        # 只有按期收到的包进入接收端，保留发送端的原始包顺序。
        delivered = {row["packet_id"] for row in report["events"] if row["status"] == "received"}
        received_wire = [raw for key, raw in zip(packet_ids, wire) if key in delivered]
        quality = None
        status = "failed_no_received_packets"
        if received_wire:
            received = unpack_feature_packets(received_wire, manifest, device=device)
            audit = audit_quantized_reception(received, manifest)
            status = audit.status
            if (allow_public_zero_fill and label == "packet_loss"
                    and status == "failed_public_incomplete"):
                missing_public_elements = int(audit.public_missing_mask.sum().item())
                # 缺块已由拆包器填零。反量化只使用接收包携带的比例。
                # 原审计状态仍用于基线，放行副本仅用于本次降级重建。
                if {"public", "individual"} <= received.quantization_by_branch.keys():
                    fallback_audit = replace(audit, decoder_ready=True)
                    fallback_features, fallback_public, fallback_individual = restore_quantized(
                        received, fallback_audit, parts, decomposer, device,
                    )
                    public_fallback_mse = mse(decoder(fallback_features), gop)
                    if recover_public:
                        # 同一个接收结果只替换公共分支缺失位置。
                        # 个体残差沿用置零对照，隔离公共补全的效果。
                        public_mask = audit.public_missing_mask.to(fallback_public.device)
                        restored_public = recover_public_spatial(fallback_public, public_mask)
                        restored = decomposer.decode(
                            FeatureParts(restored_public, fallback_individual, parts.meta),
                        )
                        public_spatial_mse = mse(decoder(restored), gop)
                        if compare_joint:
                            # 只读取接收侧残差与掩码，两种恢复共享同一接收结果。
                            residual_mask = audit.individual_missing_mask
                            residual_mask = residual_mask.to(fallback_individual.device)
                            fallback_individual_missing = int(residual_mask.sum().item())
                            if fallback_individual_missing:
                                restored_individual = recover_individual_residual(
                                    fallback_individual, residual_mask,
                                )
                                residual_features = decomposer.decode(FeatureParts(
                                    fallback_public, restored_individual, parts.meta,
                                ))
                                joint_features = decomposer.decode(FeatureParts(
                                    restored_public, restored_individual, parts.meta,
                                ))
                                residual_only_mse = mse(decoder(residual_features), gop)
                                joint_mse = mse(decoder(joint_features), gop)
                            else:
                                # 个体分支完整时，两组配对各自使用同一重建。
                                residual_only_mse = public_fallback_mse
                                joint_mse = public_spatial_mse
            # 接收端已处理缺失分片。只对可解码的 GOP 计算像素误差。
            if audit.decoder_ready:
                features, public_rx, individual_rx = restore_quantized(
                    received, audit, parts, decomposer, device,
                )
                quality = mse(decoder(features), gop)
                if recover_residual and label == "packet_loss":
                    # 恢复和置零共享同一次接收，以及同一个公共分支。
                    # 缺失掩码来自个体分支审计，不能通过像素或特征值是否为零来推断。
                    mask = audit.individual_missing_mask.to(individual_rx.device)
                    missing_elements = int(mask.sum().item())
                    if missing_elements:
                        recovered = recover_individual_residual(individual_rx, mask)
                        restored = decomposer.decode(FeatureParts(public_rx, recovered, parts.meta))
                        recovered_mse = mse(decoder(restored), gop)
                    else:
                        recovered_mse = quality
        measurements[label] = (status, quality)

    clean_status, clean_mse = measurements["no_loss"]
    if clean_status != "complete" or clean_mse is None:
        raise RuntimeError(f"GOP {gop_id} 无丢失对照未完整接收：{clean_status}")
    # 差值使用同一个 GOP 的无丢失 int8 MSE，隔离量化本身的影响。
    status, lossy_mse = measurements["packet_loss"]
    report = events["packet_loss"]
    row = {
        "gop_id": gop_id, "start_frame": start_frame, "seed": seed,
        "target_loss_fraction": loss, "actual_loss_fraction": report["realized_loss_fraction"],
        "sent_bytes": report["actual_wire_bytes"], "receiver_status": status,
        "decoder_ready": lossy_mse is not None, "no_loss_int8_mse": clean_mse,
        "packet_loss_mse": lossy_mse,
        "delta_mse": None if lossy_mse is None else lossy_mse - clean_mse,
    }
    if recover_residual:
        row.update({
            "missing_individual_elements": missing_elements,
            "recovered_mse": recovered_mse,
            # 正值表示恢复后的 MSE 低于置零基线。
            "mse_reduction_vs_zero_fill": (
                None if recovered_mse is None else lossy_mse - recovered_mse
            ),
        })
    if allow_public_zero_fill:
        row.update({
            "missing_public_elements": missing_public_elements,
            "public_zero_fill_mse": public_fallback_mse,
            "public_zero_fill_delta_mse": (
                None if public_fallback_mse is None else public_fallback_mse - clean_mse
            ),
        })
    if recover_public:
        row.update({
            "public_spatial_mse": public_spatial_mse,
            "public_spatial_mse_reduction": (
                None if public_spatial_mse is None else public_fallback_mse - public_spatial_mse
            ),
        })
    if compare_joint:
        row.update({
            "fallback_individual_missing_elements": fallback_individual_missing,
            "fallback_residual_only_mse": residual_only_mse,
            "fallback_joint_mse": joint_mse,
        })
    return row, events


def summarize_run(rows, traces):
    """汇总同一目标丢失率和种子下的逐 GOP 指标及包事件。

    rows 与 traces 按 GOP 对应，且至少包含一组结果。
    返回可写入 JSON 的字典。没有可解码 GOP 时，配对 MSE 为 None。
    """
    decoded = [row for row in rows if row["decoder_ready"]]
    failed = len(rows) - len(decoded)
    counts = dict.fromkeys(("received", "lost", "late", "unsent"), 0)
    for trace in traces:
        for status, count in trace["packet_loss"]["status_counts"].items():
            counts[status] += count

    # 实际包丢失率的分母是已发送的包。迟到包已发送，未发送包不计入。
    sent_packets = counts["received"] + counts["lost"] + counts["late"]

    def decoded_mean(field):
        # 三个误差均值都使用相同的可解码 GOP，保持逐 GOP 配对。
        return sum(row[field] for row in decoded) / len(decoded) if decoded else None

    summary = {
        "target_loss_fraction": rows[0]["target_loss_fraction"],
        "seed": rows[0]["seed"],
        "gop_count": len(rows),
        "decoded_gop_count": len(decoded),
        "failed_gop_count": failed,
        "failed_gop_fraction": failed / len(rows),
        "sent_packets": sent_packets,
        "lost_packets": counts["lost"],
        "packet_status_counts": counts,
        "actual_packet_loss_fraction": counts["lost"] / sent_packets if sent_packets else None,
        "sent_bytes": sum(row["sent_bytes"] for row in rows),
        "paired_no_loss_mse": decoded_mean("no_loss_int8_mse"),
        "decoded_packet_loss_mse": decoded_mean("packet_loss_mse"),
        "mean_delta_mse": decoded_mean("delta_mse"),
    }
    if "recovered_mse" in rows[0]:
        # 只统计实际缺失个体残差且可解码的 GOP，完整接收的样本不会稀释收益。
        eligible = [row for row in decoded if row["missing_individual_elements"]]
        summary["residual_recovery_gop_count"] = len(eligible)
        summary["mean_mse_reduction_vs_zero_fill"] = (
            sum(row["mse_reduction_vs_zero_fill"] for row in eligible) / len(eligible)
            if eligible else None
        )
    if "public_zero_fill_mse" in rows[0]:
        fallback = [row for row in rows if row["public_zero_fill_mse"] is not None]
        summary["public_zero_fill_gop_count"] = len(fallback)
        summary["failed_gop_count_after_public_zero_fill"] = failed - len(fallback)
        summary["mean_public_zero_fill_delta_mse"] = (
            sum(row["public_zero_fill_delta_mse"] for row in fallback) / len(fallback)
            if fallback else None
        )
    return summary


def main():
    """从命令行读取实验配置，并将逐 GOP 指标与回放事件写入本地。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frames", type=int, nargs="+", default=[100, 400, 800])
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--loss-fraction", type=float, default=0.1)
    parser.add_argument("--recover-residual", action="store_true",
                        help="追加时间零和残差恢复对照")
    parser.add_argument("--allow-public-zero-fill", action="store_true",
                        help="追加公共分支缺块置零降级对照")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recover-public-spatial", action="store_true",
                        help="追加公共分支邻域补全对照")
    parser.add_argument("--compare-joint-recovery", action="store_true",
                        help="比较公共缺块样本的四种恢复策略")
    args = parser.parse_args()
    if not 0 <= args.loss_fraction <= 1:
        parser.error("--loss-fraction 必须在 0 到 1 之间")
    device = torch.device(args.device)
    encoder, decoder, checkpoint = load_swin_models(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, traces = [], []
    for gop_id, start_frame in enumerate(args.start_frames):
        gop = load_video_as_gop(args.video, 128, 4, start_frame).to(device)
        row, events = evaluate_gop(
            encoder, decoder, gop, gop_id, start_frame, args.seed, args.loss_fraction,
            recover_residual=args.recover_residual,
            allow_public_zero_fill=args.allow_public_zero_fill,
            recover_public=args.recover_public_spatial,
            compare_joint=args.compare_joint_recovery,
        )
        rows.append(row)
        traces.append({"gop_id": gop_id, "start_frame": start_frame, **events})
    with (args.output_dir / "gop_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "events.json").write_text(
        json.dumps({
            "video_sha256": file_sha256(args.video),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "device": str(device), "gops": traces,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    failed = sum(not row["decoder_ready"] for row in rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summarize_run(rows, traces), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"失败 GOP：{failed}/{len(rows)}。输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
