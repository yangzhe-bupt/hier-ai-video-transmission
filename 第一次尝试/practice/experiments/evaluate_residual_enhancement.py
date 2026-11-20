# ======================================================================
# 文件：evaluate_residual_enhancement.py
# 功能：在固定 GOP 字节上限下比较公共档和公共档加低码率残差。
# 函数汇总：
#   - make_layered_plan：保留公共包和副本，将残差包追加到发送末尾。
#   - measure：记录像素 MSE 和相邻帧差分 MSE。
#   - evaluate_gop：对同一视频、同一丢包轨迹做成对验证。
#   - summarize：分别汇总公共成功、增强启用、丢包降级和质量指标。
# 输出：gop_metrics.csv、summary.csv、summary.json、events.json。
# ======================================================================

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

import torch

from evaluate_public_base_budget import BUDGET, OVERHEAD, STARTS, build_public_plan, replay_public_plan
from evaluate_adaptive_public_base import DEADLINE_MS, DELAY_MS, LINK_RATE
from train_packet_aware_decoder import load_clips, write_csv
from accept_biran_swin import file_sha256, mse
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.layered_public_codec import decode_layered_gop, encode_individual_enhancement
from hier_video.transmission.network_replay import FeaturePacket, LinkSample
from hier_video.transmission.public_base_codec import encode_public_gop
from hier_video.transmission.public_protection_control import PUBLIC_BASE_PROFILES
from train_swin_video import load_swin_models


SEEDS = range(334, 342)
RATES = (0.0, 0.1, 0.3)


def make_layered_plan(public_plan, enhancement_wire, gop_id):
    """在公共发送计划末尾放一个增强包。计划字节计入该包的应用层和配置开销。"""
    packet_id = f"{gop_id}:individual:4"
    cost = len(enhancement_wire) + OVERHEAD
    total = public_plan.planned_bytes + cost
    if total > BUDGET:
        raise ValueError(f"GOP {gop_id} 增强计划需 {total} 字节，超出 {BUDGET} 字节")
    packet = FeaturePacket(packet_id, "individual", len(enhancement_wire))
    return replace(public_plan, packets=(*public_plan.packets, packet),
                   raw_by_id={**public_plan.raw_by_id, packet_id: enhancement_wire},
                   planned_bytes=total), packet_id


def measure(decoder, features, target):
    """返回像素误差和相邻帧差分误差。差分衡量逐像素时间变化。"""
    if features is None:
        return None, None
    reconstructed = decoder(features)
    if reconstructed.shape != target.shape or not torch.isfinite(reconstructed).all().item():
        raise RuntimeError("解码器输出形状或数值无效")
    differences = reconstructed[:, 1:] - reconstructed[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    return mse(reconstructed, target), mse(differences, target_diff)


@torch.no_grad()
def evaluate_gop(encoder, decoder, clip, start_frame, gop_id):
    """对每档比较使用相同公共包的两种链路，逐例核对公共包结果。"""
    parts = PublicResidualDecomposer().encode(encoder(clip))
    enhancement_wire = encode_individual_enhancement(parts.individual, gop_id=gop_id, start_frame=start_frame)
    rows, events = [], []
    for profile in PUBLIC_BASE_PROFILES:
        wire = encode_public_gop(parts.public, profile, gop_id=gop_id, start_frame=start_frame)
        base_plan = build_public_plan(wire, gop_id, add_copies=True)
        layered_plan, enhancement_id = make_layered_plan(base_plan, enhancement_wire, gop_id)

        for rate in RATES:
            for seed in ((SEEDS.start,) if rate == 0 else SEEDS):
                link = LinkSample(LINK_RATE, rate, DELAY_MS, OVERHEAD)
                settings = {"trace_id": f"residual-enhancement-{gop_id}", "seed": seed,
                            "link": link, "deadline_ms": DEADLINE_MS}
                base_wire, _, base_report = replay_public_plan(base_plan, profile, **settings)
                layered_wire, _, layer_report = replay_public_plan(layered_plan, profile, **settings)
                layer_events = layer_report["events"]
                if layer_events[:-1] != base_report["events"] or layer_events[-1]["packet_id"] != enhancement_id:
                    raise RuntimeError("公共包原有的发送事件发生了变化")
                enhancement_status = layer_events[-1]["status"]
                arrived = enhancement_status == "received"
                layer_public_wire = layered_wire[:-1] if arrived else layered_wire
                received_enhancement = layered_wire[-1] if arrived else None
                if layer_public_wire != base_wire or (arrived and received_enhancement != enhancement_wire):
                    raise RuntimeError("分层接收包归属不一致")

                base, base_status = decode_layered_gop(base_wire, device=clip.device)
                layered, layer_status = decode_layered_gop(
                    layer_public_wire, received_enhancement, device=clip.device)
                if (base is None) != (layered is None):
                    raise RuntimeError("增强包改变了公共档的解码可用性")
                if not arrived and base is not None and not torch.equal(base, layered):
                    raise RuntimeError("增强包缺失时未精确回退到公共特征")
                if arrived and base is not None and layer_status != "with_enhancement":
                    raise RuntimeError("有效增强包未能应用")

                baseline_mse, baseline_diff = measure(decoder, base, clip)
                layered_mse, layered_diff = measure(decoder, layered, clip)
                key = {"gop_id": gop_id, "start_frame": start_frame, "profile": profile,
                       "loss_rate": rate, "seed": seed}
                rows.append({
                    **key, "baseline_status": base_status, "layered_status": layer_status,
                    "enhancement_status": enhancement_status,
                    "baseline_bytes": base_report["actual_wire_bytes"],
                    "layered_bytes": layer_report["actual_wire_bytes"],
                    "enhancement_bytes": len(enhancement_wire) + OVERHEAD,
                    "baseline_mse": baseline_mse, "layered_mse": layered_mse,
                    "baseline_difference_mse": baseline_diff, "layered_difference_mse": layered_diff,
                })
                events.append({**key, "replica_sources": base_plan.source_ids,
                               "baseline": base_report, "layered": layer_report})
    return rows, events


def summarize(rows):
    """同时保留全体可解码样本及基础、增强都收齐的配对子集。"""
    stats = []
    for rate in RATES:
        for profile in PUBLIC_BASE_PROFILES:
            current = [row for row in rows if (row["profile"], row["loss_rate"]) == (profile, rate)]
            paired = [row for row in current if row["baseline_mse"] is not None]
            enhanced = [row for row in paired if row["layered_status"] == "with_enhancement"]
            rollup = {
                "profile": profile, "loss_rate": rate, "cases": len(current),
                "decoded": len(paired), "enhanced": len(enhanced),
                "fallback": len(paired) - len(enhanced),
                "mean_baseline_bytes": mean(row["baseline_bytes"] for row in current),
                "mean_layered_bytes": mean(row["layered_bytes"] for row in current),
                "max_layered_bytes": max(row["layered_bytes"] for row in current),
                "improved_pixel_cases": sum(row["layered_mse"] < row["baseline_mse"] for row in enhanced),
                "improved_difference_cases": sum(
                    row["layered_difference_mse"] < row["baseline_difference_mse"] for row in enhanced),
            }
            for label, group in (("decoded", paired), ("enhanced", enhanced)):
                for metric in ("baseline_mse", "layered_mse", "baseline_difference_mse", "layered_difference_mse"):
                    rollup[f"{label}_{metric}"] = mean(row[metric] for row in group) if group else None
            stats.append(rollup)
    return stats


def main():
    parser = argparse.ArgumentParser(description="公共基础档与个体残差增强的预算和重建验证")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    clips = load_clips(args.video, STARTS, device)
    rows, events = [], []
    for gop_id, (clip, start) in enumerate(zip(clips, STARTS)):
        results, traces = evaluate_gop(encoder, decoder, clip, start, gop_id)
        rows.extend(results)
        events.extend(traces)
        print(f"GOP {gop_id + 1}/{len(STARTS)} 起始帧 {start}，{len(results)} 组配对", flush=True)

    summary = summarize(rows)
    config = {
        "video": str(args.video.resolve()), "video_sha256": file_sha256(args.video),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device), "starts": list(STARTS), "seeds": list(SEEDS),
        "budget_bytes": BUDGET, "overhead_bytes": OVERHEAD, "link_rate_bps": LINK_RATE,
        "deadline_ms": DEADLINE_MS, "delay_ms": DELAY_MS,
        "enhancement": "three 4x4 int8 pooled temporal residual frames, fourth as negative sum",
        "results": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "gop_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"结果已保存：{args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
