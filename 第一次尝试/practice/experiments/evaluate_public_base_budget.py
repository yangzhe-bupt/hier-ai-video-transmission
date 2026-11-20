# ======================================================================
# 文件：evaluate_public_base_budget.py
# 功能：在 1 Mbps 等效 GOP 预算下验证公共特征独立重建和固定副本保护。
# 函数汇总：
#   - build_public_plan：预先计算公共包发送计划及副本字节。
#   - replay_public_plan：回放计划并对按时到达的公共包去重。
#   - replay_public：复用网络回放，逐包记账，只用收到的公共包解码。
#   - evaluate_gop：测量三个档位的无丢包率失真和低码率档位丢包表现。
#   - summarize：记录覆盖率、可配对 MSE、实际发送字节与副本数。
# 输出：clean.csv、gop_metrics.csv、summary.csv、summary.json、events.json。
# ======================================================================

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

import torch

from train_packet_aware_decoder import load_clips, write_csv
from accept_biran_swin import file_sha256, mse
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.network_replay import FeaturePacket, LinkSample, replay_feature_packet_events
from hier_video.transmission.public_base_codec import decode_public_gop, encode_public_gop
from train_swin_video import load_swin_models


BUDGET = 1_000_000 * 4 // (8 * 24)
OVERHEAD = 28
PROFILES = ("int8", "int4", "pool2_int8")
SEEDS = range(314, 322)
RATES = (0.0, 0.1)
STARTS = range(880, 16721, 1056)


@dataclass(frozen=True)
class PublicSendPlan:
    """回放使用的公共原包、可选副本与序列化字节映射。"""

    packets: tuple[FeaturePacket, ...]
    raw_by_id: dict[str, bytes]
    source_ids: dict[str, str]
    planned_bytes: int


def build_public_plan(wire, gop_id, add_copies, budget=BUDGET):
    """构建公共包计划。富余字节足够时按轮转顺序加整包副本。"""
    originals = tuple(FeaturePacket(f"{gop_id}:public:{index}", "public", len(raw))
                      for index, raw in enumerate(wire))
    raw_by_id = {packet.packet_id: raw for packet, raw in zip(originals, wire)}
    planned = sum(packet.payload_bytes + OVERHEAD for packet in originals)
    if planned > budget:
        raise ValueError("公共包已经超出 GOP 预算")
    replicas = []
    source_ids = {}
    if add_copies:
        offset = gop_id % len(originals)
        for packet in originals[offset:] + originals[:offset]:
            cost = packet.payload_bytes + OVERHEAD
            if planned + cost <= budget:
                replica = FeaturePacket(f"{packet.packet_id}:replica", "public", packet.payload_bytes)
                replicas.append(replica)
                source_ids[replica.packet_id] = packet.packet_id
                planned += cost

    return PublicSendPlan((*originals, *replicas), raw_by_id, source_ids, planned)


def replay_public_plan(plan, profile, *, trace_id, seed, link, deadline_ms, budget=BUDGET):
    """回放计划，用准时到达的公共包去重后的字节进入解码器。"""
    report = replay_feature_packet_events(
        plan.packets, tuple(packet.packet_id for packet in plan.packets),
        trace_id=trace_id, seed=seed, decision_time_ms=0,
        deadline_ms=deadline_ms, budget_bytes=budget, link=link,
        feature_interface_version=f"public-base-{profile}",
    )
    if not report["actual_wire_bytes"] <= plan.planned_bytes <= budget:
        raise RuntimeError("低码率计划和实际发送量不一致")
    if report["decision"]["selected_wire_bytes"] != plan.planned_bytes:
        raise RuntimeError("计划和回放使用的包开销不一致")
    arrived = {plan.source_ids.get(event["packet_id"], event["packet_id"])
               for event in report["events"] if event["status"] == "received"}
    received = tuple(raw for packet_id, raw in plan.raw_by_id.items() if packet_id in arrived)
    return received, plan.source_ids, report


def replay_public(wire, profile, gop_id, rate, seed, add_copies):
    """运行原固定档评测所用的回放设置。"""
    plan = build_public_plan(wire, gop_id, add_copies)
    received, sources, report = replay_public_plan(
        plan, profile, trace_id=f"public-base-{gop_id}", seed=seed,
        link=LinkSample(100_000_000, rate, 20, OVERHEAD), deadline_ms=100,
    )
    if report["actual_wire_bytes"] != plan.planned_bytes:
        raise RuntimeError("固定档回放未能按计划发送")
    return received, sources, report


@torch.no_grad()
def evaluate_gop(encoder, decoder, clip, gop_id, start):
    """评估无丢包量化代价，再在同一丢包抽样下测量有无副本的收益。"""
    features = encoder(clip)
    public = PublicResidualDecomposer().encode(features).public
    clean = {"gop_id": gop_id, "start_frame": start, "full_float_mse": mse(decoder(features), clip),
             "public_float_mse": mse(decoder(public.expand_as(features)), clip)}
    rows, events = [], []
    for profile in PROFILES:
        wire = encode_public_gop(public, profile, gop_id=gop_id, start_frame=start)
        base_bytes = sum(len(raw) + OVERHEAD for raw in wire)
        decoded, status = decode_public_gop(wire[::-1], device=clip.device)
        if decoded is None or decoded.shape != features.shape:
            raise RuntimeError("独立公共包回环失败")
        reconstruction = decoder(decoded)
        if reconstruction.shape != clip.shape or not torch.isfinite(reconstruction).all().item():
            raise RuntimeError("公共包未能独立重建四帧")
        clean[f"{profile}_bytes"] = base_bytes
        clean[f"{profile}_mse"] = mse(reconstruction, clip)
        if base_bytes > BUDGET:
            continue

        for rate in RATES:
            for seed in ((SEEDS.start,) if rate == 0 else SEEDS):
                for add_copies in (False, True):
                    received, source_ids, report = replay_public(wire, profile, gop_id, rate, seed, add_copies)
                    output, status = decode_public_gop(received, device=clip.device)
                    key = {"gop_id": gop_id, "start_frame": start, "profile": profile,
                           "loss_rate": rate, "seed": seed, "allow_copies": add_copies}
                    rows.append({
                        **key, "replicas": len(source_ids), "sent_bytes": report["actual_wire_bytes"],
                        "public_received": len(received), "receiver_status": status,
                        "mse": mse(decoder(output), clip) if output is not None else None,
                    })
                    events.append({**key, "replica_sources": source_ids, "replay": report})
    return clean, rows, events


def summarize(rows):
    """按丢包率和档位比较覆盖率，配对 MSE使用全部低码率策略均成功的样本。"""
    cases = {}
    for row in rows:
        case = (row["gop_id"], row["seed"])
        action = (row["profile"], row["allow_copies"])
        cases.setdefault(row["loss_rate"], {}).setdefault(case, {})[action] = row
    summary = []
    actions = {(profile, copies) for profile in PROFILES[1:] for copies in (False, True)}
    for rate, group in sorted(cases.items()):
        if any(set(case) != actions for case in group.values()):
            raise RuntimeError("低码率发送方案缺失，无法配对")
        paired = [case for case in group.values() if all(row["mse"] is not None for row in case.values())]
        for profile, copies in sorted(actions):
            current = [case[profile, copies] for case in group.values()]
            summary.append({
                "loss_rate": rate, "profile": profile, "allow_copies": copies,
                "cases": len(current), "decoded": sum(row["mse"] is not None for row in current),
                "mean_sent_bytes": mean(row["sent_bytes"] for row in current),
                "mean_replicas": mean(row["replicas"] for row in current),
                "paired_cases": len(paired),
                "paired_mse": mean(case[profile, copies]["mse"] for case in paired) if paired else None,
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description="公共特征 1 Mbps 等效预算验证")
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
    clean_rows, rows, events = [], [], []
    for gop_id, (start, clip) in enumerate(zip(STARTS, clips)):
        clean, comparisons, traces = evaluate_gop(encoder, decoder, clip, gop_id, start)
        clean_rows.append(clean)
        rows.extend(comparisons)
        events.extend(traces)
        print(f"GOP {gop_id + 1}/{len(STARTS)} 起始帧 {start} 独立重建完成", flush=True)

    summary = summarize(rows)
    config = {
        "video": str(args.video.resolve()), "video_sha256": file_sha256(args.video),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device), "gop_start_frames": list(STARTS), "seeds": list(SEEDS),
        "budget_bytes": BUDGET, "overhead_bytes": OVERHEAD,
        "link": "100 Mbps independent-loss replay with 100 ms deadline and a separate 1 Mbps equivalent GOP byte cap",
        "clean_means": {key: mean(row[key] for row in clean_rows)
                        for key in clean_rows[0] if key.endswith(("mse", "bytes"))},
        "results": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "clean.csv", clean_rows)
    write_csv(args.output_dir / "gop_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"结果保存于 {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
