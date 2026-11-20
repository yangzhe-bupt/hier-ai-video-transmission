# ======================================================================
# 文件：evaluate_layered_mtu.py
# 功能：核算现有分层档按 IPv4 路径 MTU 分片后的预算及解码覆盖率。
# 函数汇总：
#   - fragment_plan：生成实际 UDP 负载与网络回放描述。
#   - receive_fragmented：返回按时重组后的特征与包事件。
#   - replay_plan：按数据报模拟丢失，重组并解码收齐的应用层原包。
#   - run：连续读取视频，核算原方案，回放预算内的对照方案。
# 输出：budget.csv、gop_metrics.csv、summary.json、events.json。
# ======================================================================

import argparse
import json
from pathlib import Path
from statistics import mean

import cv2
import torch

from evaluate_adaptive_public_base import DEADLINE_MS, DELAY_MS, LINK_RATE, PHASES
from evaluate_public_base_budget import BUDGET, build_public_plan
from evaluate_residual_enhancement import make_layered_plan
from train_packet_aware_decoder import write_csv
from accept_biran_swin import file_sha256, mse
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.layered_public_codec import decode_layered_gop, encode_individual_enhancement
from hier_video.transmission.network_replay import FeaturePacket, LinkSample, replay_feature_packet_events
from hier_video.transmission.packetization import deserialize_packet
from hier_video.transmission.public_base_codec import encode_public_gop
from hier_video.transmission.udp_feature_fragments import (
    FRAGMENT_HEADER, IPV4_UDP_BYTES, fragment_feature_packet, reassemble_feature_packets,
)
from train_swin_video import load_swin_models
from video_feature_demo import center_crop_frames, frames_to_gop


SEEDS = (342, 343, 344, 345)
POLICIES = ("int4_keep_copy", "int4_keep_enhancement", "pool2_layered")


def fragment_plan(plan, gop_id, mtu):
    """
    将应用层发送计划转换为 UDP 负载和回放描述。
    到达的原包和副本分片均带同一原包身份，模拟事件仍分开发送和统计。
    """
    raw_by_packet = {}
    fragments_by_source = {}
    for source, raw in plan.raw_by_id.items():
        header = deserialize_packet(raw).header
        if header.gop_id != gop_id or header.packet_id in raw_by_packet:
            raise ValueError("发送计划包含重复或跨 GOP 原包")
        raw_by_packet[header.packet_id] = raw
        fragments_by_source[source] = fragment_feature_packet(
            raw, gop_id=gop_id, packet_id=header.packet_id, path_mtu=mtu)
    transmissions = {}
    descriptors = []
    for packet in plan.packets:
        source = plan.source_ids.get(packet.packet_id, packet.packet_id)
        for index, payload in enumerate(fragments_by_source[source]):
            identity = f"{packet.packet_id}:frag:{index}"
            transmissions[identity] = payload
            descriptors.append(FeaturePacket(identity, packet.group_id, len(payload)))
    cost = sum(len(payload) + IPV4_UDP_BYTES for payload in transmissions.values())
    return tuple(descriptors), transmissions, raw_by_packet, cost


def receive_fragmented(fragmented, *, device, gop_id, mtu, rate, seed):
    """逐数据报回放，返回按时重组并恢复的解码器输入、状态和包事件。"""
    packets, datagrams, originals, planned = fragmented
    report = replay_feature_packet_events(
        packets, tuple(datagrams), trace_id=f"layered-mtu-{gop_id}", seed=seed,
        decision_time_ms=0, deadline_ms=DEADLINE_MS, budget_bytes=BUDGET,
        link=LinkSample(LINK_RATE, rate, DELAY_MS, IPV4_UDP_BYTES),
        feature_interface_version="public-layered-fragments-v1",
    )
    sent = report["actual_wire_bytes"]
    if not sent <= planned <= BUDGET or report["decision"]["selected_wire_bytes"] != planned:
        raise RuntimeError("分片回放发送字节与计划不一致")
    arrived = [datagrams[row["packet_id"]] for row in report["events"] if row["status"] == "received"]
    restored = reassemble_feature_packets(reversed(arrived), gop_id=gop_id, path_mtu=mtu)
    if any(packet_id not in originals or raw != originals[packet_id] for packet_id, raw in restored.items()):
        raise RuntimeError("重组后的原包字节与发送源不同")
    base = tuple(restored[index] for index in range(4) if index in restored)
    features, status = decode_layered_gop(base, restored.get(4), device=device)
    return features, {
        "datagrams": len(packets), "received_datagrams": len(arrived),
        "lost_datagrams": report["status_counts"]["lost"],
        "late_datagrams": report["status_counts"]["late"], "sent_bytes": sent,
        "restored_public": len(base), "enhancement_reassembled": 4 in restored,
        "status": status,
    }, report


def replay_plan(fragmented, decoder, clip, gop_id, mtu, rate, seed):
    """在已有覆盖率对照中同时计算成功输出样本的 MSE。"""
    features, metrics, report = receive_fragmented(
        fragmented, device=clip.device, gop_id=gop_id, mtu=mtu, rate=rate, seed=seed)
    return {**metrics, "mse": mse(decoder(features), clip) if features is not None else None}, report


def summarize(rows):
    """保留全体覆盖率，以及三种方案都输出样本上的配对 MSE。"""
    result = []
    for phase in [name for name, _ in PHASES] + ["all"]:
        cases = {}
        for row in rows:
            if phase in ("all", row["phase"]):
                cases.setdefault((row["gop_id"], row["seed"]), {})[row["policy"]] = row
        paired = [case for case in cases.values() if all(case[policy]["mse"] is not None for policy in POLICIES)]
        for policy in POLICIES:
            group = [case[policy] for case in cases.values()]
            result.append({
                "phase": phase, "policy": policy, "cases": len(group),
                "decoded": sum(row["mse"] is not None for row in group),
                "enhanced": sum(row["status"] == "with_enhancement" for row in group),
                "mean_sent_bytes": mean(row["sent_bytes"] for row in group),
                "max_sent_bytes": max(row["sent_bytes"] for row in group),
                "paired_count": len(paired),
                "paired_mse": mean(case[policy]["mse"] for case in paired) if paired else None,
            })
    return result


@torch.no_grad()
def run(args):
    """核算原分层档和可回放档，并在同一连续片段上保存逐包证据。"""
    if args.start_frame < 0 or args.frames <= 0 or args.frames % 16:
        raise ValueError("需要非负起始帧和可平分给四阶段的 GOP 数")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    decomposer = PublicResidualDecomposer()
    rows, events, budgets = [], [], []
    gops = args.frames // 4

    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 24) > 0.001:
            raise ValueError("需要可读取的 24 fps 源视频")
        for _ in range(args.start_frame):
            if not cap.grab():
                raise ValueError("无法到达连续片段起始帧")
        for gop_id in range(gops):
            frames = []
            start = args.start_frame + gop_id * 4
            for offset in range(4):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"源帧 {start + offset} 读取失败")
                frames.append(frame)

            clip = frames_to_gop(center_crop_frames(frames, 128)).to(device)
            parts = decomposer.encode(encoder(clip))
            residual = encode_individual_enhancement(parts.individual, gop_id=gop_id, start_frame=start)
            int4 = encode_public_gop(parts.public, "int4", gop_id=gop_id, start_frame=start)
            pool = encode_public_gop(parts.public, "pool2_int8", gop_id=gop_id, start_frame=start)
            int4_protected = build_public_plan(int4, gop_id, add_copies=True)
            int4_unprotected = build_public_plan(int4, gop_id, add_copies=False)
            pool_protected = build_public_plan(pool, gop_id, add_copies=True)
            int4_full, _ = make_layered_plan(int4_protected, residual, gop_id)
            int4_motion, _ = make_layered_plan(int4_unprotected, residual, gop_id)
            pool_full, _ = make_layered_plan(pool_protected, residual, gop_id)
            candidates = {"int4_full_before_mtu": int4_full, "int4_keep_copy": int4_protected,
                          "int4_keep_enhancement": int4_motion, "pool2_layered": pool_full}

            fragmented = {}
            for name, plan in candidates.items():
                payload_plan = fragment_plan(plan, gop_id, args.mtu)
                descriptors, _, _, cost = payload_plan
                budgets.append({
                    "gop_id": gop_id, "start_frame": start, "policy": name,
                    "logical_packets": len(plan.packets), "datagrams": len(descriptors),
                    "before_mtu_bytes": plan.planned_bytes,
                    "fragment_headers_bytes": len(descriptors) * FRAGMENT_HEADER.size,
                    "ipv4_udp_bytes": len(descriptors) * IPV4_UDP_BYTES,
                    "after_mtu_bytes": cost, "fits": cost <= BUDGET,
                    "max_datagram_ip_bytes": max(packet.payload_bytes + IPV4_UDP_BYTES for packet in descriptors),
                })
                if cost <= BUDGET and name in POLICIES:
                    fragmented[name] = payload_plan

            phase, rate = PHASES[gop_id // (gops // len(PHASES))]
            for seed in SEEDS:
                for policy, plan in fragmented.items():
                    result, report = replay_plan(plan, decoder, clip, gop_id, args.mtu, rate, seed)
                    key = {"gop_id": gop_id, "start_frame": start, "phase": phase,
                           "loss_fraction": rate, "policy": policy, "seed": seed}
                    rows.append({**key, **result})
                    events.append({**key, "report": report})
            if (gop_id + 1) % (gops // len(PHASES)) == 0:
                print(f"{phase} 阶段完成，已处理 {gop_id + 1}/{gops} 组", flush=True)
    finally:
        cap.release()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "budget.csv", budgets)
    if rows:
        write_csv(args.output_dir / "gop_metrics.csv", rows)
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    complete = len(rows) == gops * len(SEEDS) * len(POLICIES)
    config = {"video": str(args.video.resolve()), "video_sha256": file_sha256(args.video),
              "checkpoint": str(args.checkpoint.resolve()), "first_frame": args.start_frame,
              "frames": args.frames, "seeds": SEEDS, "path_mtu": args.mtu,
              "budget": BUDGET, "fragment_header_bytes": FRAGMENT_HEADER.size,
              "ipv4_udp_bytes": IPV4_UDP_BYTES, "link_rate_bps": LINK_RATE, "deadline_ms": DEADLINE_MS,
              "loss_model": "independent UDP datagram loss with stable fragment IDs",
              "complete_replay_comparison": complete, "results": summarize(rows) if complete else None}
    (args.output_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"分片计划及回放结果已保存：{args.output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="MTU 分片后的低码率分层发送验证")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=10240)
    parser.add_argument("--frames", type=int, default=384)
    parser.add_argument("--mtu", type=int, default=1500)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
