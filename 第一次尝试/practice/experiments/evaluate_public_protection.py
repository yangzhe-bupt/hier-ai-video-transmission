# ======================================================================
# 文件：evaluate_public_protection.py
# 功能：评估在相同 GOP 字节预算下，复制公共包并减少个体包的效果。
# 函数汇总：
#   - build_protection_plan：按序列化长度，用个体包的预算换取公共包副本。
#   - evaluate_policy：回放发送事件、合并副本并计算接收状态和重建 MSE。
#   - evaluate_received_packets：按已到达的原始包 ID 审计和解码。
#   - summarize：统计公共分支完整率、解码覆盖率和配对 MSE。
#   - main：读取固定视频和检查点，逐 GOP 运行两种发送策略。
# 输出：逐 GOP 指标、可重放的事件记录和汇总。
# 运行：从项目根目录通过 practice/run.py 启动。
# ======================================================================

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

import torch

from train_packet_aware_decoder import load_clips, prepare_packets, write_csv
from accept_biran_swin import file_sha256, mse, restore_quantized
from hier_video.transmission.network_replay import FeaturePacket, LinkSample, replay_feature_packet_events
from hier_video.transmission.packetization import unpack_feature_packets
from hier_video.transmission.reception_audit import audit_quantized_reception
from train_swin_video import load_swin_models


def build_protection_plan(packets, max_copies=None):
    """
    参数：
        packets：同一 GOP 的所有原始包及其序列化长度。
        max_copies：副本上限，None 表示为所有符合预算的公共包安排副本。
    返回：
        offered：原始包和候选副本，供网络回放记录。
        selected：实际发送的包 ID。
        replica_sources：副本 ID 到原始 ID 的映射。
    """
    public = [packet for packet in packets if packet.group_id.endswith(":public")]
    available = [packet for packet in packets if packet.group_id.endswith(":individual")]
    replicas = []
    replica_sources = {}
    omitted = set()

    # 先安排较大的公共包，每次用能够覆盖副本开销的最小个体包支付。
    for packet in sorted(public, key=lambda item: item.payload_bytes, reverse=True):
        if max_copies is not None and len(replicas) >= max_copies:
            break
        candidates = [item for item in available if item.payload_bytes >= packet.payload_bytes]
        if not candidates:
            continue
        donor = min(candidates, key=lambda item: item.payload_bytes)
        available.remove(donor)
        omitted.add(donor.packet_id)

        replica_id = f"{packet.packet_id}:replica"
        replica_sources[replica_id] = packet.packet_id
        replicas.append(FeaturePacket(replica_id, packet.group_id, packet.payload_bytes))

    offered = (*packets, *replicas)
    selected = tuple(item.packet_id for item in offered if item.packet_id not in omitted)
    return offered, selected, replica_sources


@torch.no_grad()
def evaluate_policy(prepared, decoder, clip, gop_id, rate, seed, protection):
    """
    对给定 GOP 回放一种策略，返回指标与包事件。
    两种策略使用同一接收规则。公共特征缺块且两支量化参数齐全时可置零解码。
    """
    _, manifest, _, wire, originals = prepared
    budget = sum(map(len, wire))
    if protection:
        offered, selected, replica_sources = build_protection_plan(originals)
    else:
        offered = originals
        selected = tuple(item.packet_id for item in originals)
        replica_sources = {}

    report = replay_feature_packet_events(
        offered, selected, trace_id=f"acceptance-gop-{gop_id}", seed=seed,
        decision_time_ms=0, deadline_ms=1000, budget_bytes=budget,
        link=LinkSample(100_000_000, rate),
        feature_interface_version=manifest.interface_version,
    )
    sent_bytes = report["actual_wire_bytes"]
    expected_bytes = sum(item.payload_bytes for item in offered if item.packet_id in selected)
    if sent_bytes != expected_bytes or sent_bytes > budget:
        raise RuntimeError("实际发送字节与计划或预算不一致")

    # 任何一个副本收到即可恢复对应原始包。拆包器只接收一次原始字节。
    arrived = {
        replica_sources.get(event["packet_id"], event["packet_id"])
        for event in report["events"] if event["status"] == "received"
    }
    metrics = evaluate_received_packets(prepared, decoder, clip, arrived)
    metrics.update(
        budget_bytes=budget,
        sent_bytes=sent_bytes,
        replica_count=len(replica_sources),
        lost_count=report["status_counts"]["lost"],
    )
    return metrics, {"replica_sources": replica_sources, "replay": report}


@torch.no_grad()
def evaluate_received_packets(prepared, decoder, clip, arrived):
    """用到达的原始包 ID 拆包并解码，统计公共缺块和重建 MSE。"""
    parts, manifest, decomposer, wire, originals = prepared
    public_ids = {item.packet_id for item in originals if item.group_id.endswith(":public")}
    received_wire = [raw for item, raw in zip(originals, wire) if item.packet_id in arrived]
    metrics = {
        "missing_public": len(public_ids - arrived),
        "strict_ready": False,
        "receiver_status": "failed_no_received_packets",
        "mse": None,
    }
    if received_wire:
        received = unpack_feature_packets(received_wire, manifest, device=clip.device)
        audit = audit_quantized_reception(received, manifest)
        metrics["receiver_status"] = audit.status
        metrics["strict_ready"] = audit.decoder_ready
        if metrics["missing_public"] != len(audit.missing_public_group_ids):
            raise RuntimeError("按包 ID 统计的公共缺块与拆包审计不一致")
        allow_zero_fill = (
            audit.status == "failed_public_incomplete"
            and {"public", "individual"} <= received.quantization_by_branch.keys()
        )
        if audit.decoder_ready or allow_zero_fill:
            features, _, _ = restore_quantized(
                received, replace(audit, decoder_ready=True), parts, decomposer, clip.device,
            )
            metrics["mse"] = mse(decoder(features), clip)
    return metrics


def summarize(rows, rates):
    """按标称丢包率汇总。配对 MSE 只使用两边均能输出的相同 GOP 和种子。"""
    summary = []
    for rate in rates:
        cases = [row for row in rows if row["loss_rate"] == rate]
        paired = [row for row in cases if row["baseline_mse"] is not None
                  and row["protected_mse"] is not None]
        result = {"loss_rate": rate, "cases": len(cases), "paired_cases": len(paired)}
        for name in ("baseline", "protected"):
            result[f"{name}_public_complete"] = sum(row[f"{name}_missing_public"] == 0 for row in cases)
            result[f"{name}_strict_ready"] = sum(row[f"{name}_strict_ready"] for row in cases)
            result[f"{name}_zero_fill_ready"] = sum(row[f"{name}_mse"] is not None for row in cases)
            result[f"{name}_paired_mse"] = mean(row[f"{name}_mse"] for row in paired) if paired else None
            result[f"{name}_mean_bytes"] = mean(row[f"{name}_sent_bytes"] for row in cases)
        result["mean_replicas"] = mean(row["protected_replica_count"] for row in cases)
        summary.append(result)
    return summary


def main():
    """使用指定视频与权重运行本地发送策略对照。"""
    parser = argparse.ArgumentParser(description="公共分支复制保护的本地配对实验")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    starts = list(range(100, 1201, 100))
    rates = (0.0, 0.1, 0.2, 0.3)
    seeds = (314, 315, 316)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    clips = load_clips(args.video, starts, device)
    rows = []
    events = []

    for gop_id, (start, clip) in enumerate(zip(starts, clips)):
        prepared = prepare_packets(encoder, clip, gop_id, start)
        for rate in rates:
            for seed in seeds:
                case = {"gop_id": gop_id, "start_frame": start, "loss_rate": rate, "seed": seed}
                baseline, base_events = evaluate_policy(prepared, decoder, clip, gop_id, rate, seed, False)
                protected, protected_events = evaluate_policy(prepared, decoder, clip, gop_id, rate, seed, True)

                # 相同原始包在两种策略下使用同一丢包结果，允许个体包主动省略。
                base_status = {item["packet_id"]: item["status"] for item in base_events["replay"]["events"]}
                for item in protected_events["replay"]["events"]:
                    if item["packet_id"] in base_status and item["status"] != "unsent":
                        if (item["status"] == "lost") != (base_status[item["packet_id"]] == "lost"):
                            raise RuntimeError("共享原始包的丢包轨迹不一致")

                rows.append({
                    **case,
                    **{f"baseline_{key}": value for key, value in baseline.items()},
                    **{f"protected_{key}": value for key, value in protected.items()},
                })
                events.append({**case, "baseline": base_events, "protected": protected_events})
        print(f"起始帧 {start} 配对评测完成", flush=True)

    if not any(row["protected_replica_count"] for row in rows):
        raise RuntimeError("所有 GOP 均未安排任何公共包副本")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows, rates)
    write_csv(args.output_dir / "gop_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "video": str(args.video.resolve()),
            "video_sha256": file_sha256(args.video),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "device": str(device),
            "starts": starts,
            "seeds": seeds,
            "loss_model": "independent_packet_loss",
            "budget": "GOP bytes of original serialized packets, including application packet headers",
            "results": summary,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"对照指标与包事件已保存：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
