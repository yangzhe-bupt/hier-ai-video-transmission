# ======================================================================
# 文件：evaluate_public_protection_burst.py
# 功能：在独立测试视频上比较复制 0、2、4 个公共包。
# 函数汇总：
#   - loss_patterns：生成等丢失数量的分散轨迹和周期突发轨迹。
#   - evaluate_gop：用相同的发送槽位掩码回放全部策略。
#   - summarize：合计接收状态及相同样本上的重建误差。
#   - main：加载测试视频与检查点，运行实验并保存逐例证据。
# 输入：测试视频、Swin/CNN 检查点、输出目录。
# 输出：summary.csv、summary.json、gop_metrics.csv、send_plans.json。
# 运行：从项目根目录通过 practice/run.py 启动。
# ======================================================================

import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean

import torch

from evaluate_public_protection import build_protection_plan, evaluate_received_packets
from train_packet_aware_decoder import load_clips, prepare_packets, write_csv
from accept_biran_swin import file_sha256
from train_swin_video import load_swin_models


COPY_COUNTS = (0, 2, 4)
LOSS_PERCENTS = (10, 20, 30)
TRACE_SEED = 314


def loss_patterns(slot_count, gop_id):
    """
    参数：
        slot_count：每种策略实际发出的包数，一个包占一个发送槽位。
        gop_id：GOP 序号，用于产生可重放的分散丢包位置。
    返回：
        依次生成 (轨迹种类, 丢包百分比, 重放编号, 丢失槽位)。
        突发轨迹每 slot_count 个槽位出现一段连续丢失，遍历所有相位。
    """
    yield "clean", 0, 0, ()
    for percent in LOSS_PERCENTS:
        lost_count, remainder = divmod(slot_count * percent, 100)
        if remainder:
            raise ValueError("包数无法表示所指定的精确丢失比例")

        rng = Random(f"{TRACE_SEED}:{gop_id}:{percent}")
        for phase in range(slot_count):
            scattered = tuple(sorted(rng.sample(range(slot_count), lost_count)))
            burst = tuple((phase + offset) % slot_count for offset in range(lost_count))
            # 取模表示周期突发可能跨 GOP 边界，首尾槽位属于同一周期模型。
            yield "scattered", percent, phase, scattered
            yield "burst", percent, phase, burst


def evaluate_gop(encoder, decoder, clip, gop_id, start_frame):
    """
    对同一 GOP 生成三种等包数的发送计划，再逐轨迹比较。
    输出每个轨迹和策略的指标，并保存可重建槽位顺序的包 ID。
    """
    prepared = prepare_packets(encoder, clip, gop_id, start_frame)
    _, _, _, wire, packets = prepared
    budget = sum(map(len, wire))
    plans = {}
    for copies in COPY_COUNTS:
        offered, selected, sources = build_protection_plan(packets, max_copies=copies)
        sizes = {packet.packet_id: packet.payload_bytes for packet in offered}
        sent_bytes = sum(sizes[packet_id] for packet_id in selected)
        if len(sources) != copies or len(selected) != len(packets) or sent_bytes > budget:
            raise RuntimeError(f"GOP {gop_id} 的 {copies} 副本策略不满足发送计划")
        plans[copies] = {
            "slot_packet_ids": selected,
            "replica_sources": sources,
            "deselected_ids": [item.packet_id for item in packets if item.packet_id not in selected],
            "sent_bytes": sent_bytes,
        }

    rows = []
    for channel, percent, phase, lost_slots in loss_patterns(len(packets), gop_id):
        lost = set(lost_slots)
        for copies, plan in plans.items():
            arrived = {
                plan["replica_sources"].get(packet_id, packet_id)
                for position, packet_id in enumerate(plan["slot_packet_ids"]) if position not in lost
            }
            metrics = evaluate_received_packets(prepared, decoder, clip, arrived)
            rows.append({
                "gop_id": gop_id,
                "start_frame": start_frame,
                "channel": channel,
                "loss_pct": percent,
                "phase": phase,
                "lost_slots": " ".join(map(str, lost_slots)),
                "replicas": copies,
                "budget_bytes": budget,
                "sent_bytes": plan["sent_bytes"],
                **metrics,
            })
    return rows, {"gop_id": gop_id, "start_frame": start_frame, "budget_bytes": budget, "plans": plans}


def summarize(rows):
    """
    汇总每组条件的公共分支完整次数、严格及降级解码次数。
    MSE 只使用三种策略都能解码的相同 GOP 与轨迹。
    """
    cases = {}
    for row in rows:
        condition = (row["channel"], row["loss_pct"])
        case = (row["gop_id"], row["phase"])
        cases.setdefault(condition, {}).setdefault(case, {})[row["replicas"]] = row

    summary = []
    for (channel, percent), grouped in sorted(cases.items()):
        triplets = list(grouped.values())
        if any(set(case) != set(COPY_COUNTS) for case in triplets):
            raise RuntimeError("策略对照组不完整")
        paired = [case for case in triplets if all(row["mse"] is not None for row in case.values())]
        for copies in COPY_COUNTS:
            current = [case[copies] for case in triplets]
            summary.append({
                "channel": channel,
                "loss_pct": percent,
                "replicas": copies,
                "cases": len(current),
                "paired_cases": len(paired),
                "public_complete": sum(row["missing_public"] == 0 for row in current),
                "strict_ready": sum(row["strict_ready"] for row in current),
                "zero_fill_ready": sum(row["mse"] is not None for row in current),
                "paired_mse": mean(case[copies]["mse"] for case in paired) if paired else None,
                "mean_sent_bytes": mean(row["sent_bytes"] for row in current),
            })
    return summary


def main():
    """读取固定起点的 GOP，推理并存档。"""
    parser = argparse.ArgumentParser(description="比较公共包预算分配在分散与突发丢包下的效果")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    # Tears of Steel 共 17620 帧，在约 5%～95% 进度均匀取 16 个 GOP。
    starts = list(range(880, 16721, 1056))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    clips = load_clips(args.video, starts, device)
    rows = []
    plans = []
    for gop_id, (start_frame, clip) in enumerate(zip(starts, clips)):
        gop_rows, plan = evaluate_gop(encoder, decoder, clip, gop_id, start_frame)
        rows.extend(gop_rows)
        plans.append(plan)
        print(f"GOP {gop_id + 1}/{len(starts)}，起始帧 {start_frame}，{len(gop_rows)} 条记录", flush=True)

    summary = summarize(rows)
    metadata = {
        "video": str(args.video.resolve()),
        "video_sha256": file_sha256(args.video),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "start_frames": starts,
        "replicas": COPY_COUNTS,
        "scattered_seed": TRACE_SEED,
        "channel": "Exact counts, slot-index erasures, all burst phases per GOP",
        "timing": "No delay or deadline simulation in this slot-trace experiment",
        "budget": "Original serialized packet bytes including application packet headers",
        "results": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "gop_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "send_plans.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
    print(f"评测结果已保存：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
