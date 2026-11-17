# ======================================================================
# 文件：evaluate_adaptive_public_protection.py
# 功能：用固定网络轨迹联调 int8 特征包、保护决策、包事件回放及接收审计。
# 函数汇总：
#   - build_send_plans：为同一个 GOP 构建三个合法的发送计划。
#   - replay_step：执行一次发送并返回实际开销、包事件与重建指标。
#   - run_episode：让四种策略在同一序列上各自演化并保存反馈。
#   - summarize：按场景汇总覆盖率、字节和共同样本的 MSE。
# 输入：独立测试视频、第 314 轮权重及实验结果目录。
# 输出：逐 GOP 决策、逐包事件、汇总及实验配置。
# ======================================================================

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

import torch

from evaluate_public_protection import build_protection_plan, evaluate_received_packets
from train_packet_aware_decoder import load_clips, prepare_packets, write_csv
from accept_biran_swin import file_sha256
from hier_video.transmission.network_replay import FeaturePacket, LinkSample, replay_feature_packet_events
from hier_video.transmission.public_protection_control import (
    PROTECTION_LEVELS, ProtectionDecision, choose_protection_level,
)
from train_swin_video import load_swin_models


SEEDS = (314, 315, 316, 317, 318, 319)
STEPS_PER_PHASE = 4
PACKET_OVERHEAD = 28
ONE_MBPS_BUDGET = 1_000_000 * 4 // (8 * 24)
POLICIES = ("adaptive", "fixed_0", "fixed_2", "fixed_4")


@dataclass(frozen=True)
class Phase:
    """回放器使用的链路真值，以及发送端提前拿到的约束。"""

    name: str
    loss_fraction: float
    rate_bps: int = 100_000_000
    deadline_ms: int = 100
    budget_bytes: int | None = None


@dataclass(frozen=True)
class SendPlan:
    """一次完整档位的包描述、选择和应用层加配置开销后的字节数。"""

    offered: tuple[FeaturePacket, ...]
    selected: tuple[str, ...]
    replica_sources: dict[str, str]
    wire_bytes: int


PHASES = (
    Phase("clean", 0.0),
    Phase("moderate_loss", 0.10),
    Phase("high_loss", 0.30),
    Phase("recovered", 0.0),
    Phase("short_deadline", 0.0, rate_bps=8_000_000, deadline_ms=60),
    Phase("one_mbps_budget", 0.0, budget_bytes=ONE_MBPS_BUDGET),
)


def build_send_plans(packets):
    """
    参数：packets 为同一 GOP 的完整原始包描述。
    返回：以副本数索引的全部可实现档位，计划成本含每包配置的传输开销。
    """
    plans = {}
    for copies in PROTECTION_LEVELS:
        offered, selected, sources = build_protection_plan(packets, max_copies=copies)
        if len(sources) != copies:
            continue
        chosen = set(selected)
        wire_bytes = sum(item.payload_bytes + PACKET_OVERHEAD for item in offered
                         if item.packet_id in chosen)
        plans[copies] = SendPlan(offered, selected, sources, wire_bytes)
    return plans


def replay_step(prepared, decoder, clip, decision, plans, phase, seed, step, budget):
    """
    回放当前 GOP 的选择，使用收到的原包或副本进行解码。
    返回逐例接收指标、完整包事件报告以及本次可用的真实观测反馈。
    丢包反馈只除以已经发出的包，迟到包单独统计。
    """
    _, manifest, _, _, originals = prepared
    plan = plans.get(decision.selected_copies)
    offered = plan.offered if plan else originals
    selected = plan.selected if plan else ()
    sources = plan.replica_sources if plan else {}
    planned_wire = plan.wire_bytes if plan else 0
    report = replay_feature_packet_events(
        offered, selected, trace_id=f"adaptive-public-{step}", seed=seed,
        decision_time_ms=0, deadline_ms=phase.deadline_ms, budget_bytes=budget,
        link=LinkSample(phase.rate_bps, phase.loss_fraction, one_way_delay_ms=20,
                        packet_overhead_bytes=PACKET_OVERHEAD),
        feature_interface_version=manifest.interface_version,
    )
    actual_wire = report["actual_wire_bytes"]
    if not actual_wire <= planned_wire <= budget or report["decision"]["selected_wire_bytes"] != planned_wire:
        raise RuntimeError("发送计划和回放的预算统计不一致")

    arrived = {sources.get(row["packet_id"], row["packet_id"]) for row in report["events"]
               if row["status"] == "received"}
    metrics = evaluate_received_packets(prepared, decoder, clip, arrived)
    counts = report["status_counts"]
    sent_count = counts["received"] + counts["lost"] + counts["late"]
    feedback = counts["lost"] / sent_count if sent_count else None
    selected_unsent = sum(row["selected"] and row["status"] == "unsent" for row in report["events"])
    return {
        "planned_wire_bytes": planned_wire,
        "actual_wire_bytes": actual_wire,
        "received_count": counts["received"],
        "lost_count": counts["lost"],
        "late_count": counts["late"],
        "selected_unsent": selected_unsent,
        "observed_loss": feedback,
        **metrics,
    }, {"replica_sources": sources, "replay": report}


def run_episode(samples, decoder, seed):
    """
    对同一批 GOP 比较自适应和固定 0、2、4 副本策略。
    每种策略只读取自己的历史反馈。没有实际发包时保留最近有效反馈。
    """
    latest_loss = dict.fromkeys(POLICIES)
    rows = []
    events = []
    for step, (clip, start, prepared, plans) in enumerate(samples):
        phase = PHASES[step // STEPS_PER_PHASE]
        costs = {copies: plan.wire_bytes for copies, plan in plans.items()}
        budget = phase.budget_bytes if phase.budget_bytes is not None else costs[0]
        for policy in POLICIES:
            previous = latest_loss[policy]
            if policy == "adaptive":
                decision = choose_protection_level(costs, budget, previous)
            else:
                requested = int(policy.removeprefix("fixed_"))
                fits = requested in costs and costs[requested] <= budget
                decision = ProtectionDecision(requested, requested if fits else None,
                                              "fixed_level" if fits else "fixed_level_unavailable")

            metrics, event = replay_step(prepared, decoder, clip, decision, plans, phase, seed, step, budget)
            if metrics["observed_loss"] is not None:
                latest_loss[policy] = metrics["observed_loss"]
            case = {
                "seed": seed,
                "step": step,
                "start_frame": start,
                "phase": phase.name,
                "policy": policy,
                "true_loss": phase.loss_fraction,
                "prior_loss": previous,
                "preferred_copies": decision.preferred_copies,
                "selected_copies": decision.selected_copies,
                "decision_reason": decision.reason,
                "budget_bytes": budget,
            }
            rows.append({**case, **metrics})
            events.append({**case, "wire_bytes_by_copies": costs, **event})
    return rows, events


def summarize(rows):
    """
    各场景统计覆盖率、选档次数及平均发送量。
    配对 MSE 使用四种策略都成功输出的相同 GOP 和种子。
    """
    summary = []
    for name in [phase.name for phase in PHASES] + ["all"]:
        cases = {}
        for row in rows:
            if name == "all" or row["phase"] == name:
                cases.setdefault((row["seed"], row["step"]), {})[row["policy"]] = row
        if any(set(case) != set(POLICIES) for case in cases.values()):
            raise RuntimeError("对照缺少策略记录")
        paired = [case for case in cases.values() if all(item["mse"] is not None for item in case.values())]
        for policy in POLICIES:
            group = [case[policy] for case in cases.values()]
            summary.append({
                "phase": name,
                "policy": policy,
                "gops": len(group),
                "selected_0": sum(item["selected_copies"] == 0 for item in group),
                "selected_2": sum(item["selected_copies"] == 2 for item in group),
                "selected_4": sum(item["selected_copies"] == 4 for item in group),
                "skipped": sum(item["selected_copies"] is None for item in group),
                "public_complete": sum(item["missing_public"] == 0 for item in group),
                "strict_ready": sum(item["strict_ready"] for item in group),
                "zero_fill_ready": sum(item["mse"] is not None for item in group),
                "late_packets": sum(item["late_count"] for item in group),
                "selected_unsent": sum(item["selected_unsent"] for item in group),
                "mean_wire_bytes": mean(item["actual_wire_bytes"] for item in group),
                "paired_gops": len(paired),
                "paired_mse": mean(case[policy]["mse"] for case in paired) if paired else None,
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description="公共包保护档位与网络回放联调")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    # 取 24 个四帧 GOP。各场景连续分配 4 个采样 GOP，每条轨迹按同一顺序重放。
    starts = [720 + 704 * index for index in range(len(PHASES) * STEPS_PER_PHASE)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    clips = load_clips(args.video, starts, device)
    samples = []
    for step, (clip, start) in enumerate(zip(clips, starts)):
        prepared = prepare_packets(encoder, clip, step, start)
        plans = build_send_plans(prepared[-1])
        if set(plans) != set(PROTECTION_LEVELS):
            raise RuntimeError(f"起始帧 {start} 缺少所需的保护档位")
        samples.append((clip, start, prepared, plans))

    rows = []
    events = []
    for seed in SEEDS:
        seed_rows, seed_events = run_episode(samples, decoder, seed)
        rows.extend(seed_rows)
        events.extend(seed_events)
        print(f"轨迹种子 {seed} 完成，{len(seed_rows)} 条 GOP 记录", flush=True)

    summary = summarize(rows)
    config = {
        "video": str(args.video.resolve()),
        "video_sha256": file_sha256(args.video),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "start_frames": starts,
        "seeds": SEEDS,
        "phases": [phase.__dict__ for phase in PHASES],
        "steps_per_phase": STEPS_PER_PHASE,
        "packet_overhead_bytes": PACKET_OVERHEAD,
        "feedback": "last valid GOP loss estimate; no packets sent retains prior estimate",
        "deadline": "relative to GOP decision; clock and send queue reset per GOP",
        "results": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "gop_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"联调记录已保存：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
