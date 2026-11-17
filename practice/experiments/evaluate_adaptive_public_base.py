# ======================================================================
# 文件：evaluate_adaptive_public_base.py
# 功能：在 1 Mbps 合成链路和 GOP 字节预算下验证公共分支自适应。
# 函数汇总：
#   - prepare_samples：为每个 GOP 预编码画质档和保护档。
#   - run_trace：回放状态变化并用上游历史反馈选档。
#   - summarize：比较解码覆盖、配对 MSE、发送量和档位切换。
# 输出：gop_metrics.csv、summary.csv、summary.json 和 events.json。
# ======================================================================

import argparse
from collections import deque
import json
from pathlib import Path
from statistics import mean

import torch

from evaluate_public_base_budget import BUDGET, OVERHEAD, STARTS, build_public_plan, replay_public_plan
from train_packet_aware_decoder import load_clips, write_csv
from accept_biran_swin import file_sha256, mse
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.network_replay import LinkSample
from hier_video.transmission.public_base_codec import decode_public_gop, encode_public_gop
from hier_video.transmission.public_protection_control import PUBLIC_BASE_PROFILES, choose_public_base_profile
from train_swin_video import load_swin_models


PHASES = (("clean", 0.0), ("loss_10", 0.10), ("loss_30", 0.30), ("recovered", 0.0))
POLICIES = ("adaptive", *PUBLIC_BASE_PROFILES)
SEEDS = range(322, 334)
LINK_RATE = 1_000_000
DELAY_MS = 20
DEADLINE_MS = 200
FEEDBACK_WINDOW = 3


@torch.no_grad()
def prepare_samples(encoder, clips):
    """为同一批视频样本构建发送计划，字节预算计入计划中的副本。"""
    samples = []
    decomposer = PublicResidualDecomposer()
    for gop_id, (start_frame, clip) in enumerate(zip(STARTS, clips)):
        public = decomposer.encode(encoder(clip)).public
        plans = {}
        for profile in PUBLIC_BASE_PROFILES:
            wire = encode_public_gop(public, profile, gop_id=gop_id, start_frame=start_frame)
            plans[profile] = build_public_plan(wire, gop_id, add_copies=True)
        samples.append((start_frame, clip, plans))
    return samples


@torch.no_grad()
def run_trace(samples, decoder, seed):
    """
    只使用历史原包的丢失量估算发送前的网络状态。
    由于档位间原包同 ID、同抽样且全数发出，三种策略共享观测口径。
    """
    history = deque(maxlen=FEEDBACK_WINDOW)
    previous_profile = None
    rows, events = [], []
    for step, (start_frame, clip, plans) in enumerate(samples):
        phase, true_loss = PHASES[step // (len(samples) // len(PHASES))]
        link = LinkSample(LINK_RATE, true_loss, DELAY_MS, OVERHEAD)
        prior_loss = (sum(lost for lost, _ in history) / sum(sent for _, sent in history)
                      if history else None)
        plan_costs = {profile: plan.planned_bytes for profile, plan in plans.items()}
        decision = choose_public_base_profile(plan_costs, BUDGET, prior_loss)
        if decision.selected_profile is None:
            raise RuntimeError("两种公共保护档均超出本次固定预算")

        observations = set()
        for policy in POLICIES:
            profile = decision.selected_profile if policy == "adaptive" else policy
            plan = plans[profile]
            received, replica_sources, report = replay_public_plan(
                plan, profile, trace_id=f"adaptive-public-base-{step}", seed=seed,
                link=link, deadline_ms=DEADLINE_MS,
            )
            source_events = [event for event in report["events"] if event["packet_id"] in plan.raw_by_id]
            source_outcomes = tuple(event["status"] for event in source_events)
            if len(source_outcomes) != 4 or "unsent" in source_outcomes:
                raise RuntimeError("原包反馈口径不满足此次对照")
            observations.add(source_outcomes)
            features, status = decode_public_gop(received, device=clip.device)
            quality = mse(decoder(features), clip) if features is not None else None
            switched = policy == "adaptive" and previous_profile is not None and previous_profile != profile
            key = {"seed": seed, "step": step, "phase": phase, "start_frame": start_frame, "policy": policy}
            rows.append({
                **key, "selected_profile": profile, "switched": switched,
                "true_loss": true_loss, "prior_original_loss": prior_loss,
                "original_lost": source_outcomes.count("lost"), "original_sent": len(source_outcomes),
                "replicas": len(replica_sources), "planned_bytes": plan.planned_bytes,
                "actual_bytes": report["actual_wire_bytes"],
                "late_packets": report["status_counts"]["late"], "receiver_status": status, "mse": quality,
            })
            events.append({
                **key, "selected_profile": profile, "profile_costs": plan_costs,
                "prior_original_loss": prior_loss, "preferred_profile": decision.preferred_profile,
                "adaptive_reason": decision.reason, "replica_sources": replica_sources, "replay": report,
            })

        if len(observations) != 1:
            raise RuntimeError("同一轨迹的原包结果不一致")
        common = observations.pop()
        history.append((common.count("lost"), len(common)))
        previous_profile = decision.selected_profile
    return rows, events


def summarize(rows):
    """覆盖率以全部样本为分母，MSE 单独使用三方共同成功的配对样本。"""
    summary = []
    for phase in [name for name, _ in PHASES] + ["all"]:
        cases = {}
        for row in rows:
            if phase == "all" or row["phase"] == phase:
                cases.setdefault((row["seed"], row["step"]), {})[row["policy"]] = row
        if any(set(case) != set(POLICIES) for case in cases.values()):
            raise RuntimeError("对照缺少策略")
        paired = [case for case in cases.values() if all(row["mse"] is not None for row in case.values())]
        for policy in POLICIES:
            selected = [case[policy] for case in cases.values()]
            summary.append({
                "phase": phase, "policy": policy, "gops": len(selected),
                "decoded": sum(row["mse"] is not None for row in selected),
                "selected_int4": sum(row["selected_profile"] == "int4" for row in selected),
                "selected_pool2": sum(row["selected_profile"] == "pool2_int8" for row in selected),
                "switches": sum(row["switched"] for row in selected),
                "mean_sent_bytes": mean(row["actual_bytes"] for row in selected),
                "max_sent_bytes": max(row["actual_bytes"] for row in selected),
                "late_packets": sum(row["late_packets"] for row in selected),
                "paired_gops": len(paired),
                "paired_mse": mean(case[policy]["mse"] for case in paired) if paired else None,
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description="公共低码率档位的自适应联调")
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
    samples = prepare_samples(encoder, clips)
    rows, events = [], []
    for seed in SEEDS:
        results, traces = run_trace(samples, decoder, seed)
        rows.extend(results)
        events.extend(traces)
        print(f"种子 {seed} 回放完成，{len(results)} 条策略记录", flush=True)

    summary = summarize(rows)
    metadata = {
        "video": str(args.video.resolve()), "video_sha256": file_sha256(args.video),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device), "gop_start_frames": list(STARTS), "seeds": list(SEEDS),
        "budget_bytes": BUDGET, "transport_overhead_per_packet": OVERHEAD,
        "rate_bps": LINK_RATE, "one_way_delay_ms": DELAY_MS, "deadline_ms": DEADLINE_MS,
        "timing": "Per-GOP synthetic replay; sampled GOPs and independent clock origins",
        "phases": PHASES, "feedback_window": FEEDBACK_WINDOW,
        "results": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "gop_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"结果已保存：{args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
