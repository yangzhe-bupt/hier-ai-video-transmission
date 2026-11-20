# ======================================================================
# 文件：reconstruct_adaptive_mtu_video.py
# 功能：按 MTU 分片后的成本选档，比较自适应与固定档位的连续播放。
# 函数汇总：
#   - prepare_plans：为同一 GOP 准备 int4 增强档与池化保护档。
#   - play_step：模拟传输、重组和接收回退，输出四帧播放画面。
#   - summarize：统计播放 MSE、冻结、切档及发送成本。
#   - run：连续读取源视频并保存视频、逐 GOP 和逐数据报记录。
# 输出：种子 342 的对照 MP4，所有种子的 timeline.csv、events.json、summary.json。
# ======================================================================

import argparse
from collections import deque
from contextlib import ExitStack
import json
import math
from pathlib import Path
from statistics import mean

import cv2
import torch

from accept_biran_swin import file_sha256, mse
from evaluate_adaptive_public_base import DEADLINE_MS, DELAY_MS, FEEDBACK_WINDOW, LINK_RATE, PHASES
from evaluate_layered_mtu import SEEDS, fragment_plan, receive_fragmented
from evaluate_public_base_budget import BUDGET, build_public_plan
from evaluate_residual_enhancement import make_layered_plan
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.layered_public_codec import encode_individual_enhancement
from hier_video.transmission.public_base_codec import GOP_FRAMES, encode_public_gop
from hier_video.transmission.public_protection_control import choose_public_base_profile
from reconstruct_adaptive_layered_video import CROP, FPS, render_gop, write_tensor_video
from reconstruct_video import verify_output_video
from train_packet_aware_decoder import write_csv
from train_swin_video import load_swin_models
from video_feature_demo import center_crop_frames, frames_to_gop


MTU = 1500
POLICIES = ("adaptive", "int4", "pool2_int8")


def prepare_plans(parts, step, start_frame):
    """保留两档的增强包，只给池化档加公共副本，成本使用分片后的字节。"""
    enhancement = encode_individual_enhancement(parts.individual, gop_id=step, start_frame=start_frame)
    plans = {}
    for profile in POLICIES[1:]:
        wire = encode_public_gop(parts.public, profile, gop_id=step, start_frame=start_frame)
        base = build_public_plan(wire, step, add_copies=profile == "pool2_int8")
        layered, _ = make_layered_plan(base, enhancement, step)
        plan = fragment_plan(layered, step, MTU)
        if plan[-1] > BUDGET:
            raise RuntimeError(f"固定对照 {profile} 在 GOP {step} 超出分片预算")
        plans[profile] = plan
    return plans


def play_step(plan, decoder, target, last_frame, *, step, seed, rate):
    """
    回放实际 UDP 负载。只将已发送原发公共数据报纳入控制反馈。
    公共块缺失时冻结上一输出帧，增强缺失时按原接收规则回退。
    """
    features, metrics, report = receive_fragmented(
        plan, device=target.device, gop_id=step, mtu=MTU, rate=rate, seed=seed)
    originals = [
        row for row in report["events"]
        if row["group_id"] == "public" and ":replica:" not in row["packet_id"]
        and row["status"] != "unsent"
    ]
    played = render_gop(decoder, features, target, last_frame)
    feedback = (sum(row["status"] == "lost" for row in originals), len(originals))
    return played, metrics, report, feedback


def summarize(rows):
    """逐种子统计连续冻结，所有计划的播放 MSE 均包含冻结帧。"""
    result = {}
    for phase in [name for name, _ in PHASES] + ["all"]:
        result[phase] = {}
        for policy in POLICIES:
            subset = [row for row in rows if row["policy"] == policy
                      and (phase == "all" or row["phase"] == phase)]
            longest = 0
            runs = dict.fromkeys(SEEDS, 0)
            for row in subset:
                seed = row["seed"]
                runs[seed] = runs[seed] + GOP_FRAMES if row["frozen"] else 0
                longest = max(longest, runs[seed])
            result[phase][policy] = {
                "gops": len(subset),
                "int4_gops": sum(row["profile"] == "int4" for row in subset),
                "pool2_gops": sum(row["profile"] == "pool2_int8" for row in subset),
                "switches": sum(row["switched"] for row in subset),
                "decoded_gops": sum(not row["frozen"] for row in subset),
                "enhanced_gops": sum(row["status"] == "with_enhancement" for row in subset),
                "fallback_gops": sum(row["status"] == "base_only" for row in subset),
                "frozen_frames": GOP_FRAMES * sum(row["frozen"] for row in subset),
                "longest_freeze_frames": longest,
                "playback_mse": mean(row["playback_mse"] for row in subset),
                "mean_sent_bytes": mean(row["sent_bytes"] for row in subset),
                "max_sent_bytes": max(row["sent_bytes"] for row in subset),
                "late_datagrams": sum(row["late_datagrams"] for row in subset),
            }
    return result


@torch.no_grad()
def run(args):
    """
    参数：本地视频、检查点、独立输出目录、片段起始帧和帧数。
    每组读取一次并共用编码结果，各种子与策略保留自己的播放历史。
    """
    if args.start_frame < 0 or args.frames <= 0 or args.frames % (GOP_FRAMES * len(PHASES)):
        raise ValueError("帧数应为正并能均分给四个 GOP 阶段")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    decomposer = PublicResidualDecomposer()
    gops = args.frames // GOP_FRAMES
    feedback = {seed: deque(maxlen=FEEDBACK_WINDOW) for seed in SEEDS}
    prior_profiles = dict.fromkeys(SEEDS)
    displayed = {(seed, policy): None for seed in SEEDS for policy in POLICIES}
    rows, events = [], []
    paths = {name: args.output_dir / f"{name}_mtu1500.mp4" for name in ("original", *POLICIES)}

    with ExitStack() as stack:
        cap = cv2.VideoCapture(str(args.video))
        stack.callback(cap.release)
        if not cap.isOpened() or not math.isclose(cap.get(cv2.CAP_PROP_FPS), FPS, abs_tol=1e-3):
            raise ValueError("需要可读取的 24 fps 源视频")
        for _ in range(args.start_frame):
            if not cap.grab():
                raise ValueError("无法到达连续片段起始帧")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        writers = {}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for name, path in paths.items():
            writer = cv2.VideoWriter(str(path), fourcc, FPS, (CROP, CROP))
            stack.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError(f"视频写入器无法打开：{path}")
            writers[name] = writer

        for step in range(gops):
            first_frame = args.start_frame + step * GOP_FRAMES
            frames = []
            for offset in range(GOP_FRAMES):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"源帧 {first_frame + offset} 读取失败")
                frames.append(frame)
            cropped = center_crop_frames(frames, CROP)
            for frame in cropped:
                writers["original"].write(frame)
            target = frames_to_gop(cropped).to(device)
            parts = decomposer.encode(encoder(target))
            plans = prepare_plans(parts, step, first_frame)
            costs = {profile: plan[-1] for profile, plan in plans.items()}
            phase, rate = PHASES[step // (gops // len(PHASES))]

            for seed in SEEDS:
                history = feedback[seed]
                prior = sum(lost for lost, _ in history) / sum(sent for _, sent in history) if history else None
                decision = choose_public_base_profile(costs, BUDGET, prior)
                if decision.selected_profile is None:
                    raise RuntimeError("控制器找不到可回放的预算内方案")
                for policy in POLICIES:
                    profile = decision.selected_profile if policy == "adaptive" else policy
                    key = (seed, policy)
                    played, metrics, report, observed = play_step(
                        plans[profile], decoder, target, displayed[key], step=step, seed=seed, rate=rate)
                    displayed[key] = played[:, -1:].clone()
                    if seed == SEEDS[0]:
                        write_tensor_video(writers[policy], played)
                    switched = False
                    if policy == "adaptive":
                        switched = prior_profiles[seed] is not None and prior_profiles[seed] != profile
                        prior_profiles[seed] = profile
                        if observed[1]:
                            history.append(observed)
                    tags = {"seed": seed, "step": step, "first_frame": first_frame, "time_s": step * GOP_FRAMES / FPS,
                            "phase": phase, "true_loss": rate, "policy": policy, "profile": profile}
                    rows.append({
                        **tags, "switched": switched, "prior_original_datagram_loss": prior,
                        "original_datagrams_lost": observed[0], "original_datagrams_sent": observed[1],
                        "planned_bytes": costs[profile], "frozen": metrics["restored_public"] != 4,
                        "playback_mse": mse(played, target), **metrics,
                    })
                    events.append({**tags, "profile_costs": costs, "adaptive_reason": decision.reason, "replay": report})
            if (step + 1) % (gops // len(PHASES)) == 0:
                print(f"{phase} 阶段完成，已处理 {(step + 1) * GOP_FRAMES}/{args.frames} 帧", flush=True)

    videos = {name: verify_output_video(path, args.frames, CROP) for name, path in paths.items()}
    if any(not math.isclose(video["fps"], FPS, abs_tol=1e-3) for video in videos.values()):
        raise RuntimeError("写出帧率不一致")
    meta = {
        "video": str(args.video.resolve()), "video_sha256": file_sha256(args.video),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "start_frame": args.start_frame, "frames": args.frames, "fps": FPS, "mtu": MTU,
        "budget_bytes": BUDGET, "link_rate_bps": LINK_RATE, "deadline_ms": DEADLINE_MS, "delay_ms": DELAY_MS,
        "seeds": SEEDS, "mp4_seed": SEEDS[0], "device": str(device), "phases": PHASES,
        "feedback_window": FEEDBACK_WINDOW,
        "clock": "sequential video; synthetic link clock restarts per GOP",
        "videos": videos, "results": summarize(rows),
    }
    write_csv(args.output_dir / "timeline.csv", rows)
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"分片自适应视频和记录已保存：{args.output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="MTU 分片后自适应分层播放")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=10240)
    parser.add_argument("--frames", type=int, default=384)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
