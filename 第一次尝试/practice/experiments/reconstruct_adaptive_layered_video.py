# ======================================================================
# 文件：reconstruct_adaptive_layered_video.py
# 功能：连续读取视频，在模拟丢包下自适应选择公共档，比较有无残差的播放。
# 函数汇总：
#   - replay_pair：对同一 GOP 回放相同的公共包，一条链路追加增强包。
#   - render_gop：恢复视频帧，不能解码时保持上一次播放帧。
#   - summarize：汇总各网络阶段的切档、降级、冻结、字节及误差。
#   - run：组织连续读帧、反馈决策、包回放、MP4 写出和核对。
# 输出：原始裁剪、公共档和分层档 MP4，以及逐 GOP 和逐包记录。
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

from evaluate_adaptive_public_base import DEADLINE_MS, DELAY_MS, FEEDBACK_WINDOW, LINK_RATE, PHASES
from evaluate_public_base_budget import BUDGET, OVERHEAD, build_public_plan, replay_public_plan
from evaluate_residual_enhancement import make_layered_plan
from train_packet_aware_decoder import write_csv
from accept_biran_swin import file_sha256, mse
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.layered_public_codec import decode_layered_gop, encode_individual_enhancement
from hier_video.transmission.network_replay import LinkSample
from hier_video.transmission.public_base_codec import GOP_FRAMES, encode_public_gop
from hier_video.transmission.public_protection_control import PUBLIC_BASE_PROFILES, choose_public_base_profile
from reconstruct_video import verify_output_video
from train_swin_video import load_swin_models
from video_feature_demo import center_crop_frames, frames_to_gop


CROP = 128
FPS = 24
VIDEO_NAMES = {
    "original": "original_crop.mp4",
    "base": "adaptive_public.mp4",
    "layered": "adaptive_layered.mp4",
}


def replay_pair(base_plan, send_plan, enhancement_id, enhancement_wire, profile, *, step, seed, link, device):
    """
    用同一轨迹回放两条链路，核对公共包的结果和帧可用性。
    返回解码器公共输入、分层输入、状态和完整包报告。
    """
    settings = {"trace_id": f"adaptive-layered-clip-{step}", "seed": seed,
                "link": link, "deadline_ms": DEADLINE_MS}
    public_wire, _, base_report = replay_public_plan(base_plan, profile, **settings)
    received, _, layered_report = replay_public_plan(send_plan, profile, **settings)
    base_events = base_report["events"]
    layer_events = layered_report["events"]
    if enhancement_id is None:
        status = "not_selected_budget"
        if layer_events != base_events:
            raise RuntimeError("增强未选中时发送计划与基础档不同")
    else:
        if layer_events[:-1] != base_events or layer_events[-1]["packet_id"] != enhancement_id:
            raise RuntimeError("增强包改变了公共包事件")
        status = layer_events[-1]["status"]
    residual = received[-1] if status == "received" else None
    layer_public = received[:-1] if residual is not None else received
    if layer_public != public_wire or (residual is not None and residual != enhancement_wire):
        raise RuntimeError("两条链路的公共包或增强包归属不一致")

    base, base_status = decode_layered_gop(public_wire, device=device)
    layered, layer_status = decode_layered_gop(layer_public, residual, device=device)
    if (base is None) != (layered is None):
        raise RuntimeError("两条链路的公共解码可用性不一致")
    if base is not None and residual is None and not torch.equal(base, layered):
        raise RuntimeError("增强不可用时未精确回退到公共特征")

    originals = [event for event in base_events if event["packet_id"] in base_plan.raw_by_id]
    if len(originals) != 4 or any(event["status"] == "unsent" for event in originals):
        raise RuntimeError("不能使用完整公共原包反馈")
    original_losses = sum(event["status"] == "lost" for event in originals)
    return (base, layered, base_status, layer_status, status,
            original_losses, base_report, layered_report)


def render_gop(decoder, features, target, last_frame):
    """
    将已恢复特征解码为四帧。无可用公共特征时，
    重复播放上一个实际输出帧；起始缺失时用黑帧。
    """
    if features is not None:
        output = decoder(features)
    else:
        hold = last_frame if last_frame is not None else torch.zeros_like(target[:, :1])
        output = hold.expand_as(target)
    if output.shape != target.shape or not torch.isfinite(output).all().item():
        raise RuntimeError("播放帧形状或数值不合法")
    return output


def write_tensor_video(writer, frames):
    """将 [1,T,3,H,W] RGB 浮点播放帧写到 OpenCV 的 BGR 写入器。"""
    images = frames[0].permute(0, 2, 3, 1).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    for rgb in images:
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def summarize(rows):
    """
    播放 MSE 包含冻结帧，帧差 MSE 统计每个 GOP 内的三个相邻时间差。
    各阶段的内容不同，质量对比以同一阶段的两条链路为准。
    """
    result = {}
    for label in [phase for phase, _ in PHASES] + ["all"]:
        group = [row for row in rows if label == "all" or row["phase"] == label]
        frozen = run = longest = 0
        for row in group:
            run = run + GOP_FRAMES if row["frozen"] else 0
            frozen += GOP_FRAMES if row["frozen"] else 0
            longest = max(longest, run)
        result[label] = {
            "gops": len(group), "selected_int4": sum(row["profile"] == "int4" for row in group),
            "selected_pool2": sum(row["profile"] == "pool2_int8" for row in group),
            "switches": sum(row["switched"] for row in group),
            "frozen_gops": sum(row["frozen"] for row in group), "frozen_frames": frozen,
            "longest_freeze_frames": longest,
            "enhancement_used_gops": sum(row["layered_status"] == "with_enhancement" for row in group),
            "public_fallback_gops": sum(not row["frozen"] and row["layered_status"] == "base_only" for row in group),
            **{f"mean_{name}": mean(row[name] for row in group) for name in
               ("base_playback_mse", "layered_playback_mse", "base_diff_mse", "layered_diff_mse",
                "base_bytes", "layered_bytes")},
            "max_layered_bytes": max(row["layered_bytes"] for row in group),
        }
    return result


@torch.no_grad()
def run(video, checkpoint, output, start_frame, frame_count, seed):
    """
    参数：
        video、checkpoint：本地视频和模型。
        output：本次运行的新目录。
        start_frame、frame_count：连续片段的起始帧和长度。
        seed：固定网络回放种子。
    """
    if start_frame < 0 or frame_count <= 0 or frame_count % (GOP_FRAMES * len(PHASES)) or seed < 0:
        raise ValueError("起始帧、种子为非负，帧数为正且需能平均分给四个阶段")
    if output.exists():
        raise FileExistsError(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(checkpoint, device)
    encoder.eval()
    decoder.eval()
    decomposer = PublicResidualDecomposer()
    rows, events = [], []
    history = deque(maxlen=FEEDBACK_WINDOW)
    last_base = last_layered = last_profile = None
    paths = {name: output / file for name, file in VIDEO_NAMES.items()}
    gops = frame_count // GOP_FRAMES

    with ExitStack() as stack:
        cap = cv2.VideoCapture(str(video))
        stack.callback(cap.release)
        if not cap.isOpened() or not math.isclose(cap.get(cv2.CAP_PROP_FPS), FPS, abs_tol=1e-3):
            raise ValueError("需要可读取的 24 fps 源视频")
        # 顺序跳过起点前的帧，再从同一个读取器连续取片段。
        for _ in range(start_frame):
            if not cap.grab():
                raise ValueError("无法顺序到达视频起始帧")

        output.mkdir(parents=True, exist_ok=False)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = {}
        for name, path in paths.items():
            writer = cv2.VideoWriter(str(path), fourcc, FPS, (CROP, CROP))
            stack.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError(f"视频写入器无法打开：{path}")
            writers[name] = writer

        for step in range(gops):
            frames = []
            first = start_frame + step * GOP_FRAMES
            for offset in range(GOP_FRAMES):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"读取连续源帧 {first + offset} 失败")
                frames.append(frame)
            cropped = center_crop_frames(frames, CROP)
            target = frames_to_gop(cropped).to(device)
            parts = decomposer.encode(encoder(target))
            residual = encode_individual_enhancement(parts.individual, gop_id=step, start_frame=first)

            base_plans, send_plans, enhancement_ids = {}, {}, {}
            for profile in PUBLIC_BASE_PROFILES:
                wire = encode_public_gop(parts.public, profile, gop_id=step, start_frame=first)
                base = build_public_plan(wire, step, add_copies=True)
                base_plans[profile] = base
                if base.planned_bytes + len(residual) + OVERHEAD <= BUDGET:
                    send_plans[profile], enhancement_ids[profile] = make_layered_plan(base, residual, step)
                else:
                    send_plans[profile], enhancement_ids[profile] = base, None

            prior_loss = sum(history) / (len(history) * 4) if history else None
            costs = {profile: plan.planned_bytes for profile, plan in send_plans.items()}
            decision = choose_public_base_profile(costs, BUDGET, prior_loss)
            profile = decision.selected_profile
            if profile is None:
                raise RuntimeError("连续试验没有预算允许的公共发送方案")
            phase, loss = PHASES[step // (gops // len(PHASES))]
            link = LinkSample(LINK_RATE, loss, DELAY_MS, OVERHEAD)
            base, layered, base_status, layer_status, extra_status, observed, base_report, layer_report = replay_pair(
                base_plans[profile], send_plans[profile], enhancement_ids[profile], residual, profile,
                step=step, seed=seed, link=link, device=device,
            )
            history.append(observed)
            played_base = render_gop(decoder, base, target, last_base)
            played_layered = render_gop(decoder, layered, target, last_layered)
            last_base = played_base[:, -1:].clone()
            last_layered = played_layered[:, -1:].clone()
            for frame in cropped:
                writers["original"].write(frame)
            write_tensor_video(writers["base"], played_base)
            write_tensor_video(writers["layered"], played_layered)

            target_diff = target[:, 1:] - target[:, :-1]
            tags = {"step": step, "first_frame": first, "time_s": step * GOP_FRAMES / FPS,
                    "phase": phase, "true_loss": loss, "prior_loss": prior_loss, "profile": profile}
            row = {
                **tags, "switched": last_profile is not None and last_profile != profile,
                "original_losses": observed, "base_status": base_status,
                "layered_status": layer_status, "enhancement_status": extra_status, "frozen": base is None,
                "base_bytes": base_report["actual_wire_bytes"], "layered_bytes": layer_report["actual_wire_bytes"],
                "base_playback_mse": mse(played_base, target), "layered_playback_mse": mse(played_layered, target),
                "base_diff_mse": mse(played_base[:, 1:] - played_base[:, :-1], target_diff),
                "layered_diff_mse": mse(played_layered[:, 1:] - played_layered[:, :-1], target_diff),
            }
            rows.append(row)
            events.append({**tags, "decision_reason": decision.reason, "plan_costs": costs,
                           "replica_sources": base_plans[profile].source_ids,
                           "enhancement_packet_id": enhancement_ids[profile],
                           "base": base_report, "layered": layer_report})
            last_profile = profile
            if (step + 1) % (gops // len(PHASES)) == 0:
                print(f"{phase} 阶段完成，已写出 {(step + 1) * GOP_FRAMES}/{frame_count} 帧", flush=True)

    verified = {name: verify_output_video(path, frame_count, CROP) for name, path in paths.items()}
    if any(not math.isclose(item["fps"], FPS, abs_tol=1e-3) for item in verified.values()):
        raise RuntimeError("输出视频帧率错误")
    summary = summarize(rows)
    metadata = {
        "input": str(video.resolve()), "video_sha256": file_sha256(video),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "first_frame": start_frame, "last_frame": start_frame + frame_count - 1,
        "fps": FPS, "seed": seed, "device": str(device), "duration_s": frame_count / FPS,
        "budget_bytes_per_gop": BUDGET, "per_packet_overhead": OVERHEAD,
        "link_rate_bps": LINK_RATE, "deadline_ms": DEADLINE_MS, "delay_ms": DELAY_MS,
        "phases": PHASES, "feedback_window": FEEDBACK_WINDOW,
        "clock_note": "Continuous source and playback frames; link serialization clock resets per GOP",
        "output_videos": verified, "results": summary,
    }
    write_csv(output / "timeline.csv", rows)
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"对照视频与回放记录已保存：{output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="从连续视频导出公共档和分层档自适应播放")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=10240)
    parser.add_argument("--frames", type=int, default=384)
    parser.add_argument("--seed", type=int, default=342)
    args = parser.parse_args()
    run(args.video, args.checkpoint, args.output_dir, args.start_frame, args.frames, args.seed)


if __name__ == "__main__":
    main()
