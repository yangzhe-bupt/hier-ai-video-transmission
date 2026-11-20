# ======================================================================
# 文件：transmit_full_frame.py
# 功能：保留整幅画面，通过现有分层 UDP 档传输完整本地视频。
# 函数汇总：
#   - fit_frame：等比缩放、补边和记录有效画面区域。
#   - prepare_video：读到视频末尾，编码各 GOP，最后一组必要时补帧。
#   - reconstruct：按实际到达时间重组并生成原尺寸 MP4。
#   - verify_video：复查视频帧数、分辨率和 24 fps。
# 说明：传输的是缩放后的语义特征；输出以原尺寸写出，无音频。
# ======================================================================

import argparse
import json
import math
from pathlib import Path

import cv2
import torch

from evaluate_layered_mtu import fragment_plan
from evaluate_public_base_budget import BUDGET
from evaluate_residual_enhancement import make_layered_plan
from evaluate_udp_loopback import DEADLINE, GOP_SECONDS, MTU, transmit
from evaluate_public_base_budget import build_public_plan
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.layered_public_codec import decode_layered_gop, encode_individual_enhancement
from hier_video.transmission.public_base_codec import encode_public_gop
from hier_video.transmission.udp_feature_fragments import reassemble_feature_packets
from reconstruct_adaptive_layered_video import render_gop
from train_packet_aware_decoder import write_csv
from train_swin_video import load_swin_models
from video_feature_demo import frames_to_gop


SIDE = 128
FPS = 24


def fit_frame(frame):
    """保留原画面左右和上下边界，按宽高比缩放到可放入 128×128 的尺寸。"""
    h, w = frame.shape[:2]
    factor = SIDE / max(h, w)
    width, height = max(1, round(w * factor)), max(1, round(h * factor))
    x, y = (SIDE - width) // 2, (SIDE - height) // 2
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(resized, y, SIDE - height - y, x, SIDE - width - x,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas, (x, y, width, height)


def verify_video(path, size, frames):
    """重新解码 MP4 核对帧数和帧尺寸。"""
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not cap.isOpened() or not math.isclose(cap.get(cv2.CAP_PROP_FPS), FPS, abs_tol=0.001):
            raise ValueError("输出格式或帧率错误")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1::-1] != size:
                raise ValueError("输出尺寸不同")
            count += 1
    finally:
        cap.release()
    if count != frames:
        raise ValueError(f"帧数不符：{count}/{frames}")
    return count


@torch.no_grad()
def prepare_video(video, encoder, device):
    """在固定 24 fps 条件下，连续编码完整源视频。最后一组仅补齐模型输入。"""
    cap = cv2.VideoCapture(str(video))
    clips, plans, counts = [], [], []
    decomposer = PublicResidualDecomposer()
    try:
        if not cap.isOpened() or not math.isclose(cap.get(cv2.CAP_PROP_FPS), FPS, abs_tol=0.001):
            raise ValueError("需要能读取的 24 fps 源视频")
        source_size = None
        while True:
            frames = []
            for _ in range(4):
                ok, frame = cap.read()
                if not ok:
                    break
                if source_size is None:
                    source_size = frame.shape[1::-1]
                if frame.shape[1::-1] != source_size or any(n % 2 for n in source_size):
                    raise ValueError("源视频须保持固定偶数尺寸")
                canvas, box = fit_frame(frame)
                frames.append(canvas)
            if not frames:
                break
            valid = len(frames)
            frames.extend([frames[-1]] * (4 - valid))
            clip = frames_to_gop(frames)
            parts = decomposer.encode(encoder(clip.to(device)))
            index = len(plans)
            public = encode_public_gop(parts.public, "pool2_int8", gop_id=index, start_frame=index * 4)
            extra = encode_individual_enhancement(parts.individual, gop_id=index, start_frame=index * 4)
            plan, _ = make_layered_plan(build_public_plan(public, index, True), extra, index)
            _, datagrams, originals, cost = fragment_plan(plan, index, MTU)
            if cost > BUDGET:
                raise ValueError(f"GOP {index} 超出字节预算")
            clips.append(clip)
            plans.append((datagrams, originals))
            counts.append(valid)
            if valid < 4:
                break
    finally:
        cap.release()
    if not clips:
        raise ValueError("源视频为空")
    return clips, plans, counts, source_size, box


@torch.no_grad()
def reconstruct(decoder, device, clips, plans, counts, by_gop, output, size, box):
    """仅用实际收到且准时的分片重建。去除补边，输出所有真实帧。"""
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    x, y, width, height = box
    rows, last = [], None
    try:
        if not writer.isOpened():
            raise RuntimeError("无法创建 MP4")
        for i, (clip, (_, originals), count) in enumerate(zip(clips, plans, counts)):
            received = by_gop[i]
            on_time = [raw for arrival, raw in received if arrival <= i * GOP_SECONDS + DEADLINE]
            restored = reassemble_feature_packets(on_time, gop_id=i, path_mtu=MTU)
            if any(originals.get(key) != data for key, data in restored.items()):
                raise RuntimeError("重组原包不一致")
            public = [restored[key] for key in range(4) if key in restored]
            features, status = decode_layered_gop(public, restored.get(4), device=device)
            clip = clip.to(device)
            played = render_gop(decoder, features, clip, last)
            last = played[:, count - 1:count].clone()
            content = played[:, :count, :, y:y + height, x:x + width]
            reference = clip[:, :count, :, y:y + height, x:x + width]
            error = (content - reference).square().mean().item()
            rgb = (content[0].permute(0, 2, 3, 1).clamp(0, 1).mul(255).round()
                   .to(torch.uint8).cpu().numpy())
            for frame in rgb:
                image = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append({"gop": i, "first_frame": i * 4, "frames": count, "status": status,
                         "frozen": features is None, "content_mse": error,
                         "received_datagrams": len(received), "late_datagrams": len(received) - len(on_time)})
    finally:
        writer.release()
    verify_video(output, size, sum(counts))
    return rows


def main():
    parser = argparse.ArgumentParser(description="全画面视频 UDP 回环")
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
    clips, plans, counts, original_size, box = prepare_video(args.video, encoder, device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"input": str(args.video.resolve()), "checkpoint": str(args.checkpoint.resolve()),
              "device": str(device), "fps": FPS, "frames": sum(counts), "gops": len(counts),
              "output_width_height": original_size, "letterbox_xywh": box, "model_side": SIDE,
              "padding_frames_not_written": 4 - counts[-1], "results": {}}
    timeline = []
    for label, loss in (("clean", 0.0), ("loss_10", 0.1)):
        print(f"开始 {label} 全片回环，{sum(counts)} 帧", flush=True)
        sent, arrivals, by_gop = transmit(plans, loss, 346)
        filename = f"full_frame_{label}.mp4"
        rows = reconstruct(decoder, device, clips, plans, counts, by_gop,
                           args.output_dir / filename, original_size, box)
        timeline.extend({"case": label, **row} for row in rows)
        summary = {
            "mp4": filename, "verified_frames": sum(counts),
            "decoded_gops": sum(not row["frozen"] for row in rows),
            "frozen_frames": sum(row["frames"] for row in rows if row["frozen"]),
            "injected": sum(item["injected_loss"] for item in sent),
            "offered_datagrams": len(sent), "received_datagrams": len(arrivals),
            "unexpected_missing": sum(not item["injected_loss"] for item in sent) - len(arrivals),
            "mean_accounted_bytes_per_gop": sum(item["wire_bytes"] for item in sent) / len(counts),
            "max_accounted_bytes_per_gop": max(sum(len(p) + 28 for p in d.values()) for d, _ in plans),
            "content_mse": sum(row["content_mse"] * row["frames"] for row in rows) / sum(counts),
        }
        result["results"][label] = summary
        (args.output_dir / f"{label}_datagrams.json").write_text(
            json.dumps({"send": sent, "receive": arrivals}, indent=2), encoding="utf-8")
        print(summary, flush=True)
    write_csv(args.output_dir / "gop_metrics.csv", timeline)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
