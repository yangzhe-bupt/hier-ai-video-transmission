"""Reconstruct every decodable frame through the Swin + int8 packet path.

Run via practice/run.py. The final GOP is padded for inference; padded frames
are excluded from output.  The MP4 is a viewing artifact, not the feature wire.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hier_video.transmission.feature_decomposition import FeatureParts, PublicResidualDecomposer
from hier_video.transmission.feature_mapping import build_group_manifest
from hier_video.transmission.packetization import build_feature_packets, unpack_feature_packets
from hier_video.transmission.quantization import dequantize_from_metadata, quantize_symmetric
from train_swin_video import load_swin_models
from video_feature_demo import center_crop_frames, frames_to_gop


def inspect_video(path, size):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open output: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (size, size):
            raise RuntimeError("Output frame size mismatch")
        count += 1
    cap.release()
    return {"frames": count, "fps": fps, "bytes": path.stat().st_size}


def run(video, checkpoint, output, *, crop_size=128, gop_size=4, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    encoder, decoder, ckpt = load_swin_models(checkpoint, device)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("Invalid input FPS")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    original_path = output / "original_crop.mp4"
    direct_path = output / "direct.mp4"
    compressed_path = output / "compressed.mp4"
    writers = [cv2.VideoWriter(str(path), fourcc, fps, (crop_size, crop_size))
               for path in (original_path, direct_path, compressed_path)]
    if not all(writer.isOpened() for writer in writers):
        raise RuntimeError("Cannot create output videos")
    rows = []
    input_count = padded_count = 0
    sums = {"direct": 0.0, "compressed": 0.0}
    total_values = 0
    decomposer = PublicResidualDecomposer()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            while True:
                frames = []
                for _ in range(gop_size):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    break
                valid = len(frames)
                gop_start = input_count
                frames = center_crop_frames(frames, crop_size)
                padded_count += gop_size - valid
                while len(frames) < gop_size:
                    frames.append(frames[-1].copy())
                gop = frames_to_gop(frames).to(device)
                features = encoder(gop)
                direct = decoder(features)
                parts = decomposer.encode(features)
                pq, iq = quantize_symmetric(parts.public), quantize_symmetric(parts.individual)
                manifest = build_group_manifest(pq.tensor.shape, iq.tensor.shape)
                packets = build_feature_packets(pq.tensor, iq.tensor, manifest,
                    sequence_id=gop_start, gop_id=gop_start, gop_start_frame=gop_start,
                    quantization_by_branch={"public": pq.metadata, "individual": iq.metadata})
                wire = [packet.serialize() for packet in packets]
                received = unpack_feature_packets(wire, manifest, device=device)
                if received.missing_group_ids:
                    raise RuntimeError(f"No-loss receiver lost groups: {received.missing_group_ids}")
                restored = decomposer.decode(FeatureParts(
                    dequantize_from_metadata(received.public, dict(received.quantization_by_branch["public"])),
                    dequantize_from_metadata(received.individual, dict(received.quantization_by_branch["individual"])),
                    parts.meta))
                compressed = decoder(restored)
                for name, result in (("direct", direct), ("compressed", compressed)):
                    difference = result[:, :valid] - gop[:, :valid]
                    sums[name] += difference.square().sum().item()
                total_values += gop[:, :valid].numel()
                def images(tensor):
                    return tensor[0, :valid].permute(0, 2, 3, 1).clamp(0, 1).mul(255).round().byte().cpu().numpy()
                outputs = (frames[:valid], images(direct), images(compressed))
                for i in range(valid):
                    writers[0].write(outputs[0][i])
                    writers[1].write(cv2.cvtColor(outputs[1][i], cv2.COLOR_RGB2BGR))
                    writers[2].write(cv2.cvtColor(outputs[2][i], cv2.COLOR_RGB2BGR))
                rows.append({"start_frame": gop_start, "valid_frames": valid,
                    "padded_frames": gop_size-valid, "packets": len(packets),
                    "payload_bytes": pq.tensor.numel()+iq.tensor.numel(),
                    "wire_bytes": sum(len(item) for item in wire),
                    "direct_mse": (direct[:, :valid]-gop[:, :valid]).square().mean().item(),
                    "compressed_mse": (compressed[:, :valid]-gop[:, :valid]).square().mean().item()})
                input_count += valid
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-started
    finally:
        cap.release()
        for writer in writers:
            writer.release()
    if not rows:
        raise RuntimeError("No input frames decoded")
    checked = {name: inspect_video(path, crop_size) for name, path in
               (("original", original_path), ("direct", direct_path), ("compressed", compressed_path))}
    if any(item["frames"] != input_count for item in checked.values()):
        raise RuntimeError("Output frame count mismatch")
    if any(not math.isclose(item["fps"], fps, rel_tol=1e-3, abs_tol=1e-3)
           for item in checked.values()):
        raise RuntimeError("Output FPS mismatch")
    with (output / "gop_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint_step_or_epoch": ckpt["epoch"],
        "checkpoint_training_unit": ckpt.get("training_unit", "epoch"),
        "checkpoint_training_scope": ckpt.get("training_scope", "unspecified"),
        "checkpoint_trained": ckpt.get("train_mse") is not None and ckpt.get("val_mse") is not None,
        "checkpoint_path": str(Path(checkpoint).resolve()), "device": str(device),
        "input_decoded_frames": input_count, "output_videos": checked,
        "gops": len(rows), "padded_frames_not_written": padded_count,
        "source_nominal_fps": fps, "direct_mse": sums["direct"]/total_values,
        "compressed_mse": sums["compressed"]/total_values,
        "payload_bytes": sum(row["payload_bytes"] for row in rows),
        "wire_bytes": sum(row["wire_bytes"] for row in rows),
        "processing_seconds_excluding_setup_and_verification": elapsed,
        "processing_fps": input_count/elapsed,
        "limitations": ["No network loss or recovery network", "MP4 size is not transmitted feature bytes",
                        "Input only includes successfully decoded frames", "Frame order is sequential by input read and output write"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--gop-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(args.video, args.checkpoint, args.output_dir, crop_size=args.crop_size,
        gop_size=args.gop_size, device=args.device)


if __name__ == "__main__":
    main()
