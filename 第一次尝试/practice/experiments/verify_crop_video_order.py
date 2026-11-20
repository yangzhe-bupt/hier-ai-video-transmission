"""Cross-check exported crop video against source frames in decode order.

MP4 is lossy, so frame values are compared by MSE rather than exact equality.
The paired error should be below the error from a one-frame shift.  This is a
supplement to the reconstruction script's sequential read/write invariant.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def decode(path, crop_size=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if crop_size is not None:
            h, w = frame.shape[:2]
            top, left = (h - crop_size) // 2, (w - crop_size) // 2
            frame = frame[top:top + crop_size, left:left + crop_size]
        frames.append(frame)
    capture.release()
    return frames, fps


def pair_mse(first, second):
    a = first.astype(np.float32) / 255.0
    b = second.astype(np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--crop-video", required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, source_fps = decode(args.source, args.crop_size)
    exported, exported_fps = decode(args.crop_video)
    if len(source) != len(exported):
        raise AssertionError("Source and exported videos have different frame counts")
    if not source:
        raise AssertionError("Source contains no decodable frames")
    if any(frame.shape != source[0].shape for frame in exported):
        raise AssertionError("Exported frame shape differs from the source crop")
    same = [pair_mse(source[i], exported[i]) for i in range(len(source))]
    lag = [pair_mse(source[i - 1], exported[i]) for i in range(1, len(source))]
    lead = [pair_mse(source[i + 1], exported[i]) for i in range(len(source) - 1)]
    report = {
        "input_decoded_frames": len(source), "output_decoded_frames": len(exported),
        "source_fps": source_fps, "output_fps": exported_fps,
        "paired_frame_mse_mean": float(np.mean(same)),
        "one_frame_lag_mse_mean": float(np.mean(lag)),
        "one_frame_lead_mse_mean": float(np.mean(lead)),
        "paired_beats_shifted_means": bool(np.mean(same) < np.mean(lag) and np.mean(same) < np.mean(lead)),
        "note": "Lossy MP4 comparison; sequential read/write is the primary order guarantee",
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
