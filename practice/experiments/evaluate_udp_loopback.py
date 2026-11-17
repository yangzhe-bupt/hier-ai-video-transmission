# ======================================================================
# 文件：evaluate_udp_loopback.py
# 功能：通过本机 UDP 回环连续发送池化 int8 分层档，验证真实到达与限速。
# 函数汇总：
#   - prepare：预编码连续 GOP，计入全部分片开销。
#   - transmit：以连续时钟和 1 Mbps 串行限速实际发送 UDP。
#   - evaluate：按 GOP 截止时间重组、解码或冻结，并统计质量。
# 输出：gop_metrics.csv、summary.json、逐数据报收发记录。
# ======================================================================

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
from random import Random
import socket
from statistics import mean
from threading import Event, Thread
from time import perf_counter, sleep

import cv2
import torch

from evaluate_adaptive_public_base import DEADLINE_MS, LINK_RATE
from evaluate_layered_mtu import fragment_plan
from evaluate_public_base_budget import BUDGET, build_public_plan
from evaluate_residual_enhancement import make_layered_plan
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.layered_public_codec import decode_layered_gop, encode_individual_enhancement
from hier_video.transmission.public_base_codec import encode_public_gop
from hier_video.transmission.udp_feature_fragments import FRAGMENT_HEADER, IPV4_UDP_BYTES, reassemble_feature_packets
from reconstruct_adaptive_layered_video import CROP, FPS, render_gop
from train_packet_aware_decoder import write_csv
from train_swin_video import load_swin_models
from video_feature_demo import center_crop_frames, frames_to_gop


MTU = 1500
GOP_SECONDS = 4 / FPS
DEADLINE = DEADLINE_MS / 1000


@torch.no_grad()
def prepare(video, encoder, device, first_frame, count):
    """从同一读取器连续取四帧。预编码让本轮测量聚焦在发送与接收。"""
    decomposer = PublicResidualDecomposer()
    clips, plans = [], []
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - FPS) > 0.001:
            raise ValueError("需要可读取的 24 fps 视频")
        for _ in range(first_frame):
            if not cap.grab():
                raise ValueError("无法到达起始帧")
        for index in range(count):
            frames = []
            for _ in range(4):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("连续视频帧不足")
                frames.append(frame)
            clip = frames_to_gop(center_crop_frames(frames, CROP)).to(device)
            parts = decomposer.encode(encoder(clip))
            start = first_frame + index * 4
            public = encode_public_gop(parts.public, "pool2_int8", gop_id=index, start_frame=start)
            residual = encode_individual_enhancement(parts.individual, gop_id=index, start_frame=start)
            layered, _ = make_layered_plan(build_public_plan(public, index, True), residual, index)
            _, datagrams, originals, cost = fragment_plan(layered, index, MTU)
            if cost > BUDGET:
                raise ValueError(f"GOP {index} 超出字节预算")
            clips.append(clip)
            plans.append((datagrams, originals))
    finally:
        cap.release()
    return clips, plans


def wait_until(target):
    """在同一单调时钟上等待绝对发送时刻。"""
    while (remaining := target - perf_counter()) > 0:
        if remaining > 0.001:
            sleep(remaining - 0.0007)


def transmit(plans, loss, seed):
    """
    原发及副本都占限速时间。模拟丢弃的包在计费后不调用 sendto。
    接收线程记录 recvfrom 返回时刻，和发送线程共享本机单调时钟。
    """
    rng = Random(seed)
    sent, received, errors = [], [], []
    stopped = Event()
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as rx, \
            closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as tx:
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(0.02)
        peer = rx.getsockname()

        def listen():
            try:
                while not stopped.is_set():
                    try:
                        payload, _ = rx.recvfrom(65535)
                        received.append((perf_counter(), payload))
                    except socket.timeout:
                        continue
            except OSError as error:
                errors.append(error)

        reader = Thread(target=listen)
        reader.start()
        origin = perf_counter() + 0.05
        available = origin
        try:
            for index, (datagrams, _) in enumerate(plans):
                release = origin + index * GOP_SECONDS
                for identity, payload in datagrams.items():
                    octets = len(payload) + IPV4_UDP_BYTES
                    if octets > MTU:
                        raise ValueError("UDP 数据报超出配置 MTU")
                    start = max(available, release, perf_counter())
                    wait_until(start + octets * 8 / LINK_RATE)
                    discarded = rng.random() < loss
                    if not discarded and tx.sendto(payload, peer) != len(payload):
                        raise OSError("UDP 数据报未完整提交")
                    available = perf_counter()
                    sent.append({
                        "gop": index, "packet_id": identity, "wire_bytes": octets,
                        "slot_start_s": start - origin, "slot_end_s": available - origin,
                        "injected_loss": discarded,
                    })
            wait_until(max(available, origin + (len(plans) - 1) * GOP_SECONDS + DEADLINE) + 0.03)
        finally:
            stopped.set()
            reader.join()
        if errors:
            raise errors[0]

    arrivals = []
    by_gop = defaultdict(list)
    for timestamp, payload in received:
        _, index, packet, fragment, _, _ = FRAGMENT_HEADER.unpack_from(payload)
        elapsed = timestamp - origin
        arrivals.append({"gop": index, "packet": packet, "fragment": fragment,
                         "received_s": elapsed, "udp_payload_bytes": len(payload)})
        by_gop[index].append((elapsed, payload))
    return sent, arrivals, by_gop


@torch.no_grad()
def evaluate(clips, plans, decoder, sent, by_gop):
    """按实到时间筛选分片，用实际收齐结果复现播放回退。"""
    sent_groups = defaultdict(list)
    for event in sent:
        sent_groups[event["gop"]].append(event)
    rows = []
    last = None
    for i, (clip, (_, originals)) in enumerate(zip(clips, plans)):
        release = i * GOP_SECONDS
        arrivals = by_gop[i]
        eligible = [payload for received, payload in arrivals if received <= release + DEADLINE]
        restored = reassemble_feature_packets(eligible, gop_id=i, path_mtu=MTU)
        if any(originals.get(key) != raw for key, raw in restored.items()):
            raise RuntimeError("重组结果与原包字节不同")
        public = [restored[k] for k in range(4) if k in restored]
        features, status = decode_layered_gop(public, restored.get(4), device=clip.device)
        played = render_gop(decoder, features, clip, last)
        last = played[:, -1:].clone()
        group = sent_groups[i]
        planned = sum(item["wire_bytes"] for item in group)
        submitted = [item for item in group if not item["injected_loss"]]
        if planned > BUDGET or len(arrivals) > len(submitted):
            raise RuntimeError("发送或接收计数不一致")
        rows.append({
            "gop": i, "status": status, "frozen": features is None,
            "playback_mse": (played - clip).square().mean().item(),
            "accounted_bytes": planned, "socket_wire_bytes": sum(item["wire_bytes"] for item in submitted),
            "injected_drops": len(group) - len(submitted), "unexpected_missing": len(submitted) - len(arrivals),
            "late_datagrams": len(arrivals) - len(eligible),
            "sender_queue_ms": max(0, group[0]["slot_start_s"] - release) * 1000,
            "last_arrival_ms": (arrivals[-1][0] - release) * 1000 if arrivals else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="固定池化分层的本机 UDP 回环")
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
    first_frame, count, seed = 10240, 96, 346
    clips, plans = prepare(args.video, encoder, device, first_frame, count)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"address": "127.0.0.1", "video": str(args.video), "checkpoint": str(args.checkpoint),
               "start_frame": first_frame, "gops": count, "fps": FPS, "mtu": MTU,
               "seed": seed, "rate_bps": LINK_RATE, "deadline_ms": DEADLINE_MS,
               "budget_bytes": BUDGET, "results": {}}
    all_rows = []
    for label, loss in (("clean", 0.0), ("loss_10", 0.1)):
        print(f"开始 UDP 回环：{label}", flush=True)
        sent, arrivals, by_gop = transmit(plans, loss, seed)
        rows = evaluate(clips, plans, decoder, sent, by_gop)
        all_rows.extend({"case": label, **row} for row in rows)
        (args.output_dir / f"{label}_datagrams.json").write_text(
            json.dumps({"send": sent, "receive": arrivals}, indent=2), encoding="utf-8")
        summary["results"][label] = {
            "configured_loss": loss, "datagrams": len(sent), "received": len(arrivals),
            "decoded": sum(not row["frozen"] for row in rows),
            "enhanced": sum(row["status"] == "with_enhancement" for row in rows),
            "fallback": sum(row["status"] == "base_only" for row in rows),
            "frozen_frames": 4 * sum(row["frozen"] for row in rows),
            "mean_playback_mse": mean(row["playback_mse"] for row in rows),
            "max_sender_queue_ms": max(row["sender_queue_ms"] for row in rows),
            "max_last_arrival_ms": max(row["last_arrival_ms"] for row in rows if row["last_arrival_ms"] is not None),
            "mean_accounted_bytes": mean(row["accounted_bytes"] for row in rows),
            **{key: sum(row[key] for row in rows) for key in ("late_datagrams", "injected_drops", "unexpected_missing")},
        }
        print(f"{label} 完成：接收 {len(arrivals)}/{len(sent)} 个数据报", flush=True)
    write_csv(args.output_dir / "gop_metrics.csv", all_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
