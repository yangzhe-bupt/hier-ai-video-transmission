# ======================================================================
# 文件：evaluate_live_udp.py
# 功能：在线读取、编码、限速发送、接收及解码，测量实际画面完成时间。
# 函数汇总：
#   - produce：按源帧时钟读帧，编码后将分片入队。
#   - send：按 1 Mbps 串行消费发送队列，并实际调用 UDP sendto。
#   - receive：记录实际收包时间，通过条件变量通知播放线程。
#   - run_case：按实际时间解码、判断期限并保持上一帧。
# 输出：逐 GOP 时延、逐数据报收发记录及汇总。
# ======================================================================

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
from queue import Queue
from random import Random
import socket
from statistics import mean
from threading import Condition, Event
from time import perf_counter

import cv2
import torch

from evaluate_udp_loopback import DEADLINE, GOP_SECONDS, LINK_RATE, MTU, wait_until
from evaluate_layered_mtu import fragment_plan
from evaluate_public_base_budget import BUDGET, build_public_plan
from evaluate_residual_enhancement import make_layered_plan
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.layered_public_codec import decode_layered_gop, encode_individual_enhancement
from hier_video.transmission.public_base_codec import encode_public_gop
from hier_video.transmission.udp_feature_fragments import FRAGMENT_HEADER, IPV4_UDP_BYTES, reassemble_feature_packets
from reconstruct_adaptive_layered_video import CROP, FPS
from train_packet_aware_decoder import write_csv
from train_swin_video import load_swin_models
from video_feature_demo import center_crop_frames, frames_to_gop


FIRST = 10240
COUNT = 96
RESERVE = 0.010
SEED = 346


@torch.no_grad()
def produce(cap, encoder, device, origin, targets, outgoing, encoded, failures):
    """每帧按 24 fps 读取，四帧到齐后立即编码并准备数据报。"""
    decomposer = PublicResidualDecomposer()
    try:
        for i in range(COUNT):
            frames, captured = [], []
            for j in range(4):
                wait_until(origin + (i * 4 + j) / FPS)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("源视频帧不足")
                captured.append(perf_counter())
                frames.append(frame)
            target = frames_to_gop(center_crop_frames(frames, CROP)).to(device)
            targets.put((target, captured))
            parts = decomposer.encode(encoder(target))
            start = FIRST + i * 4
            base = encode_public_gop(parts.public, "pool2_int8", gop_id=i, start_frame=start)
            residual = encode_individual_enhancement(parts.individual, gop_id=i, start_frame=start)
            plan, _ = make_layered_plan(build_public_plan(base, i, True), residual, i)
            _, datagrams, _, cost = fragment_plan(plan, i, MTU)
            if cost > BUDGET:
                raise ValueError("分片后超出 GOP 字节预算")
            ready = perf_counter()
            encoded[i] = ready
            outgoing.put((i, ready, datagrams))
    except Exception as error:
        failures.append(error)
        targets.put(None)
        raise
    finally:
        outgoing.put(None)


def send(tx, peer, outgoing, loss, sent):
    """即使是模拟丢弃的数据报，也占用正常限速时间。"""
    rng = Random(SEED)
    while (item := outgoing.get()) is not None:
        i, ready, datagrams = item
        for identity, payload in datagrams.items():
            cost = len(payload) + IPV4_UDP_BYTES
            start = perf_counter()
            wait_until(start + cost * 8 / LINK_RATE)
            dropped = rng.random() < loss
            if not dropped and tx.sendto(payload, peer) != len(payload):
                raise OSError("UDP 数据报未完整提交")
            sent.append((i, identity, cost, ready, start, perf_counter(), dropped))


def receive(rx, condition, stopped, arrivals):
    """记录 recvfrom 返回时的本机单调时间。"""
    while not stopped.is_set():
        try:
            payload, _ = rx.recvfrom(65535)
        except socket.timeout:
            continue
        when = perf_counter()
        _, gop, packet, index, count, _ = FRAGMENT_HEADER.unpack_from(payload)
        with condition:
            arrivals[gop].append((when, packet, index, count, payload))
            condition.notify_all()


@torch.no_grad()
def run_case(video, encoder, decoder, device, loss, policy="wait_enhancement"):
    """
    参数：视频、已加载的编解码器、设备、注入丢包率及接收策略。
    wait_enhancement 等待增强，base_first 在公共包收齐后立即解码。
    两种策略都以第四帧读取后 200 ms 为播放截止时间。
    """
    if policy not in ("wait_enhancement", "base_first"):
        raise ValueError("未知接收策略")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - FPS) > 0.001:
            raise ValueError("需要可读取的 24 fps 视频")
        for _ in range(FIRST):
            if not cap.grab():
                raise ValueError("无法到达起始帧")
        arrivals, sent, failures, encoded = defaultdict(list), [], [], {}
        condition, stopped = Condition(), Event()
        outgoing, targets = Queue(), Queue()
        rows = []
        last = None
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as rx, \
                closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as tx, \
                ThreadPoolExecutor(max_workers=3) as workers:
            rx.bind(("127.0.0.1", 0))
            rx.settimeout(0.02)
            origin = perf_counter() + 0.10
            producer = workers.submit(produce, cap, encoder, device, origin, targets, outgoing, encoded, failures)
            sender = workers.submit(send, tx, rx.getsockname(), outgoing, loss, sent)
            receiver = workers.submit(receive, rx, condition, stopped, arrivals)
            try:
                for i in range(COUNT):
                    item = targets.get()
                    if item is None:
                        raise failures[0]
                    target, captured = item
                    deadline = captured[-1] + DEADLINE
                    cutoff = deadline - RESERVE
                    with condition:
                        while True:
                            on_time = [row for row in arrivals[i] if row[0] <= cutoff]
                            indices = defaultdict(set)
                            sizes = {}
                            for _, packet, index, count, _ in on_time:
                                indices[packet].add(index)
                                sizes[packet] = count
                            required = range(4 if policy == "base_first" else 5)
                            if all(key in indices and len(indices[key]) == sizes[key] for key in required):
                                break
                            if (remaining := cutoff - perf_counter()) <= 0:
                                break
                            condition.wait(remaining)

                    # 记录等待结束、特征恢复结束与同步解码完成的三个时刻。
                    decode_started = perf_counter()
                    chunks = (row[-1] for row in on_time if policy != "base_first" or row[1] < 4)
                    restored = reassemble_feature_packets(chunks, gop_id=i, path_mtu=MTU)
                    public = [restored[key] for key in range(4) if key in restored]
                    features, status = decode_layered_gop(public, restored.get(4), device=device)
                    features_ready = perf_counter()
                    candidate = decoder(features) if features is not None else None
                    if candidate is not None and device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
                    finished = perf_counter()
                    missed = candidate is not None and finished > deadline
                    if candidate is None or missed:
                        hold = last if last is not None else torch.zeros_like(target[:, :1])
                        played = hold.expand_as(target)
                    else:
                        played = candidate
                    last = played[:, -1:].clone()
                    rows.append({
                        "gop": i, "decoder_status": status, "late_decode": missed,
                        "receive_wait_ms": (decode_started - captured[-1]) * 1000,
                        "reassemble_features_ms": (features_ready - decode_started) * 1000,
                        "decode_sync_ms": (finished - features_ready) * 1000 if candidate is not None else None,
                        "frozen": candidate is None or missed,
                        "frame_assembly_ms": (captured[-1] - captured[0]) * 1000,
                        "ready_from_fourth_ms": (finished - captured[-1]) * 1000 if candidate is not None else None,
                        "ready_from_first_ms": (finished - captured[0]) * 1000 if candidate is not None else None,
                        "loss_injection": loss, "playback_mse": (played - target).square().mean().item(),
                        "fourth_s": captured[-1] - origin,
                    })
                producer.result()
                sender.result()
                wait_until(origin + (COUNT * 4 - 1) / FPS + DEADLINE + 0.05)
            finally:
                stopped.set()
            receiver.result()
    finally:
        cap.release()

    grouped = defaultdict(list)
    for event in sent:
        grouped[event[0]].append(event)
    for row in rows:
        i = row["gop"]
        group = grouped[i]
        row["encode_pack_ms"] = (encoded[i] - origin - row["fourth_s"]) * 1000
        row["sender_queue_ms"] = (group[0][4] - group[0][3]) * 1000
        row["accounted_bytes"] = sum(event[2] for event in group)
        row["injected_drops"] = sum(event[-1] for event in group)
        row["received"] = len(arrivals[i])
        row["unexpected_missing"] = len(group) - row["injected_drops"] - row["received"]
    return rows, {
        "send": [{"gop": i, "id": identity, "wire_bytes": cost, "encoded_s": ready-origin,
                  "slot_start_s": start-origin, "slot_end_s": end-origin, "injected_loss": dropped}
                 for i, identity, cost, ready, start, end, dropped in sent],
        "receive": [{"gop": i, "packet": packet, "index": index, "arrived_s": when-origin}
                    for i, group in arrivals.items() for when, packet, index, _, _ in group],
    }


def main():
    parser = argparse.ArgumentParser(description="全链路在线 UDP 回环")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-receivers", action="store_true",
                        help="比较等待增强与公共优先两种接收策略")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, _ = load_swin_models(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    # 预热算子，但不预编码试验视频。
    with torch.no_grad():
        decoder(encoder(torch.zeros(1, 4, 3, CROP, CROP, device=device)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    all_rows, summary = [], {}
    policies = ("wait_enhancement", "base_first") if args.compare_receivers else ("wait_enhancement",)
    cases = [(label, loss, policy) for label, loss in (("clean", 0.0), ("loss_10", 0.1))
             for policy in policies]
    for label, loss, policy in cases:
        case = f"{label}_{policy}" if args.compare_receivers else label
        print(f"开始在线回环 {case}", flush=True)
        rows, events = run_case(args.video, encoder, decoder, device, loss, policy)
        all_rows.extend({"case": case, "policy": policy, **row} for row in rows)
        ready = [r for r in rows if r["ready_from_fourth_ms"] is not None]
        summary[case] = {
            "gops": len(rows), "decoded_on_time": sum(not r["frozen"] for r in rows),
            "late_decode": sum(r["late_decode"] for r in rows),
            "enhanced_on_time": sum(not r["frozen"] and r["decoder_status"] == "with_enhancement" for r in rows),
            "missing_public": sum(r["frozen"] and not r["late_decode"] for r in rows),
            "mean_receive_wait_ms": mean(r["receive_wait_ms"] for r in rows),
            "mean_reassemble_features_ms": mean(r["reassemble_features_ms"] for r in rows),
            "mean_decode_sync_ms": mean(r["decode_sync_ms"] for r in ready) if ready else None,
            "frozen_frames": 4 * sum(r["frozen"] for r in rows),
            "mean_mse": mean(r["playback_mse"] for r in rows),
            "max_encoder_ms": max(r["encode_pack_ms"] for r in rows),
            "max_sender_queue_ms": max(r["sender_queue_ms"] for r in rows),
            "mean_ready_from_fourth_ms": mean(r["ready_from_fourth_ms"] for r in ready) if ready else None,
            "max_ready_from_fourth_ms": max((r["ready_from_fourth_ms"] for r in ready), default=None),
            "max_ready_from_first_ms": max((r["ready_from_first_ms"] for r in ready), default=None),
            "mean_accounted_bytes": mean(r["accounted_bytes"] for r in rows),
            "injected_drops": sum(r["injected_drops"] for r in rows),
            "unexpected_missing": sum(r["unexpected_missing"] for r in rows),
        }
        (args.output_dir / f"{case}_datagrams.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        print(summary[case], flush=True)
    write_csv(args.output_dir / "gop_metrics.csv", all_rows)
    metadata = {"video": str(args.video.resolve()), "checkpoint": str(args.checkpoint.resolve()),
                "device": str(device), "ip": "127.0.0.1", "first_frame": FIRST, "seed": SEED,
                "fps": FPS, "rate_bps": LINK_RATE, "deadline_ms": DEADLINE * 1000,
                "decode_reserve_ms": RESERVE * 1000, "results": summary}
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
