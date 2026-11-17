# ======================================================================
# 文件：train_packet_aware_decoder.py
# 功能：冻结第314轮编码器，比较常规微调与丢包感知微调。
# 函数汇总：
#   - load_clips：逐次读取固定起始帧的四帧 GOP。
#   - prepare_packets：编码、分解、int8 量化并序列化数据包。
#   - receive_features：按固定种子回放、拆包、反量化并融合两支特征。
#   - main：训练两个对照解码器，在独立视频上配对评估并保存结果。
# 输入：训练视频、验证视频、冻结的编解码器检查点。
# 输出：training.csv、gop_metrics.csv、events.json、summary.json 和候选权重。
# 运行：从项目根目录通过 practice/run.py 启动本脚本。
# ======================================================================

import argparse
from collections import Counter
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from accept_biran_swin import file_sha256, restore_quantized
from hier_video.transmission.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.feature_mapping import build_group_manifest
from hier_video.transmission.network_replay import FeaturePacket, LinkSample, replay_feature_packet_events
from hier_video.transmission.packetization import build_feature_packets, unpack_feature_packets
from hier_video.transmission.quantization import quantize_symmetric
from hier_video.transmission.reception_audit import audit_quantized_reception
from train_swin_video import load_swin_models
from video_feature_demo import load_video_as_gop


def load_clips(path, starts, device):
    """加载固定源帧位置的 [1,4,3,128,128] 张量，统一放到模型设备。"""
    return [load_video_as_gop(path, 128, 4, start).to(device) for start in starts]


@torch.no_grad()
def prepare_packets(encoder, gop, gop_id, start):
    """返回固定编码器产生的 int8 包，以及拆包所需的映射和形状元信息。"""
    decomposer = PublicResidualDecomposer()
    parts = decomposer.encode(encoder(gop))
    public = quantize_symmetric(parts.public)
    individual = quantize_symmetric(parts.individual)
    manifest = build_group_manifest(public.tensor.shape, individual.tensor.shape)
    packets = build_feature_packets(
        public.tensor, individual.tensor, manifest,
        stream_id="acceptance-int8", sequence_id=gop_id, gop_id=gop_id,
        gop_start_frame=start,
        quantization_by_branch={"public": public.metadata, "individual": individual.metadata},
    )
    wire = [packet.serialize() for packet in packets]
    replay_packets = tuple(
        FeaturePacket(f"{gop_id}:{packet.header.branch}:{packet.header.group_id}",
                      f"{gop_id}:{packet.header.branch}", len(raw))
        for packet, raw in zip(packets, wire)
    )
    return parts, manifest, decomposer, wire, replay_packets


@torch.no_grad()
def receive_features(prepared, gop_id, loss, seed, device):
    """返回接收特征、原审计状态和包事件。无法反量化时特征返回 None。"""
    parts, manifest, decomposer, wire, packets = prepared
    report = replay_feature_packet_events(
        packets, tuple(packet.packet_id for packet in packets),
        trace_id=f"acceptance-gop-{gop_id}", seed=seed, decision_time_ms=0,
        deadline_ms=1000, budget_bytes=sum(map(len, wire)),
        link=LinkSample(100_000_000, loss),
        feature_interface_version=manifest.interface_version,
    )
    received_ids = {event["packet_id"] for event in report["events"]
                    if event["status"] == "received"}
    received_wire = [raw for packet, raw in zip(packets, wire)
                     if packet.packet_id in received_ids]
    if not received_wire:
        return None, "failed_no_received_packets", report
    received = unpack_feature_packets(received_wire, manifest, device=device)
    audit = audit_quantized_reception(received, manifest)
    # 沿用公共缺块置零降级。缺失反量化参数时仍视为无法解码。
    if not audit.decoder_ready:
        if (audit.status != "failed_public_incomplete"
                or not {"public", "individual"} <= received.quantization_by_branch.keys()):
            return None, audit.status, report
    features, _, _ = restore_quantized(
        received, replace(audit, decoder_ready=True), parts, decomposer, device,
    )
    return features, audit.status, report


def write_csv(path, rows):
    """将有表头的逐样本数据保存到本地。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """做配对微调，记录发生实际丢包和无法解码的次数。"""
    parser = argparse.ArgumentParser(description="本地包级丢包感知解码器微调")
    parser.add_argument("--train-video", required=True)
    parser.add_argument("--val-video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs 必须大于零")
    train_hash = file_sha256(args.train_video)
    val_hash = file_sha256(args.val_video)
    if train_hash == val_hash:
        parser.error("训练与验证视频必须不同")

    torch.manual_seed(314)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, original_decoder, checkpoint = load_swin_models(args.checkpoint, device)
    encoder.eval().requires_grad_(False)
    control_decoder = copy.deepcopy(original_decoder).train()
    aware_decoder = copy.deepcopy(original_decoder).train()
    control_optimizer = torch.optim.Adam(control_decoder.parameters(), lr=1e-4)
    aware_optimizer = torch.optim.Adam(aware_decoder.parameters(), lr=1e-4)
    train_starts = list(range(24, 745, 24))
    val_starts = list(range(100, 1201, 100))
    train_clips = load_clips(args.train_video, train_starts, device)
    train_packets = [prepare_packets(encoder, clip, i, start)
                     for i, (clip, start) in enumerate(zip(train_clips, train_starts))]
    train_clean = [receive_features(packets, i, 0, 314, device)[0]
                   for i, packets in enumerate(train_packets)]
    if any(features is None for features in train_clean):
        raise RuntimeError("训练无丢失对照无法解码")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    history = []
    for epoch in range(args.epochs):
        control_losses, aware_losses = [], []
        counts = Counter()
        for i, (clip, prepared, clean) in enumerate(zip(train_clips, train_packets, train_clean)):
            # 每轮更换丢包种子。两组看到相同样本，更新次数及学习率相同。
            damaged, _, report = receive_features(prepared, i, 0.1, 1000 + epoch, device)
            counts["lost_packets"] += report["status_counts"]["lost"]
            counts["damaged_gops"] += int(damaged is not None and report["status_counts"]["lost"] > 0)
            counts["unavailable_gops"] += int(damaged is None)
            control_optimizer.zero_grad(set_to_none=True)
            clean_loss = F.mse_loss(control_decoder(clean), clip)
            clean_loss.backward()
            control_optimizer.step()
            aware_optimizer.zero_grad(set_to_none=True)
            aware_clean_loss = F.mse_loss(aware_decoder(clean), clip)
            # 清洁与丢包输入等权。不可解码时只执行清洁项，保持更新次数。
            objective = aware_clean_loss if damaged is None else (
                aware_clean_loss + F.mse_loss(aware_decoder(damaged), clip)
            ) / 2
            objective.backward()
            aware_optimizer.step()
            control_losses.append(clean_loss.item())
            aware_losses.append(objective.item())
        history.append({"epoch": epoch + 1, "updates": len(train_clips),
                        "control_mse": mean(control_losses), "aware_objective": mean(aware_losses),
                        "damaged_gops": counts["damaged_gops"], "lost_packets": counts["lost_packets"],
                        "unavailable_gops": counts["unavailable_gops"]})
        print(history[-1], flush=True)

    decoders = {"original": original_decoder.eval(), "clean_finetune": control_decoder.eval(),
                "loss_aware": aware_decoder.eval()}
    rows, events = [], []
    with torch.no_grad():
        for gop_id, (start, clip) in enumerate(zip(val_starts, load_clips(args.val_video, val_starts, device))):
            prepared = prepare_packets(encoder, clip, gop_id, start)
            for loss, seeds in ((0.0, (314,)), (0.1, (314, 315, 316))):
                for seed in seeds:
                    received, status, report = receive_features(prepared, gop_id, loss, seed, device)
                    rows.append({
                        "start_frame": start, "loss_fraction": loss, "seed": seed,
                        "receiver_status": status, "decoder_ready": received is not None,
                        "lost_packets": report["status_counts"]["lost"],
                        **{name: F.mse_loss(decoder(received), clip).item() if received is not None else None
                           for name, decoder in decoders.items()},
                    })
                    events.append({"start_frame": start, "loss_fraction": loss, "report": report})
    result = {}
    for loss in (0.0, 0.1):
        condition = [row for row in rows if row["loss_fraction"] == loss]
        available = [row for row in condition if row["decoder_ready"]]
        result[str(loss)] = {
            "trials": len(condition), "unavailable": len(condition) - len(available),
            "mean_mse": {name: mean(row[name] for row in available) if available else None
                         for name in decoders},
        }
    write_csv(args.output_dir / "training.csv", history)
    write_csv(args.output_dir / "gop_metrics.csv", rows)
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    summary = {
        "checkpoint_sha256": file_sha256(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "train_sha256": train_hash, "val_sha256": val_hash, "train_starts": train_starts,
        "val_starts": val_starts, "learning_rate": 1e-4, "epochs": args.epochs,
        "device": str(device), "validation": result,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name in ("clean_finetune", "loss_aware"):
        candidate = dict(checkpoint, decoder={key: value.detach().cpu()
                                             for key, value in decoders[name].state_dict().items()},
                         decoder_finetune={"method": name, "epochs": args.epochs,
                                           "validation": result})
        if "val_mse" in candidate:
            candidate["decoder_finetune"]["parent_val_mse"] = candidate.pop("val_mse")
        torch.save(candidate, args.output_dir / f"{name}.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
