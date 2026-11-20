"""Reproducible GOP acceptance run for Swin + Biran's packet path.

Run through practice/run.py.  This script does not substitute a demo checkpoint
for the official epoch-314 checkpoint; checkpoint identity is recorded in the
result and must be inspected before interpreting quality.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import torch
import torch.nn.functional as F

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hier_video.transmission.feature_decomposition import FeatureParts, PublicResidualDecomposer
from hier_video.transmission.feature_mapping import build_group_manifest
from hier_video.transmission.network_replay import (
    FeaturePacket as ReplayPacket,
    LinkSample,
    replay_feature_packet_events,
)
from hier_video.transmission.packetization import build_feature_packets, unpack_feature_packets
from hier_video.transmission.quantization import dequantize_from_metadata, quantize_symmetric
from hier_video.transmission.reception_audit import audit_quantized_reception
from train_swin_video import load_swin_models
from video_feature_demo import load_video_as_gop


CONFIG_FIELDS = {
    "schema_version", "stream_id", "layout", "tensor_shape", "mapping_version",
    "interleave_version", "quantization",
}
RTOL = 1e-5
ATOL = 1e-6


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mse(a, b):
    return float((a - b).square().mean().item())


def max_abs(a, b):
    return float((a - b).abs().max().item())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_by_category(packet, raw):
    """Partition the exact JSON wire bytes; transport overhead is configured 0.

    A comma is assigned to the field that follows it.  The 4-byte length
    prefix and the two JSON braces belong to the fragment-header category.
    """
    fields = packet.header.as_dict()
    config = fragment = 0
    for index, key in enumerate(sorted(fields)):
        token = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        token += ":" + json.dumps(fields[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        size = len(token.encode("utf-8")) + (1 if index else 0)
        if key in CONFIG_FIELDS:
            config += size
        else:
            fragment += size
    fragment += 6  # four-byte header length and two JSON braces
    payload = len(packet.payload)
    if payload + config + fragment != len(raw):
        raise AssertionError("Serialized packet byte accounting does not balance")
    return {"data_bytes": payload, "config_bytes": config,
            "fragment_header_bytes": fragment, "transport_header_bytes": 0,
            "sent_total_bytes": len(raw)}


def restore_quantized(received, audit, parts, decomposer, device):
    if not audit.decoder_ready:
        raise RuntimeError(f"Receiver state {audit.status} cannot enter the decoder")
    public = dequantize_from_metadata(
        received.public, dict(received.quantization_by_branch["public"]),
        dtype=parts.meta.original_dtype,
    ).to(device)
    individual = dequantize_from_metadata(
        received.individual, dict(received.quantization_by_branch["individual"]),
        dtype=parts.meta.original_dtype,
    ).to(device)
    return decomposer.decode(FeatureParts(public, individual, parts.meta)), public, individual


def factor2_diagnostic(features, public):
    """Historical pool x2 / bilinear upsample, outside the production path."""
    b, t, c, h, w = features.shape
    residual = (features - public).reshape(b * t, c, h, w)
    pooled = F.avg_pool2d(residual, kernel_size=2, stride=2)
    upsampled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
    return public + upsampled.reshape(b, t, c, h, w)


def mask_slice(tensor, region):
    return tensor[
        region.b[0]:region.b[1], region.t[0]:region.t[1],
        region.c[0]:region.c[1], region.h[0]:region.h[1],
        region.w[0]:region.w[1],
    ]


def save_preview(path, original, direct, compressed, dropped):
    images = []
    for title, tensor in zip(
        ("source", "direct", "int8 no loss", "one packet lost"),
        (original, direct, compressed, dropped),
    ):
        # The specified packet belongs to t=3, so show the affected frame.
        rgb = tensor[0, -1].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        header = torch.full((24, frame.shape[1], 3), 255, dtype=torch.uint8).numpy()
        cv2.putText(header, title, (3, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.39, (20, 20, 20), 1, cv2.LINE_AA)
        images.append(cv2.vconcat((header, frame)))
    if not cv2.imwrite(str(path), cv2.hconcat(images)):
        raise RuntimeError(f"Could not write preview to {path}")


def run_one(encoder, decoder, gop, *, gop_id, start_frame, output, seed, loss_fraction,
            rate_bps, deadline_ms, device):
    decomposer = PublicResidualDecomposer()
    x = gop.to(device)
    timings = {}

    def timed(label, fn):
        sync(device)
        started = time.perf_counter()
        result = fn()
        sync(device)
        timings[label] = (time.perf_counter() - started) * 1000
        return result

    with torch.no_grad():
        features = timed("encode_ms", lambda: encoder(x))
        direct = timed("direct_decode_ms", lambda: decoder(features))
        parts = timed("decompose_ms", lambda: decomposer.encode(features))
        a_features = timed("factor1_restore_ms", lambda: decomposer.decode(parts))
        a_video = decoder(a_features)

        factor2_features = factor2_diagnostic(features, parts.public)
        factor2_video = decoder(factor2_features)

        float_manifest = build_group_manifest(parts.public.shape, parts.individual.shape)
        float_packets = timed("float_packetize_ms", lambda: build_feature_packets(
            parts.public, parts.individual, float_manifest,
            stream_id="acceptance-float", sequence_id=gop_id, gop_id=gop_id,
            gop_start_frame=start_frame,
        ))
        float_wire = timed("float_serialize_ms", lambda: [p.serialize() for p in float_packets])
        float_received = timed("float_receive_ms", lambda: unpack_feature_packets(
            float_wire, float_manifest, device=device,
        ))
        if float_received.missing_group_ids:
            raise AssertionError("No-loss float receiver reports a missing group")
        b_features = decomposer.decode(FeatureParts(
            float_received.public, float_received.individual, parts.meta,
        ))
        b_video = decoder(b_features)
        feature_diff = max_abs(a_features, b_features)
        image_diff = max_abs(a_video, b_video)
        consistent = bool(torch.allclose(a_features, b_features, rtol=RTOL, atol=ATOL)
                          and torch.allclose(a_video, b_video, rtol=RTOL, atol=ATOL))
        if not consistent:
            raise AssertionError("A split/restore and B float packet/restore differ")

        pq, iq = timed("quantize_ms", lambda: (
            quantize_symmetric(parts.public), quantize_symmetric(parts.individual)
        ))
        manifest = build_group_manifest(pq.tensor.shape, iq.tensor.shape)
        packets = timed("int8_packetize_ms", lambda: build_feature_packets(
            pq.tensor, iq.tensor, manifest, stream_id="acceptance-int8",
            sequence_id=gop_id, gop_id=gop_id, gop_start_frame=start_frame,
            quantization_by_branch={"public": pq.metadata, "individual": iq.metadata},
        ))
        wire = timed("int8_serialize_ms", lambda: [p.serialize() for p in packets])
        no_loss = timed("int8_receive_ms", lambda: unpack_feature_packets(wire, manifest, device=device))
        no_loss_audit = audit_quantized_reception(no_loss, manifest)
        assert no_loss_audit.status == "complete"
        c_features, c_public, c_individual = timed("int8_restore_ms", lambda:
            restore_quantized(no_loss, no_loss_audit, parts, decomposer, device))
        c_video = timed("int8_decode_ms", lambda: decoder(c_features))

        replay_packets = tuple(ReplayPacket(
            packet_id=f"{gop_id}:{p.header.branch}:{p.header.group_id}",
            group_id=f"{gop_id}:{p.header.branch}", payload_bytes=len(raw),
        ) for p, raw in zip(packets, wire))
        ids = tuple(p.packet_id for p in replay_packets)
        replay_args = dict(
            trace_id=f"acceptance-gop-{gop_id}", seed=seed,
            decision_time_ms=0, deadline_ms=deadline_ms,
            budget_bytes=sum(len(raw) for raw in wire),
            link=LinkSample(rate_bps=rate_bps, loss_fraction=loss_fraction),
            feature_interface_version=manifest.interface_version,
        )
        replay = replay_feature_packet_events(replay_packets, ids, **replay_args)
        reproducible = replay == replay_feature_packet_events(replay_packets, ids, **replay_args)
        if not reproducible:
            raise AssertionError("Same seed and packet IDs did not reproduce events")
        received_ids = {e["packet_id"] for e in replay["events"] if e["status"] == "received"}
        replay_wire = [raw for p, raw in zip(replay_packets, wire) if p.packet_id in received_ids]
        replay_received = unpack_feature_packets(replay_wire, manifest, device=device) if replay_wire else None
        replay_audit = audit_quantized_reception(replay_received, manifest) if replay_received else None

        # Exercise three explicit receiver abnormal/degraded paths on GOP 0.
        anomalies = {}
        if gop_id == 0:
            # A separate constrained trace proves late packets count as sent,
            # whereas packets never put on the link contribute zero bytes.
            constrained = replay_feature_packet_events(
                replay_packets, ids[:-1],
                trace_id="acceptance-deadline-and-unsent", seed=seed,
                decision_time_ms=0, deadline_ms=2,
                budget_bytes=sum(len(raw) for raw in wire),
                link=LinkSample(rate_bps=1_000_000, loss_fraction=0.0,
                                one_way_delay_ms=10),
                feature_interface_version=manifest.interface_version,
            )
            assert constrained["status_counts"]["late"] >= 1
            assert constrained["status_counts"]["unsent"] >= 1
            assert constrained["actual_wire_bytes"] == sum(
                event["wire_bytes"] for event in constrained["events"]
            )
            public_packets = [p for p in packets if p.header.branch == "public"]
            individual_packets = [p for p in packets if p.header.branch == "individual"]
            one_drop = individual_packets[-1]
            one_drop_wire = [raw for p, raw in zip(packets, wire) if p is not one_drop]
            one_drop_received = unpack_feature_packets(one_drop_wire, manifest, device=device)
            one_drop_audit = audit_quantized_reception(one_drop_received, manifest)
            dropped_features, _, _ = restore_quantized(
                one_drop_received, one_drop_audit, parts, decomposer, device,
            )
            dropped_video = decoder(dropped_features)
            expected_mask = torch.zeros_like(iq.tensor, dtype=torch.bool)
            mask_slice(expected_mask, one_drop.header.tensor_region).fill_(True)
            mask_exact = bool(torch.equal(one_drop_audit.individual_missing_mask, expected_mask))
            preserved = bool(torch.equal(
                one_drop_received.individual[~expected_mask], iq.tensor[~expected_mask]
            ) and torch.equal(one_drop_received.public, pq.tensor))
            if not mask_exact or not preserved:
                raise AssertionError("One-drop mask or received data is incorrect")
            save_preview(output / "one_drop_preview.png", x, direct, c_video, dropped_video)

            # The header remains valid and checksummed; only the public branch
            # quantization configuration is removed from every public packet.
            missing_config_packets = [
                replace(p, header=replace(p.header, quantization=None))
                if p.header.branch == "public" else p for p in packets
            ]
            missing_config_received = unpack_feature_packets(
                [p.serialize() for p in missing_config_packets], manifest, device=device,
            )
            config_audit = audit_quantized_reception(missing_config_received, manifest)
            public_incomplete = unpack_feature_packets(
                [raw for p, raw in zip(packets, wire) if p is not public_packets[0]],
                manifest, device=device,
            )
            public_audit = audit_quantized_reception(public_incomplete, manifest)
            no_individual = unpack_feature_packets(
                [raw for p, raw in zip(packets, wire) if p.header.branch == "public"],
                manifest, device=device,
            )
            no_individual_audit = audit_quantized_reception(no_individual, manifest)
            assert config_audit.status == "failed_quantization_config_missing"
            assert public_audit.status == "failed_public_incomplete"
            assert one_drop_audit.status == "degraded_individual_zero_fill"
            assert not no_individual_audit.decoder_ready
            anomalies = {
                "deadline_and_unsent": {
                    "status_counts": constrained["status_counts"],
                    "actual_sent_bytes": constrained["actual_wire_bytes"],
                    "late_packets_counted_as_sent": all(
                        event["wire_bytes"] > 0 for event in constrained["events"]
                        if event["status"] == "late"
                    ),
                    "unsent_packets_not_counted": all(
                        event["wire_bytes"] == 0 for event in constrained["events"]
                        if event["status"] == "unsent"
                    ),
                },
                "missing_config": {"status": config_audit.status, "decoder_ready": config_audit.decoder_ready},
                "public_incomplete": {"status": public_audit.status, "decoder_ready": public_audit.decoder_ready,
                                      "missing_elements": int(public_audit.public_missing_mask.sum().item())},
                "individual_one_packet_missing": {
                    "status": one_drop_audit.status, "decoder_ready": one_drop_audit.decoder_ready,
                    "packet_id": f"{gop_id}:{one_drop.header.branch}:{one_drop.header.group_id}",
                    "local_packet_id": one_drop.header.packet_id,
                    "gop_id": gop_id, "branch": one_drop.header.branch,
                    "tensor_region": one_drop.header.tensor_region.as_dict(),
                    "missing_elements": int(expected_mask.sum().item()),
                    "mask_exact": mask_exact, "received_elements_preserved": preserved,
                    "zero_fill_reconstruction_mse": mse(dropped_video, x),
                },
                "entire_individual_branch_missing": {
                    "status": no_individual_audit.status, "decoder_ready": no_individual_audit.decoder_ready,
                    "reason": "The current packet format repeats the scale only in that branch's packets",
                },
            }
            torch.save({
                "layout": "BTCHW", "mask_true_means": "missing", "gop_id": gop_id,
                "start_frame": start_frame,
                "encoder_feature": features.cpu(),
                "receiver_feature_no_loss": c_features.cpu(),
                "receiver_public_int8": no_loss.public.cpu(),
                "receiver_individual_int8": no_loss.individual.cpu(),
                "receiver_public_float32": c_public.cpu(),
                "receiver_individual_float32": c_individual.cpu(),
                "no_loss_public_missing_mask": no_loss_audit.public_missing_mask.cpu(),
                "no_loss_individual_missing_mask": no_loss_audit.individual_missing_mask.cpu(),
                "one_drop_receiver_feature": dropped_features.cpu(),
                "one_drop_public_missing_mask": one_drop_audit.public_missing_mask.cpu(),
                "one_drop_individual_missing_mask": one_drop_audit.individual_missing_mask.cpu(),
                "one_drop_decoder_feature_missing_mask": one_drop_audit.feature_missing_mask.cpu(),
                "one_drop_status": one_drop_audit.status,
                "replay_receiver_public_int8": replay_received.public.cpu() if replay_received else None,
                "replay_receiver_individual_int8": replay_received.individual.cpu() if replay_received else None,
                "replay_public_missing_mask": replay_audit.public_missing_mask.cpu() if replay_audit else None,
                "replay_individual_missing_mask": replay_audit.individual_missing_mask.cpu() if replay_audit else None,
                "replay_decoder_feature_missing_mask": replay_audit.feature_missing_mask.cpu() if replay_audit else None,
                "replay_status": replay_audit.status if replay_audit else "failed_no_packets_received",
                "quantization": {"public": pq.metadata, "individual": iq.metadata},
            }, output / "receiver_gop0.pt")

        byte_rows = [bytes_by_category(p, raw) for p, raw in zip(packets, wire)]
        byte_total = {key: sum(row[key] for row in byte_rows) for key in byte_rows[0]}
        sent_total = sum(e["wire_bytes"] for e in replay["events"])
        if sent_total != replay["actual_wire_bytes"]:
            raise AssertionError("Network event bytes do not balance")
        mapping = []
        stream_offset = 0
        for p, raw, byte_row in zip(packets, wire, byte_rows):
            h = p.header
            region = h.tensor_region
            mapping.append({
                "gop_id": gop_id, "start_frame": start_frame,
                "packet_id": f"{gop_id}:{h.branch}:{h.group_id}",
                "local_packet_id": h.packet_id, "group_id": h.group_id, "branch": h.branch,
                "stream_offset_bytes": stream_offset,
                "payload_offset_in_packet_bytes": len(raw) - len(p.payload),
                "packet_bytes": len(raw), "payload_bytes": len(p.payload),
                "payload_elements": int(torch.tensor(h.payload_shape).prod().item()),
                "b_range": list(region.b), "t_range": list(region.t),
                "c_range": list(region.c), "h_range": list(region.h), "w_range": list(region.w),
                "required": h.required, "priority": h.priority,
                **byte_row,
            })
            stream_offset += len(raw)

        metrics = {
            "gop_id": gop_id, "start_frame": start_frame,
            "input_shape": list(x.shape), "feature_shape": list(features.shape),
            "direct_mse": mse(direct, x), "factor1_mse": mse(a_video, x),
            "factor2_diagnostic_mse": mse(factor2_video, x),
            "factor1_float_packet_mse": mse(b_video, x), "factor1_int8_packet_mse": mse(c_video, x),
            "A_vs_B_feature_max_abs": feature_diff, "A_vs_B_image_max_abs": image_diff,
            "A_vs_B_feature_allclose": bool(torch.allclose(a_features, b_features, rtol=RTOL, atol=ATOL)),
            "A_vs_B_image_allclose": bool(torch.allclose(a_video, b_video, rtol=RTOL, atol=ATOL)),
            "int8_vs_A_feature_max_abs": max_abs(c_features, a_features),
            "int8_vs_A_image_max_abs": max_abs(c_video, a_video),
            "raw_feature_bytes": features.numel() * features.element_size(),
            "float_packet_wire_bytes": sum(len(raw) for raw in float_wire),
            "int8_payload_bytes": byte_total["data_bytes"],
            "int8_config_bytes": byte_total["config_bytes"],
            "int8_fragment_header_bytes": byte_total["fragment_header_bytes"],
            "int8_transport_header_bytes": byte_total["transport_header_bytes"],
            "int8_wire_bytes": byte_total["sent_total_bytes"],
            "int8_packet_count": len(packets),
            "replay_received": replay["status_counts"]["received"],
            "replay_lost": replay["status_counts"]["lost"],
            "replay_late": replay["status_counts"]["late"],
            "replay_unsent": replay["status_counts"]["unsent"],
            "replay_realized_loss_fraction": replay["realized_loss_fraction"],
            "replay_actual_sent_bytes": sent_total,
            "replay_receiver_status": replay_audit.status if replay_audit else "failed_no_packets_received",
            "replay_decoder_ready": replay_audit.decoder_ready if replay_audit else False,
        }
        report = {"metrics": metrics, "processing_ms": timings, "anomalies": anomalies,
                  "network_replay_reproducible": reproducible,
                  "network_report": replay,
                  "receiver_group_ids": {"received": list(no_loss.received_group_ids),
                                         "missing": list(no_loss.missing_group_ids)},
                  "byte_accounting_note": "JSON field tokens are divided into config and fragment; local transport overhead is zero"}
        return metrics, report, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frames", type=int, nargs="+", default=[0, 64, 124])
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--loss-fraction", type=float, default=0.1)
    parser.add_argument("--rate-bps", type=int, default=100_000_000)
    parser.add_argument("--deadline-ms", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    encoder, decoder, checkpoint = load_swin_models(args.checkpoint, device)
    metrics_rows, reports, mapping_rows = [], [], []
    for gop_id, start_frame in enumerate(args.start_frames):
        gop = load_video_as_gop(args.video, args.crop_size, args.num_frames, start_frame)
        metrics, report, mapping = run_one(
            encoder, decoder, gop, gop_id=gop_id, start_frame=start_frame,
            output=output, seed=args.seed, loss_fraction=args.loss_fraction,
            rate_bps=args.rate_bps, deadline_ms=args.deadline_ms, device=device,
        )
        metrics_rows.append(metrics)
        reports.append(report)
        mapping_rows.extend(mapping)

    def write_csv(name, rows):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    write_csv("gop_metrics.csv", metrics_rows)
    write_csv("packet_mapping.csv", mapping_rows)
    config = {
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch_or_step": checkpoint.get("epoch"),
        "checkpoint_training_unit": checkpoint.get("training_unit", "epoch"),
        "checkpoint_training_scope": checkpoint.get("training_scope", "unspecified"),
        "checkpoint_model_version": checkpoint.get("model_version"),
        "video_path": str(Path(args.video).resolve()),
        "video_sha256": file_sha256(args.video),
        "start_frames": args.start_frames, "gop_count": len(args.start_frames),
        "crop_size_px": args.crop_size, "gop_frames": args.num_frames,
        "layout": "BTCHW", "factor_production": 1,
        "factor2": "diagnostic_only_pool2_bilinear_upsample",
        "quantization": "symmetric_int8_per_branch_scale",
        "group_tile_hw_elements": [16, 16],
        "packet_schema": "packet-v1", "interleave_version": "none-v1",
        "network": {"seed": args.seed, "loss_fraction": args.loss_fraction,
                    "rate_bps": args.rate_bps, "deadline_ms": args.deadline_ms,
                    "configured_transport_header_bytes_per_packet": 0},
        "allclose": {"rtol": RTOL, "atol": ATOL}, "device": str(device),
        "limitations": ["Official epoch-314 quality only if that checkpoint is supplied",
                        "Grouping is one packet per 16x16 region, not MTU fragmentation",
                        "No learned missing-feature recovery or interleaving",
                        "Network latency is synthetic and excluded from processing_ms"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "acceptance.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (output / "events.json").write_text(json.dumps(
        [report["network_report"] for report in reports], indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "config": config, "gops": metrics_rows}, indent=2))


if __name__ == "__main__":
    main()
