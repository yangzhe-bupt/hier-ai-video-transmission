"""Evaluate the fixed Swin-314 interface with the factor=1 compression path.

The script compares a direct Swin encoder/decoder path with:

    encode -> public/residual split -> int8 -> packetize -> receive
    -> dequantize -> merge -> decode

It also writes a deterministic no-loss/packet-event report.  It does not claim
that the committed result files contain the private Swin-314 checkpoint.
Run through ``practice/run.py`` so the existing practice imports keep working.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hier_video.transmission.feature_decomposition import (  # noqa: E402
    FeatureParts,
    PublicResidualDecomposer,
)
from hier_video.transmission.feature_mapping import build_group_manifest  # noqa: E402
from hier_video.transmission.network_replay import (  # noqa: E402
    FeaturePacket as ReplayPacket,
    LinkSample,
    replay_feature_packet_events,
)
from hier_video.transmission.packetization import (  # noqa: E402
    build_feature_packets,
    unpack_feature_packets,
)
from hier_video.transmission.quantization import (  # noqa: E402
    dequantize_from_metadata,
    quantize_symmetric,
)

from train_swin_video import load_swin_models  # noqa: E402
from video_feature_demo import load_video_as_gop  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).square().mean().item())


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _merge_received(
    decomposer: PublicResidualDecomposer,
    original_parts: FeatureParts,
    received,
    device: torch.device,
    *,
    quantized: bool,
) -> tuple[torch.Tensor, dict]:
    """Dequantize received branches and merge them back to BTCHW."""
    if quantized:
        metadata = received.quantization_by_branch
        if "public" not in metadata or "individual" not in metadata:
            raise ValueError("Both branch quantization metadata are required")
        public = dequantize_from_metadata(
            received.public,
            dict(metadata["public"]),
            dtype=original_parts.meta.original_dtype,
        ).to(device)
        individual = dequantize_from_metadata(
            received.individual,
            dict(metadata["individual"]),
            dtype=original_parts.meta.original_dtype,
        ).to(device)
    else:
        public = received.public.to(dtype=original_parts.meta.original_dtype, device=device)
        individual = received.individual.to(
            dtype=original_parts.meta.original_dtype, device=device
        )
    return decomposer.decode(
        FeatureParts(public=public, individual=individual, meta=original_parts.meta)
    ), {
        "received_group_ids": list(received.received_group_ids),
        "missing_group_ids": list(received.missing_group_ids),
    }


def evaluate_gop(
    encoder,
    decoder,
    gop: torch.Tensor,
    *,
    start_frame: int,
    device: torch.device,
    gop_id: int,
    loss_fraction: float,
    seed: int,
    deadline_ms: int,
    budget_bytes: int,
    drop_one_packet: bool,
) -> tuple[dict, dict, dict]:
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        inputs = gop.to(device)
        _sync(device)
        started = time.perf_counter()
        features = encoder(inputs)
        _sync(device)
        encoded_ms = (time.perf_counter() - started) * 1000.0

        direct_started = time.perf_counter()
        direct = decoder(features)
        _sync(device)
        direct_decode_ms = (time.perf_counter() - direct_started) * 1000.0

        decomposer = PublicResidualDecomposer()
        split_started = time.perf_counter()
        parts = decomposer.encode(features)
        public_q = quantize_symmetric(parts.public)
        individual_q = quantize_symmetric(parts.individual)
        manifest = build_group_manifest(
            public_q.tensor.shape, individual_q.tensor.shape, tile_hw=(16, 16)
        )
        packets = build_feature_packets(
            public_q.tensor,
            individual_q.tensor,
            manifest,
            stream_id="swin314-evaluation",
            sequence_id=gop_id,
            gop_id=gop_id,
            gop_start_frame=start_frame,
            quantization_by_branch={
                "public": public_q.metadata,
                "individual": individual_q.metadata,
            },
        )
        wire = [packet.serialize() for packet in packets]
        float_manifest = build_group_manifest(
            parts.public.shape, parts.individual.shape, tile_hw=(16, 16)
        )
        float_packets = build_feature_packets(
            parts.public,
            parts.individual,
            float_manifest,
            stream_id="swin314-evaluation-float",
            sequence_id=gop_id,
            gop_id=gop_id,
            gop_start_frame=start_frame,
        )
        float_wire = [packet.serialize() for packet in float_packets]
        float_received = unpack_feature_packets(float_wire, float_manifest, device=device)
        float_restored_features, float_receive_info = _merge_received(
            decomposer, parts, float_received, device, quantized=False
        )
        float_packet_video = decoder(float_restored_features)
        split_packetize_ms = (time.perf_counter() - split_started) * 1000.0

        replay_packets = tuple(
            ReplayPacket(
                packet_id=f"{gop_id}:{packet.header.branch}:{packet.header.group_id}",
                group_id=f"{gop_id}:{packet.header.branch}",
                payload_bytes=len(raw),
            )
            for packet, raw in zip(packets, wire)
        )
        selected_ids = tuple(packet.packet_id for packet in replay_packets)
        link = LinkSample(rate_bps=10**12, loss_fraction=loss_fraction)
        event_report = replay_feature_packet_events(
            replay_packets,
            selected_ids,
            trace_id=f"swin314-gop-{gop_id}",
            seed=seed,
            decision_time_ms=0,
            deadline_ms=deadline_ms,
            budget_bytes=budget_bytes,
            link=link,
            feature_interface_version=manifest.interface_version,
        )
        received_ids = {
            row["packet_id"] for row in event_report["events"] if row["status"] == "received"
        }
        received_wire = [
            raw for packet, raw in zip(replay_packets, wire) if packet.packet_id in received_ids
        ]
        if not received_wire:
            raise RuntimeError("No packet reached the receiver; cannot form a feature sample")
        unpack_started = time.perf_counter()
        received = unpack_feature_packets(received_wire, manifest, device=device)
        restored_features, receive_info = _merge_received(
            decomposer, parts, received, device, quantized=True
        )
        _sync(device)
        receive_restore_ms = (time.perf_counter() - unpack_started) * 1000.0
        compressed = decoder(restored_features)
        _sync(device)

        # Explicitly exercise the requested one-packet drop on a copy of the
        # no-loss stream.  The dropped packet is the last individual group.
        dropped_packet_id = None
        drop_info = None
        if drop_one_packet:
            dropped_packet_id = replay_packets[-1].packet_id
            dropped_header = packets[-1].header
            one_drop_wire = [
                raw for packet, raw in zip(replay_packets, wire)
                if packet.packet_id != dropped_packet_id
            ]
            dropped_received = unpack_feature_packets(one_drop_wire, manifest, device=device)
            _, drop_info = _merge_received(
                decomposer, parts, dropped_received, device, quantized=True
            )
            region = dropped_header.tensor_region
            missing_mask = torch.zeros_like(individual_q.tensor, dtype=torch.bool)
            missing_mask[
                region.b[0]:region.b[1], region.t[0]:region.t[1],
                region.c[0]:region.c[1], region.h[0]:region.h[1],
                region.w[0]:region.w[1],
            ] = True
            drop_info.update({
                "gop_id": gop_id,
                "branch": dropped_header.branch,
                "tensor_region": region.as_dict(),
                "missing_elements": int(missing_mask.sum().item()),
                "mask_shape": list(missing_mask.shape),
                "mask_true_means": "missing",
                "received_elements_preserved": bool(torch.equal(
                    dropped_received.individual[~missing_mask],
                    individual_q.tensor[~missing_mask],
                )),
            })

    direct_feature_mse = 0.0
    decomposition_restored = decomposer.decode(parts)
    decomposition_video = decoder(decomposition_restored)
    rows = [
        {
            "start_frame": start_frame,
            "path": "A_direct_swin_encode_decode",
            "video_mse": _mse(direct, inputs),
            "feature_mse": direct_feature_mse,
            "wire_bytes": features.numel() * features.element_size(),
            "allclose_to_direct": True,
        },
        {
            "start_frame": start_frame,
            "path": "B_factor1_float_packetized_no_loss",
            "video_mse": _mse(float_packet_video, inputs),
            "feature_mse": _mse(float_restored_features, features),
            "wire_bytes": sum(len(raw) for raw in float_wire),
            "allclose_to_direct": bool(torch.allclose(float_packet_video, direct)),
        },
        {
            "start_frame": start_frame,
            "path": "factor1_decomposition_no_quantization",
            "video_mse": _mse(decomposition_video, inputs),
            "feature_mse": _mse(decomposition_restored, features),
            "wire_bytes": parts.public.numel() * parts.public.element_size()
            + parts.individual.numel() * parts.individual.element_size(),
            "allclose_to_direct": bool(torch.allclose(decomposition_video, direct)),
        },
        {
            "start_frame": start_frame,
            "path": "B_factor1_int8_packetized_no_loss_or_replay_loss",
            "video_mse": _mse(compressed, inputs),
            "feature_mse": _mse(restored_features, features),
            "wire_bytes": sum(len(raw) for raw in wire),
            "allclose_to_direct": bool(torch.allclose(compressed, direct, rtol=1e-5, atol=1e-6)),
        },
    ]
    config = {
        "model": "SwinGOPEncoder + GOPDecoder",
        "checkpoint_note": "Interpret quality only after verifying the checkpoint is trained",
        "layout": "BTCHW",
        "input_shape": list(inputs.shape),
        "feature_shape": list(features.shape),
        "factor": 1,
        "quantization": "symmetric-int8, one scale per branch",
        "tile_hw": [16, 16],
        "gop_id": gop_id,
        "start_frame": start_frame,
        "seed": seed,
        "loss_fraction": loss_fraction,
        "deadline_ms": deadline_ms,
        "budget_bytes": budget_bytes,
        "device": str(device),
    }
    details = {
        "rows": rows,
        "no_loss_consistency": {
            "feature_max_abs_diff_A_vs_B": _max_abs(float_restored_features, features),
            "reconstruction_max_abs_diff_A_vs_B": _max_abs(float_packet_video, direct),
            "allclose_A_vs_B": bool(
                torch.allclose(float_packet_video, direct, rtol=1e-5, atol=1e-6)
            ),
        },
        "quantized_quality": {
            "feature_max_abs_diff": _max_abs(restored_features, features),
            "reconstruction_max_abs_diff": _max_abs(compressed, direct),
            "allclose_to_direct": bool(
                torch.allclose(compressed, direct, rtol=1e-5, atol=1e-6)
            ),
        },
        "allclose_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "raw_feature_bytes": features.numel() * features.element_size(),
        "quantized_payload_bytes": public_q.tensor.numel() + individual_q.tensor.numel(),
        "serialized_header_bytes": sum(len(raw) - len(packet.payload)
                                       for packet, raw in zip(packets, wire)),
        "configured_network_header_bytes": 0,
        "packet_count": len(packets),
        "wire_bytes": sum(len(raw) for raw in wire),
        "processing_ms": {
            "encode": encoded_ms,
            "direct_decode": direct_decode_ms,
            "split_and_packetize": split_packetize_ms,
            "receive_and_restore": receive_restore_ms,
        },
        "receiver_feature_sample": {
            "shape": list(restored_features.shape),
            "dtype": str(restored_features.dtype),
            "device": str(restored_features.device),
        },
        "network_event_report": event_report,
        "receiver_group_status": receive_info,
        "specified_one_packet_drop": {
            "packet_id": dropped_packet_id,
            "group_status": drop_info,
        },
        "factor2": {
            "status": "not_run",
            "reason": "factor=2 was removed from the production path because pooling caused visible feature loss",
        },
    }
    return config, details, {"packets": packets, "wire": wire}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start-frames", type=int, nargs="+", default=[100, 400, 800])
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--output-dir", default="results/week4/biran_swin")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--deadline-ms", type=int, default=1000)
    parser.add_argument("--budget-bytes", type=int, default=10**9)
    parser.add_argument("--drop-one-packet", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, decoder, checkpoint = load_swin_models(args.checkpoint, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    reports = []
    for gop_id, start_frame in enumerate(args.start_frames):
        gop = load_video_as_gop(args.video, args.crop_size, args.num_frames, start_frame)
        config, details, _ = evaluate_gop(
            encoder,
            decoder,
            gop,
            start_frame=start_frame,
            device=device,
            gop_id=gop_id,
            loss_fraction=args.loss_fraction,
            seed=args.seed,
            deadline_ms=args.deadline_ms,
            budget_bytes=args.budget_bytes,
            drop_one_packet=args.drop_one_packet,
        )
        config["checkpoint_epoch"] = checkpoint.get("epoch")
        reports.append({"config": config, "details": details})
        all_rows.extend(details["rows"])

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "evaluation.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "events.json").write_text(
        json.dumps(
            [report["details"]["network_event_report"] for report in reports],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metrics saved to {output / 'metrics.csv'}")
    print(f"evaluation saved to {output / 'evaluation.json'}")
    print(f"events saved to {output / 'events.json'}")


if __name__ == "__main__":
    main()
