"""Bind real reference datagrams to the unchanged historical event engine."""

from dataclasses import dataclass

from hier_video.transmission.network_replay import (
    FeaturePacket,
    LinkSample,
    replay_feature_packet_events,
)

from .reference_transport import HEADER, Datagram, Offer, selected_packets


@dataclass(frozen=True)
class Transfer:
    received: tuple[Datagram, ...]
    report: dict
    binding: dict


def deliver(offer: Offer, mode: str = "full", *, rate_bps: int = 100_000_000,
            loss: float = 0.0, seed: int = 7, delay_ms: int = 20,
            deadline_ms: int = 10000, budget_bytes: int = 10_000_000,
            overhead: int = 28) -> Transfer:
    chosen = selected_packets(offer, mode)
    report = replay_feature_packet_events(
        tuple(FeaturePacket(p.packet_id, p.group_id, len(p.data)) for p in offer.packets),
        tuple(p.packet_id for p in chosen),
        trace_id=f"reference-{offer.config['session_id']}-{offer.config['gop_id']}",
        seed=seed, decision_time_ms=0, deadline_ms=deadline_ms,
        budget_bytes=budget_bytes, link=LinkSample(rate_bps, loss, delay_ms, overhead),
        feature_interface_version="ping-float-v1",
    )
    by_id = {p.packet_id: p for p in offer.packets}
    received = tuple(by_id[row["packet_id"]] for row in report["events"]
                     if row["status"] == "received")
    binding = {
        "kind": "reference_float_datagrams_bound_to_legacy_event_engine",
        "real_payload_bytes_bound": True,
        "reference_mapping_only": True, "mode": mode,
        "config_payload_bytes": sum(len(p.data) - HEADER.size for p in chosen if p.branch == "config"),
        "float_payload_bytes": sum(len(p.data) - HEADER.size for p in chosen if p.branch != "config"),
        "fragment_header_bytes": len(chosen) * HEADER.size,
        "configured_transport_overhead_bytes": len(chosen) * overhead,
        "receiver_packet_ids": [p.packet_id for p in received],
        "limitations": [
            "legacy report mock labels describe its original use; it is copied unchanged",
            "this binding uses actual reference float bytes and in-band config, not team compression",
            "synthetic fixed-selection link; no socket, ACK, FEC, retry or adaptive controller",
            "28B default models IPv4+UDP only, not Ethernet/Wi-Fi/cellular physical cost",
        ],
    }
    return Transfer(received, report, binding)
