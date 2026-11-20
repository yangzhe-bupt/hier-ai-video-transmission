"""Week-one feature-group events, without codec-layer independence assertions."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class LinkSample:
    """Synthetic link parameters used only to replay packet events."""

    rate_bps: int
    loss_fraction: float
    one_way_delay_ms: int = 0
    packet_overhead_bytes: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.rate_bps,
            self.one_way_delay_ms,
            self.packet_overhead_bytes,
        ):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "link rate, delay and overhead must be nonnegative integers"
                )
        if (
            not isinstance(self.loss_fraction, (int, float))
            or not isfinite(self.loss_fraction)
            or not 0 <= self.loss_fraction <= 1
            or isinstance(self.loss_fraction, bool)
        ):
            raise ValueError("loss_fraction must lie between zero and one")


def _lost(seed: int, trace_id: str, packet_id: str, fraction: float) -> bool:
    """Draw stable per-packet loss so policies can share the same trace."""
    key = f"{seed}\0{trace_id}\0{packet_id}".encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return draw < fraction


STATUSES = ("received", "lost", "late", "unsent")


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


@dataclass(frozen=True)
class FeaturePacket:
    packet_id: str
    group_id: str
    payload_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.packet_id, str) or not self.packet_id.strip()
            or not isinstance(self.group_id, str) or not self.group_id.strip()
            or not _integer(self.payload_bytes, 1)
        ):
            raise ValueError("packet/group IDs and positive payload bytes required")


def replay_feature_packet_events(
    packets: tuple[FeaturePacket, ...],
    selected_packet_ids: tuple[str, ...],
    *,
    trace_id: str,
    seed: int,
    decision_time_ms: int,
    deadline_ms: int,
    budget_bytes: int,
    link: LinkSample,
    feature_interface_version: str = "v1",
) -> dict[str, Any]:
    """Replay a fixed selection; actual_link is evaluator input, not feedback."""
    if (
        not packets or any(not isinstance(packet, FeaturePacket) for packet in packets)
        or not isinstance(trace_id, str) or not trace_id.strip()
        or not isinstance(feature_interface_version, str)
        or not feature_interface_version.strip()
        or not _integer(seed) or not _integer(decision_time_ms)
        or not _integer(deadline_ms, decision_time_ms + 1)
        or not _integer(budget_bytes) or not isinstance(link, LinkSample)
    ):
        raise ValueError("invalid feature trace, clock, seed or budget")
    ids = [packet.packet_id for packet in packets]
    selected = set(selected_packet_ids)
    if (
        len(ids) != len(set(ids))
        or len(selected_packet_ids) != len(selected)
        or not selected <= set(ids)
        or tuple(key for key in ids if key in selected) != selected_packet_ids
    ):
        raise ValueError("selection must contain unique offered IDs in manifest order")
    payload_bytes = sum(packet.payload_bytes for packet in packets
                        if packet.packet_id in selected)
    selected_wire = payload_bytes + len(selected) * link.packet_overhead_bytes
    if selected_wire > budget_bytes:
        raise ValueError("fixed selection exceeds payload plus overhead budget")

    sent_bytes = 0
    finish = decision_time_ms
    events = []
    for packet in packets:
        chosen = packet.packet_id in selected
        reason = (
            "not_selected" if not chosen else
            "link_unavailable" if link.rate_bps == 0 else
            "deadline_reached" if finish >= deadline_ms else None
        )
        event = {
            "packet_id": packet.packet_id,
            "group_id": packet.group_id,
            "selected": chosen,
            "payload_bytes": packet.payload_bytes,
            "wire_bytes": 0,
            "sent_at_ms": None,
            "arrived_at_ms": None,
            "status": "unsent",
            "reason": reason,
        }
        if reason is None:
            wire = packet.payload_bytes + link.packet_overhead_bytes
            sent_bytes += wire
            start = finish
            finish = decision_time_ms + (
                sent_bytes * 8000 + link.rate_bps - 1
            ) // link.rate_bps
            arrival = finish + link.one_way_delay_ms
            dropped = _lost(seed, trace_id, packet.packet_id, link.loss_fraction)
            event.update(
                wire_bytes=wire,
                sent_at_ms=start,
                arrived_at_ms=None if dropped else arrival,
                status="lost" if dropped else (
                    "received" if arrival <= deadline_ms else "late"
                ),
                reason="independent_loss" if dropped else (
                    "by_deadline" if arrival <= deadline_ms else "after_deadline"
                ),
            )
        events.append(event)
    groups = list(dict.fromkeys(packet.group_id for packet in packets))
    missing = [
        group for group in groups
        if any(row["status"] != "received" for row in events if row["group_id"] == group)
    ]
    counts = Counter(row["status"] for row in events)
    sent_count = len(events) - counts["unsent"]
    report = {
        "schema_version": "feature-network-v0",
        "evidence_kind": "fixed_selection_mock_group_synthetic_channel",
        "trace_id": trace_id,
        "seed": seed,
        "feature_interface_version": feature_interface_version,
        "clock": "synthetic_relative_ms",
        "deadline_ms": deadline_ms,
        "budget_scope": "payload_plus_configured_overhead",
        "decision": {
            "policy": "fixed_packet_selection",
            "decision_time_ms": decision_time_ms,
            "selected_packet_ids": list(selected_packet_ids),
            "selected_payload_bytes": payload_bytes,
            "selected_wire_bytes": selected_wire,
            "budget_bytes": budget_bytes,
            "observation_source": "none_fixed_selection",
        },
        "actual_link_evaluator_only": asdict(link),
        "packet_order": [packet.packet_id for packet in packets],
        "events": events,
        "status_counts": {status: counts[status] for status in STATUSES},
        "actual_wire_bytes": sent_bytes,
        "realized_loss_fraction": counts["lost"] / sent_count if sent_count else None,
        "available_group_ids": [group for group in groups if group not in missing],
        "missing_group_ids": missing,
        "limitations": [
            "group mapping is supplied separately, not inferred as BL/EL",
            "mock groups are not actual MTU packetization or compressed bytes",
            "only configured payload/header costs; no session, ACK, FEC or retry costs",
            "unavailable group is fully masked; no fragment-level recovery claim",
            "synthetic fixed-selection replay, not a measured link or adaptive controller",
        ],
    }
    validate_feature_event_report(report)
    return report


def validate_feature_event_report(report: dict[str, Any]) -> None:
    """Check attribution, timing and accounting before converting events to masks."""
    if (
        report.get("schema_version") != "feature-network-v0"
        or report.get("clock") != "synthetic_relative_ms"
        or report.get("budget_scope") != "payload_plus_configured_overhead"
        or not isinstance(report.get("trace_id"), str)
        or not report["trace_id"].strip()
        or not _integer(report.get("seed"))
        or not isinstance(report.get("feature_interface_version"), str)
        or not report["feature_interface_version"].strip()
    ):
        raise ValueError("unsupported feature event schema, clock or budget")
    deadline = report["deadline_ms"]
    decision = report["decision"]
    now = decision["decision_time_ms"]
    budget = decision["budget_bytes"]
    if (
        not _integer(now) or not _integer(deadline, now + 1)
        or not _integer(budget)
    ):
        raise ValueError("invalid event clock or budget")
    events = report["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("feature events must cover all offered groups")
    ids = [row["packet_id"] for row in events]
    packet_order = report.get("packet_order")
    selected_ids = decision["selected_packet_ids"]
    if (
        not isinstance(packet_order, list)
        or ids != packet_order
        or len(ids) != len(set(ids))
        or len(selected_ids) != len(set(selected_ids))
    ):
        raise ValueError("duplicate event or selected packet ID")
    selected = set(selected_ids)
    if (
        not selected <= set(ids)
        or [key for key in ids if key in selected] != selected_ids
    ):
        raise ValueError("selected packet IDs differ from manifest order")
    link = LinkSample(**report["actual_link_evaluator_only"])
    actual = selected_wire = selected_payload = 0
    previous_finish = now
    for row in events:
        FeaturePacket(row["packet_id"], row["group_id"], row["payload_bytes"])
        if (
            row["status"] not in STATUSES
            or type(row["selected"]) is not bool
            or row["selected"] != (row["packet_id"] in selected)
            or not _integer(row["wire_bytes"])
            or not _integer(row["payload_bytes"], 1)
        ):
            raise ValueError("invalid packet event attribution")
        if row["selected"]:
            selected_payload += row["payload_bytes"]
            selected_wire += row["payload_bytes"] + link.packet_overhead_bytes
        if row["status"] == "unsent":
            expected_reason = (
                "not_selected" if not row["selected"] else
                "link_unavailable" if link.rate_bps == 0 else
                "deadline_reached" if previous_finish >= deadline else None
            )
            if (
                expected_reason is None or row["reason"] != expected_reason
                or row["wire_bytes"] != 0 or row["sent_at_ms"] is not None
                or row["arrived_at_ms"] is not None
            ):
                raise ValueError("unsent event has bytes, timestamps or wrong reason")
            continue
        expected_lost = _lost(
            report["seed"], report["trace_id"], row["packet_id"],
            link.loss_fraction,
        )
        if (
            not row["selected"] or link.rate_bps <= 0
            or row["wire_bytes"] != row["payload_bytes"] + link.packet_overhead_bytes
            or not _integer(row["sent_at_ms"])
            or row["sent_at_ms"] != previous_finish
            or row["sent_at_ms"] >= deadline
        ):
            raise ValueError("sent event has inconsistent bytes or start time")
        actual += row["wire_bytes"]
        previous_finish = now + (
            actual * 8000 + link.rate_bps - 1
        ) // link.rate_bps
        expected_arrival = previous_finish + link.one_way_delay_ms
        if row["status"] == "lost":
            if (
                not expected_lost
                or row["arrived_at_ms"] is not None
                or row["reason"] != "independent_loss"
            ):
                raise ValueError("lost event cannot have a receipt timestamp")
        else:
            if expected_lost:
                raise ValueError("received/late event contradicts deterministic loss draw")
            status = "received" if expected_arrival <= deadline else "late"
            reason = "by_deadline" if status == "received" else "after_deadline"
            if (
                not _integer(row["arrived_at_ms"])
                or row["arrived_at_ms"] != expected_arrival
                or row["status"] != status or row["reason"] != reason
            ):
                raise ValueError("received/late event differs from deadline")
    counts = {status: sum(row["status"] == status for row in events)
              for status in STATUSES}
    sent_count = len(events) - counts["unsent"]
    groups = list(dict.fromkeys(row["group_id"] for row in events))
    missing = [group for group in groups if any(
        row["status"] != "received" for row in events if row["group_id"] == group
    )]
    reported_counts = report.get("status_counts")
    if (
        not _integer(report.get("actual_wire_bytes"))
        or not _integer(decision.get("selected_payload_bytes"))
        or not _integer(decision.get("selected_wire_bytes"))
        or not _integer(decision.get("budget_bytes"))
        or not isinstance(reported_counts, dict)
        or set(reported_counts) != set(STATUSES)
        or any(not _integer(value) for value in reported_counts.values())
        or selected_wire > budget or actual > selected_wire
        or actual != report["actual_wire_bytes"]
        or selected_payload != decision["selected_payload_bytes"]
        or selected_wire != decision["selected_wire_bytes"]
        or counts != reported_counts
        or report["realized_loss_fraction"] != (
            counts["lost"] / sent_count if sent_count else None
        )
        or report["missing_group_ids"] != missing
        or report["available_group_ids"] != [
            group for group in groups if group not in missing
        ]
    ):
        raise ValueError("event groups, status totals or budget accounting differ")
