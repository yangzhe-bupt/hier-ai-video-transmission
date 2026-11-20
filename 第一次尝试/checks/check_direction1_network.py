"""Run the direction-one network and reference transport smoke checks."""

from hier_video.transmission import (
    FeaturePacket,
    LinkSample,
    replay_feature_packet_events,
)


def main() -> None:
    packets = (
        FeaturePacket("base-0", "base", 1000),
        FeaturePacket("enhancement-0", "enhancement", 1000),
    )
    report = replay_feature_packet_events(
        packets,
        ("base-0", "enhancement-0"),
        trace_id="direction1-smoke",
        seed=7,
        decision_time_ms=0,
        deadline_ms=1000,
        budget_bytes=2056,
        link=LinkSample(
            rate_bps=1_000_000,
            loss_fraction=0.0,
            one_way_delay_ms=20,
            packet_overhead_bytes=28,
        ),
    )
    assert report["status_counts"] == {
        "received": 2,
        "lost": 0,
        "late": 0,
        "unsent": 0,
    }
    assert report["actual_wire_bytes"] == 2056
    print("direction-one network replay smoke check passed")


if __name__ == "__main__":
    main()
