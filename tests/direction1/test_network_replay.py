import copy
import unittest

from hier_video.transmission.network_replay import (
    LinkSample,
    FeaturePacket,
    replay_feature_packet_events,
    validate_feature_event_report,
)


def packets():
    return (
        FeaturePacket("p0", "g0", 1000),
        FeaturePacket("p1", "g0", 1000),
        FeaturePacket("p2", "g1", 1000),
    )


def replay(**overrides):
    arguments = {
        "packets": packets(),
        "selected_packet_ids": ("p0", "p1", "p2"),
        "trace_id": "feature-week1",
        "seed": 7,
        "decision_time_ms": 0,
        "deadline_ms": 1000,
        "budget_bytes": 4000,
        "link": LinkSample(1_000_000, 0, 0, 28),
    }
    arguments.update(overrides)
    return replay_feature_packet_events(**arguments)


class FeatureEventsTests(unittest.TestCase):
    def test_link_rejects_invalid_numbers(self):
        for args in (
            (-1, 0), (True, 0), (1.5, 0), (1, float("nan")),
            (1, -0.1), (1, 1.1), (1, True), (1, 0, -1), (1, 0, 0, 1.0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                LinkSample(*args)

    def test_all_received_without_asserting_codec_layers(self):
        result = replay()
        self.assertEqual(result["schema_version"], "feature-network-v0")
        self.assertEqual(result["available_group_ids"], ["g0", "g1"])
        self.assertEqual(result["missing_group_ids"], [])
        self.assertEqual(result["actual_wire_bytes"], 3084)
        self.assertEqual(result["decision"]["selected_payload_bytes"], 3000)
        self.assertNotIn("independent_gop", result)
        self.assertNotIn("decoded_prefix_count", result)
        validate_feature_event_report(result)

    def test_one_missing_packet_makes_its_whole_group_unavailable(self):
        result = replay(selected_packet_ids=("p0", "p2"))
        self.assertEqual(result["available_group_ids"], ["g1"])
        self.assertEqual(result["missing_group_ids"], ["g0"])
        self.assertEqual(result["events"][1]["status"], "unsent")
        self.assertEqual(result["events"][1]["reason"], "not_selected")
        self.assertEqual(result["events"][1]["wire_bytes"], 0)

    def test_all_lost_counts_sent_bytes_but_excludes_unsent_from_loss_rate(self):
        result = replay(
            selected_packet_ids=("p0", "p2"), link=LinkSample(1_000_000, 1),
        )
        self.assertEqual(result["status_counts"], {
            "received": 0, "lost": 2, "late": 0, "unsent": 1,
        })
        self.assertEqual(result["realized_loss_fraction"], 1)
        self.assertEqual(result["actual_wire_bytes"], 2000)
        self.assertEqual(result["missing_group_ids"], ["g0", "g1"])

    def test_late_and_unsent_are_distinct(self):
        result = replay(link=LinkSample(4000, 0, 0, 0), budget_bytes=3000)
        self.assertEqual([item["status"] for item in result["events"]], [
            "late", "unsent", "unsent",
        ])
        self.assertEqual(result["events"][0]["arrived_at_ms"], 2000)
        self.assertEqual(result["events"][1]["reason"], "deadline_reached")
        self.assertEqual(result["actual_wire_bytes"], 1000)
        self.assertEqual(result["realized_loss_fraction"], 0)

    def test_arrival_exactly_at_deadline_is_received(self):
        result = replay(
            packets=(FeaturePacket("p0", "g0", 1000),),
            selected_packet_ids=("p0",), link=LinkSample(8000, 0),
        )
        self.assertEqual(result["events"][0]["status"], "received")

    def test_disconnected_link_sends_no_bytes(self):
        result = replay(link=LinkSample(0, 0))
        self.assertEqual(result["actual_wire_bytes"], 0)
        self.assertEqual(result["realized_loss_fraction"], None)
        self.assertTrue(all(item["reason"] == "link_unavailable"
                            for item in result["events"]))

    def test_delayed_receipt_does_not_count_as_physical_loss(self):
        result = replay(link=LinkSample(1_000_000, 0, 1001))
        self.assertEqual(result["status_counts"]["late"], 3)
        self.assertEqual(result["status_counts"]["lost"], 0)
        self.assertEqual(result["realized_loss_fraction"], 0)

    def test_wire_budget_includes_configured_headers(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            replay(budget_bytes=3000)
        result = replay(budget_bytes=3084)
        self.assertEqual(result["decision"]["selected_wire_bytes"], 3084)
        self.assertEqual(result["budget_scope"], "payload_plus_configured_overhead")

    def test_subset_keeps_same_loss_draw_for_common_packet_ids(self):
        all_packets = replay(link=LinkSample(1_000_000, 0.4))
        subset = replay(
            selected_packet_ids=("p0", "p2"), link=LinkSample(1_000_000, 0.4),
        )
        statuses = {item["packet_id"]: item["status"] for item in all_packets["events"]}
        self.assertEqual(
            [item["status"] for item in subset["events"] if item["selected"]],
            [statuses[key] for key in ("p0", "p2")],
        )

    def test_identical_input_replays_identically(self):
        self.assertEqual(replay(), replay())

    def test_rejects_unknown_duplicate_and_out_of_order_selection(self):
        for ids in (("bad",), ("p0", "p0"), ("p2", "p0")):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                replay(selected_packet_ids=ids)

    def test_rejects_bad_offer_and_clock(self):
        for arguments in (
            {"packets": (packets()[0], packets()[0])},
            {"packets": ()},
            {"seed": -1},
            {"seed": True},
            {"decision_time_ms": -1},
            {"deadline_ms": 0},
            {"budget_bytes": -1},
            {"trace_id": ""},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                replay(**arguments)
        for arguments in (("", "g0", 1), ("p0", "", 1), ("p0", "g0", 0),
                          ("p0", "g0", True)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                FeaturePacket(*arguments)

    def test_empty_selection_is_a_valid_no_send_result(self):
        result = replay(selected_packet_ids=(), budget_bytes=0)
        self.assertEqual(result["actual_wire_bytes"], 0)
        self.assertEqual(result["missing_group_ids"], ["g0", "g1"])

    def test_report_rejects_forged_groups_times_and_byte_sums(self):
        valid = replay()
        edits = (
            ("actual_wire_bytes", 1),
            ("available_group_ids", []),
            ("missing_group_ids", ["g0"]),
            ("realized_loss_fraction", 0.9),
        )
        for key, value in edits:
            damaged = copy.deepcopy(valid)
            damaged[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_feature_event_report(damaged)

    def test_report_rejects_inconsistent_clock_selection_and_receipt(self):
        for edit in (
            lambda obj: obj.update(deadline_ms=0),
            lambda obj: obj.update(events=[]),
            lambda obj: obj["events"].append(copy.deepcopy(obj["events"][0])),
            lambda obj: obj["decision"].update(selected_packet_ids=["p2", "p0"]),
            lambda obj: obj["events"][0].update(selected=False),
        ):
            damaged = copy.deepcopy(replay())
            edit(damaged)
            with self.assertRaises(ValueError):
                validate_feature_event_report(damaged)
        damaged = replay(selected_packet_ids=())
        damaged["events"][0]["wire_bytes"] = 1
        with self.assertRaises(ValueError):
            validate_feature_event_report(damaged)
        damaged = replay(link=LinkSample(1_000_000, 1))
        damaged["events"][0]["arrived_at_ms"] = 8
        with self.assertRaises(ValueError):
            validate_feature_event_report(damaged)
        valid = replay()
        for key, value in (("arrived_at_ms", 1001), ("wire_bytes", 2),
                           ("status", "recovered")):
            damaged = copy.deepcopy(valid)
            damaged["events"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_feature_event_report(damaged)

    def test_report_recomputes_loss_and_requires_trace_metadata(self):
        damaged = replay(link=LinkSample(1_000_000, 0))
        damaged["events"][0]["status"] = "lost"
        damaged["events"][0]["reason"] = "independent_loss"
        damaged["events"][0]["arrived_at_ms"] = None
        damaged["status_counts"] = {
            "received": 2, "lost": 1, "late": 0, "unsent": 0,
        }
        damaged["realized_loss_fraction"] = 1 / 3
        damaged["available_group_ids"] = ["g1"]
        damaged["missing_group_ids"] = ["g0"]
        with self.assertRaises(ValueError):
            validate_feature_event_report(damaged)
        damaged = replay(link=LinkSample(1_000_000, 1))
        damaged["events"][0]["status"] = "received"
        damaged["events"][0]["reason"] = "by_deadline"
        damaged["events"][0]["arrived_at_ms"] = 8
        with self.assertRaises(ValueError):
            validate_feature_event_report(damaged)
        for key, value in (
            ("trace_id", ""),
            ("seed", -1),
            ("feature_interface_version", ""),
        ):
            damaged = replay()
            damaged[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_feature_event_report(damaged)

    def test_non_integer_serialization_boundaries_match_packet_replay(self):
        result = replay(
            packets=(
                FeaturePacket("p0", "g0", 6),
                FeaturePacket("p1", "g1", 1),
            ),
            selected_packet_ids=("p0", "p1"),
            link=LinkSample(32_000, 0),
            deadline_ms=2,
            budget_bytes=7,
        )
        self.assertEqual([row["status"] for row in result["events"]], [
            "received", "unsent",
        ])
        self.assertEqual(result["actual_wire_bytes"], 6)
        self.assertEqual(result["events"][0]["arrived_at_ms"], 2)

    def test_report_rejects_reordered_events_and_float_accounting(self):
        damaged = replay()
        damaged["events"] = list(reversed(damaged["events"]))
        with self.assertRaises(ValueError):
            validate_feature_event_report(damaged)
        for path in (
            ("actual_wire_bytes",),
            ("decision", "selected_payload_bytes"),
            ("decision", "selected_wire_bytes"),
            ("events", 0, "wire_bytes"),
            ("events", 0, "arrived_at_ms"),
            ("events", 0, "sent_at_ms"),
            ("events", 0, "payload_bytes"),
            ("status_counts", "received"),
            ("status_counts", "unsent"),
        ):
            damaged = replay()
            target = damaged
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = float(target[path[-1]])
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_feature_event_report(damaged)


if __name__ == "__main__":
    unittest.main()
