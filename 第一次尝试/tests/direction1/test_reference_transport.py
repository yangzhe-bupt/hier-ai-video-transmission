import json
import unittest

import torch

from hier_video.transmission.reference_layers import restore_layers, split_layers
from hier_video.transmission.reference_transport import (
    HEADER, _validate_config, make_offer, receive, selected_packets, unpack_packet,
)
from hier_video.transmission.reference_delivery import deliver


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.parts = split_layers(torch.randn(1, 4, 3, 8, 8))
        self.offer = make_offer(self.parts, session_id=8, gop_id=2, model_id="test")

    def run_case(self, mode="full", **kwargs):
        return deliver(self.offer, mode=mode, **kwargs)

    def test_no_loss_base_full_matches_same_processing(self):
        for mode in ("base", "full"):
            result = self.run_case(mode)
            recovered = receive(result.received, expected_session=8, expected_gop=2,
                                expected_model="test")
            self.assertTrue(torch.equal(recovered.features, restore_layers(self.parts, mode)))
            self.assertEqual(recovered.state, mode)
            self.assertEqual(recovered.missing_base_elements, 0)
            self.assertEqual(recovered.missing_enhancement_elements,
                             0 if mode == "full" else self.parts.enhancement.numel())
            self.assertEqual(result.report["actual_wire_bytes"],
                             sum(len(p.data) + 28 for p in selected_packets(self.offer, mode)))
            self.assertEqual(result.report["status_counts"]["lost"], 0)

    def test_packets_mtu_and_header_and_payload(self):
        self.assertTrue(all(len(p.data) <= 1200 for p in self.offer.packets))
        self.assertGreater(HEADER.size, 0)
        counts = {"config": 0, "base": 0, "enhancement": 0}
        for p in self.offer.packets:
            row = unpack_packet(p.data)
            counts[p.branch] += len(row["payload"])
        self.assertEqual(counts["base"], self.parts.base.numel() * 4)
        self.assertEqual(counts["enhancement"], self.parts.enhancement.numel() * 4)
        self.assertGreater(counts["config"], 0)

    def test_receiver_is_order_independent_and_identical_duplicates_ok(self):
        result = self.run_case()
        rx = list(reversed(result.received)) + [result.received[0]]
        a = receive(rx)
        self.assertTrue(torch.equal(a.features, restore_layers(self.parts, "full")))

    def test_missing_base_blocks_el_and_masks_only_missing_bytes(self):
        result = self.run_case()
        base_id = next(p.packet_id for p in self.offer.packets if p.branch == "base")
        rx = [p for p in result.received if p.packet_id != base_id]
        restored = receive(rx)
        self.assertEqual(restored.state, "base_incomplete")
        self.assertGreater(restored.missing_base_elements, 0)
        self.assertTrue(torch.equal(restored.features, restored.base.expand(self.parts.shape)))
        self.assertTrue(torch.count_nonzero(restored.applied_enhancement).item() == 0)
        known = ~restored.base_missing
        self.assertTrue(torch.equal(restored.base[known], self.parts.base[known]))
        self.assertEqual(torch.count_nonzero(restored.base[~known]).item(), 0)

    def test_missing_el_falls_back_and_retains_base(self):
        result = self.run_case()
        rx = [p for p in result.received if p.branch != "enhancement"]
        restored = receive(rx)
        self.assertEqual(restored.state, "base")
        self.assertTrue(torch.equal(restored.features, restore_layers(self.parts, "base")))

    def test_config_required_and_stream_binding(self):
        result = self.run_case()
        with self.assertRaises(ValueError):
            receive([p for p in result.received if p.branch != "config"])
        for opts in ({"expected_session": 99}, {"expected_gop": 88},
                     {"expected_model": "wrong"}):
            with self.assertRaises(ValueError):
                receive(result.received, **opts)

    def test_corruption_length_identity_and_conflicting_duplicate_rejected(self):
        result = self.run_case()
        sample = result.received[-1]
        from dataclasses import replace
        bad = bytearray(sample.data)
        bad[-1] ^= 1
        for packet in (replace(sample, data=bytes(bad)),
                       replace(sample, data=sample.data[:-1]),
                       replace(sample, packet_id="forged")):
            with self.assertRaises(ValueError):
                receive(list(result.received[:-1]) + [packet])

    def test_late_lost_unsent_and_budget(self):
        for opts, status in (
            ({"loss": 1.0}, "lost"),
            ({"delay_ms": 20000}, "late"),
            ({"rate_bps": 0}, "unsent"),
        ):
            result = self.run_case(**opts)
            self.assertEqual(len(result.received), 0)
            self.assertGreater(result.report["status_counts"][status], 0)
        base = self.run_case("base")
        el_events = [r for r in base.report["events"] if "enhancement" in r["group_id"]]
        self.assertTrue(el_events and all(r["reason"] == "not_selected" for r in el_events))
        with self.assertRaises(ValueError):
            self.run_case(budget_bytes=1)

    def test_seed_replay_and_receiver_does_not_accept_missing_source_payload(self):
        a = self.run_case(loss=0.3, seed=4)
        b = self.run_case(loss=0.3, seed=4)
        self.assertEqual(a.report, b.report)
        self.assertEqual(a.received, b.received)
        self.assertEqual(len(a.received), a.report["status_counts"]["received"])

    def test_invalid_offer(self):
        for kwargs in ({"mtu": HEADER.size + 1}, {"session_id": -1},
                       {"gop_id": True}, {"model_id": ""}):
            with self.assertRaises(ValueError):
                make_offer(self.parts, **kwargs)
        with self.assertRaises(ValueError):
            selected_packets(self.offer, "el")

    def test_reconstructed_tensor_memory_is_bounded_before_allocation(self):
        config = dict(self.offer.config)
        config.update(
            input_shape=[1, 4096, 1, 1024, 1024], factor=1024,
            base_shape=[1, 1, 1, 1024, 1024],
            enhancement_shape=[1, 4096, 1, 1, 1],
            bytes={"base": 4194304, "enhancement": 16384},
        )
        with self.assertRaises(ValueError):
            _validate_config(config, 8, 2)


if __name__ == "__main__":
    unittest.main()
