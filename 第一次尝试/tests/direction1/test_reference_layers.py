import unittest

import torch

from Biran.feature_decomposition import PublicResidualDecomposer
from hier_video.transmission.reference_layers import (
    ReferenceBackend, restore_layers, split_layers,
)


class LayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24)
        self.z = torch.randn(2, 4, 3, 8, 12, requires_grad=True)

    def test_shapes_and_reference_mapping(self):
        parts = split_layers(self.z, factor=2)
        self.assertEqual(parts.base.shape, (2, 1, 3, 8, 12))
        self.assertEqual(parts.enhancement.shape, (2, 4, 3, 4, 6))
        self.assertTrue(torch.equal(parts.base, self.z.mean(1, keepdim=True)))
        self.assertEqual(restore_layers(parts, "base").shape, self.z.shape)
        self.assertEqual(restore_layers(parts, "full").shape, self.z.shape)

    def test_matches_existing_biran_implementation(self):
        for factor in (1, 2):
            decomposer = PublicResidualDecomposer(factor)
            expected = decomposer.encode(self.z)
            actual = split_layers(self.z, factor=factor)
            self.assertTrue(torch.equal(actual.base, expected.public))
            self.assertTrue(torch.equal(actual.enhancement, expected.individual))
            self.assertTrue(torch.equal(
                restore_layers(actual, "full"), decomposer.decode(expected),
            ))

    def test_full_is_lossy_but_factor_one_is_exact_with_tolerance(self):
        lossy = restore_layers(split_layers(self.z), "full")
        self.assertFalse(torch.allclose(lossy, self.z))
        exact = restore_layers(split_layers(self.z, factor=1), "full")
        self.assertTrue(torch.allclose(exact, self.z, atol=1e-6))

    def test_gradient_and_input_unchanged(self):
        before = self.z.detach().clone()
        restore_layers(split_layers(self.z), "full").square().mean().backward()
        self.assertTrue(torch.isfinite(self.z.grad).all())
        self.assertTrue(torch.equal(before, self.z.detach()))

    def test_backend_is_injectable(self):
        class IdentityResidual(ReferenceBackend):
            def split(self, z, factor):
                return super().split(z, 1)
        parts = split_layers(self.z, backend=IdentityResidual())
        self.assertEqual(parts.factor, 1)
        self.assertTrue(torch.allclose(restore_layers(parts, "full"), self.z, atol=1e-6))

    def test_invalid_inputs(self):
        invalid = [
            torch.zeros(1, 3, 4, 4), torch.zeros(1, 0, 3, 8, 8),
            torch.zeros(1, 4, 3, 7, 8), torch.zeros(1, 4, 3, 8, 8).double(),
            torch.full((1, 4, 3, 8, 8), float("nan")),
        ]
        for z in invalid:
            with self.subTest(shape=z.shape), self.assertRaises(ValueError):
                split_layers(z)
        for factor in (0, True, 1.5):
            with self.assertRaises(ValueError):
                split_layers(self.z, factor=factor)
        with self.assertRaises(ValueError):
            restore_layers(split_layers(self.z), "enhancement_only")


if __name__ == "__main__":
    unittest.main()
