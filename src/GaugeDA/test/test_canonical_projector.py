"""Tests for the canonical orientation projector (Module 01)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.GaugeDA.models.canonical_projector import CanonicalOrientationProjector


class TestCanonicalProjector(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.projector = CanonicalOrientationProjector(
            filter_window=25,
            sampling_rate=50.0,
            heading_mode="pca",
        ).eval()

    def test_antipodal_gravity_maps_to_plus_z(self):
        """g ~ -e_z must not be treated as already aligned."""
        x = torch.zeros(1, 6, 64)
        x[:, 2, :] = -9.81
        out = self.projector(x, sampling_rate=50.0)
        self.assertAlmostEqual(float(out[0, 2, 0]), 9.81, places=2)
        self.assertLess(float(out[:, 0, :].abs().max()), 1e-3)
        self.assertLess(float(out[:, 1, :].abs().max()), 1e-3)

    def test_yaw_rotation_invariance(self):
        """A yaw rotation about gravity must be canonicalized away."""
        t = torch.linspace(-1.0, 1.0, 64)
        # Skewed horizontal signal with a well-defined principal axis and a
        # non-zero third moment (the sign rule needs a stable heading).
        a = 0.5 + t.pow(2)
        b = 0.01 * t
        hx = a
        hy = b
        base = torch.zeros(1, 6, 64)
        base[0, 0] = hx
        base[0, 1] = hy
        base[0, 2] = 9.81

        for angle in (0.7, -1.2):
            cos_a = torch.cos(torch.tensor(angle))
            sin_a = torch.sin(torch.tensor(angle))
            rotated = base.clone()
            rotated[0, 0] = cos_a * hx - sin_a * hy
            rotated[0, 1] = sin_a * hx + cos_a * hy

            canon_base = self.projector(base, sampling_rate=50.0)
            canon_rot = self.projector(rotated, sampling_rate=50.0)
            self.assertTrue(
                torch.allclose(canon_base, canon_rot, atol=1e-3),
                msg=f"angle={angle} max diff={(canon_base - canon_rot).abs().max().item():.3e}",
            )

    def test_gravity_window_is_rate_aware(self):
        p50 = CanonicalOrientationProjector(
            sampling_rate=50.0, heading_mode="none", filter_seconds=0.5
        )
        p200 = CanonicalOrientationProjector(
            sampling_rate=200.0, heading_mode="none", filter_seconds=0.5
        )
        self.assertGreater(p200.filter_window, p50.filter_window)
        self.assertEqual(p50.filter_window % 2, 1)
        self.assertEqual(p200.filter_window % 2, 1)

    def test_explicit_window_is_kept(self):
        proj = CanonicalOrientationProjector(
            filter_window=15, sampling_rate=100.0, heading_mode="none"
        )
        self.assertEqual(proj._get_window(200.0), 15)

    def test_output_shape(self):
        x = torch.randn(3, 6, 128)
        out = self.projector(x, sampling_rate=50.0)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
