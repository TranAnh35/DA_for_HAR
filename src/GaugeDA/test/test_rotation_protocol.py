"""Tests for the rotation evaluator's protocol helpers."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.GaugeDA.experiments.evaluate_rotation_robustness import (
    _build_model,
    generate_random_rotation_matrix,
)


class TestRotationMatrix(unittest.TestCase):
    def test_is_orthogonal_with_unit_determinant(self):
        rng = np.random.default_rng(0)
        for angle in (0.0, 30.0, 90.0, 180.0):
            r = generate_random_rotation_matrix(angle, torch.device("cpu"), rng)
            self.assertEqual(tuple(r.shape), (3, 3))
            eye = torch.eye(3, dtype=r.dtype)
            self.assertTrue(torch.allclose(r @ r.T, eye, atol=1e-5))
            self.assertAlmostEqual(float(torch.det(r)), 1.0, places=5)

    def test_seeded_rotations_are_deterministic(self):
        a = generate_random_rotation_matrix(45.0, torch.device("cpu"), np.random.default_rng(7))
        b = generate_random_rotation_matrix(45.0, torch.device("cpu"), np.random.default_rng(7))
        self.assertTrue(torch.allclose(a, b))

    def test_identity_at_zero_angle(self):
        r = generate_random_rotation_matrix(0.0, torch.device("cpu"), np.random.default_rng(0))
        self.assertTrue(torch.allclose(r, torch.eye(3), atol=1e-6))


class TestModelBuilder(unittest.TestCase):
    def test_builds_with_and_without_projector(self):
        ckpt = {
            "args": {
                "d_model": 8,
                "d_state": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "filter_window": None,
                "heading_mode": "pca",
            }
        }
        with_proj = _build_model(ckpt, 3, 6, 50.0, True, torch.device("cpu"))
        without_proj = _build_model(ckpt, 3, 6, 50.0, False, torch.device("cpu"))
        self.assertTrue(with_proj.use_projector)
        self.assertFalse(without_proj.use_projector)
        x = torch.randn(2, 6, 32)
        self.assertEqual(with_proj(x)[0].shape, (2, 3))
        self.assertEqual(without_proj(x)[0].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
