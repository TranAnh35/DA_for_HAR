"""Tests for the continuous-time ST-LieOp operator (Module 02)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.st_lie_operator import (
    ContinuousSSMCell,
    STLieOperatorBlock,
    _as_dt_scale,
    ssm_recurrent_scan,
)


class TestDeltaTScale(unittest.TestCase):
    def test_scalar_broadcast(self):
        u = torch.zeros(2, 5, 3)
        self.assertEqual(tuple(_as_dt_scale(0.02, u).shape), (1, 1, 1))

    def test_batch_vector_broadcast(self):
        u = torch.zeros(2, 5, 3)
        scale = _as_dt_scale(torch.tensor([0.02, 0.01]), u)
        self.assertEqual(tuple(scale.shape), (2, 1, 1))

    def test_batch_matrix_broadcast(self):
        u = torch.zeros(2, 5, 3)
        scale = _as_dt_scale(torch.full((2, 1, 1), 0.02), u)
        self.assertEqual(tuple(scale.shape), (2, 1, 1))

    def test_unsupported_shape_raises(self):
        u = torch.zeros(2, 5, 3)
        with self.assertRaises(ValueError):
            _as_dt_scale(torch.zeros(2, 2, 2, 2), u)


class TestContinuousSSMCell(unittest.TestCase):
    def test_forward_shape_and_finite(self):
        torch.manual_seed(0)
        cell = ContinuousSSMCell(d_model=8, d_state=4).eval()
        u = torch.randn(2, 7, 8)
        out = cell(u, delta_t=0.02)
        self.assertEqual(out.shape, (2, 7, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_physical_delta_t_changes_output(self):
        torch.manual_seed(0)
        cell = ContinuousSSMCell(d_model=4, d_state=3).eval()
        u = torch.randn(1, 6, 4)
        out_fast = cell(u, delta_t=0.001)
        out_slow = cell(u, delta_t=0.5)
        self.assertFalse(torch.allclose(out_fast, out_slow, atol=1e-5))

    def test_none_delta_t_still_runs(self):
        torch.manual_seed(0)
        cell = ContinuousSSMCell(d_model=4, d_state=3).eval()
        out = cell(torch.randn(1, 6, 4))
        self.assertEqual(out.shape, (1, 6, 4))


class TestSTLieOperatorBlock(unittest.TestCase):
    def test_residual_forward_shape(self):
        torch.manual_seed(0)
        block = STLieOperatorBlock(d_in=6, d_model=8, d_state=4).eval()
        x = torch.randn(2, 6, 16)
        out = block(x, delta_t=0.02)
        self.assertEqual(out.shape, (2, 8, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_recurrent_scan_shapes(self):
        B, T, dm, ds = 2, 5, 3, 4
        A_bar = torch.rand(B, T, dm, ds) * 0.9
        deltaB_u = torch.randn(B, T, dm, ds)
        C_mat = torch.randn(B, T, ds)
        D = torch.ones(dm)
        u = torch.randn(B, T, dm)
        out = ssm_recurrent_scan(A_bar, deltaB_u, C_mat, D, u)
        self.assertEqual(out.shape, (B, T, dm))


if __name__ == "__main__":
    unittest.main()
