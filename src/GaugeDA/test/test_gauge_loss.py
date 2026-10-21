"""Tests for the gauge-invariant UDA loss (class-conditional prototype alignment)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.GaugeDA.losses.gauge_invariant_loss import GaugeInvariantUDALoss


class TestGaugeInvariantLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.criterion = GaugeInvariantUDALoss(
            lambda_gauge=0.5,
            lambda_ent=0.1,
            pseudo_threshold=0.0,
        )

    def _batch(self, num_classes=3, n=16, dim=5):
        return (
            torch.randn(n, num_classes, requires_grad=True),
            torch.randint(0, num_classes, (n,)),
            torch.randn(n, dim),
            torch.randn(n, dim),
            torch.randn(n, num_classes),
        )

    def test_loss_da_equals_loss_proto_and_keys_present(self):
        loss, metrics = self.criterion(*self._batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["loss_da"], metrics["loss_proto"], places=7)
        self.assertAlmostEqual(metrics["loss_gauge"], metrics["loss_proto"], places=7)
        for key in ("loss_total", "loss_src", "loss_da", "loss_proto", "loss_ent",
                    "proto_classes", "pseudo_coverage", "proto_fallback"):
            self.assertIn(key, metrics)

    def test_class_conditional_path_has_valid_classes(self):
        logits_s = torch.randn(32, 4, requires_grad=True)
        y_s = torch.randint(0, 4, (32,))
        z_s = torch.randn(32, 5, requires_grad=True)
        z_t = torch.randn(32, 5, requires_grad=True)
        logits_t = torch.randn(32, 4)
        loss, metrics = self.criterion(logits_s, y_s, z_s, z_t, logits_t)
        self.assertGreaterEqual(metrics["proto_classes"], 1.0)
        self.assertEqual(metrics["proto_fallback"], 0.0)
        loss.backward()
        self.assertIsNotNone(logits_s.grad)
        self.assertIsNotNone(z_t.grad)

    def test_global_fallback_when_no_class_overlap(self):
        logits_s = torch.zeros(4, 3)
        y_s = torch.zeros(4, dtype=torch.long)  # only class 0
        logits_t = torch.tensor([[0.0, 5.0, 0.0]] * 4)  # argmax class 1
        z_s = torch.randn(4, 5)
        z_t = torch.randn(4, 5)
        _, metrics = self.criterion(logits_s, y_s, z_s, z_t, logits_t)
        self.assertEqual(metrics["proto_fallback"], 1.0)
        self.assertEqual(metrics["proto_classes"], 0.0)

    def test_disabled_branches_return_zero_da(self):
        criterion = GaugeInvariantUDALoss(
            lambda_gauge=0.0, lambda_ent=0.0, pseudo_threshold=0.0
        )
        _, metrics = criterion(*self._batch())
        self.assertEqual(metrics["loss_da"], 0.0)
        self.assertEqual(metrics["loss_ent"], 0.0)


if __name__ == "__main__":
    unittest.main()
