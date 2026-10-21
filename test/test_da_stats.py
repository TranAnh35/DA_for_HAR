"""Tests for the paired bootstrap helpers."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.da_har.benchmark.stats import paired_bootstrap, summarize


class TestPairedBootstrap(unittest.TestCase):
    def test_positive_effect(self):
        diffs = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03])
        res = paired_bootstrap(diffs, n_boot=2000, seed=0)
        self.assertAlmostEqual(res["mean"], float(diffs.mean()), places=6)
        self.assertLessEqual(res["ci_low"], res["mean"])
        self.assertGreaterEqual(res["ci_high"], res["mean"])
        self.assertLess(res["p_value"], 0.5)

    def test_zero_effect_is_not_significant(self):
        rng = np.random.default_rng(0)
        diffs = rng.normal(0.0, 0.05, size=40)
        res = paired_bootstrap(diffs, n_boot=2000, seed=3)
        self.assertGreater(res["p_value"], 0.05)

    def test_degenerate_inputs(self):
        self.assertEqual(paired_bootstrap([])["n"], 0)
        one = paired_bootstrap([0.1])
        self.assertEqual(one["n"], 1)
        self.assertTrue(np.isnan(one["ci_low"]))


class TestSummarize(unittest.TestCase):
    def test_basic(self):
        mean, std, n = summarize([0.5, 0.7, 0.9])
        self.assertAlmostEqual(mean, 0.7)
        self.assertAlmostEqual(std, 0.2)
        self.assertEqual(n, 3)

    def test_empty(self):
        mean, std, n = summarize([])
        self.assertTrue(np.isnan(mean))
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
