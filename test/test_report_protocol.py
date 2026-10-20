"""Tests for post-fix benchmark filtering / history versioning."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.generate_benchmark_report import (
    history_version,
    load_postfix_benchmarks,
)


class TestHistoryVersion(unittest.TestCase):
    def test_legacy_history_is_detected(self):
        legacy = [{"epoch": 1, "target_macro_f1": 90.0}]
        self.assertEqual(history_version(legacy), "legacy")

    def test_post_fix_history_is_detected(self):
        post_fix = [{"epoch": 1, "is_best": True, "source_val_macro_f1": 80.0}]
        self.assertEqual(history_version(post_fix), "post_fix")

    def test_missing_history(self):
        self.assertEqual(history_version(None), "missing")
        self.assertEqual(history_version([]), "missing")


class TestPostfixBenchmarkLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def _write(self, name, payload):
        with open(os.path.join(self.root, name), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_valid_and_stale_files_are_separated(self):
        self._write(
            "valid.json",
            {
                "is_target_oracle": False,
                "selection": {"uses_target_labels": False},
                "args": {
                    "dataset": "uci_har",
                    "direction": "forward",
                    "method": "coral",
                    "backbone": "cnn1d",
                    "seed": 42,
                },
                "best_target_accuracy": 90.0,
                "best_target_macro_f1": 89.0,
            },
        )
        self._write(
            "stale.json",
            {
                "args": {
                    "dataset": "uci_har",
                    "direction": "forward",
                    "method": "coral",
                    "backbone": "cnn1d",
                    "seed": 42,
                },
                "best_target_accuracy": 99.0,
                "best_target_macro_f1": 99.0,
            },
        )

        groups, stale = load_postfix_benchmarks(self.root)
        self.assertEqual(stale, 1)
        key = ("uci_har", "forward", "coral", "cnn1d")
        self.assertIn(key, groups)
        self.assertEqual(len(groups[key]), 1)
        self.assertAlmostEqual(groups[key][0][0], 90.0)

    def test_missing_directory_is_empty(self):
        groups, stale = load_postfix_benchmarks(os.path.join(self.root, "nope"))
        self.assertEqual(len(groups), 0)
        self.assertEqual(stale, 0)


if __name__ == "__main__":
    unittest.main()
