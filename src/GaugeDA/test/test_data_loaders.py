"""Tests for loader edge cases (HHAR resampling, PAMAP2 empty branch)."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.GaugeDA.data.hhar_loader import ACTIVITY_MAP, _resample_user
from src.GaugeDA.data.pamap2_loader import process_subject_file


def _hhar_frame(times_ns, label="walk"):
    n = len(times_ns)
    return pd.DataFrame(
        {
            "Creation_Time": np.asarray(times_ns, dtype=np.int64),
            "x": np.ones(n, dtype=float),
            "y": np.zeros(n, dtype=float),
            "z": np.zeros(n, dtype=float),
            "gt": np.array([label] * n, dtype=object),
        }
    )


class TestHHARResample(unittest.TestCase):
    def test_dual_rate_signals_share_common_grid(self):
        acc_times = np.arange(0, 1.0, 1.0 / 200.0) * 1e9
        gyro_times = np.arange(0, 1.0, 1.0 / 100.0) * 1e9
        acc = _hhar_frame(acc_times)
        gyro = _hhar_frame(gyro_times)

        out = _resample_user(acc, gyro, target_hz=100.0, window_size=32)
        self.assertIsNotNone(out)
        signals, labels = out
        self.assertEqual(signals.shape[0], 6)
        self.assertEqual(signals.shape[1], labels.shape[0])
        self.assertGreaterEqual(signals.shape[1], 32)
        self.assertTrue(np.all(labels == ACTIVITY_MAP["walk"]))

    def test_too_short_overlap_returns_none(self):
        acc = _hhar_frame([0, 10])
        gyro = _hhar_frame([0, 10])
        self.assertIsNone(_resample_user(acc, gyro, target_hz=100.0, window_size=32))


class TestPAMAP2EmptyBranch(unittest.TestCase):
    SOURCE_COLS = [4, 5, 6, 10, 11, 12]
    TARGET_COLS = [21, 22, 23, 27, 28, 29]

    def _write_dat(self, activity, rows=3):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False, encoding="utf-8"
        )
        for _ in range(rows):
            values = [0.0] * 54
            values[1] = float(activity)
            handle.write(" ".join(str(v) for v in values) + "\n")
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def test_too_short_file_returns_four_arrays(self):
        path = self._write_dat(activity=1, rows=3)
        result = process_subject_file(
            path, 0, self.SOURCE_COLS, self.TARGET_COLS, window_size=128, step_size=64
        )
        self.assertEqual(len(result), 4)
        src, tgt, labels, subjects = result
        self.assertEqual(src.shape, (0, 6, 128))
        self.assertEqual(tgt.shape, (0, 6, 128))
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(subjects.shape, (0,))

    def test_no_valid_activity_returns_four_arrays(self):
        path = self._write_dat(activity=0, rows=200)
        result = process_subject_file(
            path, 0, self.SOURCE_COLS, self.TARGET_COLS, window_size=128, step_size=64
        )
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
