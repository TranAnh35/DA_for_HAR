"""Tests for windowing, resampling, channel identity and source-only scaling."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.da_har.data.core import (
    ChannelSpec,
    DomainData,
    align_channels,
    fit_normalizer,
    macro_f1,
    make_windows,
    pad_to_slots,
    resample_signal,
)


def _domain(X, y, subjects, channels, rate=50.0, name="d"):
    return DomainData(
        name=name,
        X=X,
        y=y,
        subjects=np.asarray(subjects),
        sessions=np.full(len(y), "s"),
        channels=channels,
        sampling_rate=rate,
    )


class TestChannelSpec(unittest.TestCase):
    def test_slot_ignores_placement(self):
        wrist = ChannelSpec("imu0", "acc", "x", "wrist")
        ankle = ChannelSpec("imu0", "acc", "x", "ankle")
        self.assertEqual(wrist.slot, ankle.slot)
        self.assertNotEqual(wrist.label, ankle.label)

    def test_rejects_unknown_axis(self):
        with self.assertRaises(ValueError):
            ChannelSpec("imu0", "acc", "w", "wrist")


class TestDomainData(unittest.TestCase):
    def setUp(self):
        self.channels = (
            ChannelSpec("imu0", "acc", "x", "waist"),
            ChannelSpec("imu0", "acc", "y", "waist"),
        )
        self.X = np.random.default_rng(0).normal(size=(10, 2, 32)).astype(np.float32)
        self.y = np.arange(10) % 3
        self.dom = _domain(self.X, self.y, ["a"] * 5 + ["b"] * 5, self.channels)

    def test_shape_contract(self):
        self.assertEqual(self.dom.num_windows, 10)
        self.assertEqual(self.dom.num_channels, 2)
        self.assertEqual(self.dom.window_size, 32)
        self.assertAlmostEqual(self.dom.window_seconds, 32 / 50.0)

    def test_channel_mismatch_raises(self):
        with self.assertRaises(ValueError):
            DomainData(
                name="bad",
                X=self.X,
                y=self.y,
                subjects=np.full(10, "a"),
                sessions=np.full(10, "s"),
                channels=self.channels[:1],
                sampling_rate=50.0,
            )

    def test_subset_and_take(self):
        sub = self.dom.subset(np.arange(4))
        self.assertEqual(sub.num_windows, 4)
        tiny = self.dom.take(3, seed=1)
        self.assertEqual(tiny.num_windows, 3)
        # A subsampled domain must keep its key (used for bundle lookup).
        self.assertEqual(tiny.name, self.dom.name)


class TestWindowing(unittest.TestCase):
    def test_windows_never_straddle_activity_boundary(self):
        signal = np.arange(100, dtype=np.float32).reshape(-1, 1)
        labels = np.array([0] * 50 + [1] * 50)
        X, y, starts = make_windows(
            signal, labels, sampling_rate=10.0, window_seconds=2.0, stride_seconds=1.0
        )
        self.assertGreater(len(y), 0)
        for start, label in zip(starts, y):
            seg = labels[start : start + 20]
            self.assertTrue(np.all(seg == label), f"window at {start} mixes labels")

    def test_ignored_transient_is_dropped_and_creates_gap(self):
        signal = np.zeros((120, 1), dtype=np.float32)
        labels = np.array([0] * 40 + [-1] * 40 + [1] * 40)
        X, y, _ = make_windows(
            signal, labels, sampling_rate=10.0, window_seconds=2.0, stride_seconds=1.0
        )
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))

    def test_empty_when_too_short(self):
        X, y, starts = make_windows(
            np.zeros((5, 1), dtype=np.float32), np.zeros(5), 10.0, 2.0
        )
        self.assertEqual(X.shape[0], 0)
        self.assertEqual(y.shape[0], 0)

    def test_rejects_bad_fraction(self):
        with self.assertRaises(ValueError):
            make_windows(np.zeros((50, 1)), np.zeros(50), 10.0, 2.0, min_label_fraction=0.0)


class TestResample(unittest.TestCase):
    def test_length_scales_with_rate(self):
        x = np.random.default_rng(0).normal(size=(200, 3)).astype(np.float32)
        up = resample_signal(x, 50.0, 100.0)
        down = resample_signal(x, 100.0, 50.0)
        self.assertEqual(up.shape, (400, 3))
        self.assertEqual(down.shape, (100, 3))

    def test_identity_when_rates_match(self):
        x = np.random.default_rng(0).normal(size=(64, 2)).astype(np.float32)
        self.assertTrue(np.array_equal(resample_signal(x, 50.0, 50.0), x))


class TestNormalizer(unittest.TestCase):
    def test_source_statistics_only(self):
        rng = np.random.default_rng(0)
        src = rng.normal(loc=2.0, scale=3.0, size=(20, 2, 16)).astype(np.float32)
        tgt = rng.normal(loc=-5.0, scale=0.5, size=(20, 2, 16)).astype(np.float32)
        norm = fit_normalizer(src)
        src_n = norm.transform(src)
        tgt_n = norm.transform(tgt)
        self.assertLess(abs(float(src_n.mean())), 1e-5)
        # Target must keep its own shift; the normaliser must not absorb it.
        self.assertGreater(abs(float(tgt_n.mean())), 1.0)
        np.testing.assert_allclose(norm.mean.reshape(-1), [2.0, 2.0], atol=1.0)


class TestChannelAlignment(unittest.TestCase):
    def test_union_and_mask(self):
        a = _domain(
            np.zeros((2, 2, 8), dtype=np.float32),
            np.zeros(2, dtype=int),
            ["a", "a"],
            (ChannelSpec("imu0", "acc", "x", "wrist"), ChannelSpec("imu0", "acc", "y", "wrist")),
            name="a",
        )
        b = _domain(
            np.zeros((2, 1, 8), dtype=np.float32),
            np.zeros(2, dtype=int),
            ["b", "b"],
            (ChannelSpec("imu0", "acc", "x", "ankle"),),
            name="b",
        )
        slots, masks = align_channels([a, b])
        self.assertEqual(slots, ("imu0:acc:x", "imu0:acc:y"))
        np.testing.assert_array_equal(masks[0], [True, True])
        np.testing.assert_array_equal(masks[1], [True, False])

        padded, mask = pad_to_slots(b, slots)
        self.assertEqual(padded.shape, (2, 2, 8))
        np.testing.assert_array_equal(mask, [True, False])


class TestMacroF1(unittest.TestCase):
    def test_perfect_and_wrong(self):
        y = np.array([0, 1, 2, 2])
        self.assertAlmostEqual(macro_f1(y, y), 1.0)
        self.assertAlmostEqual(macro_f1(y, np.array([1, 2, 0, 0])), 0.0)

    def test_ignores_absent_classes(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 0])
        # class 2 absent from both -> should not dilute the average
        self.assertAlmostEqual(macro_f1(y, pred, num_classes=3), (0.8 + 2.0 / 3.0) / 2)


if __name__ == "__main__":
    unittest.main()
