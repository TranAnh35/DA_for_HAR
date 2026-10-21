"""Loader tests built on tiny synthetic fixtures (no dependency on Data/)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.da_har.data import uci_har, pamap2, wisdm, opportunity, hhar


def _uci_fixture(root: Path) -> None:
    base = root / "UCI_HAR" / "UCI HAR Dataset"
    rng = np.random.default_rng(0)
    for subset, n in (("train", 6), ("test", 4)):
        signals = base / subset / "Inertial Signals"
        signals.mkdir(parents=True)
        for stem in ("total_acc", "body_acc", "body_gyro"):
            for axis in "xyz":
                np.savetxt(
                    signals / f"{stem}_{axis}_{subset}.txt",
                    rng.normal(size=(n, 128)),
                    fmt="%.4f",
                )
        np.savetxt(base / subset / f"y_{subset}.txt", np.arange(1, n + 1), fmt="%d")
        np.savetxt(base / subset / f"subject_{subset}.txt", np.arange(1, n + 1), fmt="%d")


def _pamap2_fixture(root: Path) -> None:
    base = root / "PAMAP2_Dataset" / "Protocol"
    base.mkdir(parents=True)
    rng = np.random.default_rng(0)
    segments = [(4, 150), (0, 100), (1, 150)]  # walking, transient, lying
    total = sum(n for _, n in segments)
    arr = np.zeros((total, 54), dtype=np.float64)
    arr[:, 0] = np.arange(total) / 100.0
    arr[:, 1] = np.concatenate([np.full(n, a) for a, n in segments])
    arr[:, 2] = 70.0
    # Distinct means per placement so tests can verify column ordering.
    for c in pamap2.POSITION_COLS["wrist"]:
        arr[:, c] = 5.0 + rng.normal(scale=0.05, size=total)
    for c in pamap2.POSITION_COLS["ankle"]:
        arr[:, c] = -5.0 + rng.normal(scale=0.05, size=total)
    np.savetxt(base / "subject101.dat", arr, fmt="%.5f")


def _wisdm_fixture(root: Path) -> None:
    base = root / "wisdm-dataset" / "raw"
    rng = np.random.default_rng(0)
    for device in ("phone", "watch"):
        for sensor in ("accel", "gyro"):
            folder = base / device / sensor
            folder.mkdir(parents=True)
            lines = []
            for offset, activity in enumerate(("A", "D")):
                t0 = 1_000_000_000 + offset * 100_000_000_000
                for i in range(120):
                    ts = t0 + i * 50_000_000
                    x, y, z = rng.normal(size=3)
                    lines.append(
                        f"1600,{activity},{ts},{x:.5f},{y:.5f},{z:.5f};"
                    )
            (folder / f"data_1600_{sensor}_{device}.txt").write_text("\n".join(lines))


def _opportunity_fixture(root: Path) -> None:
    base = root / "OpportunityUCIDataset" / "dataset"
    base.mkdir(parents=True)
    n = 400
    arr = np.zeros((n, 250), dtype=np.float64)
    arr[:, 0] = np.arange(n) * (1000.0 / 30.0)
    arr[:, 1:19] = 1.0
    labels = np.concatenate(
        [np.full(100, 1.0), np.full(100, 2.0), np.full(100, 4.0), np.full(100, 5.0)]
    )
    arr[:, 243] = labels
    np.savetxt(base / "S1-ADL1.dat", arr, fmt="%.3f")


def _hhar_fixture(root: Path, gap_seconds: float = 0.0) -> None:
    base = root / "heterogeneity" / "Activity recognition exp"
    base.mkdir(parents=True)
    header = "Index,Arrival_Time,Creation_Time,x,y,z,User,Model,Device,gt"
    rng = np.random.default_rng(0)
    acc_lines = [header]
    gyro_lines = [header]
    idx = 0
    for model in ("s3", "nexus4"):
        for activity in ("walk", "sit"):
            t0 = 1_424_696_600_000_000_000
            offsets = [0.0]
            if gap_seconds and model == "s3" and activity == "walk":
                offsets = [0.0, gap_seconds]
            for offset in offsets:
                for i in range(300):
                    t = t0 + int(offset * 1e9) + i * 10_000_000  # 100 Hz in ns
                    x, y, z = rng.normal(size=3)
                    gx, gy, gz = rng.normal(size=3)
                    acc_lines.append(f"{idx},{t // 1000},{t},{x:.5f},{y:.5f},{z:.5f},a,{model},{model}_1,{activity}")
                    gyro_lines.append(f"{idx},{t // 1000},{t},{gx:.5f},{gy:.5f},{gz:.5f},a,{model},{model}_1,{activity}")
                    idx += 1
    (base / "Phones_accelerometer.csv").write_text("\n".join(acc_lines))
    (base / "Phones_gyroscope.csv").write_text("\n".join(gyro_lines))


class TestUCILoader(unittest.TestCase):
    def test_domains_and_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _uci_fixture(Path(tmp))
            bundle = uci_har.load_bundle(root=tmp)
            self.assertEqual(bundle.name, "uci_har")
            self.assertEqual(len(bundle.label_names), 6)
            self.assertEqual(len(bundle.domains), 6)
            any_dom = next(iter(bundle.domains.values()))
            self.assertEqual(any_dom.num_channels, 9)
            self.assertEqual(any_dom.window_size, 128)
            self.assertAlmostEqual(any_dom.sampling_rate, 50.0)
            self.assertTrue(all(k.startswith("subject:") for k in bundle.domains))


class TestPAMAP2Loader(unittest.TestCase):
    def test_transient_removed_and_runs_not_bridged(self):
        with tempfile.TemporaryDirectory() as tmp:
            _pamap2_fixture(Path(tmp))
            bundle = pamap2.load_bundle(
                root=tmp, sensors=("wrist",), window_seconds=0.5, stride_seconds=0.25
            )
            dom = bundle.get("subject:101")
            self.assertEqual(dom.num_channels, 6)
            self.assertEqual(set(np.unique(dom.y).tolist()), {0, 3})  # lying, walking
            self.assertEqual(dom.num_windows, 10)  # 5 walking + 5 lying, no transient

    def test_cross_placement_slots_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            _pamap2_fixture(Path(tmp))
            wrist = pamap2.load_bundle(root=tmp, sensors=("wrist",))
            ankle = pamap2.load_bundle(root=tmp, sensors=("ankle",))
            self.assertEqual(
                wrist.get("subject:101").slot_names(),
                ankle.get("subject:101").slot_names(),
            )
            self.assertNotEqual(
                wrist.get("subject:101").channels[0].placement,
                ankle.get("subject:101").channels[0].placement,
            )

    def test_multi_sensor_columns_follow_requested_order(self):
        # Regression: columns used to be sorted by file index, so
        # sensors=("ankle", "wrist") silently put wrist data in imu0.
        with tempfile.TemporaryDirectory() as tmp:
            _pamap2_fixture(Path(tmp))
            bundle = pamap2.load_bundle(root=tmp, sensors=("ankle", "wrist"))
            dom = bundle.get("subject:101")
            self.assertEqual(
                [c.placement for c in dom.channels],
                ["ankle"] * 6 + ["wrist"] * 6,
            )
            ankle_block = dom.X[:, :6, :].mean()
            wrist_block = dom.X[:, 6:, :].mean()
            self.assertLess(float(ankle_block), 0.0)   # fixture ankle = -5
            self.assertGreater(float(wrist_block), 0.0)  # fixture wrist = +5


class TestWISDMLoader(unittest.TestCase):
    def test_cross_device_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            _wisdm_fixture(Path(tmp))
            bundle = wisdm.load_bundle(root=tmp)
            self.assertEqual(set(bundle.domains), {"device:phone", "device:watch"})
            dom = bundle.get("device:phone")
            self.assertEqual(dom.num_channels, 6)
            self.assertEqual(len(bundle.label_names), 18)
            # two activities (walking, sitting) x 2 windows each
            self.assertEqual(set(np.unique(dom.y).tolist()), {0, 3})
            self.assertGreater(dom.num_windows, 0)


class TestOpportunityLoader(unittest.TestCase):
    def test_labels_and_subject_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            _opportunity_fixture(Path(tmp))
            bundle = opportunity.load_bundle(root=tmp, sensors=("RKN",), window_seconds=1.0)
            dom = bundle.get("subject:S1")
            self.assertEqual(dom.num_channels, 3)
            self.assertEqual(set(np.unique(dom.y).tolist()), {0, 1, 2, 3})
            self.assertEqual(set(np.unique(dom.sessions).tolist()), {"ADL1"})
            self.assertGreater(dom.num_windows, 0)


class TestHHARLoader(unittest.TestCase):
    def test_streaming_cross_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            _hhar_fixture(Path(tmp))
            bundle = hhar.load_bundle(root=tmp, use_cache=False, chunk_size=64)
            self.assertEqual(set(bundle.domains), {"device:s3", "device:nexus4"})
            dom = bundle.get("device:s3")
            self.assertEqual(dom.num_channels, 6)
            self.assertEqual(set(np.unique(dom.y).tolist()), {1, 3})  # sit, walk
            self.assertEqual(set(np.unique(dom.subjects).tolist()), {"a"})
            self.assertGreater(dom.num_windows, 0)

    def test_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "heterogeneity" / "Activity recognition exp").mkdir(parents=True)
            with self.assertRaises(ValueError):
                hhar.load_bundle(root=tmp, use_cache=False)

    def test_recording_gap_is_not_interpolated(self):
        # One hour of dead time inside the same (user, device, activity) group
        # must NOT be resampled into hundreds of thousands of fake windows.
        with tempfile.TemporaryDirectory() as tmp:
            _hhar_fixture(Path(tmp), gap_seconds=3600.0)
            bundle = hhar.load_bundle(root=tmp, use_cache=False)
            dom = bundle.get("device:s3")
            # two walked runs of 300 samples at stride 64 -> 3 windows each,
            # plus the sit run -> 3; a bridged interpolation would give ~5.6k.
            self.assertLess(dom.num_windows, 20)
            self.assertGreaterEqual(dom.num_windows, 6)


if __name__ == "__main__":
    unittest.main()
