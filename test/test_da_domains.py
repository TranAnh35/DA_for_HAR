"""Tests for task definitions (source/target domain selection)."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.da_har.benchmark import domains as D
from src.da_har.data.core import ChannelSpec, DatasetBundle, DomainData, make_imu_channels


def _bundle(keys, num_classes=3, n=8, window=32, channels=None):
    channels = channels or make_imu_channels("imu0", "waist", kinds=("acc",))
    rng = np.random.default_rng(0)
    domains = {}
    for key in keys:
        X = rng.normal(size=(n, len(channels), window)).astype(np.float32)
        y = rng.integers(0, num_classes, size=n)
        domains[key] = DomainData(
            name=key,
            X=X,
            y=y,
            subjects=np.array([f"{key}-s{i}" for i in range(n)]),
            sessions=np.full(n, "s"),
            channels=channels,
            sampling_rate=50.0,
        )
    return DatasetBundle(name="toy", domains=domains, label_names=tuple(f"c{i}" for i in range(num_classes)))


class TestBuildTasks(unittest.TestCase):
    def test_uci_tasks_are_subject_disjoint(self):
        tasks = D.build_tasks("uci_har", "cross_subject")
        self.assertEqual(len(tasks), 2)
        for spec in tasks:
            self.assertTrue(spec.source.keys and spec.target.keys)
            self.assertFalse(set(spec.source.keys) & set(spec.target.keys))
            self.assertTrue(all(k.startswith("subject:") for k in spec.source.keys))

    def test_hhar_only_cross_device(self):
        tasks = D.build_tasks("hhar", "cross_device")
        self.assertEqual({t.name for t in tasks}, {"hhar/s3_to_nexus4", "hhar/nexus4_to_s3"})
        with self.assertRaises(ValueError):
            D.build_tasks("hhar", "cross_subject")

    def test_pamap2_cross_placement_uses_different_sensors(self):
        tasks = D.build_tasks("pamap2", "cross_placement")
        self.assertTrue(tasks)
        for spec in tasks:
            src_sensor = spec.source.loader["sensors"]
            tgt_sensor = spec.target.loader["sensors"]
            self.assertNotEqual(src_sensor, tgt_sensor)
            self.assertEqual(set(spec.source.keys), set(spec.target.keys))

    def test_wisdm_and_opportunity_shifts(self):
        self.assertEqual(len(D.build_tasks("wisdm", "cross_device")), 2)
        self.assertEqual(len(D.build_tasks("wisdm", "cross_subject")), 2)
        self.assertEqual(len(D.build_tasks("opportunity", "cross_subject")), 2)
        self.assertEqual(len(D.build_tasks("opportunity", "cross_session")), 1)

    def test_unknown_dataset_and_shift(self):
        with self.assertRaises(ValueError):
            D.build_tasks("nope", "cross_subject")
        with self.assertRaises(ValueError):
            D.build_tasks("uci_har", "cross_device")

    def test_max_windows_is_injected_into_loaders(self):
        specs = D.build_tasks("uci_har", "cross_subject", max_windows=123)
        for spec in specs:
            self.assertEqual(spec.source.loader["max_windows_per_domain"], 123)
            self.assertEqual(spec.target.loader["max_windows_per_domain"], 123)
        # ... and the uncapped builder stays uncapped.
        plain = D.build_tasks("uci_har", "cross_subject")
        self.assertNotIn("max_windows_per_domain", plain[0].source.loader)


class TestMaterialize(unittest.TestCase):
    def test_materialize_combines_domains(self):
        bundle = _bundle(["device:a", "device:b"])
        spec = D.TaskSpec(
            "toy/a_to_b",
            "toy",
            "cross_device",
            D.DomainRequest(("device:a",)),
            D.DomainRequest(("device:b",)),
        )
        with mock.patch.object(D, "_load_cached", return_value=bundle):
            task = spec.materialize()
        self.assertEqual(task.source.num_windows, 8)
        self.assertEqual(task.target.num_windows, 8)
        self.assertEqual(task.num_classes, 3)
        self.assertEqual(task.summary()["source_slots"], ["imu0:acc:x", "imu0:acc:y", "imu0:acc:z"])

    def test_unknown_domain_raises(self):
        bundle = _bundle(["device:a"])
        spec = D.TaskSpec(
            "toy/x",
            "toy",
            "cross_device",
            D.DomainRequest(("device:a",)),
            D.DomainRequest(("device:missing",)),
        )
        with mock.patch.object(D, "_load_cached", return_value=bundle):
            with self.assertRaises(KeyError):
                spec.materialize()


class TestLoaderCacheKey(unittest.TestCase):
    def test_lists_become_hashable(self):
        req = D.DomainRequest(("a",), {"subjects": [1, 2, 3], "sensors": ("wrist",)})
        key = D._loader_key(req)
        hash(key)  # must not raise
        self.assertIn(("subjects", (1, 2, 3)), key)

    def test_two_equal_requests_share_key(self):
        a = D.DomainRequest(("a",), {"subjects": [1, 2]})
        b = D.DomainRequest(("a",), {"subjects": [1, 2]})
        self.assertEqual(D._loader_key(a), D._loader_key(b))


if __name__ == "__main__":
    unittest.main()
