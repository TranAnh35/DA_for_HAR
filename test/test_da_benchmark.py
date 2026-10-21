"""End-to-end tests for the source-selected benchmark loop (CPU, synthetic)."""

import dataclasses
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.da_har.benchmark.domains import DomainRequest, TaskData, TaskSpec
from src.da_har.benchmark.methods import build_method
from src.da_har.benchmark.models import MaskedCNN1D
from src.da_har.benchmark.protocol import subject_disjoint_split
from src.da_har.benchmark.trainer import TrainConfig, train_task
from src.da_har.data.core import ChannelSpec, DomainData, make_imu_channels


def _domain(name, n, subjects, rng, shift=0.0, num_classes=3, window=64):
    channels = make_imu_channels("imu0", "waist", kinds=("acc", "gyro"))
    X = rng.normal(size=(n, len(channels), window)).astype(np.float32) + shift
    y = rng.integers(0, num_classes, size=n)
    return DomainData(
        name=name,
        X=X,
        y=y,
        subjects=np.asarray(subjects),
        sessions=np.full(n, "s"),
        channels=channels,
        sampling_rate=50.0,
    )


def _task(rng, seed=0):
    src = _domain("src", 48, ["s1"] * 16 + ["s2"] * 16 + ["s3"] * 16, rng, shift=0.0)
    tgt = _domain("tgt", 36, ["t1"] * 18 + ["t2"] * 18, rng, shift=1.0)
    spec = TaskSpec(
        "toy/a_to_b",
        "toy",
        "cross_subject",
        DomainRequest(("device:a",)),
        DomainRequest(("device:b",)),
    )
    return TaskData(spec=spec, source=src, target=tgt, label_names=("c0", "c1", "c2"))


class TestProtocol(unittest.TestCase):
    def test_subject_disjoint_split(self):
        rng = np.random.default_rng(0)
        dom = _domain("d", 60, ["s1"] * 20 + ["s2"] * 20 + ["s3"] * 20, rng)
        split = subject_disjoint_split(dom, val_fraction=0.34, seed=1)
        self.assertTrue(split.subject_disjoint)
        train_subj = set(dom.subjects[split.train].tolist())
        val_subj = set(dom.subjects[split.val].tolist())
        self.assertFalse(train_subj & val_subj)
        self.assertEqual(len(train_subj) + len(val_subj), 3)

    def test_single_subject_falls_back(self):
        rng = np.random.default_rng(0)
        dom = _domain("d", 30, ["s1"] * 30, rng)
        split = subject_disjoint_split(dom, val_fraction=0.2, seed=0)
        self.assertFalse(split.subject_disjoint)
        self.assertGreater(len(split.val), 0)
        self.assertGreater(len(split.train), 0)


class TestModelAndMethods(unittest.TestCase):
    def test_masked_forward(self):
        model = MaskedCNN1D(in_slots=6, num_classes=4, hidden=8)
        x = torch.randn(5, 6, 64)
        mask = torch.tensor([True, True, True, False, False, False])
        logits, feats = model(x, mask)
        self.assertEqual(logits.shape, (5, 4))
        self.assertEqual(feats.shape, (5, model.feat_dim))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            build_method("does_not_exist", feat_dim=8, num_classes=3)


class TestTrainTask(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.task = _task(self.rng)
        self.config = TrainConfig(
            epochs=3, batch_size=8, lr=1e-3, hidden=8, dropout=0.0, device="cpu", seed=7
        )

    def test_result_contract(self):
        result = train_task(self.task, "source_only", self.config)
        self.assertFalse(result["is_target_oracle"])
        self.assertEqual(result["selection"]["metric"], "source_val_macro_f1")
        self.assertFalse(result["protocol"]["target_labels_in_selection"])
        self.assertFalse(result["protocol"]["target_labels_in_training"])
        self.assertEqual(result["data"]["source_train_windows"] + result["data"]["source_val_windows"],
                         self.task.source.num_windows)
        for key in ("target_accuracy", "target_macro_f1", "source_val_macro_f1"):
            self.assertIn(key, result["metrics"])
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["label_names"], ["c0", "c1", "c2"])

    def test_adaptation_methods_run(self):
        for method in ("coral", "mmd", "dann"):
            result = train_task(self.task, method, self.config)
            self.assertIn("target_macro_f1", result["metrics"])

    def test_target_labels_do_not_affect_selection(self):
        before = train_task(self.task, "source_only", self.config)

        shuffled = np.random.default_rng(123).permutation(self.task.target.y)
        tampered_target = dataclasses.replace(self.task.target, y=shuffled)
        tampered = TaskData(
            spec=self.task.spec,
            source=self.task.source,
            target=tampered_target,
            label_names=self.task.label_names,
        )
        after = train_task(tampered, "source_only", self.config)

        self.assertEqual(before["selection"]["best_epoch"], after["selection"]["best_epoch"])
        self.assertEqual(
            before["metrics"]["source_val_macro_f1"],
            after["metrics"]["source_val_macro_f1"],
        )
        # ... while the reported target metric does change (it used the labels).
        self.assertNotEqual(
            before["metrics"]["target_accuracy"],
            after["metrics"]["target_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
