"""Tests for source-only model selection and the label-free split protocol."""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.selection import (
    evaluate,
    forward_model,
    is_improvement,
    selection_score,
    split_source_train_val,
)


class DummyDataset:
    """Minimal dataset exposing ``domain_ids`` and ``labels`` like loaders do."""

    def __init__(self, domains, labels):
        self.domain_ids = np.asarray(domains)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.domain_ids)

    def __getitem__(self, index):
        return self.domain_ids[index], self.labels[index]


def make_dataset(n_domains=6, per_domain=10, n_classes=3, rare_domain=None):
    domains, labels = [], []
    for d in range(n_domains):
        for i in range(per_domain):
            domains.append(d)
            if rare_domain is not None and d == rare_domain:
                labels.append(n_classes - 1)
            else:
                labels.append(i % (n_classes - 1))
    return DummyDataset(domains, labels)


class TestSplitProtocol(unittest.TestCase):
    def test_domains_are_disjoint_between_train_and_val(self):
        dataset = make_dataset()
        train, val, info = split_source_train_val(dataset, val_fraction=0.2, seed=42)
        self.assertEqual(info["split"], "subject_held_out")
        train_domains = {int(dataset.domain_ids[i]) for i in train.indices}
        val_domains = {int(dataset.domain_ids[i]) for i in val.indices}
        self.assertEqual(train_domains & val_domains, set())
        self.assertEqual(info["n_train"], len(train.indices))
        self.assertEqual(info["n_val"], len(val.indices))

    def test_rare_class_stays_in_training(self):
        dataset = make_dataset(n_domains=6, per_domain=10, n_classes=3, rare_domain=5)
        train, _, info = split_source_train_val(dataset, val_fraction=0.2, seed=42)
        self.assertIn(5, info["forced_train_domains"])
        self.assertIn(2, info["train_classes"])
        train_domains = {int(dataset.domain_ids[i]) for i in train.indices}
        self.assertIn(5, train_domains)

    def test_invalid_fraction_raises(self):
        dataset = make_dataset()
        with self.assertRaises(ValueError):
            split_source_train_val(dataset, val_fraction=0.0)

    def test_fallback_random_split_without_domain_ids(self):
        dataset = TensorDataset(torch.arange(20).unsqueeze(1), torch.zeros(20, dtype=torch.long))
        train, val, info = split_source_train_val(dataset, val_fraction=0.25, seed=0)
        self.assertEqual(info["split"], "random")
        self.assertEqual(len(train) + len(val), 20)


class TestSelectionScores(unittest.TestCase):
    def test_loss_is_negated_so_larger_is_better(self):
        self.assertGreater(selection_score("source_loss", {"loss": 0.5}),
                           selection_score("source_loss", {"loss": 1.5}))

    def test_macro_f1_and_accuracy(self):
        self.assertEqual(selection_score("source_macro_f1", {"macro_f1": 80.0}), 80.0)
        self.assertEqual(selection_score("source_accuracy", {"accuracy": 80.0}), 80.0)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            selection_score("nope", {"accuracy": 1.0})

    def test_is_improvement(self):
        self.assertTrue(is_improvement("source_macro_f1", {"macro_f1": 0.8}, best=0.7))
        self.assertFalse(is_improvement("source_macro_f1", {"macro_f1": 0.6}, best=0.7))


class _AcceptsKwargs(nn.Module):
    def forward(self, x, delta_t=None, sampling_rate=None):
        return x, x


class _Plain(nn.Module):
    def forward(self, x):
        return x, x


class TestForwardAndEvaluate(unittest.TestCase):
    def test_forward_model_handles_baseline_without_kwargs(self):
        x = torch.randn(2, 3, 8)
        out_kwargs, _ = forward_model(_AcceptsKwargs(), x, delta_t=0.02, sampling_rate=50.0)
        out_plain, _ = forward_model(_Plain(), x, delta_t=0.02, sampling_rate=50.0)
        self.assertTrue(torch.allclose(out_kwargs, x))
        self.assertTrue(torch.allclose(out_plain, x))

    def test_evaluate_returns_metrics(self):
        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 2)

            def forward(self, x, **_):
                return self.fc(x), x

        x = torch.randn(8, 3)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = evaluate(Classifier(), loader, torch.device("cpu"), use_amp=False)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()
