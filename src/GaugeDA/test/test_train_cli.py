"""CLI contract tests for train_benchmark.py (independent projector ablation)."""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "src", "GaugeDA", "train_benchmark.py")


class TestNoProjectorFlag(unittest.TestCase):
    def test_help_lists_no_projector(self):
        proc = subprocess.run(
            [sys.executable, TRAIN_SCRIPT, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--no_projector", proc.stdout)

    def test_no_projector_rejected_for_cnn1d(self):
        proc = subprocess.run(
            [
                sys.executable,
                TRAIN_SCRIPT,
                "--no_projector",
                "--backbone",
                "cnn1d",
                "--epochs",
                "1",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--no_projector", proc.stderr)


if __name__ == "__main__":
    unittest.main()
