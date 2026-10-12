import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation import compare  # noqa: E402


class TestFindCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "checkpoints").mkdir()
        (self.root / "runs").mkdir()
        patcher = mock.patch.object(compare, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def touch(self, rel):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        time.sleep(0.01)  # distinct mtimes
        return path

    def test_precedence(self):
        legacy = self.touch("reinforce")
        self.assertEqual(compare.find_checkpoint("reinforce", {}), legacy)
        self.touch("checkpoints/reinforce_20260101_000000.pt")
        newest = self.touch("checkpoints/reinforce_20260102_000000.pt")
        self.assertEqual(compare.find_checkpoint("reinforce", {}), newest)
        self.assertEqual(compare.find_checkpoint("reinforce", {"reinforce": Path("x.pt")}), Path("x.pt"))

    def test_prefix_does_not_leak_between_algorithms(self):
        self.touch("checkpoints/trpo_20260101_000000.pt")
        self.assertIsNone(compare.find_checkpoint("ppo", {}))

    def test_run_matching(self):
        ckpt = self.touch("checkpoints/ppo_20260102_000000.pt")
        (self.root / "runs" / "ppo_20260101_000000").mkdir()
        self.assertEqual(compare.find_run("ppo", ckpt, {})[1], "latest run")
        (self.root / "runs" / ckpt.stem).mkdir()
        self.assertEqual(compare.find_run("ppo", ckpt, {}), (self.root / "runs" / ckpt.stem, "matches checkpoint"))
        self.assertEqual(compare.find_run("sac", Path("checkpoints/sac_1.pt"), {})[1], "none")


class TestHelpers(unittest.TestCase):
    def test_moving_average(self):
        np.testing.assert_allclose(compare.moving_average(np.array([1.0, 2, 3, 4]), 2), [1.5, 2.5, 3.5])
        np.testing.assert_allclose(compare.moving_average(np.array([1.0, 3]), 50), [2.0])  # window > length

    def test_spread_labels_keeps_order_and_gap(self):
        placed = compare.spread_labels([0.0, 0.1, 0.05, 5.0], min_gap=1.0)
        order = np.argsort(placed)
        np.testing.assert_array_equal(order, np.argsort([0.0, 0.1, 0.05, 5.0], kind="stable"))
        self.assertTrue(np.all(np.diff(np.sort(placed)) >= 1.0 - 1e-6))

    def test_colors_are_fixed_per_algorithm(self):
        self.assertEqual(compare.algo_color("reinforce"), compare.SERIES[0])
        self.assertEqual(compare.algo_color("sac"), compare.SERIES[4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
