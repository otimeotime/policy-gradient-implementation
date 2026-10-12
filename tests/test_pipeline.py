import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pipeline  # noqa: E402
from training.learners import LEARNERS  # noqa: E402


class TestSearchSpaces(unittest.TestCase):
    def test_every_key_is_a_valid_hyperparameter(self):
        for algo, space in pipeline.SEARCH_SPACES.items():
            self.assertTrue(set(space) <= set(LEARNERS[algo].valid_hparams()), algo)

    def test_every_algorithm_has_a_space(self):
        self.assertEqual(set(pipeline.SEARCH_SPACES), set(LEARNERS))

    def test_samples_are_reproducible_and_in_range(self):
        self.assertEqual(pipeline.sample_config("ppo", 0, 0), {})  # trial 0 = defaults
        a, b = pipeline.sample_config("ppo", 3, 0), pipeline.sample_config("ppo", 3, 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, pipeline.sample_config("ppo", 4, 0))
        self.assertNotEqual(a, pipeline.sample_config("trpo", 3, 0))
        for trial in range(1, 50):
            lr = pipeline.sample_config("ppo", trial, 0)["learning_rate"]
            self.assertTrue(3e-5 * 0.99 <= lr <= 1e-3 * 1.01)
            json.dumps(pipeline.sample_config("sac", trial, 0))  # plain Python values

    def test_fit_to_budget(self):
        self.assertEqual(pipeline.fit_to_budget("ppo", {"batch_episodes": 8}, 100_000)["batch_episodes"], 8)
        self.assertEqual(pipeline.fit_to_budget("ppo", {"batch_episodes": 8}, 2_000)["batch_episodes"], 1)
        sac = pipeline.fit_to_budget("sac", {}, 2_000)
        self.assertEqual((sac["start_steps"], sac["update_after"], sac["buffer_size"]), (200, 100, 2_000))
        sac = pipeline.fit_to_budget("sac", {}, 1_000_000)
        self.assertEqual((sac["start_steps"], sac["update_after"], sac["buffer_size"]), (10_000, 1_000, 50_000))
        self.assertEqual(pipeline.fit_to_budget("reinforce", {"learning_rate": 0.1}, 10), {"learning_rate": 0.1})


class TestCurves(unittest.TestCase):
    def test_mean_curve_on_common_range(self):
        h1 = (np.array([100.0, 200, 300, 400]), np.array([0.0, 1, 2, 3]))
        h2 = (np.array([150.0, 250, 350]), np.array([2.0, 3, 4]))
        x, mean, low, high = pipeline.mean_curve([h1, h2], smooth=1, points=3)
        np.testing.assert_allclose(x, [150, 250, 350])
        np.testing.assert_allclose(low, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(high, [2, 3, 4])
        np.testing.assert_allclose(mean, (low + high) / 2)


class TestResume(unittest.TestCase):
    def test_finished_run_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "trial_00"
            run.mkdir()
            pipeline.write_json(run / "result.json", {"status": "ok", "score": 5.0})
            with mock.patch.object(pipeline, "make_learner") as make:
                result = pipeline.run_one("ppo", {}, run, max_steps=10, train_seed=0, eval_seeds=[0],
                                          device="cpu", label="x")
            make.assert_not_called()
            self.assertEqual(result["score"], 5.0)

    def test_failed_run_is_recorded_and_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "trial_01"
            with mock.patch.object(pipeline, "make_learner", side_effect=RuntimeError("boom")), \
                    self.assertWarns(UserWarning):
                result = pipeline.run_one("ppo", {}, run, max_steps=10, train_seed=0, eval_seeds=[0],
                                          device="cpu", label="x")
            self.assertEqual(result["status"], "failed")
            self.assertTrue((run / "error.txt").exists())
            self.assertFalse((run / "result.json").exists())  # retried on the next start

    def test_best_trial(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for trial, score in [(0, 1.0), (1, 7.0), (2, 7.0), (3, None)]:
                run = pipeline.trial_dir(out, "ppo", trial)
                run.mkdir(parents=True)
                pipeline.write_json(run / "config.json", {"config": {"t": trial}, "hparams": {}})
                if score is not None:
                    pipeline.write_json(run / "result.json", {"status": "ok", "score": score})
            best = pipeline.best_trial(out, "ppo")
            self.assertEqual(best["trial"], 1)  # ties go to the earlier trial
            statuses = [r["status"] for r in pipeline.tuning_results(out, "ppo")]
            self.assertEqual(statuses, ["ok", "ok", "ok", "unfinished"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
