import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from training.factory import available_algorithms, load_agent, make_learner, save_checkpoint  # noqa: E402
from training.learners import LEARNERS, Learner, register  # noqa: E402

HW = (20, 20)  # smallest-ish input the CNN accepts, keeps tests fast


def fake_episode(learner, length, terminated=True):
    """Feed `length` random steps into learner.observe, returns the stats of the last call."""
    stats = None
    for t in range(length):
        obs = torch.randint(0, 256, (*HW, 3), dtype=torch.uint8)
        action = learner.act(obs)
        last = t == length - 1
        stats = learner.observe(obs, action, 1.0, obs, terminated and last, (not terminated) and last)
    return stats


class TestFactory(unittest.TestCase):
    def test_all_algorithms_registered(self):
        self.assertEqual(available_algorithms(), ["actor_critic", "ppo", "reinforce", "sac", "trpo"])

    def test_make_each_algorithm(self):
        for algo in available_algorithms():
            learner = make_learner(algo, None, "cpu", input_hw=HW)
            self.assertIsInstance(learner, LEARNERS[algo])
            self.assertEqual(learner.name, algo)

    def test_defaults_and_overrides(self):
        learner = make_learner("trpo", None, "cpu", input_hw=HW, max_kl=0.05)
        self.assertEqual(learner.agent.max_kl, 0.05)
        self.assertEqual(learner.agent.exploration_epsilon, 0.0)  # trpo default
        self.assertEqual(learner.batch_episodes, 4)

    def test_unknown_algorithm_and_hparam_raise(self):
        with self.assertRaises(ValueError):
            make_learner("dqn", None, "cpu")
        with self.assertRaises(ValueError):
            make_learner("reinforce", None, "cpu", batch_episodes=2)

    def test_duplicate_register_raises(self):
        with self.assertRaises(ValueError):
            register("trpo")(type("Dup", (Learner,), {}))

    def test_checkpoint_roundtrip(self):
        learner = make_learner("trpo", None, "cpu", input_hw=HW, max_kl=0.05)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trpo.pt")
            save_checkpoint(path, learner)
            agent = load_agent(path, None, "cpu")
            with self.assertRaises(ValueError):
                load_agent(path, None, "cpu", algo="reinforce")
        self.assertEqual(agent.max_kl, 0.05)
        self.assertFalse(agent.training)
        for k, v in learner.agent.state_dict().items():
            torch.testing.assert_close(agent.state_dict()[k], v)

    def test_bare_state_dict_needs_algo(self):
        learner = make_learner("reinforce", None, "cpu", input_hw=HW)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old")
            torch.save(learner.agent.state_dict(), path)
            with self.assertRaises(ValueError):
                load_agent(path, None, "cpu")


class TestLearners(unittest.TestCase):
    def test_reinforce_updates_once_per_episode(self):
        learner = make_learner("reinforce", None, "cpu", input_hw=HW, exploration_epsilon=0.0)
        with mock.patch.object(learner.agent, "update", return_value=1.5) as update:
            stats = fake_episode(learner, 5)
        update.assert_called_once()
        observations, actions, rewards, num_steps = update.call_args.args[0]
        self.assertEqual((len(observations), len(actions), num_steps), (5, 5, 5))
        self.assertEqual(stats, {"loss": 1.5})

    def test_actor_critic_updates_every_step(self):
        learner = make_learner("actor_critic", None, "cpu", input_hw=HW, exploration_epsilon=0.0)
        with mock.patch.object(learner.agent, "update", return_value=(1.0, 3.0)) as update:
            stats = fake_episode(learner, 4)
        self.assertEqual(update.call_count, 4)
        gamma = learner.agent.discount_factor
        self.assertEqual([c.args[1] for c in update.call_args_list], [1.0, gamma, gamma ** 2, gamma ** 3])
        self.assertEqual(stats, {"critic_loss": 1.0, "actor_loss": 3.0})
        self.assertEqual(learner.I, 1.0)  # reset for the next episode

    def test_trpo_pads_a_batch_of_episodes(self):
        learner = make_learner("trpo", None, "cpu", input_hw=HW, batch_episodes=2)
        with mock.patch.object(learner.agent, "update", return_value={"kl": 0.0}) as update:
            self.assertIsNone(fake_episode(learner, 3, terminated=True))
            update.assert_not_called()
            stats = fake_episode(learner, 5, terminated=False)
        update.assert_called_once()
        obs, acts, rewards, lengths, last_obs, terminated = update.call_args.args
        self.assertEqual(obs.shape, (2, 5, *HW, 3))
        self.assertEqual(acts.shape, (2, 5, 3))
        self.assertEqual(lengths, [3, 5])
        self.assertEqual(last_obs.shape, (2, *HW, 3))
        self.assertEqual(terminated, [True, False])
        self.assertTrue(torch.all(rewards[0, 3:] == 0))
        self.assertEqual(stats, {"kl": 0.0})
        self.assertEqual(learner.episodes, [])

    def test_trpo_real_update(self):
        learner = make_learner("trpo", None, "cpu", input_hw=HW, batch_episodes=2)
        fake_episode(learner, 3)
        stats = fake_episode(learner, 4, terminated=False)
        self.assertLessEqual(stats["kl"], learner.agent.max_kl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
