import os
import sys
import unittest
from unittest import mock

import torch
from torch.distributions import Normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from training.factory import make_learner  # noqa: E402
from training.sac import ReplayBuffer, SoftActorCriticAgent  # noqa: E402

HW = (20, 20)  # smallest-ish input the CNN accepts, keeps tests fast


def make_agent(**kwargs):
    torch.manual_seed(0)
    return SoftActorCriticAgent(env=None, device="cpu", input_hw=HW, **kwargs)


def frame(i):
    return torch.full((2,), i, dtype=torch.uint8)


def fill(buffer, episode_lengths, truncate_last=False):
    """Stores episodes whose frame t carries the id t, so s' can be checked. Returns the true transitions."""
    truth, step = [], 0
    for e, n in enumerate(episode_lengths):
        for t in range(n):
            obs, next_obs = frame(step), frame(step + 1)
            last = t == n - 1
            truncated = last and truncate_last and e == len(episode_lengths) - 1
            terminated = last and not truncated
            buffer.store(obs, torch.full((3,), float(step)), float(step), next_obs, terminated, truncated)
            truth.append((step, step + 1, terminated))
            step += 1
        step += 100  # the next episode starts from an unrelated frame
    return truth


class TestReplayBuffer(unittest.TestCase):
    def check_batch(self, buffer, batch_size=256):
        obs, acts, rewards, next_obs, dones = buffer.sample(batch_size)
        for o, a, r, n, d in zip(obs, acts, rewards, next_obs, dones):
            step = int(o[0])
            self.assertEqual(int(n[0]), step + 1)  # s' is the frame after s, also across episode ends
            self.assertEqual(a[0].item(), step)
            self.assertEqual(r.item(), step)
        return obs, dones

    def test_next_obs_across_episodes(self):
        buffer = ReplayBuffer(100, (2,), 3)
        truth = fill(buffer, [3, 4])
        obs, dones = self.check_batch(buffer)
        terminal_steps = {s for s, _, term in truth if term}
        for o, d in zip(obs, dones):
            self.assertEqual(bool(d), int(o[0]) in terminal_steps)

    def test_truncated_is_not_done_but_has_next_obs(self):
        buffer = ReplayBuffer(100, (2,), 3)
        fill(buffer, [3], truncate_last=True)
        obs, dones = self.check_batch(buffer)
        self.assertTrue(torch.all(dones == 0))
        self.assertIn(2, set(obs[:, 0].tolist()))  # the truncated last step is still sampleable

    def test_newest_unfinished_transition_is_excluded(self):
        buffer = ReplayBuffer(100, (2,), 3)
        for t in range(3):
            buffer.store(frame(t), torch.full((3,), float(t)), float(t), frame(t + 1), False, False)
        self.assertEqual(buffer.num_sampleable(), 2)
        obs, _ = self.check_batch(buffer)
        self.assertNotIn(2, set(obs[:, 0].tolist()))

    def test_wraparound(self):
        buffer = ReplayBuffer(5, (2,), 3)
        fill(buffer, [4, 6])  # 10 transitions into 5 slots
        self.assertEqual(len(buffer), 5)
        obs, _ = self.check_batch(buffer)
        self.assertTrue(set(obs[:, 0].tolist()) <= {105, 106, 107, 108, 109})
        self.assertEqual(set(buffer.final_next), {(9 % 5)})  # stale episode ends were dropped

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(5, (2,), 3).sample(1)


class TestPolicy(unittest.TestCase):
    def test_actions_in_bounds(self):
        agent = make_agent()
        obs = torch.randint(0, 256, (64, *HW, 3), dtype=torch.uint8)
        actions, _ = agent.policy(obs)
        low, high = torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0])
        self.assertTrue(torch.all(actions >= low) and torch.all(actions <= high))
        self.assertEqual(agent.get_action(obs[0]).shape, (3,))
        rand = torch.stack([agent.random_action() for _ in range(200)])
        self.assertTrue(torch.all(rand >= low) and torch.all(rand <= high))

    def test_log_prob_change_of_variables(self):
        # log pi(a) = log N(u) - log |da/du|, with a = center + scale * tanh(u)
        agent = make_agent()
        obs = torch.randint(0, 256, (8, *HW, 3), dtype=torch.uint8)
        torch.manual_seed(5)
        actions, log_probs = agent.policy(obs)

        x = obs.float().permute(0, 3, 1, 2) / 255.0
        hidden = agent.head[:-1](agent.features(x))
        mean, log_std = agent.head[-1](hidden), agent.log_std_layer(hidden).clamp(-20, 2)
        u = torch.atanh((actions - agent.action_center) / agent.action_scale)
        jac = agent.action_scale * (1 - torch.tanh(u) ** 2)
        expected = (Normal(mean, log_std.exp()).log_prob(u) - jac.log()).sum(-1)
        torch.testing.assert_close(log_probs, expected, rtol=1e-3, atol=1e-3)

    def test_deterministic_uses_mean(self):
        agent = make_agent()
        obs = torch.randint(0, 256, (*HW, 3), dtype=torch.uint8)
        self.assertTrue(torch.equal(agent.get_action(obs, deterministic=True),
                                    agent.get_action(obs, deterministic=True)))


class TestUpdates(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        torch.manual_seed(1)
        n = 16
        self.batch = (torch.randint(0, 256, (n, *HW, 3), dtype=torch.uint8),
                      torch.rand(n, 3), torch.randn(n),
                      torch.randint(0, 256, (n, *HW, 3), dtype=torch.uint8),
                      (torch.rand(n) < 0.3).float())

    def test_terminal_target_is_reward(self):
        obs, acts, rewards, next_obs, _ = self.batch
        y = self.agent.compute_q_targets(rewards, next_obs, torch.ones_like(rewards))
        torch.testing.assert_close(y, rewards)
        self.assertFalse(y.requires_grad)

    def test_q_loss_decreases(self):
        obs, acts, rewards, next_obs, dones = self.batch
        y = self.agent.compute_q_targets(rewards, next_obs, dones)
        first = self.agent.update_q_functions(obs, acts, y)
        for _ in range(30):
            last = self.agent.update_q_functions(obs, acts, y)
        self.assertLess(last["q1_loss"], first["q1_loss"])
        self.assertLess(last["q2_loss"], first["q2_loss"])

    def test_policy_step_does_not_change_q(self):
        q_before = [p.clone() for p in self.agent.q1.parameters()]
        pi_before = [p.clone() for p in self.agent.policy_parameters()]
        self.agent.update_policy(self.batch[0])
        for p, b in zip(self.agent.q1.parameters(), q_before):
            self.assertTrue(torch.equal(p, b))
            self.assertTrue(p.requires_grad)  # unfrozen again
        self.assertTrue(any(not torch.equal(p, b) for p, b in zip(self.agent.policy_parameters(), pi_before)))

    def test_polyak_update(self):
        agent = make_agent(polyak=0.9)
        with torch.no_grad():
            for p in agent.q1.parameters():
                p.add_(1.0)
        target_before = [p.clone() for p in agent.q1_target.parameters()]
        agent.update_targets()
        for p_t, p_t0, p in zip(agent.q1_target.parameters(), target_before, agent.q1.parameters()):
            torch.testing.assert_close(p_t, 0.9 * p_t0 + 0.1 * p)

    def test_targets_start_equal_and_are_frozen(self):
        for q, q_t in ((self.agent.q1, self.agent.q1_target), (self.agent.q2, self.agent.q2_target)):
            for p, p_t in zip(q.parameters(), q_t.parameters()):
                self.assertTrue(torch.equal(p, p_t))
                self.assertFalse(p_t.requires_grad)

    def test_full_update(self):
        stats = self.agent.update(self.batch)
        self.assertEqual(set(stats), {"q1_loss", "q2_loss", "policy_objective", "entropy"})


class TestLearner(unittest.TestCase):
    def test_schedule(self):
        learner = make_learner("sac", None, "cpu", input_hw=HW, start_steps=5,
                               update_after=4, update_every=3, batch_size=2)
        obs = torch.randint(0, 256, (*HW, 3), dtype=torch.uint8)
        with mock.patch.object(learner.agent, "random_action", wraps=learner.agent.random_action) as rand, \
                mock.patch.object(learner.agent, "update", return_value={"q1_loss": 1.0}) as update:
            results = []
            for t in range(10):
                action = learner.act(obs)
                results.append(learner.observe(obs, action, 0.0, obs, False, t == 9))
        self.assertEqual(rand.call_count, 5)       # first start_steps actions are random
        self.assertEqual(update.call_count, 6)     # at steps 6 and 9, 3 updates each
        self.assertEqual(results[-1], {"q1_loss": 1.0})
        self.assertTrue(all(r is None for r in results[:-1]))
        self.assertEqual(len(learner.buffer), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
