import os
import sys
import unittest

import torch
from torch.nn.utils import parameters_to_vector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from training.ppo import PPOAgent  # noqa: E402

HW = (20, 20)  # smallest-ish input the CNN accepts, keeps tests fast


def make_agent(**kwargs):
    torch.manual_seed(0)
    return PPOAgent(env=None, device="cpu", input_hw=HW, **kwargs)


def make_batch(num_ep=2, max_len=4):
    torch.manual_seed(1)
    obs = torch.randint(0, 256, (num_ep, max_len, *HW, 3), dtype=torch.uint8)
    acts = torch.randn(num_ep, max_len, 3)
    rewards = torch.randn(num_ep, max_len)
    return obs, acts, rewards


class TestClipObjective(unittest.TestCase):
    def test_hand_computed(self):
        agent = make_agent(clip_ratio=0.2)
        ratio = torch.tensor([1.5, 0.5, 0.5, 1.5, 1.1])
        adv = torch.tensor([2.0, 2.0, -2.0, -2.0, 2.0])
        obj = agent.clip_objective(ratio.log(), torch.zeros(5), adv)
        # A > 0: capped at (1 + eps) A.  A < 0: capped at (1 - eps) A.  Inside the range: r A.
        expected = torch.tensor([1.2 * 2, 0.5 * 2, 0.8 * -2, 1.5 * -2, 1.1 * 2])
        torch.testing.assert_close(obj, expected)

    def test_no_gradient_once_clipped(self):
        agent = make_agent(clip_ratio=0.2)
        log_probs = torch.tensor([1.5, 0.5, 1.1]).log().requires_grad_()
        adv = torch.tensor([2.0, -2.0, 2.0])
        agent.clip_objective(log_probs, torch.zeros(3), adv).sum().backward()
        self.assertEqual(log_probs.grad[0].item(), 0.0)  # r > 1 + eps with A > 0
        self.assertEqual(log_probs.grad[1].item(), 0.0)  # r < 1 - eps with A < 0
        self.assertNotEqual(log_probs.grad[2].item(), 0.0)


class TestUpdatePolicy(unittest.TestCase):
    def setUp(self):
        self.obs, self.acts, adv = (t.flatten(0, 1) for t in make_batch())
        self.adv = adv.flatten()

    def surrogate(self, agent, old_log_probs, adv):
        log_probs = agent.policy_distribution(self.obs).log_prob(self.acts).sum(-1)
        return ((log_probs - old_log_probs).exp() * adv).mean()

    def test_improves_objective_and_leaves_critic(self):
        agent = make_agent(target_kl=None, policy_epochs=5, normalize_advantages=False)
        old_log_probs = agent.policy_distribution(self.obs).log_prob(self.acts).sum(-1)
        critic_before = parameters_to_vector(agent.critic_head.parameters()).detach().clone()

        stats = agent.update_policy(self.obs, self.acts, self.adv, old_log_probs)

        self.assertGreater(self.surrogate(agent, old_log_probs, self.adv), self.adv.mean())
        self.assertEqual(stats["policy_epochs"], 5)
        torch.testing.assert_close(parameters_to_vector(agent.critic_head.parameters()), critic_before)

    def test_early_stop_on_kl(self):
        agent = make_agent(target_kl=1e-12, policy_epochs=10)
        old_log_probs = agent.policy_distribution(self.obs).log_prob(self.acts).sum(-1)
        stats = agent.update_policy(self.obs, self.acts, self.adv, old_log_probs)
        self.assertEqual(stats["policy_epochs"], 1)
        self.assertGreater(stats["kl"], 1.5e-12)


class TestUpdate(unittest.TestCase):
    def test_end_to_end_with_padding_and_bootstrap(self):
        agent = make_agent()
        obs, acts, rewards = make_batch(num_ep=3, max_len=5)
        last_obs = torch.randint(0, 256, (3, *HW, 3), dtype=torch.uint8)
        policy_before = parameters_to_vector(agent.policy_parameters()).detach().clone()
        critic_before = parameters_to_vector(agent.critic_head.parameters()).detach().clone()

        stats = agent.update(obs, acts, rewards, lengths=[5, 3, 1],
                             last_observations=last_obs, terminated=[True, False, True])

        self.assertEqual(set(stats), {"policy_objective", "kl", "clip_fraction", "policy_epochs", "value_loss"})
        self.assertFalse(torch.equal(parameters_to_vector(agent.policy_parameters()), policy_before))
        self.assertFalse(torch.equal(parameters_to_vector(agent.critic_head.parameters()), critic_before))

    def test_padding_garbage_is_ignored(self):
        obs, acts, rewards = make_batch(num_ep=2, max_len=4)
        obs2, acts2, rewards2 = obs.clone(), acts.clone(), rewards.clone()
        obs2[0, 2:] = 255
        acts2[0, 2:] = 1e3
        rewards2[0, 2:] = 1e3

        results = []
        for o, a, r in [(obs, acts, rewards), (obs2, acts2, rewards2)]:
            agent = make_agent(critic_epochs=1, policy_epochs=2)
            torch.manual_seed(3)  # same minibatch order
            agent.update(o, a, r, lengths=[2, 4])
            results.append(parameters_to_vector(agent.parameters()).detach())
        torch.testing.assert_close(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
