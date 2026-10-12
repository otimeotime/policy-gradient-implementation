import os
import sys
import unittest

import torch
from torch.nn.utils import parameters_to_vector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from training.trust_region import TrustRegionAgent  # noqa: E402

HW = (20, 20)  # smallest-ish input the CNN accepts, keeps tests fast


def make_agent(**kwargs):
    torch.manual_seed(0)
    return TrustRegionAgent(env=None, device="cpu", input_hw=HW, **kwargs)


def make_batch(num_ep=2, max_len=4):
    torch.manual_seed(1)
    obs = torch.randint(0, 256, (num_ep, max_len, *HW, 3), dtype=torch.uint8)
    acts = torch.randn(num_ep, max_len, 3)
    rewards = torch.randn(num_ep, max_len)
    return obs, acts, rewards


def log_std_only(agent, values):
    # Flat vector that is zero everywhere except the log_std entries (the last ones).
    v = torch.zeros(sum(p.numel() for p in agent.policy_parameters()))
    v[-agent.log_std.numel():] = values
    return v


class TestFisherVectorProduct(unittest.TestCase):
    def test_log_std_block_is_exact(self):
        # For a Gaussian, d^2 KL / d log_std^2 = 2 per action dim, with no cross terms at theta_old.
        agent = make_agent()
        obs = make_batch()[0].flatten(0, 1)
        v = log_std_only(agent, torch.tensor([1.0, -2.0, 0.5]))
        hv = agent.fisher_vector_product(obs, v)
        torch.testing.assert_close(hv, (2.0 + agent.cg_damping) * v, rtol=1e-4, atol=1e-5)

    def test_symmetric_and_positive(self):
        agent = make_agent()
        obs = make_batch()[0].flatten(0, 1)
        torch.manual_seed(2)
        n = sum(p.numel() for p in agent.policy_parameters())
        u, v = torch.randn(n), torch.randn(n)
        uHv = u @ agent.fisher_vector_product(obs, v)
        vHu = v @ agent.fisher_vector_product(obs, u)
        torch.testing.assert_close(uHv, vHu, rtol=1e-3, atol=1e-4)
        self.assertGreaterEqual((v @ agent.fisher_vector_product(obs, v)).item(),
                                agent.cg_damping * (v @ v).item() * (1 - 1e-4))


class TestConjugateGradient(unittest.TestCase):
    def test_exact_on_log_std_block(self):
        agent = make_agent()
        obs = make_batch()[0].flatten(0, 1)
        g = log_std_only(agent, torch.tensor([0.3, -1.0, 2.0]))
        x = agent.conjugate_gradient(obs, g)
        torch.testing.assert_close(x, g / (2.0 + agent.cg_damping), rtol=1e-4, atol=1e-6)

    def test_reduces_residual(self):
        agent = make_agent(cg_iters=10)
        obs, acts, rewards = make_batch()
        g = agent.estimate_grad(obs, acts, rewards)
        x = agent.conjugate_gradient(obs.flatten(0, 1), g)
        residual = g - agent.fisher_vector_product(obs.flatten(0, 1), x)
        self.assertLess(residual.norm().item(), 0.1 * g.norm().item())

    def test_zero_gradient_gives_zero_direction(self):
        agent = make_agent()
        obs = make_batch()[0].flatten(0, 1)
        n = sum(p.numel() for p in agent.policy_parameters())
        x = agent.conjugate_gradient(obs, torch.zeros(n))
        self.assertTrue(torch.all(x == 0))


class TestLineSearch(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        obs, acts, adv = make_batch()
        self.obs, self.acts, self.adv = obs.flatten(0, 1), acts.flatten(0, 1), adv.flatten()
        g = self.agent.estimate_grad(obs, acts, adv)
        self.x = self.agent.conjugate_gradient(self.obs, g)
        self.params = self.agent.policy_parameters()
        self.before = parameters_to_vector(self.params).detach().clone()

    def test_accepted_step_respects_kl_and_improves(self):
        stats = self.agent.line_search(self.obs, self.acts, self.adv, 2, self.x)
        self.assertIsNotNone(stats["backtrack_step"])
        self.assertLessEqual(stats["kl"], self.agent.max_kl)
        self.assertGreater(stats["surrogate_improvement"], 0)
        self.assertFalse(torch.equal(parameters_to_vector(self.params), self.before))

    def test_rejected_step_restores_params(self):
        # A step against the gradient can never improve the surrogate.
        stats = self.agent.line_search(self.obs, self.acts, self.adv, 2, -self.x)
        self.assertIsNone(stats["backtrack_step"])
        torch.testing.assert_close(parameters_to_vector(self.params), self.before)


class TestFitValueFunction(unittest.TestCase):
    def test_loss_decreases(self):
        agent = make_agent(critic_epochs=50, critic_batch_size=4)
        obs = make_batch()[0].flatten(0, 1)
        targets = torch.linspace(-1, 1, obs.shape[0])
        with torch.no_grad():
            before = (agent.value(obs) - targets).pow(2).mean()
        agent.fit_value_function(obs, targets)
        with torch.no_grad():
            after = (agent.value(obs) - targets).pow(2).mean()
        self.assertLess(after.item(), 0.5 * before.item())

    def test_does_not_touch_policy(self):
        agent = make_agent()
        obs = make_batch()[0].flatten(0, 1)
        before = parameters_to_vector(agent.policy_parameters()).detach().clone()
        agent.fit_value_function(obs, torch.ones(obs.shape[0]))
        torch.testing.assert_close(parameters_to_vector(agent.policy_parameters()), before)


class TestUpdate(unittest.TestCase):
    def test_end_to_end_with_padding_and_bootstrap(self):
        agent = make_agent()
        obs, acts, rewards = make_batch(num_ep=3, max_len=5)
        last_obs = torch.randint(0, 256, (3, *HW, 3), dtype=torch.uint8)
        policy_before = parameters_to_vector(agent.policy_parameters()).detach().clone()
        critic_before = parameters_to_vector(agent.critic_head.parameters()).detach().clone()

        stats = agent.update(obs, acts, rewards, lengths=[5, 3, 1],
                             last_observations=last_obs, terminated=[True, False, True])

        self.assertEqual(set(stats), {"backtrack_step", "kl", "surrogate_improvement", "value_loss"})
        self.assertLessEqual(stats["kl"], agent.max_kl)
        self.assertFalse(torch.equal(parameters_to_vector(agent.policy_parameters()), policy_before))
        self.assertFalse(torch.equal(parameters_to_vector(agent.critic_head.parameters()), critic_before))
        self.assertTrue(all(p.grad is None for p in agent.policy_parameters()))

    def test_padding_garbage_is_ignored(self):
        obs, acts, rewards = make_batch(num_ep=2, max_len=4)
        obs2, acts2, rewards2 = obs.clone(), acts.clone(), rewards.clone()
        obs2[0, 2:] = 255
        acts2[0, 2:] = 1e3
        rewards2[0, 2:] = 1e3

        results = []
        for o, a, r in [(obs, acts, rewards), (obs2, acts2, rewards2)]:
            agent = make_agent(critic_epochs=1)
            torch.manual_seed(3)  # same minibatch order in fit_value_function
            agent.update(o, a, r, lengths=[2, 4])
            results.append(parameters_to_vector(agent.parameters()).detach())
        torch.testing.assert_close(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
