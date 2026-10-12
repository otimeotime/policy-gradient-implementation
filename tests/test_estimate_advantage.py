import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from training.trust_region import TrustRegionAgent  # noqa: E402


def make_agent(gamma=0.95, lam=0.97):
    return TrustRegionAgent(env=None, device="cpu", discount_factor=gamma, lam=lam)


class TestEstimateAdvantage(unittest.TestCase):
    def test_runs_with_default_constructor(self):
        # The agent should be usable as constructed, without patching attributes.
        agent = make_agent()
        r = torch.tensor([[1.0, 1.0]])
        v = torch.zeros_like(r)
        agent.estimate_advantage(r, v)

    def test_lambda_zero_is_td_residual(self):
        agent = make_agent(gamma=0.9, lam=0.0)
        r = torch.tensor([[1.0, 2.0, 3.0]])
        v = torch.tensor([[0.5, 1.0, 1.5]])
        adv = agent.estimate_advantage(r, v)
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), V(s_T) = 0
        expected = torch.tensor([[1.0 + 0.9 * 1.0 - 0.5,
                                  2.0 + 0.9 * 1.5 - 1.0,
                                  3.0 + 0.0 - 1.5]])
        torch.testing.assert_close(adv, expected)

    def test_lambda_one_is_return_minus_value(self):
        agent = make_agent(gamma=0.9, lam=1.0)
        r = torch.tensor([[1.0, 2.0, 3.0]])
        v = torch.tensor([[0.5, 1.0, 1.5]])
        adv = agent.estimate_advantage(r, v)
        g2 = 3.0
        g1 = 2.0 + 0.9 * g2
        g0 = 1.0 + 0.9 * g1
        expected = torch.tensor([[g0, g1, g2]]) - v
        torch.testing.assert_close(adv, expected)

    def test_hand_computed_gae(self):
        gamma, lam = 0.99, 0.95
        agent = make_agent(gamma=gamma, lam=lam)
        r = torch.tensor([[1.0, 0.0, -1.0]])
        v = torch.tensor([[0.2, 0.4, 0.1]])
        last = torch.tensor([0.3])
        adv = agent.estimate_advantage(r, v, last_values=last)
        d2 = -1.0 + gamma * 0.3 - 0.1
        d1 = 0.0 + gamma * 0.1 - 0.4
        d0 = 1.0 + gamma * 0.4 - 0.2
        a2 = d2
        a1 = d1 + gamma * lam * a2
        a0 = d0 + gamma * lam * a1
        torch.testing.assert_close(adv, torch.tensor([[a0, a1, a2]]))

    def test_padding_is_ignored_and_zeroed(self):
        agent = make_agent(gamma=0.9, lam=0.8)
        r = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0, 1.0]])
        v = torch.tensor([[0.5, 0.5, 99.0, 99.0],   # garbage in padded slots
                          [0.1, 0.2, 0.3, 0.4]])
        adv = agent.estimate_advantage(r, v, lengths=[2, 4])
        # Episode 0 alone, unpadded, must give the same first two values.
        adv0 = agent.estimate_advantage(r[:1, :2], v[:1, :2])
        torch.testing.assert_close(adv[0, :2], adv0[0])
        self.assertTrue(torch.all(adv[0, 2:] == 0))

    def test_zero_length_episode(self):
        agent = make_agent(gamma=0.9, lam=0.8)
        r = torch.ones(1, 3)
        adv = agent.estimate_advantage(r, torch.zeros_like(r), lengths=[0])
        self.assertTrue(torch.all(adv == 0))

    def test_no_gradient_leaks_into_advantages(self):
        agent = make_agent(gamma=0.9, lam=0.8)
        r = torch.ones(1, 3, requires_grad=True)
        v = torch.zeros(1, 3, requires_grad=True)
        last = torch.ones(1, requires_grad=True)   # e.g. critic(final_obs)
        adv = agent.estimate_advantage(r, v, last_values=last)
        self.assertFalse(adv.requires_grad)

    def test_bad_inputs_raise(self):
        agent = make_agent(gamma=0.9, lam=0.8)
        r = torch.ones(2, 3)
        with self.assertRaises(ValueError):
            agent.estimate_advantage(r, torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            agent.estimate_advantage(r, torch.zeros_like(r), lengths=[3])
        with self.assertRaises(ValueError):
            agent.estimate_advantage(r, torch.zeros_like(r), lengths=[4, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
