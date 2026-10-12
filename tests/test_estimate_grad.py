import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from training.trust_region import TrustRegionAgent  # noqa: E402

HW = (20, 20)  # smallest-ish input the CNN accepts, keeps tests fast


def make_agent():
    torch.manual_seed(0)
    return TrustRegionAgent(env=None, device="cpu", input_hw=HW)


def make_batch(num_ep=2, max_len=4):
    torch.manual_seed(1)
    obs = torch.randint(0, 256, (num_ep, max_len, *HW, 3), dtype=torch.uint8)
    acts = torch.randn(num_ep, max_len, 3)
    adv = torch.randn(num_ep, max_len)
    return obs, acts, adv


def reference_grad(agent, obs, acts, adv, lengths):
    # Literal formula: 1/|D| * sum_tau sum_t grad log pi(a_t|s_t) * A_t, one sample at a time.
    params = agent.policy_parameters()
    total = [torch.zeros_like(p) for p in params]
    for ep, n in enumerate(lengths):
        for t in range(n):
            lp = agent.distribution(obs[ep, t]).log_prob(acts[ep, t]).sum()
            for acc, g in zip(total, torch.autograd.grad(lp, params)):
                acc += g * adv[ep, t]
    return torch.cat([g.reshape(-1) for g in total]) / len(lengths)


class TestEstimateGrad(unittest.TestCase):
    def test_matches_per_sample_formula(self):
        agent = make_agent()
        obs, acts, adv = make_batch()
        g = agent.estimate_grad(obs, acts, adv)
        expected = reference_grad(agent, obs, acts, adv, [4, 4])
        torch.testing.assert_close(g, expected, rtol=1e-4, atol=1e-6)

    def test_chunking_does_not_change_result(self):
        agent = make_agent()
        obs, acts, adv = make_batch()
        g_full = agent.estimate_grad(obs, acts, adv, lengths=[3, 4])
        agent.policy_batch_size = 2  # 7 valid steps -> chunks of 2, 2, 2, 1
        g_chunked = agent.estimate_grad(obs, acts, adv, lengths=[3, 4])
        torch.testing.assert_close(g_full, g_chunked, rtol=1e-4, atol=1e-6)

    def test_shape_is_flat_policy_params(self):
        agent = make_agent()
        g = agent.estimate_grad(*make_batch())
        n = sum(p.numel() for p in agent.policy_parameters())
        self.assertEqual(g.shape, (n,))

    def test_padding_is_ignored(self):
        agent = make_agent()
        obs, acts, adv = make_batch()
        lengths = [2, 4]
        g = agent.estimate_grad(obs, acts, adv, lengths=lengths)
        # Garbage in padded slots must not change anything.
        obs2, acts2, adv2 = obs.clone(), acts.clone(), adv.clone()
        obs2[0, 2:] = 255
        acts2[0, 2:] = 1e3
        adv2[0, 2:] = 1e3
        g2 = agent.estimate_grad(obs2, acts2, adv2, lengths=lengths)
        torch.testing.assert_close(g, g2)
        torch.testing.assert_close(
            g, reference_grad(agent, obs, acts, adv, lengths), rtol=1e-4, atol=1e-6
        )

    def test_zero_advantage_gives_zero_grad(self):
        agent = make_agent()
        obs, acts, adv = make_batch()
        g = agent.estimate_grad(obs, acts, torch.zeros_like(adv))
        self.assertTrue(torch.all(g == 0))

    def test_does_not_touch_param_grad(self):
        agent = make_agent()
        agent.estimate_grad(*make_batch())
        self.assertTrue(all(p.grad is None for p in agent.parameters()))

    def test_step_along_grad_increases_surrogate(self):
        agent = make_agent()
        obs, acts, adv = make_batch()
        params = agent.policy_parameters()

        def surrogate():
            with torch.no_grad():
                lp = agent.distribution(obs.flatten(0, 1)).log_prob(acts.flatten(0, 1)).sum(-1)
                return (lp * adv.flatten()).sum() / obs.shape[0]

        g = agent.estimate_grad(obs, acts, adv)
        before = surrogate()
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(
                torch.nn.utils.parameters_to_vector(params) + 1e-4 * g, params
            )
        self.assertGreater(surrogate(), before)

    def test_wrong_lengths_raise(self):
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.estimate_grad(*make_batch(), lengths=[4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
