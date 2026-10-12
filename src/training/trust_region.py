from training.on_policy import OnPolicyAgent
import gymnasium as gym
import torch
from torch.distributions import Normal, kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class TrustRegionAgent(OnPolicyAgent):
    def __init__(
            self, 
            env: gym.Env,
            in_channels=3, 
            n_actions=3, 
            input_hw=(96, 96),
            learning_rate: float = 1e-3,
            discount_factor: float = 0.95,
            exploration_epsilon: float = 0.05,
            device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
            critic_learning_rate: float = 1e-3,
            vine_factor: int = 1,
            lam: float = 0.97,
            max_kl: float = 0.01,
            backtrack_coeff: float = 0.8,
            backtrack_iters: int = 10,
            cg_iters: int = 10,
            cg_damping: float = 0.1,
            critic_epochs: int = 5,
            critic_batch_size: int = 256,
            policy_batch_size: int = 512,
            fisher_fraction: float = 0.1
            ):
        super().__init__(
            env=env, 
            in_channels=in_channels, 
            n_actions=n_actions, 
            input_hw=input_hw,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_epsilon=exploration_epsilon,
            device=device,
            critic_learning_rate=critic_learning_rate,
            lam=lam,
            critic_epochs=critic_epochs,
            critic_batch_size=critic_batch_size,
            policy_batch_size=policy_batch_size)

        self.vine_factor = vine_factor
        self.q_samples = list()
        self.max_kl = max_kl                    # delta
        self.backtrack_coeff = backtrack_coeff  # alpha
        self.backtrack_iters = backtrack_iters  # K
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping            # H + damping * I, keeps CG stable
        self.fisher_fraction = fisher_fraction  # share of states used for Hessian-vector products

    def estimate_q_a(self, history):
    # Estimate q-function and advantage function
        observations, actions, rewards, num_steps = history
        G = 0.0
        for t in reversed(range(num_steps)):
            G = rewards[t] + self.discount_factor * G
            self.q_samples.append((observations[t], actions[t], G))

    def estimate_grad(self, observations, actions, advantages, lengths=None):
        """
        Policy gradient g = 1/|D| * sum_tau sum_t grad log pi(a_t|s_t) * A_t.

        observations: (num_episodes, max_episode_length, H, W, C)
        actions: (num_episodes, max_episode_length, action_dim)
        advantages: (num_episodes, max_episode_length), e.g. from estimate_advantage
        lengths: optional number of valid timesteps in each episode.
            Use this when trajectories are padded.

        Returns:
            g: flat vector (num_policy_params,), in the order of policy_parameters()
        """
        advantages = torch.as_tensor(advantages, device=self.device).detach().float()
        num_ep, max_len = advantages.shape

        # Keep only the valid timesteps, so padding never reaches the network.
        mask = self.valid_mask(lengths, num_ep, max_len)
        obs = torch.as_tensor(observations, device=self.device)[mask]
        acts = torch.as_tensor(actions, device=self.device).float()[mask]

        adv = advantages[mask]

        # Accumulate over chunks so only policy_batch_size states are in the graph at once.
        params = self.policy_parameters()
        total = [torch.zeros_like(p) for p in params]
        for start in range(0, obs.shape[0], self.policy_batch_size):
            chunk = slice(start, start + self.policy_batch_size)
            log_probs = self.distribution(obs[chunk]).log_prob(acts[chunk]).sum(-1)  # joint over action dims
            surrogate = (log_probs * adv[chunk]).sum() / num_ep
            grads = torch.autograd.grad(surrogate, params)  # leaves .grad untouched
            for acc, g in zip(total, grads):
                acc += g
        return torch.cat([g.reshape(-1) for g in total])

    def fisher_vector_product(self, observations, vector):
        """
        H v, where H is the Hessian of the sample-average KL(pi_old || pi_theta) at theta = theta_old.
        Computed by differentiating (grad KL) . v, so H is never built.

        observations: (num_steps, H, W, C), valid timesteps only
        vector: flat (num_policy_params,)
        """
        params = self.policy_parameters()
        dist = self.distribution(observations)
        old_dist = Normal(dist.mean.detach(), dist.stddev.detach())
        kl = kl_divergence(old_dist, dist).sum(-1).mean()

        grads = torch.autograd.grad(kl, params, create_graph=True)
        grad_v = torch.cat([g.reshape(-1) for g in grads]) @ vector
        hv = torch.autograd.grad(grad_v, params)
        return torch.cat([h.reshape(-1) for h in hv]) + self.cg_damping * vector

    def conjugate_gradient(self, observations, g, tol=1e-10):
        """
        Line 8: x ~= H^-1 g by conjugate gradient, using only Hessian-vector products.

        observations: (num_steps, H, W, C), valid timesteps only
        g: flat policy gradient from estimate_grad

        Returns:
            x: flat (num_policy_params,)
        """
        g = g.detach()
        x = torch.zeros_like(g)
        r = g.clone()  # residual g - Hx, with x = 0
        p = g.clone()
        rr = r @ r

        for _ in range(self.cg_iters):
            if rr < tol:
                break
            Hp = self.fisher_vector_product(observations, p)
            step = rr / (p @ Hp)
            x += step * p
            r -= step * Hp
            rr_new = r @ r
            p = r + (rr_new / rr) * p
            rr = rr_new

        return x

    def line_search(self, observations, actions, advantages, num_episodes, x, fisher_observations=None):
        """
        Line 9: theta <- theta + alpha^j * sqrt(2 delta / x^T H x) * x, with the smallest j in 0..K
        that improves the surrogate and keeps the sample-average KL <= delta.
        If no j works, the policy is left at theta_k.

        observations: (num_steps, H, W, C), valid timesteps only
        actions: (num_steps, action_dim)
        advantages: (num_steps,)
        num_episodes: |D|, same normalization as estimate_grad
        x: search direction from conjugate_gradient
        fisher_observations: states for x^T H x, the same ones conjugate_gradient used.
            Defaults to observations.

        Returns:
            dict with the accepted backtracking step (None if rejected), its KL and surrogate improvement
        """
        params = self.policy_parameters()
        old_params = parameters_to_vector(params).detach()
        advantages = advantages.detach()

        old_dist = self.policy_distribution(observations)
        old_log_probs = old_dist.log_prob(actions).sum(-1)
        # The ratio pi_theta / pi_old is 1 at theta_k.
        old_surrogate = advantages.sum() / num_episodes

        if fisher_observations is None:
            fisher_observations = observations
        xHx = x @ self.fisher_vector_product(fisher_observations, x)
        full_step = torch.sqrt(2 * self.max_kl / (xHx + 1e-8)) * x

        for j in range(self.backtrack_iters + 1):
            vector_to_parameters(old_params + self.backtrack_coeff ** j * full_step, params)

            dist = self.policy_distribution(observations)
            ratio = (dist.log_prob(actions).sum(-1) - old_log_probs).exp()
            surrogate = (ratio * advantages).sum() / num_episodes
            kl = kl_divergence(old_dist, dist).sum(-1).mean()

            if surrogate > old_surrogate and kl <= self.max_kl:
                return {"backtrack_step": j, "kl": kl.item(),
                        "surrogate_improvement": (surrogate - old_surrogate).item()}

        vector_to_parameters(old_params, params)  # no step accepted: keep theta_k
        return {"backtrack_step": None, "kl": 0.0, "surrogate_improvement": 0.0}

    def update(self, observations, actions, rewards, lengths=None, last_observations=None, terminated=None):
        """
        One TRPO iteration (lines 5-10) on a padded batch of trajectories collected with the current policy.
        Arguments as in OnPolicyAgent.prepare_batch.

        Returns:
            dict of stats for logging
        """
        # Lines 5-6: rewards-to-go R_t and advantages from the current V_phi
        batch = self.prepare_batch(observations, actions, rewards, lengths, last_observations, terminated)
        mask, obs, acts = batch["mask"], batch["obs"], batch["acts"]
        advantages = batch["advantages"]

        # Line 7: policy gradient
        g = self.estimate_grad(observations, actions, advantages, batch["lengths"])

        # Line 8: x ~= H^-1 g, with H estimated on a random subset of states.
        # Every CG iteration is a double backward pass, so this is where most of the time goes.
        num_fisher = max(1, int(self.fisher_fraction * obs.shape[0]))
        fisher_obs = obs[torch.randperm(obs.shape[0], device=self.device)[:num_fisher]]
        x = self.conjugate_gradient(fisher_obs, g)

        # Line 9: backtracking line search on theta
        stats = self.line_search(obs, acts, advantages[mask], mask.shape[0], x, fisher_observations=fisher_obs)

        # Line 10: fit V_phi to the rewards-to-go
        stats["value_loss"] = self.fit_value_function(obs, batch["rewards_to_go"][mask])
        return stats
