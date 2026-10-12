from training.on_policy import OnPolicyAgent
import gymnasium as gym
import torch


class PPOAgent(OnPolicyAgent):
    def __init__(
            self,
            env: gym.Env,
            in_channels=3,
            n_actions=3,
            input_hw=(96, 96),
            learning_rate: float = 3e-4,
            discount_factor: float = 0.95,
            exploration_epsilon: float = 0.05,
            device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
            critic_learning_rate: float = 1e-3,
            lam: float = 0.97,
            clip_ratio: float = 0.2,
            policy_epochs: int = 10,
            target_kl: float | None = 0.01,
            normalize_advantages: bool = True,
            critic_epochs: int = 5,
            critic_batch_size: int = 256,
            policy_batch_size: int = 256
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
            policy_batch_size=policy_batch_size)  # minibatch size of the policy's Adam steps

        self.clip_ratio = clip_ratio                    # epsilon
        self.policy_epochs = policy_epochs
        self.target_kl = target_kl                      # None disables early stopping
        self.normalize_advantages = normalize_advantages

    def clip_objective(self, log_probs, old_log_probs, advantages):
        """
        Per-sample PPO-Clip objective min(r A, g(eps, A)), with r = pi_theta / pi_theta_k and
        g(eps, A) = (1 + eps) A if A >= 0 else (1 - eps) A.

        log_probs: (num_steps,) log pi_theta(a_t|s_t), with gradient
        old_log_probs: (num_steps,) log pi_theta_k(a_t|s_t), constant
        advantages: (num_steps,)
        """
        ratio = (log_probs - old_log_probs).exp()
        g = torch.where(advantages >= 0, (1 + self.clip_ratio) * advantages, (1 - self.clip_ratio) * advantages)
        return torch.min(ratio * advantages, g)

    def update_policy(self, observations, actions, advantages, old_log_probs):
        """
        Line 6: theta <- argmax_theta mean_t min(r A, g(eps, A)), by policy_epochs passes of minibatch Adam.
        Stops early once the approximate KL(pi_k || pi_theta) exceeds 1.5 * target_kl.

        observations: (num_steps, H, W, C), valid timesteps only
        actions: (num_steps, action_dim)
        advantages: (num_steps,)
        old_log_probs: (num_steps,) log pi_theta_k(a_t|s_t)

        Returns:
            dict with the last minibatch's objective, the KL and clip fraction after the last epoch,
            and how many epochs ran
        """
        advantages = advantages.detach()
        old_log_probs = old_log_probs.detach()
        num_steps = observations.shape[0]
        if self.normalize_advantages and num_steps > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(1, self.policy_epochs + 1):
            for idx in torch.randperm(num_steps, device=self.device).split(self.policy_batch_size):
                log_probs = self.distribution(observations[idx]).log_prob(actions[idx]).sum(-1)
                objective = self.clip_objective(log_probs, old_log_probs[idx], advantages[idx]).mean()
                self.optimizer.zero_grad()
                (-objective).backward()  # gradient ascent on the objective
                self.optimizer.step()

            log_probs = self.policy_distribution(observations).log_prob(actions).sum(-1)
            kl = (old_log_probs - log_probs).mean().item()
            if self.target_kl is not None and kl > 1.5 * self.target_kl:
                break

        clipped = ((log_probs - old_log_probs).exp() - 1).abs() > self.clip_ratio
        return {"policy_objective": objective.item(), "kl": kl,
                "clip_fraction": clipped.float().mean().item(), "policy_epochs": epoch}

    def update(self, observations, actions, rewards, lengths=None, last_observations=None, terminated=None):
        """
        One PPO iteration (lines 4-7) on a padded batch of trajectories collected with the current policy.
        Arguments as in OnPolicyAgent.prepare_batch.

        Returns:
            dict of stats for logging
        """
        # Lines 4-5: rewards-to-go R_t and advantages from the current V_phi
        batch = self.prepare_batch(observations, actions, rewards, lengths, last_observations, terminated)
        mask, obs, acts = batch["mask"], batch["obs"], batch["acts"]
        old_log_probs = self.policy_distribution(obs).log_prob(acts).sum(-1)

        # Line 6: maximize the PPO-Clip objective
        stats = self.update_policy(obs, acts, batch["advantages"][mask], old_log_probs)

        # Line 7: fit V_phi to the rewards-to-go
        stats["value_loss"] = self.fit_value_function(obs, batch["rewards_to_go"][mask])
        return stats
