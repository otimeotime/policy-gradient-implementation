from training.agent import Agent
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal


class OnPolicyAgent(Agent):
    """
    Shared parts of the on-policy actor-critic methods (TRPO, PPO):
    the value function V_phi, rewards-to-go, GAE advantages, padding masks and the value regression.
    Subclasses implement the policy step and update().
    """
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
            lam: float = 0.97,
            critic_epochs: int = 5,
            critic_batch_size: int = 256,
            policy_batch_size: int = 512
            ):
        super().__init__(
            env=env, 
            in_channels=in_channels, 
            n_actions=n_actions, 
            input_hw=input_hw,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_epsilon=exploration_epsilon,
            device=device)

        self.lam = lam
        self.critic_epochs = critic_epochs
        self.critic_batch_size = critic_batch_size
        self.policy_batch_size = policy_batch_size  # chunk size for policy forward/backward passes

        # Value function V_phi, a separate network from the policy.
        self.critic_features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flat = self.critic_features(torch.zeros(1, in_channels, *input_hw)).shape[1]

        self.critic_head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # Agent.__init__ already moved the actor to device and built self.optimizer over the actor params only
        # (the critic didn't exist yet), so the critic gets its own device move and optimizer.
        self.critic_features.to(self.device)
        self.critic_head.to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_features.parameters()) + list(self.critic_head.parameters()),
            lr=critic_learning_rate)

    def value(self, x):
        """Critic V(s, phi). Not named forward: Agent.distribution() uses forward() as the actor's mean."""
        x = torch.as_tensor(x, device=self.device).float()
        if x.dim() == 3:               # single obs (H, W, C) -> (1, H, W, C)
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W)
        return self.critic_head(self.critic_features(x / 255.0)).squeeze(-1)


    # Compute reward to go
    # Parameters:
    #   trajectories: vector of vectors, each vector consists rewards accross one episode.
    # Return:
    #   reward_to_go: vector of vectors, each vector consists reward to go at each time step of its corresponding epsiode

    def compute_reward_to_go(self, trajectories: torch.tensor):
        reward_to_go = torch.empty_like(trajectories)

        for i in range(trajectories.shape[0]):
            running_return = torch.zeros((), dtype=trajectories.dtype, device=trajectories.device)

            for t in reversed(range(trajectories.shape[1])):
                running_return = trajectories[i, t] + self.discount_factor * running_return
                reward_to_go[i, t] = running_return

        return reward_to_go

    def estimate_advantage(self, trajectories, values, lengths=None, last_values=None): 
        """
        trajectories: (num_episodes, max_episode_length)
            Rewards r_t for each episode.
        values: same shape as trajectories
            Critic estimates V(s_t).
        lengths: optional number of valid timesteps in each episode.
            Use this when trajectories are padded.
        last_values: (num_episodes,)
            Bootstrap value V(s_T) after each trajectory.
            Use 0 for genuinely terminated episodes.

        Returns:
            advantages: same shape as trajectories.
            Padded positions are set to zero.
        """
        rewards = torch.as_tensor(trajectories).detach()
        values = values.detach().to(
            device=rewards.device, dtype=rewards.dtype
        )

        num_ep, max_len = rewards.shape

        if values.shape != rewards.shape:
            raise ValueError("values and trajectories must have the same shape")

        if lengths is None:
            lengths = [max_len] * num_ep
        else:
            lengths = [int(n) for n in lengths]

        if last_values is None:
            # Correct only when every trajectory ends terminally.
            last_values = torch.zeros(
                num_ep, device=rewards.device, dtype=rewards.dtype
            )
        else:
            last_values = torch.as_tensor(
                last_values, device=rewards.device, dtype=rewards.dtype
            ).detach().reshape(-1)

        if len(lengths) != num_ep or last_values.numel() != num_ep:
            raise ValueError("Need one length and bootstrap value per episode")

        advantages = torch.zeros_like(rewards)

        for ep in range(num_ep):
            length = lengths[ep]

            if not 0 <= length <= max_len:
                raise ValueError("Invalid episode length")

            # Initialize the recursion at the end of this trajectory.
            gae = torch.zeros(
                (), device=rewards.device, dtype=rewards.dtype
            )
            next_value = last_values[ep]

            for t in reversed(range(length)):
                delta = (
                    rewards[ep, t]
                    + self.discount_factor * next_value
                    - values[ep, t]
                )

                gae = delta + self.discount_factor * self.lam * gae
                advantages[ep, t] = gae

                # At the preceding timestep, V(s_t) is the next value.
                next_value = values[ep, t]

        return advantages

    def policy_parameters(self):
        return [*self.features.parameters(), *self.head.parameters(), self.log_std]

    def policy_distribution(self, observations):
        """pi(.|s) for many states without gradient tracking, computed in chunks of policy_batch_size."""
        with torch.no_grad():
            mean = torch.cat([self.forward(chunk) for chunk in observations.split(self.policy_batch_size)])
            return Normal(mean, self.log_std.exp().expand_as(mean))

    def valid_mask(self, lengths, num_ep, max_len):
        """(num_ep, max_len) bool mask, True at the real (non-padded) timesteps."""
        if lengths is None:
            lengths = [max_len] * num_ep
        lengths = torch.as_tensor(lengths, device=self.device).reshape(-1)
        if lengths.numel() != num_ep:
            raise ValueError("Need one length per episode")
        return torch.arange(max_len, device=self.device) < lengths.unsqueeze(1)

    def fit_value_function(self, observations, rewards_to_go):
        """
        Fit the value function: phi <- argmin_phi mean_t (V_phi(s_t) - R_t)^2,
        by critic_epochs passes of minibatch Adam over the valid timesteps.

        observations: (num_steps, H, W, C), valid timesteps only
        rewards_to_go: (num_steps,)

        Returns:
            mean squared error of the last minibatch
        """
        rewards_to_go = rewards_to_go.detach().float()
        num_steps = observations.shape[0]

        for _ in range(self.critic_epochs):
            for idx in torch.randperm(num_steps, device=self.device).split(self.critic_batch_size):
                loss = (self.value(observations[idx]) - rewards_to_go[idx]).pow(2).mean()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()

        return loss.item()

    def prepare_batch(self, observations, actions, rewards, lengths=None, last_observations=None, terminated=None):
        """
        Rewards-to-go and advantages for a padded batch of trajectories (the steps before the policy update).

        observations: (num_episodes, max_episode_length, H, W, C)
        actions: (num_episodes, max_episode_length, action_dim)
        rewards: (num_episodes, max_episode_length)
        lengths: optional number of valid timesteps in each episode.
        last_observations: optional (num_episodes, H, W, C), the state after each trajectory's last step.
            Episodes that did not terminate are bootstrapped with V(last_observation).
            If omitted, every episode is treated as terminated.
        terminated: optional (num_episodes,) bool. If omitted with last_observations given,
            every episode is bootstrapped.

        Returns:
            dict with
                mask: (num_episodes, max_episode_length) bool, True at valid timesteps
                lengths: (num_episodes,)
                obs, acts: valid timesteps only, (num_steps, H, W, C) and (num_steps, action_dim)
                rewards_to_go, advantages: padded, (num_episodes, max_episode_length)
        """
        rewards = torch.as_tensor(rewards, device=self.device).float()
        num_ep, max_len = rewards.shape
        mask = self.valid_mask(lengths, num_ep, max_len)
        lengths = mask.sum(1)
        rewards = rewards * mask  # padded rewards must not leak into rewards-to-go

        obs = torch.as_tensor(observations, device=self.device)[mask]
        acts = torch.as_tensor(actions, device=self.device).float()[mask]

        # Rewards-to-go R_t
        # The per-timestep recursions run on the CPU: thousands of tiny GPU kernels are far slower.
        rewards_to_go = self.compute_reward_to_go(rewards.cpu()).to(self.device)

        # Advantages from the current V_phi
        with torch.no_grad():
            values = torch.zeros_like(rewards)
            values[mask] = torch.cat([self.value(chunk) for chunk in obs.split(self.critic_batch_size)])

            last_values = torch.zeros(num_ep, device=self.device)
            if last_observations is not None:
                if terminated is None:
                    bootstrap = torch.ones(num_ep, dtype=torch.bool, device=self.device)
                else:
                    bootstrap = ~torch.as_tensor(terminated, dtype=torch.bool, device=self.device).reshape(-1)
                if bootstrap.any():
                    last_obs = torch.as_tensor(last_observations, device=self.device)[bootstrap]
                    last_values[bootstrap] = self.value(last_obs)

        advantages = self.estimate_advantage(
            rewards.cpu(), values.cpu(), lengths.cpu(), last_values.cpu()
        ).to(self.device)

        return {"mask": mask, "lengths": lengths, "obs": obs, "acts": acts,
                "rewards_to_go": rewards_to_go, "advantages": advantages}
