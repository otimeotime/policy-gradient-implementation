from training.agent import Agent
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn

class ActorCriticAgent(Agent):
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
            critic_learning_rate: float = 1e-3
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
        """Critic v(s, w). Not named forward: Agent.distribution() uses forward() as the actor's mean."""
        # Move as uint8 (4x less to copy), then convert to float on the device
        x = torch.as_tensor(x, device=self.device).float()
        if x.dim() == 3:               # single obs (H, W, C) -> (1, H, W, C)
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W)
        return self.critic_head(self.critic_features(x / 255.0)).squeeze(-1)

    def update(self, transition: tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool], I: float):
        """One-step actor-critic update. I is γ^t: start it at 1 each episode and multiply by γ after every step."""
        observation, action, reward, next_observation, terminated = transition

        # δ = R + γ·v(S', w) - v(S, w)   (v(S') = 0 if S' is terminal; the target is held fixed, no gradient)
        value = self.value(observation)
        with torch.no_grad():
            next_value = 0.0 if terminated else self.value(next_observation)
            target = reward + self.discount_factor * next_value
        td_error = target - value

        # Critic: w ← w + α_w · δ · ∇v(S, w)   (= gradient descent on ½δ²)
        critic_loss = 0.5 * td_error.pow(2).sum()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor: θ ← θ + α_θ · I · δ · ∇ln π(A|S, θ)   (δ is a constant here)
        action = torch.as_tensor(action, device=self.device)
        log_prob = self.distribution(observation).log_prob(action).sum()  # joint over action dims
        actor_loss = -(I * td_error.detach().sum() * log_prob)
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        return critic_loss.item(), actor_loss.item()
