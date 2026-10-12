import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

class Agent(nn.Module):
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
            ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_channels, *input_hw)).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions))

        self.log_std = nn.Parameter(torch.zeros(n_actions))  # state-independent std, learned
        self.to(device)                                      # before creating the optimizer, so it tracks the device params
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_epsilon = exploration_epsilon
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    @property
    def device(self):
        return next(self.parameters()).device

    def distribution(self, observation):
        mean = self.forward(observation)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def get_action(self, observation, deterministic: bool = False):
        """Returns an action tensor (action_dim,) on the agent's device, unclipped."""
        if not deterministic and np.random.random() < self.exploration_epsilon:
            return torch.as_tensor(self.env.action_space.sample(), device=self.device)
        with torch.no_grad():
            dist = self.distribution(observation)
            action = dist.mean if deterministic else dist.sample()
        return action.squeeze(0)

    def forward(self, x):
        # Move as uint8 (4x less to copy), then convert to float on the device
        x = torch.as_tensor(x, device=self.device).float()
        if x.dim() == 3:               # single obs (H, W, C) -> (1, H, W, C)
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W)
        return self.head(self.features(x / 255.0))
    