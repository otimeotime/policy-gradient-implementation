from training.agent import Agent
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class ReinforceAgent(Agent):
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
            super().__init__(
                env=env, 
                in_channels=in_channels, 
                n_actions=n_actions, 
                input_hw=input_hw,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                exploration_epsilon=exploration_epsilon,
                device=device)

    def update(self, history: tuple[list[torch.Tensor], list[torch.Tensor], list[float], int]):
        observations, actions, rewards, num_steps = history
        # 1. Discounted returns G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...  (computed backwards)
        returns = np.zeros(num_steps, dtype=np.float32)
        G = 0.0
        for t in reversed(range(num_steps)):
            G = rewards[t] + self.discount_factor * G
            returns[t] = G
        returns = torch.as_tensor(returns, device=self.device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalizing cuts variance a lot

        # 2. log π(a_t | s_t) for every step, as one batch, WITH gradient tracking
        obs = torch.stack(observations)                        # (T, H, W, C) uint8, already on device
        acts = torch.stack(actions)                            # (T, action_dim)
        log_probs = self.distribution(obs).log_prob(acts).sum(-1)  # (T,), joint over action dims

        # 3. Surrogate loss: its gradient is -Σ G_t ∇log π(a_t|s_t)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()   # clear gradients from the previous update
        loss.backward()              # autograd computes ∇θ of the loss
        self.optimizer.step()        # θ ← θ - lr · ∇θ L   (= ascent on return)
        return loss.item()
