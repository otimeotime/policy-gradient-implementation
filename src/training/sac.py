import copy
import math

from training.agent import Agent
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    """
    Replay buffer D of (s, a, r, s', d) transitions, kept on the CPU.

    Frames are stored once: inside an episode, s' of one transition is s of the next one,
    so only the final s' of each episode is stored separately. This halves the memory
    (about 27 KB per CarRacing frame), but it requires store() to be called with consecutive
    transitions, the way the training loop produces them.
    """
    def __init__(self, capacity: int, obs_shape, action_dim: int, device: str | torch.device = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)  # where sampled batches go
        self.frames = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)   # d: s' is terminal
        self.ended = torch.zeros(capacity, dtype=torch.bool)   # episode ended here (terminal or truncated)
        self.final_next = {}                                   # slot -> s' of an episode's last transition
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def store(self, obs, action, reward, next_obs, terminated, truncated):
        """Line 7: store (s, a, r, s', d) in D, overwriting the oldest transition when full."""
        i = self.ptr
        self.final_next.pop(i, None)
        self.frames[i] = torch.as_tensor(obs).cpu()
        self.actions[i] = torch.as_tensor(action).cpu().float()
        self.rewards[i] = float(reward)
        self.dones[i] = bool(terminated)
        self.ended[i] = bool(terminated or truncated)
        if self.ended[i]:
            self.final_next[i] = torch.as_tensor(next_obs).cpu()

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def num_sampleable(self):
        # The newest transition's s' is the next frame, which is not stored yet unless the episode ended.
        newest = (self.ptr - 1) % self.capacity
        return self.size if self.size and self.ended[newest] else max(self.size - 1, 0)

    def sample(self, batch_size: int):
        """
        Line 11: a random batch B = {(s, a, r, s', d)} from D.

        Returns:
            obs (B, H, W, C) uint8, actions (B, action_dim), rewards (B,), next_obs (B, H, W, C) uint8,
            dones (B,) float, all on self.device
        """
        n = self.num_sampleable()
        if n == 0:
            raise ValueError("Replay buffer has no complete transitions yet")
        oldest = (self.ptr - self.size) % self.capacity
        idx = (oldest + torch.randint(0, n, (batch_size,))) % self.capacity

        next_obs = self.frames[(idx + 1) % self.capacity]  # fancy indexing copies
        for k in self.ended[idx].nonzero().flatten().tolist():
            next_obs[k] = self.final_next[int(idx[k])]

        return (self.frames[idx].to(self.device), self.actions[idx].to(self.device),
                self.rewards[idx].to(self.device), next_obs.to(self.device),
                self.dones[idx].float().to(self.device))


class QNetwork(nn.Module):
    """Q_phi(s, a): CNN features of s, concatenated with a, then an MLP."""
    def __init__(self, in_channels=3, n_actions=3, input_hw=(96, 96)):
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
            nn.Linear(n_flat + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, x, action):
        x = x.float().permute(0, 3, 1, 2)  # (N, H, W, C) uint8 -> (N, C, H, W)
        return self.head(torch.cat([self.features(x / 255.0), action], dim=-1)).squeeze(-1)


class SoftActorCriticAgent(Agent):
    """
    SAC with a tanh-squashed Gaussian policy whose mean and log std both depend on the state,
    twin Q-functions and Polyak-averaged target Q-functions. The entropy coefficient alpha is fixed.
    """
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(
            self,
            env: gym.Env,
            in_channels=3,
            n_actions=3,
            input_hw=(96, 96),
            learning_rate: float = 1e-3,
            discount_factor: float = 0.95,
            device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
            q_learning_rate: float = 1e-3,
            alpha: float = 0.2,
            polyak: float = 0.995,
            action_low: tuple = (-1.0, 0.0, 0.0),   # CarRacing: steer, gas, brake
            action_high: tuple = (1.0, 1.0, 1.0)
            ):
        super().__init__(
            env=env,
            in_channels=in_channels,
            n_actions=n_actions,
            input_hw=input_hw,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_epsilon=0.0,  # SAC explores through its entropy bonus
            device=device)

        self.alpha = alpha
        self.polyak = polyak  # rho

        # The base head gives the mean; the log std comes from the same hidden layer.
        # Agent's state-independent log_std is unused here.
        self.log_std_layer = nn.Linear(self.head[-1].in_features, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_parameters(), lr=learning_rate)

        low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.register_buffer("action_center", (high + low) / 2)
        self.register_buffer("action_scale", (high - low) / 2)

        # Lines 1-2: Q_phi1, Q_phi2 and their targets, initialized equal
        self.q1 = QNetwork(in_channels, n_actions, input_hw).to(self.device)
        self.q2 = QNetwork(in_channels, n_actions, input_hw).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=q_learning_rate)

    def policy_parameters(self):
        return [*self.features.parameters(), *self.head.parameters(), *self.log_std_layer.parameters()]

    def policy(self, observation, deterministic: bool = False):
        """
        a ~ pi_theta(.|s) by the reparametrization trick, so a is differentiable w.r.t. theta.

        Returns:
            action: (N, action_dim), inside [action_low, action_high]
            log_prob: (N,) log pi_theta(a|s), including the tanh and rescaling Jacobians
        """
        x = torch.as_tensor(observation, device=self.device).float()
        if x.dim() == 3:               # single obs (H, W, C) -> (1, H, W, C)
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W)
        hidden = self.head[:-1](self.features(x / 255.0))
        mean = self.head[-1](hidden)
        log_std = self.log_std_layer(hidden).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

        dist = Normal(mean, log_std.exp())
        u = mean if deterministic else dist.rsample()
        # log(1 - tanh(u)^2), written to stay finite for large |u|
        log_det_tanh = 2 * (math.log(2) - u - F.softplus(-2 * u))
        log_prob = (dist.log_prob(u) - log_det_tanh - self.action_scale.log()).sum(-1)
        action = self.action_center + self.action_scale * torch.tanh(u)
        return action, log_prob

    def get_action(self, observation, deterministic: bool = False):
        """Returns an action tensor (action_dim,) on the agent's device, already inside the action bounds."""
        with torch.no_grad():
            action, _ = self.policy(observation, deterministic)
        return action.squeeze(0)

    def random_action(self):
        """Uniform over the action box, for the initial exploration steps."""
        u = torch.rand_like(self.action_center)
        return self.action_center + self.action_scale * (2 * u - 1)

    def compute_q_targets(self, rewards, next_observations, dones):
        """
        Line 12: y = r + gamma (1 - d) (min_i Q_targ,i(s', a') - alpha log pi(a'|s')), a' ~ pi(.|s').
        """
        with torch.no_grad():
            next_actions, next_log_probs = self.policy(next_observations)
            q_next = torch.min(self.q1_target(next_observations, next_actions),
                               self.q2_target(next_observations, next_actions))
            return rewards + self.discount_factor * (1 - dones) * (q_next - self.alpha * next_log_probs)

    def update_q_functions(self, observations, actions, targets):
        """Line 13: one gradient step on mean (Q_phi_i(s, a) - y)^2, for i = 1, 2."""
        q1_loss = (self.q1(observations, actions) - targets).pow(2).mean()
        q2_loss = (self.q2(observations, actions) - targets).pow(2).mean()
        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()
        return {"q1_loss": q1_loss.item(), "q2_loss": q2_loss.item()}

    def update_policy(self, observations):
        """Line 14: one gradient ascent step on mean (min_i Q_phi_i(s, a~(s)) - alpha log pi(a~(s)|s))."""
        # The Q-functions are only differentiated through, not trained, in this step.
        for q in (self.q1, self.q2):
            q.requires_grad_(False)

        actions, log_probs = self.policy(observations)
        q = torch.min(self.q1(observations, actions), self.q2(observations, actions))
        objective = (q - self.alpha * log_probs).mean()
        self.optimizer.zero_grad()
        (-objective).backward()
        self.optimizer.step()

        for q in (self.q1, self.q2):
            q.requires_grad_(True)
        return {"policy_objective": objective.item(), "entropy": -log_probs.mean().item()}

    def update_targets(self):
        """Line 15: phi_targ,i <- rho phi_targ,i + (1 - rho) phi_i."""
        with torch.no_grad():
            for q, q_target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, p_target in zip(q.parameters(), q_target.parameters()):
                    p_target.lerp_(p, 1 - self.polyak)

    def update(self, batch):
        """
        One SAC gradient step (lines 12-15) on a batch sampled from the replay buffer.

        batch: (obs, actions, rewards, next_obs, dones) as returned by ReplayBuffer.sample

        Returns:
            dict of stats for logging
        """
        obs, actions, rewards, next_obs, dones = (torch.as_tensor(t, device=self.device) for t in batch)
        actions, rewards, dones = actions.float(), rewards.float(), dones.float()

        targets = self.compute_q_targets(rewards, next_obs, dones)
        stats = self.update_q_functions(obs, actions, targets)
        stats.update(self.update_policy(obs))
        self.update_targets()
        return stats
