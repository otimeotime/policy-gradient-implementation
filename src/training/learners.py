"""
Learners connect an agent to the shared training loop.

Every algorithm consumes experience differently (REINFORCE per episode, actor-critic per step,
TRPO and PPO per batch of episodes, SAC from a replay buffer), so each one gets a Learner that
buffers transitions and calls its agent's update() at the right time. The training loop only
ever calls act() and observe().

To add an algorithm: write its Agent, then a Learner subclass decorated with @register("name").
"""
import inspect
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch

from training.agent import Agent
from training.actor_critic import ActorCriticAgent
from training.ppo import PPOAgent
from training.reinforce import ReinforceAgent
from training.sac import ReplayBuffer, SoftActorCriticAgent
from training.trust_region import TrustRegionAgent

LEARNERS: dict[str, type["Learner"]] = {}


def register(name: str):
    """Class decorator that makes a Learner available to the factory under `name`."""
    def decorator(cls):
        if name in LEARNERS:
            raise ValueError(f"Algorithm {name!r} is already registered")
        cls.name = name
        LEARNERS[name] = cls
        return cls
    return decorator


class Learner(ABC):
    name: str                   # set by @register
    agent_cls: type[Agent]
    defaults: dict = {}         # this algorithm's hyperparameters, overridable from the command line
    learner_keys: tuple = ()    # hyperparameters the learner uses itself, not passed to the agent

    def __init__(self, env: gym.Env, device: str | torch.device, **hparams):
        unknown = set(hparams) - set(self.valid_hparams())
        if unknown:
            raise ValueError(f"Unknown hyperparameters for {self.name}: {sorted(unknown)}. "
                             f"Valid: {self.valid_hparams()}")
        self.hparams = {**self.defaults, **hparams}
        agent_kwargs = {k: v for k, v in self.hparams.items() if k not in self.learner_keys}
        self.agent = self.agent_cls(env, device=device, **agent_kwargs)

    @classmethod
    def valid_hparams(cls) -> list[str]:
        agent_args = inspect.signature(cls.agent_cls.__init__).parameters
        return sorted({*agent_args, *cls.learner_keys} - {"self", "env", "device"})

    def act(self, obs):
        return self.agent.get_action(obs)

    @abstractmethod
    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> dict | None:
        """Called after every env step. Returns training stats when an update finished, else None."""


@register("reinforce")
class ReinforceLearner(Learner):
    """Monte Carlo policy gradient: one update at the end of every episode."""
    agent_cls = ReinforceAgent
    defaults = {"learning_rate": 0.01}

    def __init__(self, env, device, **hparams):
        super().__init__(env, device, **hparams)
        self._reset()

    def _reset(self):
        self.observations, self.actions, self.rewards = [], [], []

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        if not (terminated or truncated):
            return None

        loss = self.agent.update((self.observations, self.actions, self.rewards, len(self.rewards)))
        self._reset()
        return {"loss": loss}


@register("actor_critic")
class ActorCriticLearner(Learner):
    """One-step actor-critic: an update after every step, stats averaged over the episode."""
    agent_cls = ActorCriticAgent

    def __init__(self, env, device, **hparams):
        super().__init__(env, device, **hparams)
        self.I = 1.0  # gamma^t
        self.losses = []

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        losses = self.agent.update((obs, action, reward, next_obs, terminated), self.I)
        self.I *= self.agent.discount_factor
        self.losses.append(losses)
        if not (terminated or truncated):
            return None

        critic_loss, actor_loss = np.mean(self.losses, axis=0)
        self.I = 1.0
        self.losses = []
        return {"critic_loss": float(critic_loss), "actor_loss": float(actor_loss)}


class EpisodeBatchLearner(Learner):
    """
    On-policy batch methods: collect batch_episodes episodes with the current policy,
    then one agent.update() on all of them, padded to the same length.
    """
    # Random epsilon actions are not drawn from pi, which breaks the importance ratio pi_theta / pi_k.
    defaults = {"exploration_epsilon": 0.0, "batch_episodes": 4}
    learner_keys = ("batch_episodes",)

    def __init__(self, env, device, **hparams):
        super().__init__(env, device, **hparams)
        self.batch_episodes = self.hparams["batch_episodes"]
        self.episodes = []
        self._reset()

    def _reset(self):
        self.observations, self.actions, self.rewards = [], [], []

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        if not (terminated or truncated):
            return None

        # next_obs is the state after the last step, used to bootstrap episodes that were cut off.
        self.episodes.append((self.observations, self.actions, self.rewards, next_obs, terminated))
        self._reset()
        if len(self.episodes) < self.batch_episodes:
            return None

        stats = self.agent.update(*self._pad(self.episodes))
        self.episodes = []
        return stats

    def _pad(self, episodes):
        """Stack a list of episodes into the padded (num_episodes, max_len, ...) tensors update() expects."""
        device = self.agent.device
        lengths = [len(rewards) for _, _, rewards, _, _ in episodes]
        num_ep, max_len = len(episodes), max(lengths)
        first_obs = torch.as_tensor(episodes[0][0][0])
        first_act = torch.as_tensor(episodes[0][1][0])

        observations = torch.zeros((num_ep, max_len, *first_obs.shape), dtype=first_obs.dtype, device=device)
        actions = torch.zeros((num_ep, max_len, *first_act.shape), dtype=torch.float32, device=device)
        rewards = torch.zeros((num_ep, max_len), dtype=torch.float32, device=device)

        for i, (obs, acts, rews, _, _) in enumerate(episodes):
            n = lengths[i]
            observations[i, :n] = torch.stack([torch.as_tensor(o, device=device) for o in obs])
            actions[i, :n] = torch.stack([torch.as_tensor(a, device=device).float() for a in acts])
            rewards[i, :n] = torch.as_tensor(rews, dtype=torch.float32, device=device)

        last_observations = torch.stack([torch.as_tensor(ep[3], device=device) for ep in episodes])
        terminated = [bool(ep[4]) for ep in episodes]
        return observations, actions, rewards, lengths, last_observations, terminated


@register("trpo")
class TrustRegionLearner(EpisodeBatchLearner):
    agent_cls = TrustRegionAgent


@register("ppo")
class PPOLearner(EpisodeBatchLearner):
    agent_cls = PPOAgent


@register("sac")
class SoftActorCriticLearner(Learner):
    """
    SAC: every step goes into a replay buffer. The first start_steps actions are uniformly random,
    and after update_after steps it runs update_every gradient steps every update_every env steps.
    Stats are averaged over the gradient steps of each episode.
    """
    agent_cls = SoftActorCriticAgent
    defaults = {"buffer_size": 50_000, "batch_size": 128, "start_steps": 10_000,
                "update_after": 1_000, "update_every": 50}
    learner_keys = tuple(defaults)

    def __init__(self, env, device, **hparams):
        super().__init__(env, device, **hparams)
        self.buffer = None  # created on the first step, once the observation shape is known
        self.total_steps = 0
        self.stats = []

    def act(self, obs):
        # Lines 4-5, with uniform random actions at the start for better exploration
        if self.total_steps < self.hparams["start_steps"]:
            return self.agent.random_action()
        return self.agent.get_action(obs)

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        if self.buffer is None:
            self.buffer = ReplayBuffer(self.hparams["buffer_size"], tuple(obs.shape),
                                       len(self.agent.action_center), device=self.agent.device)
        # Line 7: store (s, a, r, s', d), with d = terminated: a time-limit cut is not a terminal state
        self.buffer.store(obs, action, reward, next_obs, terminated, truncated)
        self.total_steps += 1

        # Lines 9-11: when it is time, sample batches and update
        every = self.hparams["update_every"]
        if self.total_steps >= self.hparams["update_after"] and self.total_steps % every == 0:
            for _ in range(every):
                self.stats.append(self.agent.update(self.buffer.sample(self.hparams["batch_size"])))

        if not (terminated or truncated) or not self.stats:
            return None
        stats = {k: float(np.mean([s[k] for s in self.stats])) for k in self.stats[0]}
        self.stats = []
        return stats
