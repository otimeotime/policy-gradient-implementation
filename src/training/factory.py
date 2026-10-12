"""
Factory for RL algorithms: build a learner or agent by name, and save/load checkpoints
that remember which algorithm and hyperparameters produced them.
"""
from pathlib import Path

import gymnasium as gym
import torch

from training.agent import Agent
from training.learners import LEARNERS, Learner


def available_algorithms() -> list[str]:
    return sorted(LEARNERS)


def make_learner(algo: str, env: gym.Env, device: str | torch.device, **hparams) -> Learner:
    """Create the learner (and its agent) registered as `algo`. hparams override the algorithm's defaults."""
    if algo not in LEARNERS:
        raise ValueError(f"Unknown algorithm {algo!r}. Available: {', '.join(available_algorithms())}")
    return LEARNERS[algo](env, device, **hparams)


def save_checkpoint(path: str | Path, learner: Learner):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algo": learner.name,
        "hparams": learner.hparams,
        "state_dict": learner.agent.state_dict(),
    }, path)


def load_agent(path: str | Path, env: gym.Env, device: str | torch.device, algo: str | None = None) -> Agent:
    """
    Rebuild the agent saved at `path`, in eval mode.
    Old checkpoints that are a bare state_dict carry no algorithm name, so they need `algo`.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if "state_dict" in checkpoint:
        if algo is not None and algo != checkpoint["algo"]:
            raise ValueError(f"Checkpoint was trained with {checkpoint['algo']!r}, not {algo!r}")
        algo, hparams, state_dict = checkpoint["algo"], checkpoint["hparams"], checkpoint["state_dict"]
    else:
        if algo is None:
            raise ValueError(f"{path} is a bare state_dict, pass the algorithm it was trained with")
        hparams, state_dict = {}, checkpoint

    agent = make_learner(algo, env, device, **hparams).agent
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent
