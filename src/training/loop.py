"""The training loop shared by every algorithm: it only talks to a Learner through act() and observe()."""
import random

import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(learner, env, *, episodes=None, max_steps=None, seed=None, writer=None,
                 checkpoint_fn=None, save_every=0, progress=True):
    """
    Train until `episodes` episodes or `max_steps` env steps are done, whichever comes first.
    An episode in progress always finishes, so the last one can go past max_steps.

    seed: seeds the first env.reset (and with it the track sequence)
    writer: optional TensorBoard SummaryWriter
    checkpoint_fn: optional callable, called every save_every episodes

    Returns:
        list of {"episode", "env_steps", "return", "length"}, one per episode,
        env_steps counted at the end of the episode
    """
    if episodes is None and max_steps is None:
        raise ValueError("Give episodes, max_steps or both")

    history = []
    total_steps = 0
    bar = tqdm(total=max_steps or episodes, unit="step" if max_steps else "ep", disable=not progress)

    while (episodes is None or len(history) < episodes) and (max_steps is None or total_steps < max_steps):
        episode = len(history)
        obs, info = env.reset(seed=seed if episode == 0 else None)
        done = False
        episode_return, length = 0.0, 0
        stats = None
        while not done:
            action = learner.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # obs is the state the action was taken in, not the next one
            stats = learner.observe(obs, action, reward, next_obs, terminated, truncated) or stats
            obs = next_obs
            episode_return += reward
            length += 1

        total_steps += length
        history.append({"episode": episode, "env_steps": total_steps, "return": episode_return, "length": length})

        if writer is not None:
            writer.add_scalar("episode/return", episode_return, episode)
            writer.add_scalar("episode/length", length, episode)
            for key, value in (stats or {}).items():
                if value is not None:
                    writer.add_scalar(f"train/{key}", value, episode)

        if checkpoint_fn is not None and save_every and (episode + 1) % save_every == 0:
            checkpoint_fn()

        bar.update(length if max_steps else 1)
        bar.set_postfix(ret=f"{episode_return:.0f}")

    bar.close()
    return history
