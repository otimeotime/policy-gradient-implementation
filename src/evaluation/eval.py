"""
Evaluate a checkpoint saved by train.py, with deterministic (mean) actions, and record videos.

    python src/evaluation/eval.py --checkpoint checkpoints/trpo_20261013_120000.pt
    python src/evaluation/eval.py --checkpoint reinforce --algo reinforce   # old bare state_dict
"""
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))  # so `environment` and `training` are importable when running this file directly

import argparse

import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from environment.env import CarEnv
from training.factory import available_algorithms, load_agent

ROOT = SRC.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--algo", choices=available_algorithms(), default=None,
                        help="only needed for old checkpoints that are a bare state_dict")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "eval_logs")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    env = CarEnv(device=device, domain_randomization=False,
                 render_mode=None if args.no_video else "rgb_array")
    if not args.no_video:
        env = RecordVideo(
            env,
            video_folder=str(args.video_dir),
            name_prefix=args.checkpoint.stem,
            episode_trigger=lambda x: True
        )
    env = RecordEpisodeStatistics(env)

    agent = load_agent(args.checkpoint, env, device, algo=args.algo)

    print(f"Starting evaluation of {args.checkpoint} for {args.episodes} episodes...")
    if not args.no_video:
        print(f"Video will be saved to: {args.video_dir}")

    for episode_num in range(args.episodes):
        obs, info = env.reset(seed=None if args.seed is None else args.seed + episode_num)
        returns = 0.0
        step_count = 0

        episode_over = False
        while not episode_over:
            action = agent.get_action(obs, deterministic=True)  # mean action, no sampling noise

            obs, reward, terminated, truncated, info = env.step(action)
            returns += reward
            step_count += 1

            episode_over = terminated or truncated

        print(f"Episode {episode_num + 1}: {step_count} steps, returns = {returns:.2f}")

    env.close()

    print(f"\nEvaluation Summary:")
    print(f"Episode duration: {list(env.time_queue)}")
    print(f"Episode returns: {list(env.return_queue)}")
    print(f"Episode lengths: {list(env.length_queue)}")

    avg_reward = np.mean(env.return_queue)
    avg_length = np.mean(env.length_queue)
    std_reward = np.std(env.return_queue)

    print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f} steps")
    print(f"Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1f}")


if __name__ == "__main__":
    main()
