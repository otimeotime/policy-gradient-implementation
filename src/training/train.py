"""
Train any registered algorithm on CarRacing.

    python src/training/train.py --algo reinforce
    python src/training/train.py --algo trpo --episodes 2000 --set max_kl=0.02 --set batch_episodes=8
    python src/training/train.py --algo sac --max-steps 500000
"""
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))  # so `environment` and `training` are importable when running this file directly

import argparse
import ast
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from environment.env import CarEnv
from training.factory import available_algorithms, make_learner, save_checkpoint
from training.loop import run_training, seed_everything

ROOT = SRC.parent  # repo root, independent of the working directory


def parse_hparam(item: str):
    """'key=value' -> (key, value), with value parsed as a Python literal when possible."""
    key, sep, value = item.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {item!r}")
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass  # keep as a string
    return key, value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", required=True, choices=available_algorithms())
    parser.add_argument("--episodes", type=int, default=None, help="default 1000 when --max-steps is not given")
    parser.add_argument("--max-steps", type=int, default=None, help="env-step budget, the same for every algorithm")
    parser.add_argument("--set", dest="hparams", type=parse_hparam, action="append", default=[],
                        metavar="KEY=VALUE", help="override a hyperparameter, can be repeated")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--save-every", type=int, default=100, help="episodes between checkpoints, 0 = only at the end")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="where to save the model, default checkpoints/<run name>.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes is None and args.max_steps is None:
        args.episodes = 1000
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True  # input size is fixed, let cuDNN pick the fastest kernels

    seed_everything(args.seed)
    env = CarEnv(device=device, domain_randomization=args.domain_randomization)
    if args.seed is not None:
        env.action_space.seed(args.seed)

    learner = make_learner(args.algo, env, device, **dict(args.hparams))

    run_name = f"{args.algo}_{datetime.now():%Y%m%d_%H%M%S}"
    checkpoint = args.checkpoint or ROOT / "checkpoints" / f"{run_name}.pt"
    writer = SummaryWriter(log_dir=ROOT / "runs" / run_name)
    writer.add_text("hparams", str(learner.hparams))
    print(f"Training {args.algo} on {device} with {learner.hparams}")
    print(f"TensorBoard: runs/{run_name}, checkpoint: {checkpoint}")

    run_training(learner, env, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed, writer=writer,
                 checkpoint_fn=lambda: save_checkpoint(checkpoint, learner), save_every=args.save_every)

    writer.close()
    env.close()
    save_checkpoint(checkpoint, learner)
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
