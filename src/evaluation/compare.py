"""
Evaluate the trained policy of every algorithm on the same tracks and write a comparison report.

    python src/evaluation/compare.py
    python src/evaluation/compare.py --episodes 50 --video
    python src/evaluation/compare.py --algos ppo sac --checkpoint ppo=checkpoints/ppo_best.pt

Checkpoints are found per algorithm, first match wins:
    1. --checkpoint ALGO=PATH
    2. the newest checkpoints/<algo>_*.pt (what train.py writes)
    3. a bare file named <algo> or <algo>.pt in the repo root (like the old `reinforce` file)
Algorithms without a checkpoint are skipped with a warning.

Learning curves come from the TensorBoard run with the checkpoint's name (runs/<checkpoint stem>),
or, when there is none, the newest runs/<algo>_* (marked "latest run" in the report). --run ALGO=DIR overrides.

Every algorithm plays the same episodes: episode i is reset with seed --seed + i, with deterministic
(mean) actions and no domain randomization.

Outputs in --output-dir (default eval_results/<time>/):
    episodes.csv          one row per evaluation episode
    summary.csv / .md     one row per algorithm
    returns.png           evaluation return of every episode, with the mean and its 95% CI
    learning_curves.png   smoothed training return against environment steps
"""
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))  # so `environment` and `training` are importable when running this file directly

import argparse
import csv
import math
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from environment.env import CarEnv
from training.factory import available_algorithms, load_agent

ROOT = SRC.parent
SOLVED_RETURN = 900  # CarRacing counts as solved at 900+ points

# Fixed color per algorithm, in the order the project introduces them, so a color always means the
# same algorithm whichever subset is plotted. Validated as a categorical palette (light surface).
ALGO_ORDER = ["reinforce", "actor_critic", "trpo", "ppo", "sac"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, TEXT, TEXT_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def algo_color(algo):
    order = ALGO_ORDER + sorted(set(available_algorithms()) - set(ALGO_ORDER))
    i = order.index(algo)
    return SERIES[i] if i < len(SERIES) else MUTED


def parse_mapping(item: str):
    """'algo=path' -> (algo, Path(path))"""
    algo, sep, path = item.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"Expected ALGO=PATH, got {item!r}")
    return algo, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", choices=available_algorithms(), default=None,
                        help="default: every registered algorithm")
    parser.add_argument("--checkpoint", type=parse_mapping, action="append", default=[], metavar="ALGO=PATH")
    parser.add_argument("--run", type=parse_mapping, action="append", default=[], metavar="ALGO=DIR",
                        help="TensorBoard run to take the learning curve from")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=50, help="moving-average window, in episodes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video", action="store_true", help="record the first episode of each algorithm")
    return parser.parse_args()


# ---------------------------------------------------------------- finding the trained policies

def find_checkpoint(algo, overrides):
    if algo in overrides:
        return overrides[algo]
    saved = sorted((ROOT / "checkpoints").glob(f"{algo}_*.pt"), key=lambda p: p.stat().st_mtime)
    if saved:
        return saved[-1]
    for legacy in (ROOT / algo, ROOT / f"{algo}.pt"):
        if legacy.is_file():
            return legacy
    return None


def find_run(algo, checkpoint, overrides):
    """Returns (run dir or None, how it was matched)."""
    if algo in overrides:
        return overrides[algo], "given"
    matching = ROOT / "runs" / checkpoint.stem
    if matching.is_dir():
        return matching, "matches checkpoint"
    runs = sorted((d for d in (ROOT / "runs").glob(f"{algo}_*") if d.is_dir()), key=lambda d: d.name)
    if runs:
        return runs[-1], "latest run"
    return None, "none"


def read_learning_curve(run_dir):
    """(env steps at the end of each episode, episode returns) from a TensorBoard run, or None."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})  # 0 = keep every point
    events.Reload()
    tags = events.Tags()["scalars"]
    if "episode/return" not in tags:
        return None
    returns = np.array([e.value for e in events.Scalars("episode/return")])
    if "episode/length" in tags:
        steps = np.cumsum([e.value for e in events.Scalars("episode/length")])[:len(returns)]
    else:
        steps = np.arange(1, len(returns) + 1) * 1000.0  # CarRacing's time limit
    return steps, returns[:len(steps)]


def moving_average(x, window):
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / window  # entry i averages x[i : i + window]


# ---------------------------------------------------------------- evaluation

def evaluate(agent, env, seeds, verbose=True):
    rows = []
    for i, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        episode_return, steps, done = 0.0, 0, False
        start = time.time()
        while not done:
            action = agent.get_action(obs, deterministic=True)  # mean action, no sampling noise
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            done = terminated or truncated
        rows.append({"episode": i, "seed": seed, "return": episode_return, "length": steps,
                     "terminated": bool(terminated), "seconds": round(time.time() - start, 2)})
        if verbose:
            print(f"  episode {i + 1}/{len(seeds)} (seed {seed}): {steps} steps, return {episode_return:.1f}")
    return rows


def summarize(algo, rows, checkpoint, run, run_match, curve, smooth):
    returns = np.array([r["return"] for r in rows])
    n = len(returns)
    std = returns.std(ddof=1) if n > 1 else 0.0
    summary = {
        "algo": algo,
        "episodes": n,
        "mean_return": returns.mean(),
        "std_return": std,
        "ci95": 1.96 * std / math.sqrt(n),
        "median_return": float(np.median(returns)),
        "min_return": returns.min(),
        "max_return": returns.max(),
        "solved_rate": float(np.mean(returns >= SOLVED_RETURN)),
        "positive_rate": float(np.mean(returns > 0)),
        "mean_length": float(np.mean([r["length"] for r in rows])),
        "checkpoint": str(checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint),
        "run": f"{run.name} ({run_match})" if run else "",
        "train_episodes": "",
        "train_env_steps": "",
        "final_train_return": "",
        "best_train_return": "",
    }
    if curve is not None:
        steps, train_returns = curve
        smoothed = moving_average(train_returns, smooth)
        summary.update(train_episodes=len(train_returns), train_env_steps=int(steps[-1]),
                       final_train_return=smoothed[-1], best_train_return=smoothed.max())
    return summary


# ---------------------------------------------------------------- report files

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=1):
    if value == "" or value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path, summaries, args, has_curves):
    ranked = sorted(summaries, key=lambda s: s["mean_return"], reverse=True)
    lines = [
        f"# CarRacing evaluation, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{args.episodes} episodes per algorithm on the same tracks (seeds {args.seed}–{args.seed + args.episodes - 1}), "
        f"deterministic actions. Solved = return ≥ {SOLVED_RETURN}. "
        f"Training returns are a {args.smooth}-episode moving average.",
        "",
        "| Rank | Algorithm | Mean return ± 95% CI | Std | Median | Min | Max | Solved | Return > 0 "
        "| Mean length | Train episodes | Train env steps | Final train return | Best train return |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, s in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {s['algo']} | {s['mean_return']:.1f} ± {s['ci95']:.1f} | {s['std_return']:.1f} "
            f"| {s['median_return']:.1f} | {s['min_return']:.1f} | {s['max_return']:.1f} "
            f"| {s['solved_rate']:.0%} | {s['positive_rate']:.0%} | {s['mean_length']:.0f} "
            f"| {fmt(s['train_episodes'])} | {fmt(s['train_env_steps'])} "
            f"| {fmt(s['final_train_return'])} | {fmt(s['best_train_return'])} |")
    lines += ["", "| Algorithm | Checkpoint | TensorBoard run |", "|---|---|---|"]
    lines += [f"| {s['algo']} | `{s['checkpoint']}` | {s['run'] or '–'} |" for s in ranked]
    lines += ["", "![Evaluation returns](returns.png)", ""]
    if has_curves:
        lines += ["![Learning curves](learning_curves.png)", ""]
    path.write_text("\n".join(lines))


def print_table(summaries):
    ranked = sorted(summaries, key=lambda s: s["mean_return"], reverse=True)
    header = f"{'algorithm':<14}{'mean ± 95% CI':>18}{'median':>9}{'min':>9}{'max':>9}{'solved':>8}{'> 0':>6}"
    print("\n" + header + "\n" + "-" * len(header))
    for s in ranked:
        ci = f"{s['mean_return']:.1f} ± {s['ci95']:.1f}"
        print(f"{s['algo']:<14}{ci:>18}{s['median_return']:>9.1f}{s['min_return']:>9.1f}"
              f"{s['max_return']:>9.1f}{s['solved_rate']:>8.0%}{s['positive_rate']:>6.0%}")


# ---------------------------------------------------------------- charts

def format_steps(value, _=None):
    """Env-step tick labels: 0, 1.25k, 200k, 1.5M."""
    if value == 0:
        return "0"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3g}M"
    return f"{value / 1e3:.3g}k"


STEPS_FORMATTER = matplotlib.ticker.FuncFormatter(format_steps)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, length=0)


def plot_returns(path, episodes, summaries):
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.8 * len(summaries)), dpi=150, facecolor=SURFACE)
    draw_returns(ax, episodes, summaries)
    ax.set_title("Evaluation returns on the same tracks, mean with 95% CI", loc="left",
                 color=TEXT, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def draw_returns(ax, episodes, summaries, value_key="return"):
    """One row per algorithm: a dot per evaluation episode, the mean as a tick with its 95% CI."""
    ranked = sorted(summaries, key=lambda s: s["mean_return"])  # best at the top
    style_axes(ax)
    ax.grid(False, axis="y")

    rng = np.random.default_rng(0)
    for y, s in enumerate(ranked):
        returns = np.array([r[value_key] for r in episodes if r["algo"] == s["algo"]])
        jitter = rng.uniform(-0.18, 0.18, len(returns))
        ax.scatter(returns, y + jitter, s=36, color=algo_color(s["algo"]), alpha=0.75,
                   edgecolors=SURFACE, linewidths=1, zorder=3)
        ax.plot([s["mean_return"] - s["ci95"], s["mean_return"] + s["ci95"]], [y + 0.32] * 2,
                color=TEXT, linewidth=1.5, solid_capstyle="round", zorder=4)
        ax.plot([s["mean_return"]] * 2, [y + 0.24, y + 0.40], color=TEXT, linewidth=2, zorder=4)
        ax.annotate(f"mean {s['mean_return']:.0f}", (s["mean_return"] + s["ci95"], y + 0.32),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT_SECONDARY)

    ax.axvline(SOLVED_RETURN, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"solved ({SOLVED_RETURN})", (SOLVED_RETURN, len(ranked) - 0.5), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(ranked)), [s["algo"] for s in ranked], fontsize=10)
    ax.set_ylim(-0.6, len(ranked) - 0.3)
    ax.set_xlabel("Evaluation return per episode", color=TEXT_SECONDARY)


def spread_labels(ys, min_gap):
    """Nudge end-of-line label positions apart so they don't overlap, keeping their order."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)
    for _ in range(50):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            gap = placed[b] - placed[a]
            if gap < min_gap:
                shift = (min_gap - gap) / 2
                placed[a] -= shift
                placed[b] += shift
                moved = True
        if not moved:
            break
    return placed


def plot_learning_curves(path, curves, smooth):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ends = []
    for algo, (steps, returns) in curves.items():
        smoothed = moving_average(returns, smooth)
        x = steps[len(steps) - len(smoothed):]  # each average is plotted at its last episode
        ax.plot(x, smoothed, color=algo_color(algo), linewidth=1.5, label=algo, zorder=3)
        ax.scatter([x[-1]], [smoothed[-1]], s=36, color=algo_color(algo), edgecolors=SURFACE,
                   linewidths=1.5, zorder=4)
        ends.append((algo, x[-1], smoothed[-1]))

    ax.axhline(SOLVED_RETURN, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"solved ({SOLVED_RETURN})", (0, SOLVED_RETURN), xytext=(4, 4), textcoords="offset points",
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED)

    # Direct labels at each line's end, in text ink: three of the series colors are too light for text.
    if ends:
        x_max = max(x for _, x, _ in ends)
        ax.set_xlim(right=x_max * 1.14)  # room for the labels of the longest runs
        lo, hi = ax.get_ylim()
        label_ys = spread_labels([y for _, _, y in ends], min_gap=0.045 * (hi - lo))
        for (algo, x, y), label_y in zip(ends, label_ys):
            ax.annotate(algo, (x, label_y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT_SECONDARY)

    ax.xaxis.set_major_formatter(STEPS_FORMATTER)
    ax.set_xlabel("Environment steps", color=TEXT_SECONDARY)
    ax.set_ylabel(f"Training return ({smooth}-episode average)", color=TEXT_SECONDARY)
    ax.set_title("Learning curves", loc="left", color=TEXT, fontsize=12, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=TEXT_SECONDARY,
              fontsize=9, ncols=len(curves), borderaxespad=0.3, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    device = torch.device(args.device)
    registered = available_algorithms()
    algos = args.algos or [a for a in ALGO_ORDER if a in registered] + sorted(set(registered) - set(ALGO_ORDER))
    checkpoint_overrides, run_overrides = dict(args.checkpoint), dict(args.run)
    out = args.output_dir or ROOT / "eval_results" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + i for i in range(args.episodes)]

    episodes, summaries, curves = [], [], {}
    for algo in algos:
        checkpoint = find_checkpoint(algo, checkpoint_overrides)
        if checkpoint is None or not checkpoint.exists():
            warnings.warn(f"No checkpoint for {algo}, skipping it")
            continue

        print(f"\n{algo}: {checkpoint}")
        env = CarEnv(device=device, domain_randomization=False, render_mode="rgb_array" if args.video else None)
        if args.video:
            env = RecordVideo(env, video_folder=str(out / "videos"), name_prefix=algo,
                              episode_trigger=lambda i: i == 0)
        agent = load_agent(checkpoint, env, device, algo=algo)
        rows = evaluate(agent, env, seeds)
        env.close()
        del agent
        if device.type == "cuda":
            torch.cuda.empty_cache()

        run, run_match = find_run(algo, checkpoint, run_overrides)
        curve = read_learning_curve(run) if run else None
        if curve is not None:
            curves[algo] = curve
        else:
            print(f"  no learning curve found for {algo}")

        episodes += [{"algo": algo, **r} for r in rows]
        summaries.append(summarize(algo, rows, checkpoint, run if curve else None, run_match, curve, args.smooth))

    if not summaries:
        sys.exit("No checkpoints found. Train with src/training/train.py or pass --checkpoint ALGO=PATH.")

    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "summary.csv", summaries)
    write_markdown(out / "summary.md", summaries, args, bool(curves))
    plot_returns(out / "returns.png", episodes, summaries)
    if curves:
        plot_learning_curves(out / "learning_curves.png", curves, args.smooth)

    print_table(summaries)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
