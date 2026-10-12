"""
End-to-end experiment: tune every algorithm, retrain the best configuration over several seeds,
evaluate it on held-out tracks, and write a figure and CSV tables.

    python src/pipeline.py --quick                      # a few minutes, checks that everything runs
    python src/pipeline.py                              # the real experiment, takes many hours
    python src/pipeline.py --algos ppo sac --trials 10
    python src/pipeline.py --stages report              # rebuild the figure and tables from saved results

Stages
    tune    For each algorithm, --trials configurations: trial 0 is the algorithm's defaults, the others are
            random samples from SEARCH_SPACES. Each one trains for --tune-steps env steps and is scored by its
            mean deterministic return on --val-episodes validation tracks.
    final   The best trial's configuration is retrained from scratch with --final-seeds seeds for
            --final-steps env steps, and every run is evaluated on --test-episodes test tracks, which are
            never used for selection. --final-seeds 0 evaluates the best tuning checkpoint instead.
    report  figure and CSV tables from whatever results exist so far.

All algorithms get the same env-step budgets and the same validation and test tracks.

Every finished run writes result.json, so an interrupted pipeline resumes where it stopped when it is started
again with the same arguments; an unfinished run is redone from scratch. Delete a run's folder to redo it.

Layout of --output-dir (default experiments/<--name>):
    settings.json                     budgets and seeds, checked on resume
    tuning/<algo>/trial_<k>/          config.json, result.json, curve.csv, checkpoint.pt, tensorboard/
    final/<algo>/trial_<k>_seed_<s>/  the same, for the retrained best configuration
    tuning_results.csv                one row per trial, the best one marked
    final_episodes.csv                one row per test episode
    final_results.csv                 one row per algorithm: best configuration and test statistics
    results.png                       tuning scores, final learning curves and test returns
"""
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))  # so `environment`, `training` and `evaluation` are importable

import argparse
import csv
import gc
import json
import math
import shutil
import time
import traceback
import warnings
import zlib

import matplotlib
matplotlib.use("Agg")  # files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environment.env import CarEnv
from evaluation.compare import (ALGO_ORDER, SOLVED_RETURN, STEPS_FORMATTER, SURFACE, TEXT, TEXT_SECONDARY,
                                algo_color, draw_returns, evaluate, moving_average, spread_labels, style_axes,
                                write_csv)
from training.factory import available_algorithms, load_agent, make_learner, save_checkpoint
from training.learners import LEARNERS
from training.loop import run_training, seed_everything

ROOT = SRC.parent
VAL_SEED = 10_000   # validation track i is reset with seed VAL_SEED + i
TEST_SEED = 0       # test track i is reset with seed TEST_SEED + i, the same tracks compare.py uses


# ---------------------------------------------------------------- search spaces

class LogUniform:
    def __init__(self, low, high):
        self.low, self.high = low, high

    def sample(self, rng):
        return float(f"{math.exp(rng.uniform(math.log(self.low), math.log(self.high))):.3g}")


class Choice:
    def __init__(self, *options):
        self.options = options

    def sample(self, rng):
        return self.options[int(rng.integers(len(self.options)))]


SEARCH_SPACES = {
    "reinforce": {
        "learning_rate": LogUniform(1e-4, 3e-2),
        "discount_factor": Choice(0.95, 0.99),
    },
    "actor_critic": {
        "learning_rate": LogUniform(1e-5, 1e-3),
        "critic_learning_rate": LogUniform(1e-4, 1e-2),
        "discount_factor": Choice(0.95, 0.99),
    },
    "trpo": {
        "max_kl": Choice(0.005, 0.01, 0.02, 0.05),
        "lam": Choice(0.9, 0.95, 0.97),
        "batch_episodes": Choice(2, 4, 8),
        "critic_learning_rate": LogUniform(1e-4, 3e-3),
        "discount_factor": Choice(0.95, 0.99),
    },
    "ppo": {
        "learning_rate": LogUniform(3e-5, 1e-3),
        "clip_ratio": Choice(0.1, 0.2, 0.3),
        "policy_epochs": Choice(4, 10),
        "lam": Choice(0.9, 0.95, 0.97),
        "batch_episodes": Choice(2, 4, 8),
        "discount_factor": Choice(0.95, 0.99),
    },
    "sac": {
        "learning_rate": LogUniform(1e-4, 1e-3),
        "q_learning_rate": LogUniform(1e-4, 1e-3),
        "alpha": Choice(0.02, 0.05, 0.1, 0.2),
        "polyak": Choice(0.99, 0.995),
        "discount_factor": Choice(0.95, 0.99),
    },
}


def sample_config(algo, trial, seed):
    """Trial 0 is the defaults. Other trials are reproducible samples: the same (algo, trial, seed), the same config."""
    if trial == 0:
        return {}
    rng = np.random.default_rng([seed, zlib.crc32(algo.encode()), trial])
    return {key: dist.sample(rng) for key, dist in SEARCH_SPACES.get(algo, {}).items()}


def fit_to_budget(algo, config, max_steps):
    """Scale the settings that only make sense for long runs, so short budgets still train."""
    hparams = dict(config)
    defaults = LEARNERS[algo].defaults
    budget_episodes = max(1, max_steps // 1000)  # CarRacing episodes are at most 1000 steps
    if "batch_episodes" in defaults:
        # at least a few updates in the budget
        batch = hparams.get("batch_episodes", defaults["batch_episodes"])
        hparams["batch_episodes"] = max(1, min(batch, budget_episodes // 4))
    if algo == "sac":
        hparams["start_steps"] = min(hparams.get("start_steps", defaults["start_steps"]), max_steps // 10)
        hparams["update_after"] = min(hparams.get("update_after", defaults["update_after"]), max_steps // 20)
        hparams["buffer_size"] = min(hparams.get("buffer_size", defaults["buffer_size"]), max_steps)
    return hparams


# ---------------------------------------------------------------- one training run

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=float))


def read_curve(run_dir):
    with open(run_dir / "curve.csv") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["env_steps"]) for r in rows]), np.array([float(r["return"]) for r in rows])


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(algo, config, run_dir, *, max_steps, train_seed, eval_seeds, device, label):
    """
    Train `algo` with `config` for max_steps env steps, then evaluate on eval_seeds.
    Returns the result dict; a finished run (result.json exists) is loaded instead of rerun.
    """
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"[done] {label}")
        return read_json(result_path)

    if run_dir.exists():
        shutil.rmtree(run_dir)  # leftovers of an interrupted attempt
    run_dir.mkdir(parents=True)
    hparams = fit_to_budget(algo, config, max_steps)
    print(f"[run]  {label}: {config or 'defaults'}")
    start = time.time()
    learner = env = None
    try:
        seed_everything(train_seed)
        env = CarEnv(device=device, domain_randomization=False)
        env.action_space.seed(train_seed)
        learner = make_learner(algo, env, device, **hparams)
        write_json(run_dir / "config.json", {"algo": algo, "config": config, "hparams": learner.hparams,
                                             "max_steps": max_steps, "train_seed": train_seed})

        writer = SummaryWriter(log_dir=run_dir / "tensorboard")
        history = run_training(learner, env, max_steps=max_steps, seed=train_seed, writer=writer)
        writer.close()
        save_checkpoint(run_dir / "checkpoint.pt", learner)
        write_csv(run_dir / "curve.csv", history)

        eval_env = CarEnv(device=device, domain_randomization=False)
        episodes = evaluate(learner.agent, eval_env, eval_seeds, verbose=False)
        eval_env.close()
    except Exception as error:  # one broken configuration must not stop a long experiment
        (run_dir / "error.txt").write_text(traceback.format_exc())
        warnings.warn(f"{label} failed: {error!r}, see {run_dir / 'error.txt'}")
        return {"status": "failed", "error": repr(error)}
    finally:
        if env is not None:
            env.close()
        del learner
        free_memory()

    returns = [e["return"] for e in episodes]
    result = {
        "status": "ok",
        "score": float(np.mean(returns)),
        "score_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "episodes": episodes,
        "train_steps": history[-1]["env_steps"],
        "train_episodes": len(history),
        "minutes": (time.time() - start) / 60,
    }
    write_json(result_path, result)
    print(f"       score {result['score']:.1f} ± {result['score_std']:.1f} in {result['minutes']:.1f} min")
    return result


# ---------------------------------------------------------------- stages

def trial_dir(out, algo, trial):
    return out / "tuning" / algo / f"trial_{trial:02d}"


def final_dir(out, algo, trial, seed):
    return out / "final" / algo / f"trial_{trial:02d}_seed_{seed}"


def tune(args, out, algos):
    val_seeds = [VAL_SEED + i for i in range(args.val_episodes)]
    # Trial-major order: after the first round every algorithm has its defaults scored.
    for trial in range(args.trials):
        for algo in algos:
            run_one(algo, sample_config(algo, trial, args.seed), trial_dir(out, algo, trial),
                    max_steps=args.tune_steps, train_seed=args.seed + trial, eval_seeds=val_seeds,
                    device=args.device, label=f"tune {algo} trial {trial}/{args.trials - 1}")


def tuning_results(out, algo):
    """Every trial on disk for algo, finished or failed, sorted by trial number."""
    rows = []
    for run in sorted((out / "tuning" / algo).glob("trial_*")):
        if not (run / "config.json").exists():
            continue
        config = read_json(run / "config.json")
        result = read_json(run / "result.json") if (run / "result.json").exists() else None
        status = "ok" if result else ("failed" if (run / "error.txt").exists() else "unfinished")
        rows.append({"trial": int(run.name.split("_")[1]), "dir": run, "config": config["config"],
                     "hparams": config["hparams"], "status": status, "result": result})
    return rows


def best_trial(out, algo):
    finished = [r for r in tuning_results(out, algo) if r["status"] == "ok"]
    return max(finished, key=lambda r: (r["result"]["score"], -r["trial"])) if finished else None


def final(args, out, algos):
    test_seeds = [TEST_SEED + i for i in range(args.test_episodes)]
    for algo in algos:
        best = best_trial(out, algo)
        if best is None:
            warnings.warn(f"No finished tuning trial for {algo}, skipping its final runs")
            continue
        print(f"\n{algo}: best is trial {best['trial']} with validation score {best['result']['score']:.1f}")

        if args.final_seeds == 0:
            evaluate_best_checkpoint(args, out, algo, best, test_seeds)
            continue
        for seed in range(args.final_seeds):
            run_one(algo, best["config"], final_dir(out, algo, best["trial"], seed),
                    max_steps=args.final_steps, train_seed=args.seed + 1000 + seed, eval_seeds=test_seeds,
                    device=args.device, label=f"final {algo} seed {seed + 1}/{args.final_seeds}")


def evaluate_best_checkpoint(args, out, algo, best, test_seeds):
    """--final-seeds 0: test the best tuning run itself instead of retraining it."""
    run_dir = out / "final" / algo / f"trial_{best['trial']:02d}_tuning"
    if (run_dir / "result.json").exists():
        print(f"[done] test {algo} tuning checkpoint")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    env = CarEnv(device=args.device, domain_randomization=False)
    agent = load_agent(best["dir"] / "checkpoint.pt", env, args.device)
    episodes = evaluate(agent, env, test_seeds, verbose=False)
    env.close()
    del agent
    free_memory()

    shutil.copy(best["dir"] / "config.json", run_dir / "config.json")
    shutil.copy(best["dir"] / "curve.csv", run_dir / "curve.csv")
    returns = [e["return"] for e in episodes]
    write_json(run_dir / "result.json", {
        "status": "ok", "score": float(np.mean(returns)),
        "score_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "episodes": episodes, "train_steps": best["result"]["train_steps"],
        "train_episodes": best["result"]["train_episodes"], "minutes": best["result"]["minutes"]})


def final_runs(out, algo, best):
    """Finished final runs of the current best configuration."""
    runs = []
    for run in sorted((out / "final" / algo).glob(f"trial_{best['trial']:02d}_*")):
        if (run / "result.json").exists():
            runs.append((run, read_json(run / "result.json")))
    return runs


# ---------------------------------------------------------------- report

def report(args, out, algos):
    tuning_rows, final_episode_rows, summaries, tuning_points, curves = [], [], [], [], {}

    for algo in algos:
        trials = tuning_results(out, algo)
        best = best_trial(out, algo)
        for t in trials:
            result = t["result"] or {}
            is_best = best is not None and t["trial"] == best["trial"]
            tuning_rows.append({
                "algo": algo, "trial": t["trial"], "status": t["status"], "best": is_best,
                "val_return": result.get("score", ""), "val_std": result.get("score_std", ""),
                "train_steps": result.get("train_steps", ""), "train_episodes": result.get("train_episodes", ""),
                "minutes": round(result["minutes"], 1) if "minutes" in result else "",
                "config": json.dumps(t["config"] or "defaults"),
            })
            if t["status"] == "ok":
                tuning_points.append({"algo": algo, "trial": t["trial"], "return": result["score"], "best": is_best})
        if best is None:
            continue

        runs = final_runs(out, algo, best)
        if not runs:
            continue
        seed_means = []
        for run, result in runs:
            seed_means.append(result["score"])
            for e in result["episodes"]:
                final_episode_rows.append({"algo": algo, "run": run.name, "episode": e["episode"],
                                           "track_seed": e["seed"], "return": e["return"],
                                           "length": e["length"], "terminated": e["terminated"]})
        curves[algo] = [read_curve(run) for run, _ in runs]

        returns = np.array([r["return"] for r in final_episode_rows if r["algo"] == algo])
        n = len(returns)
        std = float(returns.std(ddof=1)) if n > 1 else 0.0
        summaries.append({
            "algo": algo,
            "best_trial": best["trial"],
            "val_return": best["result"]["score"],
            "seeds": len(runs),
            "test_episodes": n,
            "mean_return": float(returns.mean()),
            "ci95": 1.96 * std / math.sqrt(n),
            "std_return": std,
            "std_across_seeds": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            "median_return": float(np.median(returns)),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "solved_rate": float(np.mean(returns >= SOLVED_RETURN)),
            "positive_rate": float(np.mean(returns > 0)),
            "mean_length": float(np.mean([r["length"] for r in final_episode_rows if r["algo"] == algo])),
            "train_steps_per_seed": int(np.mean([r["train_steps"] for _, r in runs])),
            "train_minutes_total": round(sum(r["minutes"] for _, r in runs), 1),
            "config": json.dumps(best["config"] or "defaults"),
        })

    if tuning_rows:
        write_csv(out / "tuning_results.csv", tuning_rows)
    if final_episode_rows:
        write_csv(out / "final_episodes.csv", final_episode_rows)
    if summaries:
        rounded = [{k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()} for s in summaries]
        write_csv(out / "final_results.csv", rounded)
    if tuning_points or summaries:
        plot_results(out / "results.png", tuning_points, curves, final_episode_rows, summaries, args)

    print_summary(summaries)
    print(f"\nTables and figure in {out}")


def mean_curve(histories, smooth, points=200):
    """Smooth each seed's curve, then the mean and min-max across seeds on their common step range."""
    smoothed = []
    for steps, returns in histories:
        y = moving_average(returns, smooth)
        smoothed.append((steps[len(steps) - len(y):], y))  # each average sits at its last episode
    lo, hi = max(x[0] for x, _ in smoothed), min(x[-1] for x, _ in smoothed)
    grid = np.linspace(lo, hi, points) if hi > lo else np.array([hi])
    ys = np.stack([np.interp(grid, x, y) for x, y in smoothed])
    return grid, ys.mean(0), ys.min(0), ys.max(0)


def draw_tuning(ax, points, algos):
    """One row per algorithm, a dot per trial; the chosen trial is ringed and labeled."""
    rows = [a for a in algos if any(p["algo"] == a for p in points)]
    best = {p["algo"]: p for p in points if p["best"]}
    rows.sort(key=lambda a: best[a]["return"] if a in best else -math.inf)  # best at the top
    style_axes(ax)
    ax.grid(False, axis="y")
    for y, algo in enumerate(rows):
        values = [p["return"] for p in points if p["algo"] == algo]
        ax.scatter(values, [y] * len(values), s=36, color=algo_color(algo), alpha=0.75,
                   edgecolors=SURFACE, linewidths=1, zorder=3)
        if algo in best:
            b = best[algo]
            ax.scatter([b["return"]], [y], s=110, facecolors="none", edgecolors=TEXT, linewidths=1.5, zorder=4)
            ax.annotate(f"trial {b['trial']}: {b['return']:.0f}", (b["return"], y), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(rows)), rows, fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("Validation return (mean over validation tracks)", color=TEXT_SECONDARY)


def draw_final_curves(ax, curves, smooth):
    style_axes(ax)
    ends = []
    for algo, histories in curves.items():
        x, mean, low, high = mean_curve(histories, smooth)
        color = algo_color(algo)
        if len(histories) > 1:
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=1.5, label=algo, zorder=3)
        ax.scatter([x[-1]], [mean[-1]], s=36, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ends.append((algo, x[-1], mean[-1]))

    if ends:
        x_max = max(x for _, x, _ in ends)
        ax.set_xlim(right=x_max * 1.14)  # room for the end labels
        lo, hi = ax.get_ylim()
        for (algo, x, _), label_y in zip(ends, spread_labels([y for _, _, y in ends], 0.05 * (hi - lo))):
            ax.annotate(algo, (x, label_y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT_SECONDARY)
    ax.xaxis.set_major_formatter(STEPS_FORMATTER)
    ax.set_xlabel("Environment steps", color=TEXT_SECONDARY)
    ax.set_ylabel(f"Training return\n({smooth}-episode average)", color=TEXT_SECONDARY)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=TEXT_SECONDARY,
              fontsize=9, ncols=max(1, len(curves)), borderaxespad=0.3, handlelength=1.5)


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def plot_results(path, tuning_points, curves, final_episodes, summaries, args):
    n = max(1, len({p["algo"] for p in tuning_points} | {s["algo"] for s in summaries}))
    panels = [("tuning", 1.0 + 0.55 * n)] * bool(tuning_points)
    panels += [("curves", 4.0)] * bool(curves) + [("test", 1.0 + 0.55 * n)] * bool(summaries)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, sum(h for _, h in panels) + 1.2 * len(panels)),
                             dpi=150, facecolor=SURFACE, height_ratios=[h for _, h in panels], squeeze=False)
    seeds = plural(args.final_seeds, "seed") if args.final_seeds else "the tuning run"

    for ax, (kind, _) in zip(axes[:, 0], panels):
        if kind == "tuning":
            draw_tuning(ax, tuning_points, ALGO_ORDER + sorted(set(available_algorithms()) - set(ALGO_ORDER)))
            title = (f"Tuning: {plural(args.trials, 'configuration')} per algorithm, "
                     f"{args.tune_steps:,} env steps each, scored on {plural(args.val_episodes, 'validation track')}")
        elif kind == "curves":
            draw_final_curves(ax, curves, args.smooth)
            title = f"Best configuration retrained: mean over {seeds}" + (", band = min to max" if
                                                                          args.final_seeds > 1 else "")
        else:
            draw_returns(ax, final_episodes, summaries)
            ax.set_xlabel("Test return per episode (all seeds)", color=TEXT_SECONDARY)
            title = f"Test: {plural(args.test_episodes, 'held-out track')} per run, mean with 95% CI"
        ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=28 if kind == "curves" else 12)

    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def print_summary(summaries):
    if not summaries:
        print("\nNo final results yet.")
        return
    ranked = sorted(summaries, key=lambda s: s["mean_return"], reverse=True)
    header = (f"{'algorithm':<14}{'trial':>6}{'val':>8}{'test mean ± 95% CI':>22}{'seed std':>10}"
              f"{'median':>9}{'solved':>8}")
    print("\n" + header + "\n" + "-" * len(header))
    for s in ranked:
        ci = f"{s['mean_return']:.1f} ± {s['ci95']:.1f}"
        print(f"{s['algo']:<14}{s['best_trial']:>6}{s['val_return']:>8.1f}{ci:>22}{s['std_across_seeds']:>10.1f}"
              f"{s['median_return']:>9.1f}{s['solved_rate']:>8.0%}")


# ---------------------------------------------------------------- main

QUICK = {"trials": 2, "tune_steps": 1000, "val_episodes": 1, "final_seeds": 1, "final_steps": 2000,
         "test_episodes": 2, "smooth": 1, "name": "quick"}
SETTINGS_KEYS = ("tune_steps", "val_episodes", "final_steps", "test_episodes", "seed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", choices=available_algorithms(), default=None,
                        help="default: every registered algorithm")
    parser.add_argument("--stages", nargs="+", choices=["tune", "final", "report"], default=["tune", "final", "report"])
    parser.add_argument("--trials", type=int, default=6, help="configurations per algorithm, trial 0 = defaults")
    parser.add_argument("--tune-steps", type=int, default=100_000, help="env steps per tuning run")
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--final-seeds", type=int, default=3, help="0 = test the best tuning run, no retraining")
    parser.add_argument("--final-steps", type=int, default=500_000, help="env steps per final run")
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--smooth", type=int, default=20, help="moving-average window of the learning curves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default="main", help="experiment name, results go to experiments/<name>")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help=f"tiny budgets to check the pipeline: {QUICK}")
    args = parser.parse_args()
    if args.quick:
        for key, value in QUICK.items():
            if parser.get_default(key) == getattr(args, key):  # explicit flags still win
                setattr(args, key, value)
    return args


def check_settings(out, args):
    """Resuming with different budgets would mix incomparable runs, so refuse."""
    settings = {k: getattr(args, k) for k in SETTINGS_KEYS}
    path = out / "settings.json"
    if path.exists():
        saved = read_json(path)
        changed = {k: (saved.get(k), v) for k, v in settings.items() if saved.get(k) != v}
        if changed:
            sys.exit(f"{out} was run with different settings {changed} (saved, now). "
                     f"Use another --name, or delete the folder to start over.")
    else:
        write_json(path, settings)


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True  # input size is fixed, let cuDNN pick the fastest kernels
    registered = available_algorithms()
    algos = args.algos or [a for a in ALGO_ORDER if a in registered] + sorted(set(registered) - set(ALGO_ORDER))
    out = args.output_dir or ROOT / "experiments" / args.name
    out.mkdir(parents=True, exist_ok=True)
    check_settings(out, args)

    tune_total = len(algos) * args.trials * args.tune_steps
    final_total = len(algos) * args.final_seeds * args.final_steps
    print(f"Experiment {out}\nAlgorithms: {', '.join(algos)}\n"
          f"Budget: {tune_total:,} tuning + {final_total:,} final env steps")

    if "tune" in args.stages:
        tune(args, out, algos)
    if "final" in args.stages:
        final(args, out, algos)
    if "report" in args.stages:
        report(args, out, algos)


if __name__ == "__main__":
    main()
