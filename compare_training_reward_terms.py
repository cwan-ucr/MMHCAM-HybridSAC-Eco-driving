"""
Compare reward-term training curves across algorithms.

By default this scans logs/sumo-intersection/1/*/train.csv and writes:
  - reward_term_curves.png
  - reward_terms_long.csv
  - reward_terms_tail_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    ("episode_reward", "Episode reward", "Reward"),
    ("r_speed_mean", "Speed reward", "Mean reward / CAV-step"),
    ("r_accel_mean", "Acceleration reward", "Mean reward / CAV-step"),
    ("r_jerk_mean", "Jerk reward", "Mean reward / CAV-step"),
    ("r_safety_mean", "Safety reward", "Mean reward / CAV-step"),
    ("r_lc_mean", "Lane-change reward", "Mean reward / CAV-step"),
    ("r_idle_mean", "Idle reward", "Mean reward / CAV-step"),
    ("r_energy_mean", "Energy reward", "Mean reward / CAV-step"),
    ("r_terminal_mean", "Terminal reward", "Mean reward / CAV-step"),
)


def _smooth(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=max(1, window), min_periods=1, center=True).mean()


def collect_results(root: Path, methods: list[str] | None) -> pd.DataFrame:
    frames = []
    method_filter = set(methods) if methods else None
    for train_csv in sorted(root.glob("*/train.csv")):
        method = train_csv.parent.name
        if method_filter is not None and method not in method_filter:
            continue
        df = pd.read_csv(train_csv)
        if "step" not in df.columns:
            print(f"Skipping {train_csv}: missing step column")
            continue
        available = ["step", *(metric for metric, _, _ in METRICS if metric in df.columns)]
        frame = df[available].copy()
        frame.insert(0, "method", method)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No usable train.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


def make_long_table(results: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    rows = []
    for method, method_df in results.groupby("method"):
        method_df = method_df.sort_values("step")
        for metric, _, _ in METRICS:
            if metric not in method_df.columns:
                continue
            values = pd.to_numeric(method_df[metric], errors="coerce")
            smoothed = _smooth(values, smooth_window)
            for step, value, smooth_value in zip(method_df["step"], values, smoothed):
                rows.append({
                    "method": method,
                    "step": float(step),
                    "metric": metric,
                    "value": float(value),
                    "smoothed_value": float(smooth_value),
                })
    return pd.DataFrame(rows)


def summarize_tail(long_table: pd.DataFrame, tail_episodes: int) -> pd.DataFrame:
    rows = []
    for (method, metric), group in long_table.groupby(["method", "metric"]):
        tail = group.sort_values("step").tail(tail_episodes)
        rows.append({
            "method": method,
            "metric": metric,
            "tail_episodes": len(tail),
            "step_min": float(tail["step"].min()),
            "step_max": float(tail["step"].max()),
            "mean": float(tail["value"].mean()),
            "std": float(tail["value"].std(ddof=0)),
            "last_smoothed_value": float(tail["smoothed_value"].iloc[-1]),
        })
    return pd.DataFrame(rows).sort_values(["metric", "method"])


def plot_curves(long_table: pd.DataFrame, out_path: Path, smooth_window: int) -> None:
    methods = sorted(long_table["method"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {method: cmap(i % 10) for i, method in enumerate(methods)}
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)

    for ax, (metric, title, ylabel) in zip(axes.flat, METRICS):
        metric_df = long_table[long_table["metric"] == metric]
        for method in methods:
            data = metric_df[metric_df["method"] == method].sort_values("step")
            if data.empty:
                continue
            x = data["step"].to_numpy(dtype=float)
            raw = data["value"].to_numpy(dtype=float)
            smooth = data["smoothed_value"].to_numpy(dtype=float)
            ax.plot(x, raw, color=colors[method], alpha=0.10, linewidth=0.7)
            ax.plot(x, smooth, color=colors[method], linewidth=1.8, label=method)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        if metric == "episode_reward":
            ax.set_ylim(bottom=-80, top=100)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle(
        f"Training Reward-Term Comparison | moving average window: {smooth_window} episodes",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_root(root: Path) -> Path:
    if list(root.glob("*/train.csv")):
        return root
    parent = root.parent
    candidates = [
        path for path in parent.iterdir()
        if path.is_dir() and path.name.isdigit() and list(path.glob("*/train.csv"))
    ] if parent.exists() else []
    if not candidates:
        return root
    return sorted(candidates, key=lambda path: int(path.name))[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("logs/sumo-intersection/1"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--smooth-window", type=int, default=21)
    parser.add_argument("--tail-episodes", type=int, default=50)
    args = parser.parse_args()

    root = resolve_root(args.root)
    if root != args.root:
        print(f"No train.csv under {args.root}; using {root}")
    out_dir = args.out_dir or root / "reward_term_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(root, args.methods)
    long_table = make_long_table(results, args.smooth_window)
    tail_summary = summarize_tail(long_table, args.tail_episodes)

    long_path = out_dir / "reward_terms_long.csv"
    summary_path = out_dir / "reward_terms_tail_summary.csv"
    plot_path = out_dir / "reward_term_curves.png"
    long_table.to_csv(long_path, index=False)
    tail_summary.to_csv(summary_path, index=False)
    plot_curves(long_table, plot_path, args.smooth_window)

    print(f"Compared {results['method'].nunique()} methods")
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
