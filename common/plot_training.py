"""
Post-training / online plotting utility.

Reads train.csv and eval.csv from a run's log_dir and produces a multi-panel
figure showing:
  • total episode reward (train + eval with shaded std band)
  • each per-term reward contribution (r_speed, r_accel, ... r_terminal)
  • alpha (SAC entropy coefficient) — if present
  • other diagnostic metrics (avg_speed, avg_jerk, tet_rate, total_fuel_mL)

The figure is saved as training_curves.png in the same log_dir. Designed
to be called at the end of a training run from Logger.finish(), but can
also be invoked standalone:

    python common/plot_training.py <log_dir>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Column groups to plot (label -> list of columns)
_REWARD_TERMS = [
    ('r_speed_mean', 'Speed'),
    ('r_accel_mean', 'Accel'),
    ('r_jerk_mean', 'Jerk'),
    ('r_safety_mean', 'Safety'),
    ('r_lc_mean', 'Lane-change'),
    ('r_idle_mean', 'Idle'),
    ('r_energy_mean', 'Energy'),
    ('r_terminal_mean', 'Terminal'),
]


def _smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
    """Simple moving-average smoothing; returns same-length array."""
    if len(y) < 3 or window <= 1:
        return y
    window = min(window, len(y) // 2 * 2 + 1)   # keep odd, ≤ len(y)
    if window < 3:
        return y
    pad = window // 2
    y = np.asarray(y, dtype=float)
    y_padded = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(y_padded, kernel, mode='valid')


def _plot_with_std(ax, x, y, y_std=None, label=None, color=None, smooth_window=11):
    """Plot y vs x with optional shaded ±std band and smoothing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return
    x, y = x[mask], y[mask]

    ax.plot(x, y, alpha=0.25, color=color, linewidth=1.0)   # raw
    y_smooth = _smooth(y, smooth_window)
    ax.plot(x, y_smooth, label=label, color=color, linewidth=1.8)

    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float)[mask]
        if not np.isnan(y_std).all():
            ax.fill_between(x, y_smooth - y_std, y_smooth + y_std,
                            alpha=0.15, color=color, linewidth=0)


def _trim_and_rebase_steps(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    skip_initial_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Drop first N train episodes and rebase step so kept data starts at 0."""
    if skip_initial_episodes <= 0 or df_train.empty:
        return df_train, df_eval, 0.0
    if "episode" not in df_train.columns:
        print(
            "[plot_training_curves] 'episode' column missing; "
            "cannot skip initial episodes."
        )
        return df_train, df_eval, 0.0

    df_train = df_train.copy()
    df_eval = df_eval.copy()
    kept = df_train[df_train["episode"] >= skip_initial_episodes]
    if kept.empty:
        print(
            f"[plot_training_curves] no train rows left after skipping "
            f"{skip_initial_episodes} episodes."
        )
        return kept, df_eval.iloc[0:0].copy(), 0.0

    base_step = float(kept["step"].min()) if "step" in kept.columns else 0.0
    if "step" in kept.columns:
        kept.loc[:, "step"] = kept["step"] - base_step

    if not df_eval.empty and "step" in df_eval.columns:
        df_eval = df_eval[df_eval["step"] >= base_step].copy()
        if not df_eval.empty:
            df_eval.loc[:, "step"] = df_eval["step"] - base_step

    return kept, df_eval, base_step


def plot_training_curves(
    log_dir: str | Path,
    save_path: Optional[str | Path] = None,
    skip_initial_episodes: int = 0,
) -> Optional[Path]:
    """Read train/eval CSVs in ``log_dir`` and save a multi-panel figure.

    Returns the path to the saved PNG, or None if nothing was plotted.
    """
    log_dir = Path(log_dir)
    train_csv = log_dir / "train.csv"
    eval_csv = log_dir / "eval.csv"

    df_train = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    df_eval = pd.read_csv(eval_csv) if eval_csv.exists() else pd.DataFrame()
    df_train, df_eval, _ = _trim_and_rebase_steps(
        df_train=df_train,
        df_eval=df_eval,
        skip_initial_episodes=int(skip_initial_episodes),
    )
    if df_train.empty and df_eval.empty:
        print(f"[plot_training_curves] no data in {log_dir}")
        return None

    # ── Figure layout: 4 rows × 2 cols ───────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)

    # (0, 0) Total episode reward
    ax = axes[0, 0]
    if not df_train.empty and 'episode_reward' in df_train:
        _plot_with_std(ax, df_train['step'], df_train['episode_reward'],
                       label='Train', color='#1976D2')
    if not df_eval.empty and 'episode_reward' in df_eval:
        std = df_eval.get('episode_reward_std', None)
        _plot_with_std(ax, df_eval['step'], df_eval['episode_reward'],
                       y_std=std, label='Eval (±std)', color='#D32F2F')
    ax.set_title('Total episode reward', fontweight='bold')
    ax.set_ylim(bottom=-80, top=80)  # rewards are typically non-negative; focus on that range
    ax.set_xlabel('Step'); ax.set_ylabel('Reward'); ax.grid(alpha=0.2); ax.legend()

    # (0, 1) SAC alpha (if present)
    ax = axes[0, 1]
    if 'alpha' in df_train.columns:
        _plot_with_std(ax, df_train['step'], df_train['alpha'],
                       label='alpha', color='#F57C00')
        ax.set_yscale('log')
    ax.set_title('SAC alpha (log scale)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('alpha'); ax.grid(alpha=0.2, which='both')
    if 'alpha' in df_train.columns:
        ax.legend()

    # (1, 0) Per-term rewards — train
    ax = axes[1, 0]
    cmap = plt.cm.tab10
    for i, (col, label) in enumerate(_REWARD_TERMS):
        if col in df_train.columns:
            _plot_with_std(ax, df_train['step'], df_train[col],
                           label=label, color=cmap(i))
    ax.set_title('Per-term reward mean (train)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Mean reward / CAV-step')
    ax.grid(alpha=0.2); ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.legend(fontsize=8, ncol=2)

    # (1, 1) Per-term rewards — eval
    ax = axes[1, 1]
    for i, (col, label) in enumerate(_REWARD_TERMS):
        if col in df_eval.columns:
            std_col = f"{col}_std"
            std = df_eval.get(std_col, None)
            _plot_with_std(ax, df_eval['step'], df_eval[col],
                           y_std=std, label=label, color=cmap(i))
    ax.set_title('Per-term reward mean (eval)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Mean reward / CAV-step')
    ax.grid(alpha=0.2); ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.legend(fontsize=8, ncol=2)

    # (2, 0) avg_speed
    ax = axes[2, 0]
    if 'avg_speed' in df_train.columns:
        _plot_with_std(ax, df_train['step'], df_train['avg_speed'],
                       label='Train', color='#1976D2')
    if 'avg_speed' in df_eval.columns:
        std = df_eval.get('avg_speed_std', None)
        _plot_with_std(ax, df_eval['step'], df_eval['avg_speed'],
                       y_std=std, label='Eval', color='#D32F2F')
    ax.set_title('Average speed (m/s)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('m/s'); ax.grid(alpha=0.2); ax.legend()

    # (2, 1) avg_jerk
    ax = axes[2, 1]
    if 'avg_jerk' in df_train.columns:
        _plot_with_std(ax, df_train['step'], df_train['avg_jerk'],
                       label='Train', color='#1976D2')
    if 'avg_jerk' in df_eval.columns:
        std = df_eval.get('avg_jerk_std', None)
        _plot_with_std(ax, df_eval['step'], df_eval['avg_jerk'],
                       y_std=std, label='Eval', color='#D32F2F')
    ax.set_title('Average jerk (m/s³)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('m/s³'); ax.grid(alpha=0.2); ax.legend()

    # (3, 0) tet_rate
    ax = axes[3, 0]
    if 'tet_rate' in df_train.columns:
        _plot_with_std(ax, df_train['step'], df_train['tet_rate'],
                       label='Train', color='#1976D2')
    if 'tet_rate' in df_eval.columns:
        std = df_eval.get('tet_rate_std', None)
        _plot_with_std(ax, df_eval['step'], df_eval['tet_rate'],
                       y_std=std, label='Eval', color='#D32F2F')
    ax.set_title('TET rate (fraction of CAV-steps with TTC<2s)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('rate'); ax.grid(alpha=0.2); ax.legend()

    # (3, 1) total_fuel_mL
    ax = axes[3, 1]
    if 'total_fuel_mL' in df_train.columns:
        _plot_with_std(ax, df_train['step'], df_train['total_fuel_mL'],
                       label='Train', color='#1976D2')
    if 'total_fuel_mL' in df_eval.columns:
        std = df_eval.get('total_fuel_mL_std', None)
        _plot_with_std(ax, df_eval['step'], df_eval['total_fuel_mL'],
                       y_std=std, label='Eval', color='#D32F2F')
    ax.set_title('Total fuel per episode (mL)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('mL'); ax.grid(alpha=0.2); ax.legend()

    fig.suptitle(f'Training curves  |  {log_dir.name}', fontsize=14, fontweight='bold')

    save_path = Path(save_path) if save_path else log_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot_training_curves] saved to {save_path}")
    return save_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) not in (2, 3):
        print("Usage: python common/plot_training.py <log_dir> [skip_initial_episodes]")
        sys.exit(1)
    skip = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    plot_training_curves(sys.argv[1], skip_initial_episodes=skip)
