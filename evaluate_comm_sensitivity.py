"""
Evaluate a trained CAV-V2X policy under limited V2V communication.

This script keeps the trained policy fixed and changes only the evaluation-time
communication context:

    K in {0, 1, 2, 4, 6}
    communication modes:
        full_50m, full_100m, full_150m, full_200m, front_150m

K=0 is a parameter-matched masked-context baseline. It keeps the trained
attention actor/critic architecture and the original context tensor width, but
masks all V2V neighbor tokens before action selection.

The default checkpoint is the trained CAV-V2X model:

    logs/sumo-intersection/1/cav_control/models/final.pt

Outputs are written to:

    eval_results/communication_sensitivity/<checkpoint-tag>/

The main exported files are:
    - eval_comm_sensitivity_detail.csv
    - eval_comm_sensitivity_summary.csv
    - run_level_savings_long.csv
    - run_level_savings_summary.csv
    - mixed_traffic_comm_sensitivity_heatmap.{png,pdf}

Example:
    conda run -n tdmpc2 python evaluate_comm_sensitivity.py

Override examples:
    conda run -n tdmpc2 python evaluate_comm_sensitivity.py \
        checkpoint=logs/sumo-intersection/1/cav_control/models/final.pt \
        +comm_sensitivity_k_values=[0,1,2,4,6] \
        +comm_sensitivity_ranges=[50,100,150,200] \
        +comm_sensitivity_front_range=150 \
        +comm_sensitivity_context_slots=8
"""

from __future__ import annotations

import csv
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cav-comm-sensitivity")

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from termcolor import colored

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_LOCAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL_ROOT))

_TDMPC2_ROOT = _LOCAL_ROOT.parent / "tdmpc2"
if _TDMPC2_ROOT.is_dir():
    sys.path.append(str(_TDMPC2_ROOT))

from agents.dqn import DQN
from agents.ppo import PPO
from agents.sac import SAC
from agents.tdmpc2 import TDMPC2
from common.evaluation_metrics import (
    GROUP_DETAIL_FIELDS,
    GROUP_SUMMARY_FIELDS,
    append_group_metrics,
    fuel_l_per_100km,
    group_detail_from_info,
    new_group_accumulator,
    summarize_group_metrics,
)
from common.parser import parse_cfg
from common.plotting import plot_time_space_diagram
from common.seed import set_seed
from envs import make_env


PENETRATION_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.90]
DEFAULT_K_VALUES = [0, 1, 2, 4, 6]
DEFAULT_RANGES_M = [50.0, 100.0, 150.0, 200.0]
DEFAULT_FRONT_RANGE_M = 150.0
DEFAULT_CHECKPOINT = Path("logs/sumo-intersection/1/cav_control/models/final.pt")
DEFAULT_BASELINE_DETAIL = Path("eval_results/sumo_baseline/default/eval_penetration_detail.csv")

GROUPS = ("mix", "cav", "hdv")
GROUP_LABELS = {"mix": "Mixed", "cav": "CAV", "hdv": "HDV"}

METRICS = {
    "time": {
        "label": "Travel time",
        "detail_suffix": "avg_time_s",
        "higher_is_better": False,
    },
    "stop_time": {
        "label": "Stop time",
        "detail_suffix": "avg_stop_time_s",
        "higher_is_better": False,
    },
    "speed": {
        "label": "Speed",
        "detail_suffix": "avg_speed_mps",
        "higher_is_better": True,
    },
    "fuel": {
        "label": "Fuel",
        "detail_suffix": "fuel_L_per_100km",
        "higher_is_better": False,
    },
    "jerk": {
        "label": "Jerk",
        "detail_suffix": "avg_jerk",
        "higher_is_better": False,
    },
    "tet_rate": {
        "label": "TET rate",
        "detail_suffix": "tet_rate",
        "higher_is_better": False,
    },
}

CONDITION_FIELDS = [
    "condition",
    "condition_label",
    "communication_capacity_k",
    "context_size",
    "communication_range_m",
    "comm_topology",
]

DETAIL_FIELDS = [
    *CONDITION_FIELDS,
    "penetration_target",
    "run",
    "seed",
    "episode_reward",
    "reward_per_sec",
    "episode_length",
    "avg_speed",
    "avg_jerk",
    "total_fuel_mL",
    "total_distance_m",
    "tet_rate",
    "r_energy_mean",
    "trip50_avg_time_s",
    "trip50_avg_stop_time_s",
    "trip50_avg_speed_mps",
    "trip50_num_completed",
    *GROUP_DETAIL_FIELDS,
    "scenario_total_flow_vph",
    "scenario_cav_penetration",
    "scenario_cav_flow_vph",
    "scenario_hdv_flow_vph",
]

SUMMARY_FIELDS = [
    *CONDITION_FIELDS,
    "penetration_target",
    "runs",
    "reward_mean",
    "reward_std",
    "length_mean",
    "fuel_L_per_100km_mean",
    "fuel_L_per_100km_std",
    "average_speed_mean",
    "average_speed_std",
    "average_jerk_mean",
    "average_jerk_std",
    "tet_rate_mean",
    "tet_rate_std",
    "trip50_time_mean",
    "trip50_time_std",
    "trip50_speed_mean",
    "trip50_speed_std",
    *GROUP_SUMMARY_FIELDS,
]


def _as_list(value, default: list[float | int]) -> list:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return list(value)


def _project_root() -> Path:
    try:
        return Path(hydra.utils.get_original_cwd())
    except Exception:
        return _LOCAL_ROOT


def _resolve(root: Path, path_like) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else root / path


def _condition_slug(k: int, topology: str, range_m: float) -> str:
    prefix = "front" if topology == "front_only" else "full"
    return f"K{k:02d}_{prefix}_{int(round(range_m)):03d}m"


def _condition_label(topology: str, range_m: float) -> str:
    if topology == "masked":
        return "Masked K=0"
    prefix = "Front" if topology == "front_only" else "Full"
    return f"{prefix}-{int(round(range_m))}m"


def _build_conditions(raw_cfg) -> list[dict]:
    k_values = [int(v) for v in _as_list(raw_cfg.get("comm_sensitivity_k_values"), DEFAULT_K_VALUES)]
    ranges = [float(v) for v in _as_list(raw_cfg.get("comm_sensitivity_ranges"), DEFAULT_RANGES_M)]
    front_range = float(raw_cfg.get("comm_sensitivity_front_range", DEFAULT_FRONT_RANGE_M))
    context_slots = int(raw_cfg.get("comm_sensitivity_context_slots", raw_cfg.get("context_size", 8)))
    if context_slots <= 0:
        raise ValueError("comm_sensitivity_context_slots must be positive.")
    if any(k < 0 for k in k_values):
        raise ValueError("All communication capacity K values must be non-negative.")
    if any(k > context_slots for k in k_values):
        raise ValueError(
            f"Communication capacity K values {k_values} cannot exceed "
            f"context slots {context_slots}."
        )

    conditions = []
    for k in k_values:
        if k == 0:
            conditions.append({
                "condition": "K00_masked",
                "condition_label": _condition_label("masked", 0.0),
                "communication_capacity_k": k,
                "context_size": context_slots,
                "communication_range_m": 0.0,
                # The environment still needs a valid topology; all context is
                # masked later, so the topology/range have no behavioral effect.
                "comm_topology": "full",
            })
            continue
        for range_m in ranges:
            conditions.append({
                "condition": _condition_slug(k, "full", range_m),
                "condition_label": _condition_label("full", range_m),
                "communication_capacity_k": k,
                "context_size": context_slots,
                "communication_range_m": range_m,
                "comm_topology": "full",
            })
        conditions.append({
            "condition": _condition_slug(k, "front_only", front_range),
            "condition_label": _condition_label("front_only", front_range),
            "communication_capacity_k": k,
            "context_size": context_slots,
            "communication_range_m": front_range,
            "comm_topology": "front_only",
        })
    return conditions


def _make_agent(cfg):
    agent_type = str(getattr(cfg, "agent", "sac")).lower()
    cfg.agent = agent_type
    if agent_type == "tdmpc2" and bool(getattr(cfg, "lane_action_discrete", False)):
        raise NotImplementedError(
            "lane_action_discrete is for non-MPPI baselines. "
            "TD-MPC2/MPPI still uses continuous action sampling."
        )
    if agent_type == "tdmpc2":
        return TDMPC2(cfg)
    if agent_type == "sac":
        return SAC(cfg)
    if agent_type == "ppo":
        return PPO(cfg)
    if agent_type == "dqn":
        return DQN(cfg)
    raise ValueError(f"Unknown agent: {agent_type}. Expected tdmpc2, sac, ppo, or dqn.")


def _empty_condition_outputs(condition_dir: Path) -> bool:
    detail_path = condition_dir / "eval_penetration_detail.csv"
    summary_path = condition_dir / "eval_penetration_summary.csv"
    if not detail_path.exists() or not summary_path.exists():
        return False
    try:
        detail_cols = set(pd.read_csv(detail_path, nrows=0).columns)
        summary_cols = set(pd.read_csv(summary_path, nrows=0).columns)
    except Exception:
        return False
    required = {"communication_capacity_k", "context_size"}
    return required.issubset(detail_cols) and required.issubset(summary_cols)


def _safe_nanmean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_nanstd(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(finite)) if finite else float("nan")


def _limit_context_capacity(obs: dict, capacity_k: int) -> dict:
    """Keep the original context tensor width, but mask tokens after capacity_k."""
    if capacity_k < 0:
        raise ValueError("capacity_k must be non-negative.")
    ctx = obs["ctx_obs"]
    mask = obs["ctx_mask"]
    if ctx.ndim < 3 or mask.ndim < 2:
        raise ValueError("Expected ctx_obs with shape (N, K, D) and ctx_mask with shape (N, K).")
    slots = int(mask.shape[1])
    if capacity_k >= slots:
        return obs

    limited = dict(obs)
    limited_ctx = ctx.clone()
    limited_mask = mask.clone()
    limited_ctx[:, capacity_k:, :] = 0.0
    limited_mask[:, capacity_k:] = 0.0
    limited["ctx_obs"] = limited_ctx
    limited["ctx_mask"] = limited_mask
    return limited


def _run_condition(cfg, agent, condition: dict, out_dir: Path) -> tuple[list[dict], list[dict]]:
    condition_dir = out_dir / "conditions" / str(condition["condition"])
    condition_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(cfg, "comm_sensitivity_skip_existing", True)) and _empty_condition_outputs(condition_dir):
        print(colored(f"Skipping existing condition {condition['condition']}", "yellow"))
        detail_df = pd.read_csv(condition_dir / "eval_penetration_detail.csv")
        summary_df = pd.read_csv(condition_dir / "eval_penetration_summary.csv")
        return detail_df.to_dict("records"), summary_df.to_dict("records")

    capacity_k = int(condition["communication_capacity_k"])
    cfg.context_size = int(condition["context_size"])
    cfg.communication_range_m = float(condition["communication_range_m"])
    cfg.comm_topology = str(condition["comm_topology"])
    cfg.communication = True
    cfg.attention = True
    cfg.rl_control = True
    cfg.sumo_default_cav_behavior = False

    detail_rows = []
    summary_rows = []
    runs_per_penetration = int(getattr(cfg, "eval_penetration_runs", 10))
    save_ts = bool(getattr(cfg, "comm_sensitivity_save_ts", False))
    ts_dir = condition_dir / "eval_ts_diagrams"
    if save_ts:
        ts_dir.mkdir(parents=True, exist_ok=True)

    print(colored(
        f"\nCondition {condition['condition']} | capacity K={capacity_k}, "
        f"context slots={cfg.context_size}, R={cfg.communication_range_m:g}m, "
        f"topology={cfg.comm_topology}",
        "cyan",
        attrs=["bold"],
    ))

    mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)

    for p_idx, pen in enumerate(PENETRATION_LEVELS):
        cfg.cav_penetration_min = pen
        cfg.cav_penetration_max = pen
        cfg.randomize_demand = True
        env = make_env(cfg)

        pen_rewards = []
        pen_lengths = []
        pen_fuels = []
        pen_speeds = []
        pen_jerks = []
        pen_tet_rates = []
        pen_trip50_times = []
        pen_trip50_speeds = []
        pen_group_metrics = new_group_accumulator()
        first_fcd_path = ts_dir / f"fcd_pen{int(pen * 100):02d}_run01.xml"
        first_png_path = ts_dir / f"ts_pen{int(pen * 100):02d}_run01.png"
        first_ts_pending = save_ts

        print(colored(f"  Penetration {int(pen * 100):>2d}%", "blue", attrs=["bold"]))
        for i in range(runs_per_penetration):
            eval_seed = int(cfg.seed) + 1000 + i
            env.unwrapped.set_fcd_output(str(first_fcd_path) if save_ts and i == 0 else None)
            obs, done, ep_reward, t = env.reset(seed=eval_seed), False, 0.0, 0
            info = {}

            if save_ts and i == 1 and first_ts_pending:
                plot_time_space_diagram(
                    fcd_path=str(first_fcd_path),
                    save_path=str(first_png_path),
                    edge_length=294.4,
                    episode=1,
                )
                first_ts_pending = False

            while not done:
                if callable(mark_step):
                    mark_step()
                policy_obs = _limit_context_capacity(obs, capacity_k)
                action = agent.act(policy_obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward)
                t += 1

            total_travel_time = float(info.get("total_travel_time", max(t, 1)))
            reward_per_sec = ep_reward / max(total_travel_time, 1e-6)

            row = {
                **condition,
                "penetration_target": pen,
                "run": i + 1,
                "seed": eval_seed,
                "episode_reward": ep_reward,
                "reward_per_sec": reward_per_sec,
                "episode_length": t,
                "avg_speed": float(info.get("avg_speed", float("nan"))),
                "avg_jerk": float(info.get("avg_jerk", float("nan"))),
                "total_fuel_mL": float(info.get("total_fuel_mL", float("nan"))),
                "total_distance_m": float(info.get("total_distance", float("nan"))),
                "tet_rate": float(info.get("tet_rate", float("nan"))),
                "r_energy_mean": float(info.get("r_energy_mean", float("nan"))),
                "trip50_avg_time_s": float(info.get("trip50_avg_time_s", float("nan"))),
                "trip50_avg_stop_time_s": float(info.get("trip50_avg_stop_time_s", float("nan"))),
                "trip50_avg_speed_mps": float(info.get("trip50_avg_speed_mps", float("nan"))),
                "trip50_num_completed": float(info.get("trip50_num_completed", float("nan"))),
                "scenario_total_flow_vph": float(info.get("scenario_total_flow_vph", float("nan"))),
                "scenario_cav_penetration": float(info.get("scenario_cav_penetration", float("nan"))),
                "scenario_cav_flow_vph": float(info.get("scenario_cav_flow_vph", float("nan"))),
                "scenario_hdv_flow_vph": float(info.get("scenario_hdv_flow_vph", float("nan"))),
            }
            row.update(group_detail_from_info(info))
            detail_rows.append(row)

            pen_rewards.append(ep_reward)
            pen_lengths.append(t)
            pen_speeds.append(row["avg_speed"])
            pen_jerks.append(row["avg_jerk"])
            pen_tet_rates.append(row["tet_rate"])
            pen_fuels.append(fuel_l_per_100km(row["total_fuel_mL"], row["total_distance_m"]))
            pen_trip50_times.append(row["trip50_avg_time_s"])
            pen_trip50_speeds.append(row["trip50_avg_speed_mps"])
            append_group_metrics(pen_group_metrics, row)

            print(colored(
                f"    Run {i + 1:>2d}  R: {ep_reward:>8.2f}  Len: {t:>4d}  "
                f"trip50_t: {row['trip50_avg_time_s']:>6.2f}s  "
                f"trip50_v: {row['trip50_avg_speed_mps']:>5.2f}m/s",
                "yellow",
            ))

        summary_row = {
            **condition,
            "penetration_target": pen,
            "runs": runs_per_penetration,
            "reward_mean": _safe_nanmean(pen_rewards),
            "reward_std": _safe_nanstd(pen_rewards),
            "length_mean": _safe_nanmean(pen_lengths),
            "fuel_L_per_100km_mean": _safe_nanmean(pen_fuels),
            "fuel_L_per_100km_std": _safe_nanstd(pen_fuels),
            "average_speed_mean": _safe_nanmean(pen_speeds),
            "average_speed_std": _safe_nanstd(pen_speeds),
            "average_jerk_mean": _safe_nanmean(pen_jerks),
            "average_jerk_std": _safe_nanstd(pen_jerks),
            "tet_rate_mean": _safe_nanmean(pen_tet_rates),
            "tet_rate_std": _safe_nanstd(pen_tet_rates),
            "trip50_time_mean": _safe_nanmean(pen_trip50_times),
            "trip50_time_std": _safe_nanstd(pen_trip50_times),
            "trip50_speed_mean": _safe_nanmean(pen_trip50_speeds),
            "trip50_speed_std": _safe_nanstd(pen_trip50_speeds),
        }
        summary_row.update(summarize_group_metrics(pen_group_metrics))
        summary_rows.append(summary_row)

        if save_ts and first_ts_pending:
            env.unwrapped.set_fcd_output(None)
            _ = env.reset(seed=int(cfg.seed) + p_idx * 1000 + 99999)
            plot_time_space_diagram(
                fcd_path=str(first_fcd_path),
                save_path=str(first_png_path),
                edge_length=294.4,
                episode=1,
            )
        env.close()

    _write_csv(condition_dir / "eval_penetration_detail.csv", DETAIL_FIELDS, detail_rows)
    _write_csv(condition_dir / "eval_penetration_summary.csv", SUMMARY_FIELDS, summary_rows)
    return detail_rows, summary_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_value(row: pd.Series, group: str, metric: str) -> float:
    col = f"trip50_{group}_{METRICS[metric]['detail_suffix']}"
    if col not in row.index:
        return float("nan")
    return float(row[col])


def _improvement_pct(method_value: float, baseline_value: float, higher_is_better: bool) -> float:
    if (
        not np.isfinite(method_value)
        or not np.isfinite(baseline_value)
        or abs(baseline_value) < 1e-12
    ):
        return float("nan")
    if higher_is_better:
        return (method_value - baseline_value) / abs(baseline_value) * 100.0
    return (baseline_value - method_value) / abs(baseline_value) * 100.0


def _compute_run_level_savings(detail: pd.DataFrame, baseline_detail_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not baseline_detail_path.exists():
        print(colored(
            f"Baseline detail CSV not found: {baseline_detail_path}. "
            "Skipping run-level savings.",
            "red",
        ))
        return pd.DataFrame(), pd.DataFrame()

    baseline = pd.read_csv(baseline_detail_path)
    index_cols = ["penetration_target", "run"]
    missing = [col for col in index_cols if col not in baseline.columns or col not in detail.columns]
    if missing:
        print(colored(f"Cannot compute savings; missing columns: {missing}", "red"))
        return pd.DataFrame(), pd.DataFrame()

    baseline_index = baseline.set_index(index_cols)
    savings_rows = []

    for _, row in detail.iterrows():
        key = (float(row["penetration_target"]), int(row["run"]))
        if key not in baseline_index.index:
            continue
        base_row = baseline_index.loc[key]
        if isinstance(base_row, pd.DataFrame):
            base_row = base_row.iloc[0]

        for group in GROUPS:
            for metric, spec in METRICS.items():
                method_value = _metric_value(row, group, metric)
                baseline_value = _metric_value(base_row, group, metric)
                improvement = _improvement_pct(
                    method_value,
                    baseline_value,
                    bool(spec["higher_is_better"]),
                )
                savings_rows.append({
                    "condition": row["condition"],
                    "condition_label": row["condition_label"],
                    "communication_capacity_k": int(row["communication_capacity_k"]),
                    "context_size": int(row["context_size"]),
                    "communication_range_m": float(row["communication_range_m"]),
                    "comm_topology": row["comm_topology"],
                    "penetration_target": float(row["penetration_target"]),
                    "run": int(row["run"]),
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "metric": metric,
                    "metric_label": spec["label"],
                    "baseline_value": baseline_value,
                    "method_value": method_value,
                    "improvement_pct": improvement,
                    "direction": "higher_is_better" if spec["higher_is_better"] else "lower_is_better",
                })

    savings = pd.DataFrame(savings_rows)
    if savings.empty:
        return savings, pd.DataFrame()

    summary = (
        savings
        .groupby(
            [
                "condition",
                "condition_label",
                "communication_capacity_k",
                "context_size",
                "communication_range_m",
                "comm_topology",
                "group",
                "group_label",
                "metric",
                "metric_label",
            ],
            as_index=False,
        )
        .agg(
            improvement_mean=("improvement_pct", "mean"),
            improvement_std=("improvement_pct", "std"),
            samples=("improvement_pct", "count"),
        )
    )
    return savings, summary


def _plot_mixed_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return

    data = summary[summary["group"].eq("mix")].copy()
    if data.empty:
        return

    metrics = list(METRICS)
    k_values = sorted(data["communication_capacity_k"].unique())
    condition_order = []
    for label in ["Full-50m", "Full-100m", "Full-150m", "Full-200m", "Front-150m"]:
        if label in set(data["condition_label"]):
            condition_order.append(label)
    for label in sorted(set(data["condition_label"]) - set(condition_order)):
        condition_order.append(label)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        metric_data = data[data["metric"].eq(metric)]
        pivot = metric_data.pivot_table(
            index="communication_capacity_k",
            columns="condition_label",
            values="improvement_mean",
            aggfunc="mean",
        ).reindex(index=k_values, columns=condition_order)

        values = pivot.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            vmax = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1.0)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        else:
            norm = None

        im = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_title(METRICS[metric]["label"])
        ax.set_xticks(np.arange(len(condition_order)))
        ax.set_xticklabels(condition_order, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values])
        ax.set_ylabel("K")

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Improvement (%)", rotation=90)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Mixed-Traffic Communication Sensitivity", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"mixed_traffic_comm_sensitivity_heatmap.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _print_savings_snapshot(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    mixed = summary[summary["group"].eq("mix")]
    if mixed.empty:
        return
    print(colored("\n── Mixed-Traffic Mean Improvement by Condition ──", "green", attrs=["bold"]))
    for condition in sorted(mixed["condition"].unique()):
        row = mixed[mixed["condition"].eq(condition)]
        pieces = []
        for metric in METRICS:
            match = row[row["metric"].eq(metric)]["improvement_mean"]
            if len(match):
                pieces.append(f"{metric}: {float(match.iloc[0]):.2f}%")
        print(colored(f"  {condition}: " + ", ".join(pieces), "green"))


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(raw_cfg):
    root = _project_root()

    if OmegaConf.is_missing(raw_cfg, "checkpoint"):
        raw_cfg.checkpoint = str(DEFAULT_CHECKPOINT)

    # The trained CAV-V2X checkpoint uses the communication-aware SAC setup.
    raw_cfg.agent = str(raw_cfg.get("agent", "sac"))
    raw_cfg.attention = True
    raw_cfg.communication = True
    raw_cfg.lane_action_discrete = bool(raw_cfg.get("lane_action_discrete", True))
    raw_cfg.sac_attention_include_self_token = bool(
        raw_cfg.get("sac_attention_include_self_token", True)
    )
    raw_cfg.rl_control = True
    raw_cfg.sumo_default_cav_behavior = False

    cfg = parse_cfg(raw_cfg)
    set_seed(cfg.seed)

    checkpoint = _resolve(root, cfg.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Bootstrap once with the fixed context-slot count used by the trained policy.
    conditions = _build_conditions(raw_cfg)
    context_slots = max(int(c["context_size"]) for c in conditions)
    cfg.context_size = context_slots
    cfg.communication_range_m = max(float(c["communication_range_m"]) for c in conditions)
    cfg.comm_topology = "full"
    bootstrap_env = make_env(cfg)
    bootstrap_env.close()

    agent = _make_agent(cfg)
    agent.load(str(checkpoint))
    agent.eval()

    ckpt_tag = checkpoint.stem
    default_out = root / "eval_results" / "communication_sensitivity" / ckpt_tag
    out_dir = _resolve(root, raw_cfg.get("comm_sensitivity_out_dir", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_detail = _resolve(
        root,
        raw_cfg.get("comm_sensitivity_baseline_detail", DEFAULT_BASELINE_DETAIL),
    )

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {checkpoint}", "blue", attrs=["bold"]))
    print(colored(f"Output dir: {out_dir}", "blue", attrs=["bold"]))
    print(colored(f"Baseline detail: {baseline_detail}", "blue"))
    print(colored(
        f"Conditions: {len(conditions)} | K values and communication modes fixed at evaluation time",
        "yellow",
        attrs=["bold"],
    ))

    all_detail_rows = []
    all_summary_rows = []
    for condition in conditions:
        detail_rows, summary_rows = _run_condition(cfg, agent, condition, out_dir)
        all_detail_rows.extend(detail_rows)
        all_summary_rows.extend(summary_rows)

        _write_csv(out_dir / "eval_comm_sensitivity_detail.csv", DETAIL_FIELDS, all_detail_rows)
        _write_csv(out_dir / "eval_comm_sensitivity_summary.csv", SUMMARY_FIELDS, all_summary_rows)

    detail = pd.DataFrame(all_detail_rows)
    summary = pd.DataFrame(all_summary_rows)
    detail.to_csv(out_dir / "eval_comm_sensitivity_detail.csv", index=False)
    summary.to_csv(out_dir / "eval_comm_sensitivity_summary.csv", index=False)

    savings, savings_summary = _compute_run_level_savings(detail, baseline_detail)
    if not savings.empty:
        savings.to_csv(out_dir / "run_level_savings_long.csv", index=False)
        savings_summary.to_csv(out_dir / "run_level_savings_summary.csv", index=False)
        _plot_mixed_heatmap(savings_summary, out_dir)
        _print_savings_snapshot(savings_summary)

    print(colored("\nSaved communication sensitivity outputs:", "green", attrs=["bold"]))
    for path in [
        out_dir / "eval_comm_sensitivity_detail.csv",
        out_dir / "eval_comm_sensitivity_summary.csv",
        out_dir / "run_level_savings_long.csv",
        out_dir / "run_level_savings_summary.csv",
        out_dir / "mixed_traffic_comm_sensitivity_heatmap.png",
        out_dir / "mixed_traffic_comm_sensitivity_heatmap.pdf",
    ]:
        if path.exists():
            print(colored(f"  {path}", "green"))


if __name__ == "__main__":
    main()
