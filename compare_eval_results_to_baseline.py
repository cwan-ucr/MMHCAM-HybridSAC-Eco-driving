"""
Compare evaluation results against the SUMO default baseline by vehicle group.

The script scans eval_results/**/eval_penetration_summary.csv, uses
eval_results/sumo_baseline/default as the baseline, and computes improvement
rates separately for:
  - mix: all vehicles
  - cav: CAVs only
  - hdv: HDVs only

Outputs are written to eval_results/comparison_vs_sumo_baseline by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


GROUPS = ("mix", "cav", "hdv")
GROUP_LABELS = {
    "mix": "Mixed",
    "cav": "CAV",
    "hdv": "HDV",
}

METHOD_GROUPS = {
    "basic_control": {
        "label": "AV vs V2I vs CAV",
        "methods": (
            "av_control/final",
            "cav_control_v2i/final",
            "cav_control/final",
        ),
    },
    "front_topology": {
        "label": "Full V2X vs Front-only V2X",
        "methods": (
            "cav_control/final",
            "cav_control_front/final",
        ),
    },
    "cooperative_reward": {
        "label": "CAV vs Cooperative Reward",
        "methods": (
            "cav_control/final",
            "cav_control_coop/final",
            "cav_control_coop_back/final",
        ),
    },
    "control_ablation": {
        "label": "Hybrid vs Longitudinal-only vs Lane-change-only",
        "methods": (
            "cav_control_ablation_hybrid/final",
            "cav_control_ablation_longitudinal/final",
            "cav_control_ablation_lane_change/final",
        ),
    },
}

METRICS = {
    "time": {
        "summary_suffix": "time",
        "detail_suffix": "avg_time_s",
        "higher_is_better": False,
        "label": "Trip time",
    },
    "stop_time": {
        "summary_suffix": "stop_time",
        "detail_suffix": "avg_stop_time_s",
        "higher_is_better": False,
        "label": "Stop time",
    },
    "speed": {
        "summary_suffix": "speed",
        "detail_suffix": "avg_speed_mps",
        "higher_is_better": True,
        "label": "Speed",
    },
    "fuel": {
        "summary_suffix": "fuel_L_per_100km",
        "detail_suffix": "fuel_L_per_100km",
        "higher_is_better": False,
        "label": "Fuel",
    },
    "jerk": {
        "summary_suffix": "avg_jerk",
        "detail_suffix": "avg_jerk",
        "higher_is_better": False,
        "label": "Jerk",
    },
    "tet_rate": {
        "summary_suffix": "tet_rate",
        "detail_suffix": "tet_rate",
        "higher_is_better": False,
        "label": "TET rate",
    },
}

# Backward-compatible fallback for older evaluation CSVs that only had one
# aggregate set of columns. These are treated as the mixed-traffic group.
LEGACY_MIX_SUMMARY_COLUMNS = {
    "time": ("trip50_time_mean",),
    "stop_time": (),
    "speed": ("average_speed_mean", "avg_speed_mean", "avg_speed", "trip50_speed_mean"),
    "fuel": ("fuel_L_per_100km_mean", "fuel_mean_mL", "fuel_mean"),
    "jerk": ("average_jerk_mean", "avg_jerk_mean", "avg_jerk"),
    "tet_rate": ("tet_rate_mean", "tet_rate"),
}

LEGACY_MIX_DETAIL_COLUMNS = {
    "time": "trip50_avg_time_s",
    "stop_time": "trip50_avg_stop_time_s",
    "speed": "trip50_avg_speed_mps",
    "fuel": None,  # computed from total_fuel_mL / total_distance_m
    "jerk": "avg_jerk",
    "tet_rate": "tet_rate",
}


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = set(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def _method_name(summary_path: Path, root: Path) -> str:
    rel = summary_path.relative_to(root)
    return "/".join(rel.parts[:-1])


def _summary_column(group: str, metric: str) -> str:
    return f"trip50_{group}_{METRICS[metric]['summary_suffix']}_mean"


def _detail_column(group: str, metric: str) -> str:
    return f"trip50_{group}_{METRICS[metric]['detail_suffix']}"


def _legacy_fuel_column(detail: pd.DataFrame) -> str | None:
    if {"total_fuel_mL", "total_distance_m"}.issubset(detail.columns):
        denom = detail["total_distance_m"].astype(float) * 0.01
        detail["_legacy_fuel_L_per_100km"] = np.where(
            denom > 0,
            detail["total_fuel_mL"].astype(float) / denom,
            np.nan,
        )
        return "_legacy_fuel_L_per_100km"
    return None


def _fill_missing_from_detail(table: pd.DataFrame, detail_path: Path) -> pd.DataFrame:
    if not detail_path.exists():
        return table

    detail = pd.read_csv(detail_path)
    if "penetration_target" not in detail.columns:
        return table

    table = table.copy()
    for group in GROUPS:
        for metric in METRICS:
            missing_mask = table["group"].eq(group) & table[metric].isna()
            if not missing_mask.any():
                continue

            detail_col = _detail_column(group, metric)
            if detail_col not in detail.columns:
                if group != "mix":
                    continue
                if metric == "fuel":
                    detail_col = _legacy_fuel_column(detail)
                else:
                    detail_col = LEGACY_MIX_DETAIL_COLUMNS.get(metric)
                if detail_col is None or detail_col not in detail.columns:
                    continue

            fallback = detail.groupby("penetration_target")[detail_col].mean()
            table.loc[missing_mask, metric] = (
                table.loc[missing_mask, "penetration_target"].map(fallback).astype(float)
            )

    return table


def _read_metric_table(summary_path: Path) -> pd.DataFrame:
    """Read one evaluation summary into group-aware metric rows."""
    summary = pd.read_csv(summary_path)
    if "penetration_target" not in summary.columns:
        raise ValueError(f"{summary_path} missing required column penetration_target")

    base = pd.DataFrame()
    base["penetration_target"] = summary["penetration_target"].astype(float)
    base["runs"] = summary["runs"] if "runs" in summary.columns else np.nan

    group_tables = []
    for group in GROUPS:
        group_out = base.copy()
        group_out["group"] = group
        for metric in METRICS:
            col = _summary_column(group, metric)
            if col in summary.columns:
                group_out[metric] = summary[col].astype(float)
            elif group == "mix":
                legacy_col = _first_existing(
                    summary.columns,
                    LEGACY_MIX_SUMMARY_COLUMNS.get(metric, ()),
                )
                group_out[metric] = (
                    summary[legacy_col].astype(float) if legacy_col is not None else np.nan
                )
            else:
                group_out[metric] = np.nan
        group_tables.append(group_out)

    out = pd.concat(group_tables, ignore_index=True)
    return _fill_missing_from_detail(
        out,
        summary_path.with_name("eval_penetration_detail.csv"),
    )


def collect_results(root: Path) -> Dict[str, pd.DataFrame]:
    results = {}
    for summary_path in sorted(root.glob("**/eval_penetration_summary.csv")):
        if "comparison_vs_sumo_baseline" in summary_path.parts:
            continue
        method = _method_name(summary_path, root)
        results[method] = _read_metric_table(summary_path)
    return results


def _improvement_pct(method_value: float, baseline_value: float, higher_is_better: bool) -> float:
    if (
        not np.isfinite(method_value)
        or not np.isfinite(baseline_value)
        or abs(baseline_value) < 1e-12
    ):
        return np.nan
    if higher_is_better:
        return (method_value - baseline_value) / abs(baseline_value) * 100.0
    return (baseline_value - method_value) / abs(baseline_value) * 100.0


def compute_improvements(
    results: Dict[str, pd.DataFrame],
    baseline_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_method not in results:
        available = ", ".join(sorted(results))
        raise ValueError(f"Missing baseline {baseline_method!r}. Available: {available}")

    index_cols = ["penetration_target", "group"]
    baseline = results[baseline_method].set_index(index_cols)
    value_rows = []
    improvement_rows = []

    for method, table in sorted(results.items()):
        if method.startswith("sumo_baseline") and method != baseline_method:
            continue
        if method == baseline_method:
            continue

        joined = table.set_index(index_cols).join(
            baseline,
            lsuffix="_method",
            rsuffix="_baseline",
            how="inner",
        )

        for (pen, group), row in joined.iterrows():
            for metric, spec in METRICS.items():
                method_value = float(row[f"{metric}_method"])
                baseline_value = float(row[f"{metric}_baseline"])
                improvement = _improvement_pct(
                    method_value,
                    baseline_value,
                    bool(spec["higher_is_better"]),
                )
                direction = (
                    "higher_is_better" if spec["higher_is_better"] else "lower_is_better"
                )

                value_rows.append({
                    "method": method,
                    "penetration_target": pen,
                    "group": group,
                    "metric": metric,
                    "metric_label": spec["label"],
                    "baseline_value": baseline_value,
                    "method_value": method_value,
                    "direction": direction,
                })
                improvement_rows.append({
                    "method": method,
                    "penetration_target": pen,
                    "group": group,
                    "metric": metric,
                    "metric_label": spec["label"],
                    "improvement_pct": improvement,
                    "direction": direction,
                })

    return pd.DataFrame(value_rows), pd.DataFrame(improvement_rows)


def plot_improvements(improvements: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if improvements.empty:
        return

    metrics = list(METRICS)
    groups = [group for group in GROUPS if group in set(improvements["group"])]
    methods = sorted(improvements["method"].unique())
    pens = sorted(improvements["penetration_target"].unique())
    x = np.arange(len(pens), dtype=float)
    total_width = 0.82
    bar_width = total_width / max(len(methods), 1)
    cmap = plt.get_cmap("tab10")
    colors = {method: cmap(i % 10) for i, method in enumerate(methods)}

    fig, axes = plt.subplots(
        len(groups),
        len(metrics),
        figsize=(4.0 * len(metrics), 3.0 * len(groups)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, group in enumerate(groups):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            data = improvements[
                (improvements["group"] == group)
                & (improvements["metric"] == metric)
            ]
            for idx, method in enumerate(methods):
                vals = []
                for pen in pens:
                    match = data[
                        (data["method"] == method)
                        & (data["penetration_target"] == pen)
                    ]["improvement_pct"]
                    vals.append(float(match.iloc[0]) if len(match) else np.nan)
                offset = -total_width / 2 + bar_width / 2 + idx * bar_width
                ax.bar(x + offset, vals, width=bar_width, label=method, color=colors[method])

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{GROUP_LABELS[group]} | {METRICS[metric]['label']}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(p * 100)}%" for p in pens])
            if col_idx == 0:
                ax.set_ylabel("Improvement vs baseline (%)")
            ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, max(1, len(labels))))
    fig.suptitle(title, y=1.02, fontsize=15)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_average_improvements(improvements: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if improvements.empty:
        return

    avg = improvements.groupby(["method", "group", "metric"], as_index=False)[
        "improvement_pct"
    ].mean()
    metrics = list(METRICS)
    groups = [group for group in GROUPS if group in set(avg["group"])]
    methods = sorted(avg["method"].unique())
    x = np.arange(len(metrics), dtype=float)
    total_width = 0.82
    bar_width = total_width / max(len(methods), 1)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.2 * len(groups), 5.0),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )

    for group_idx, group in enumerate(groups):
        ax = axes[0, group_idx]
        data = avg[avg["group"] == group]
        for idx, method in enumerate(methods):
            vals = []
            for metric in metrics:
                match = data[
                    (data["method"] == method)
                    & (data["metric"] == metric)
                ]["improvement_pct"]
                vals.append(float(match.iloc[0]) if len(match) else np.nan)
            offset = -total_width / 2 + bar_width / 2 + idx * bar_width
            ax.bar(x + offset, vals, width=bar_width, label=method, color=cmap(idx % 10))

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(GROUP_LABELS[group])
        ax.set_xticks(x)
        ax.set_xticklabels([METRICS[m]["label"] for m in metrics], rotation=20, ha="right")
        if group_idx == 0:
            ax.set_ylabel("Mean improvement vs baseline (%)")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, max(1, len(labels))))
    fig.suptitle(title, y=1.04, fontsize=15)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    values: pd.DataFrame,
    improvements: pd.DataFrame,
    out_dir: Path,
    title: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path = out_dir / "metric_values_vs_sumo_baseline.csv"
    long_path = out_dir / "improvement_vs_sumo_baseline_long.csv"
    wide_path = out_dir / "improvement_vs_sumo_baseline_wide.csv"
    plot_path = out_dir / "improvement_vs_sumo_baseline_bars.png"
    avg_plot_path = out_dir / "improvement_vs_sumo_baseline_mean_bars.png"

    values.to_csv(values_path, index=False)
    improvements.to_csv(long_path, index=False)
    improvements.pivot_table(
        index=["penetration_target", "group", "metric"],
        columns="method",
        values="improvement_pct",
    ).reset_index().to_csv(wide_path, index=False)

    plot_improvements(improvements, plot_path, f"{title} | Improvement vs SUMO Baseline")
    plot_average_improvements(improvements, avg_plot_path, f"{title} | Mean Improvement")
    return [values_path, long_path, wide_path, plot_path, avg_plot_path]


def write_method_group_outputs(
    values: pd.DataFrame,
    improvements: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    written = []
    available_methods = set(improvements["method"].unique())
    for slug, spec in METHOD_GROUPS.items():
        methods = list(spec["methods"])
        present = [method for method in methods if method in available_methods]
        missing = [method for method in methods if method not in available_methods]
        if missing:
            print(f"Group {slug}: missing {', '.join(missing)}")
        if not present:
            print(f"Group {slug}: skipped because no requested methods are available")
            continue
        group_values = values[values["method"].isin(present)].copy()
        group_improvements = improvements[improvements["method"].isin(present)].copy()
        group_values["method"] = pd.Categorical(group_values["method"], categories=methods, ordered=True)
        group_improvements["method"] = pd.Categorical(
            group_improvements["method"], categories=methods, ordered=True
        )
        group_values = group_values.sort_values(["method", "penetration_target", "group", "metric"])
        group_improvements = group_improvements.sort_values(
            ["method", "penetration_target", "group", "metric"]
        )
        written.extend(write_outputs(
            group_values,
            group_improvements,
            out_dir / "method_groups" / slug,
            str(spec["label"]),
        ))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("eval_results"))
    parser.add_argument("--baseline", default="sumo_baseline/default")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir or root / "comparison_vs_sumo_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(root)
    values, improvements = compute_improvements(results, args.baseline)

    written = write_outputs(
        values,
        improvements,
        out_dir,
        "All Methods",
    )
    written.extend(write_method_group_outputs(values, improvements, out_dir))

    print(f"Compared {len(results) - 1} methods against {args.baseline}")
    print(f"Groups: {', '.join(GROUPS)}")
    print(f"Metrics: {', '.join(METRICS)}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
