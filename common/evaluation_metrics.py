import math

import numpy as np


GROUPS = ("mix", "cav", "hdv")

GROUP_DETAIL_SUFFIXES = (
    "avg_time_s",
    "avg_stop_time_s",
    "avg_speed_mps",
    "num_completed",
    "avg_step_speed_mps",
    "avg_jerk",
    "tet_rate",
    "total_fuel_mL",
    "total_distance_m",
    "fuel_L_per_100km",
    "step_count",
)

GROUP_DETAIL_FIELDS = [
    f"trip50_{group}_{suffix}"
    for group in GROUPS
    for suffix in GROUP_DETAIL_SUFFIXES
]

_SUMMARY_METRICS = {
    "time": "avg_time_s",
    "stop_time": "avg_stop_time_s",
    "speed": "avg_speed_mps",
    "num_completed": "num_completed",
    "avg_step_speed": "avg_step_speed_mps",
    "avg_jerk": "avg_jerk",
    "tet_rate": "tet_rate",
    "fuel_L_per_100km": "fuel_L_per_100km",
}

GROUP_SUMMARY_FIELDS = [
    f"trip50_{group}_{metric}_{stat}"
    for group in GROUPS
    for metric in _SUMMARY_METRICS
    for stat in ("mean", "std")
]


def fuel_l_per_100km(total_fuel_ml, total_distance_m):
    if total_distance_m <= 0:
        return float("nan")
    return total_fuel_ml / (total_distance_m * 0.01)


def group_detail_from_info(info):
    row = {}
    for group in GROUPS:
        prefix = f"trip50_{group}_"
        for suffix in GROUP_DETAIL_SUFFIXES:
            if suffix == "fuel_L_per_100km":
                continue
            row[f"{prefix}{suffix}"] = info.get(f"{prefix}{suffix}", 0.0)
        row[f"{prefix}fuel_L_per_100km"] = fuel_l_per_100km(
            row[f"{prefix}total_fuel_mL"],
            row[f"{prefix}total_distance_m"],
        )
    return row


def new_group_accumulator():
    return {
        group: {metric: [] for metric in _SUMMARY_METRICS}
        for group in GROUPS
    }


def append_group_metrics(accumulator, detail_row):
    for group in GROUPS:
        prefix = f"trip50_{group}_"
        for metric, detail_suffix in _SUMMARY_METRICS.items():
            accumulator[group][metric].append(detail_row[f"{prefix}{detail_suffix}"])


def _mean_std(values):
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan"), float("nan")
    return float(np.mean(finite_values)), float(np.std(finite_values))


def summarize_group_metrics(accumulator):
    summary = {}
    for group in GROUPS:
        prefix = f"trip50_{group}_"
        for metric, values in accumulator[group].items():
            mean, std = _mean_std(values)
            summary[f"{prefix}{metric}_mean"] = mean
            summary[f"{prefix}{metric}_std"] = std
    return summary
