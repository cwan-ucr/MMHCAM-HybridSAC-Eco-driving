"""
Evaluation script for a trained TD-MPC2 CAV control agent.

Usage:
    python evaluate.py task=sumo-intersection checkpoint=/path/to/model.pt
    python evaluate.py task=sumo-intersection checkpoint=/path/to/model.pt eval_episodes=20
"""

import os
import sys
import warnings
import csv
from pathlib import Path
warnings.filterwarnings('ignore')

import hydra
import numpy as np
import torch
from termcolor import colored

_TDMPC2_ROOT = os.path.join(os.path.dirname(__file__), '..', 'tdmpc2')
if os.path.isdir(_TDMPC2_ROOT):
    sys.path.insert(0, _TDMPC2_ROOT)

from common.parser import parse_cfg
from common.seed import set_seed
from common.plotting import plot_time_space_diagram
from common.evaluation_metrics import (
    GROUP_DETAIL_FIELDS,
    GROUP_SUMMARY_FIELDS,
    append_group_metrics,
    fuel_l_per_100km,
    group_detail_from_info,
    new_group_accumulator,
    summarize_group_metrics,
)
from agents.tdmpc2 import TDMPC2
from agents.sac import SAC
from agents.ppo import PPO
from agents.dqn import DQN

sys.path.insert(0, os.path.dirname(__file__))
from envs import make_env

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.', version_base=None)
def evaluate(cfg):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

    agent_type = str(getattr(cfg, 'agent', 'tdmpc2')).lower()
    cfg.agent = agent_type
    if agent_type == 'tdmpc2' and bool(getattr(cfg, 'lane_action_discrete', False)):
        raise NotImplementedError(
            "lane_action_discrete is for non-MPPI baselines. "
            "TD-MPC2/MPPI still uses continuous action sampling."
        )

    # Bootstrap environment once so cfg.episode_length/action dims are populated
    # before agent construction (required by SAC discount initialization).
    bootstrap_env = make_env(cfg)
    bootstrap_env.close()

    # Load agent
    if agent_type == 'tdmpc2':
        agent = TDMPC2(cfg)
    elif agent_type == 'sac':
        agent = SAC(cfg)
    elif agent_type == 'ppo':
        agent = PPO(cfg)
    elif agent_type == 'dqn':
        agent = DQN(cfg)
    else:
        raise ValueError(f'Unknown agent: {agent_type}. Expected tdmpc2, sac, ppo, or dqn.')
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found!'
    agent.load(cfg.checkpoint)

    # Fixed penetration experiment setup
    penetration_levels = [0.10, 0.30, 0.50, 0.70, 0.90]
    runs_per_penetration = 10
    print(colored(
        f'Evaluating by penetration: {penetration_levels} | {runs_per_penetration} runs each',
        'yellow', attrs=['bold']
    ))

    ckpt_tag = Path(cfg.checkpoint).stem
    out_dir = Path(os.getcwd()) / "eval_results" / str(cfg.exp_name) / ckpt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []
    ts_dir = out_dir / "eval_ts_diagrams"
    ts_dir.mkdir(parents=True, exist_ok=True)

    for p_idx, pen in enumerate(penetration_levels):
        # Force fixed penetration for this group; keep randomized total demand.
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
        first_fcd_path = ts_dir / f"fcd_pen{int(pen*100):02d}_run01.xml"
        first_png_path = ts_dir / f"ts_pen{int(pen*100):02d}_run01.png"
        first_ts_pending = True

        print(colored(f'\nPenetration {int(pen*100):>2d}%', 'cyan', attrs=['bold']))
        for i in range(runs_per_penetration):
            eval_seed = cfg.seed + 1000 + i
            if i == 0:
                env.unwrapped.set_fcd_output(str(first_fcd_path))
            else:
                env.unwrapped.set_fcd_output(None)
            obs, done, ep_reward, t = env.reset(seed=eval_seed), False, 0.0, 0
            info = {}
            # FCD for run-1 is flushed when run-2 reset closes the previous SUMO run.
            if i == 1 and first_ts_pending:
                plot_time_space_diagram(
                    fcd_path=str(first_fcd_path),
                    save_path=str(first_png_path),
                    edge_length=294.4,
                    episode=1,
                )
                first_ts_pending = False

            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward)
                t += 1

            total_travel_time = float(info.get('total_travel_time', max(t, 1)))
            reward_per_sec = ep_reward / max(total_travel_time, 1e-6)

            row = {
                "penetration_target": pen,
                "run": i + 1,
                "seed": eval_seed,
                "episode_reward": ep_reward,
                "reward_per_sec": reward_per_sec,
                "episode_length": t,
                "avg_speed": float(info.get('avg_speed', float('nan'))),
                "avg_jerk": float(info.get('avg_jerk', float('nan'))),
                "total_fuel_mL": float(info.get('total_fuel_mL', float('nan'))),
                "total_distance_m": float(info.get('total_distance', float('nan'))),
                "tet_rate": float(info.get('tet_rate', float('nan'))),
                "r_energy_mean": float(info.get('r_energy_mean', float('nan'))),
                "trip50_avg_time_s": float(info.get('trip50_avg_time_s', float('nan'))),
                "trip50_avg_stop_time_s": float(info.get('trip50_avg_stop_time_s', float('nan'))),
                "trip50_avg_speed_mps": float(info.get('trip50_avg_speed_mps', float('nan'))),
                "trip50_num_completed": float(info.get('trip50_num_completed', float('nan'))),
                "scenario_total_flow_vph": float(info.get('scenario_total_flow_vph', float('nan'))),
                "scenario_cav_penetration": float(info.get('scenario_cav_penetration', float('nan'))),
                "scenario_cav_flow_vph": float(info.get('scenario_cav_flow_vph', float('nan'))),
                "scenario_hdv_flow_vph": float(info.get('scenario_hdv_flow_vph', float('nan'))),
            }
            row.update(group_detail_from_info(info))
            detail_rows.append(row)

            pen_rewards.append(ep_reward)
            pen_lengths.append(t)
            pen_speeds.append(float(info.get('avg_speed', float('nan'))))
            pen_jerks.append(float(info.get('avg_jerk', float('nan'))))
            pen_tet_rates.append(float(info.get('tet_rate', float('nan'))))
            pen_fuels.append(fuel_l_per_100km(row["total_fuel_mL"], row["total_distance_m"]))
            pen_trip50_times.append(row["trip50_avg_time_s"])
            pen_trip50_speeds.append(row["trip50_avg_speed_mps"])
            append_group_metrics(pen_group_metrics, row)

            print(colored(
                f'  Run {i+1:>2d}  R: {ep_reward:>8.2f}  Len: {t:>4d}  '
                f'trip50_t: {row["trip50_avg_time_s"]:>6.2f}s  '
                f'trip50_v: {row["trip50_avg_speed_mps"]:>5.2f}m/s',
                'yellow'
            ))

        summary_row = {
            "penetration_target": pen,
            "runs": runs_per_penetration,
            "reward_mean": float(np.nanmean(pen_rewards)),
            "reward_std": float(np.nanstd(pen_rewards)),
            "length_mean": float(np.nanmean(pen_lengths)),
            "fuel_L_per_100km_mean": float(np.nanmean(pen_fuels)),
            "fuel_L_per_100km_std": float(np.nanstd(pen_fuels)),
            "average_speed_mean": float(np.nanmean(pen_speeds)),
            "average_speed_std": float(np.nanstd(pen_speeds)),
            "average_jerk_mean": float(np.nanmean(pen_jerks)),
            "average_jerk_std": float(np.nanstd(pen_jerks)),
            "tet_rate_mean": float(np.nanmean(pen_tet_rates)),
            "tet_rate_std": float(np.nanstd(pen_tet_rates)),
            "trip50_time_mean": float(np.nanmean(pen_trip50_times)),
            "trip50_time_std": float(np.nanstd(pen_trip50_times)),
            "trip50_speed_mean": float(np.nanmean(pen_trip50_speeds)),
            "trip50_speed_std": float(np.nanstd(pen_trip50_speeds)),
        }
        summary_row.update(summarize_group_metrics(pen_group_metrics))
        summary_rows.append(summary_row)
        # If only one run configured, force one extra reset to flush FCD before plotting.
        if first_ts_pending:
            env.unwrapped.set_fcd_output(None)
            _ = env.reset(seed=cfg.seed + p_idx * 1000 + 99999)
            plot_time_space_diagram(
                fcd_path=str(first_fcd_path),
                save_path=str(first_png_path),
                edge_length=294.4,
                episode=1,
            )
            first_ts_pending = False
        env.close()

    detail_path = out_dir / "eval_penetration_detail.csv"
    summary_path = out_dir / "eval_penetration_summary.csv"

    detail_fields = [
        "penetration_target", "run", "seed",
        "episode_reward", "reward_per_sec", "episode_length",
        "avg_speed", "avg_jerk", "total_fuel_mL", "total_distance_m", "tet_rate", "r_energy_mean",
        "trip50_avg_time_s", "trip50_avg_stop_time_s", "trip50_avg_speed_mps", "trip50_num_completed",
        *GROUP_DETAIL_FIELDS,
        "scenario_total_flow_vph", "scenario_cav_penetration", "scenario_cav_flow_vph", "scenario_hdv_flow_vph",
    ]
    summary_fields = [
        "penetration_target", "runs",
        "reward_mean", "reward_std",
        "length_mean",
        "fuel_L_per_100km_mean", "fuel_L_per_100km_std",
        "average_speed_mean", "average_speed_std",
        "average_jerk_mean", "average_jerk_std",
        "tet_rate_mean", "tet_rate_std",
        "trip50_time_mean", "trip50_time_std",
        "trip50_speed_mean", "trip50_speed_std",
        *GROUP_SUMMARY_FIELDS,
    ]

    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(colored('\n── Summary by Penetration ──', 'green', attrs=['bold']))
    for row in summary_rows:
        print(colored(
            f'  {int(row["penetration_target"]*100):>2d}%  '
            f'R: {row["reward_mean"]:.2f} ± {row["reward_std"]:.2f}  '
            f'fuel: {row["fuel_L_per_100km_mean"]:.2f}L/100km  '
            f'v: {row["average_speed_mean"]:.2f}m/s  '
            f'jerk: {row["average_jerk_mean"]:.2f}  '
            f'TET: {row["tet_rate_mean"]:.4f}  '
            f'trip50_t: {row["trip50_time_mean"]:.2f}s  '
            f'trip50_v: {row["trip50_speed_mean"]:.2f}m/s',
            'green'
        ))
    print(colored(f'\nSaved detailed results: {detail_path}', 'green'))
    print(colored(f'Saved summary results:  {summary_path}', 'green'))
    print(colored(f'Saved time-space diagrams: {ts_dir}', 'green'))


if __name__ == '__main__':
    evaluate()
