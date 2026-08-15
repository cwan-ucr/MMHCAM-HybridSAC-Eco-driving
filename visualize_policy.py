"""Visualize a trained policy in SUMO or SUMO-GUI."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import hydra
import torch
from termcolor import colored

_LOCAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_LOCAL_ROOT))

from agents.dqn import DQN
from agents.ppo import PPO
from agents.sac import SAC
from agents.tdmpc2 import TDMPC2
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env


def _resolve_path(value, root: str, name: str) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return str(path)


def _make_agent(cfg):
    agent_type = str(getattr(cfg, "agent", "sac")).lower()
    cfg.agent = agent_type
    if agent_type == "sac":
        return SAC(cfg)
    if agent_type == "ppo":
        return PPO(cfg)
    if agent_type == "dqn":
        return DQN(cfg)
    if agent_type == "tdmpc2":
        return TDMPC2(cfg)
    raise ValueError(f"Unknown agent: {agent_type}")


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg):
    original_cwd = hydra.utils.get_original_cwd()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    cfg.checkpoint = _resolve_path(cfg.checkpoint, original_cwd, "Checkpoint")
    if getattr(cfg, "sumo_cfg", None):
        cfg.sumo_cfg = _resolve_path(cfg.sumo_cfg, original_cwd, "SUMO cfg")
    if getattr(cfg, "route_template", None):
        cfg.route_template = _resolve_path(cfg.route_template, original_cwd, "Route template")

    print(colored("Visualizing trained policy", "cyan", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue"))
    print(colored(f"SUMO cfg:   {getattr(cfg, 'sumo_cfg', None)}", "blue"))
    print(colored(f"GUI:        {bool(getattr(cfg, 'use_gui', False))}", "blue"))

    # Bootstrap once so cfg.obs_per_agent / cfg.act_per_agent are populated.
    bootstrap_env = make_env(cfg)
    bootstrap_env.close()

    agent = _make_agent(cfg)
    agent.load(cfg.checkpoint)
    agent.eval()

    env = make_env(cfg)
    obs = env.reset(seed=cfg.seed)
    done = False
    ep_reward = 0.0
    t = 0
    info = {}
    print_every = max(int(getattr(cfg, "visualize_print_every", 100) or 100), 1)

    try:
        while not done:
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            t += 1
            if t % print_every == 0:
                n_cav = len(obs.get("cav_ids", []))
                print(
                    f"step={t:4d} active_cav={n_cav:3d} "
                    f"reward={ep_reward:9.2f} "
                    f"completed={info.get('trip50_num_completed', 0)}"
                )
    finally:
        if bool(getattr(cfg, "use_gui", False)) and bool(getattr(cfg, "visualize_keep_open", False)):
            try:
                input("Episode finished. Press Enter to close SUMO-GUI...")
            except EOFError:
                pass
        env.close()

    print(colored("Episode finished", "green", attrs=["bold"]))
    print(
        f"steps={t}, reward={ep_reward:.2f}, "
        f"trip50_completed={info.get('trip50_num_completed', 'n/a')}, "
        f"avg_speed={info.get('trip50_avg_speed_mps', float('nan')):.2f} m/s"
    )


if __name__ == "__main__":
    main()
