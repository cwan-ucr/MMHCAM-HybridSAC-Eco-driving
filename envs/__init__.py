"""
Environment factory – extended with SUMO CAV intersection support.
"""

from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(
        f"Missing dependencies for task {task}; "
        "install dependencies to use this environment."
    )

# ── NEW: SUMO environment ──────────────────────────────────────────────────
try:
    from envs.sumo_wrapper import make_env as make_sumo_env
except Exception:
    make_sumo_env = missing_dependencies


warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_multitask_env(cfg):
    """Make a multi-task environment for TD-MPC2 experiments."""
    print("Creating multi-task environment with tasks:", cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    Tries each backend in order; SUMO is checked early so that
    `sumo-*` tasks are found before falling through to errors.
    """
    gym.logger.set_level(40)

    if cfg.multitask:
        env = make_multitask_env(cfg)
    else:
        env = None
        # Try SUMO first for sumo-* tasks, then the original backends
        backends = [
            make_sumo_env,
        ]
        for fn in backends:
            try:
                env = fn(cfg)
                break
            except ValueError:
                pass
        if env is None:
            raise ValueError(
                f'Failed to make environment "{cfg.task}": '
                "please verify that dependencies are installed and "
                "that the task exists."
            )
        # Only wrap with TensorWrapper for non-SUMO envs
        # (SumoWrapper already returns tensors)
        if not cfg.task.startswith("sumo-"):
            env = TensorWrapper(env)

    try:  # Dict observation space
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except Exception:  # Box observation space
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}

    if cfg.task.startswith("sumo-"):
        # DTDE: action dim is per-CAV (the model processes each CAV independently).
        cfg.action_dim = cfg.act_per_agent
    else:
        cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(100, 5 * cfg.episode_length)
    return env
