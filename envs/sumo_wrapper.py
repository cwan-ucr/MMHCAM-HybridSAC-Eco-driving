"""
TD-MPC2 wrapper for the slot-free DTDE SUMO CAV intersection environment.

The underlying env returns a dict with variable-length per-CAV tensors:
    {
        'obs':            (N, 35)      own observations
        'ctx_obs':        (N, K, 35)    K nearest neighbors per CAV
        'ctx_mask':       (N, K)
        'entry_phase_id': (N,)
        'progress_id':    (N,)
        'cav_ids':        List[str] length N
    }
where N is the number of currently active CAVs (may change every step).

This wrapper converts numpy arrays to torch tensors and provides a
rand_act() that returns a (N, 2) tensor sized to the current active count.
"""

import numpy as np
import torch
import gymnasium as gym

from envs.sumo_env import SumoIntersectionEnv, _OBS_PER_CAV, _ACT_PER_CAV
from envs.sumo_env_fast import SumoIntersectionFastEnv
from envs.sumo_env_rw import SumoIntersectionRWEnv


# Task registry
SUMO_TASKS = {
    "sumo-intersection":         dict(max_steps=1500, desired_speed=13.89),
    "sumo-intersection-dense":   dict(max_steps=1500, desired_speed=13.89),
    "sumo-intersection-sparse":  dict(max_steps=800,  desired_speed=13.89),
}


def _to_tensor_dict(obs_dict):
    """Convert env dict output to a dict of torch tensors (cav_ids stays list)."""
    return {
        'obs':            torch.from_numpy(obs_dict['obs']).float(),
        'ctx_obs':        torch.from_numpy(obs_dict['ctx_obs']).float(),
        'ctx_mask':       torch.from_numpy(obs_dict['ctx_mask']).float(),
        'entry_phase_id': torch.from_numpy(obs_dict['entry_phase_id']).long(),
        'progress_id':    torch.from_numpy(obs_dict['progress_id']).long(),
        'cav_ids':        list(obs_dict['cav_ids']),
    }


class SumoWrapper(gym.Wrapper):
    """Wraps SumoIntersectionEnv and converts its dict output to torch tensors."""

    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._cumulative_reward = 0.0
        self._last_cav_ids = []

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self._cumulative_reward = 0.0
        tensor_dict = _to_tensor_dict(obs_dict)
        self._last_cav_ids = tensor_dict['cav_ids']
        return tensor_dict

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1 and action.size == 0:
            action = action.reshape(0, _ACT_PER_CAV)
        obs_dict, reward, done, info = self.env.step(action)

        self._cumulative_reward += reward
        info["success"] = float(self._cumulative_reward > 0)
        info["terminated"] = info.get("terminated", False)
        if "agent_rewards" in info and isinstance(info["agent_rewards"], np.ndarray):
            info["agent_rewards"] = torch.from_numpy(info["agent_rewards"]).float()

        tensor_dict = _to_tensor_dict(obs_dict)
        self._last_cav_ids = tensor_dict['cav_ids']
        return tensor_dict, reward, done, info

    def rand_act(self):
        """Return a (N_current, 2) random action aligned with the last obs."""
        n = len(self._last_cav_ids)
        if n == 0:
            return torch.zeros((0, _ACT_PER_CAV), dtype=torch.float32)
        if bool(getattr(self.cfg, "lane_action_discrete", False)):
            accel = torch.empty((n, 1), dtype=torch.float32).uniform_(-1.0, 1.0)
            lane_values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
            lane = lane_values[torch.randint(0, 3, (n,))].unsqueeze(-1)
            return torch.cat([accel, lane], dim=-1)
        return torch.empty((n, _ACT_PER_CAV), dtype=torch.float32).uniform_(-1.0, 1.0)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, **kwargs):
        return self.env.render(**kwargs)


class Timeout(gym.Wrapper):
    """Enforces a maximum episode length."""

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._t += 1
        done = done or self._t >= self.max_episode_steps
        return obs, reward, done, info

    def rand_act(self):
        return self.env.rand_act()


def make_env(cfg):
    """Factory for the slot-free SUMO CAV intersection environment."""
    if cfg.task not in SUMO_TASKS:
        raise ValueError(f"Unknown task: {cfg.task}")
    assert cfg.obs == "state", "SUMO environment only supports state observations."

    task_cfg = dict(SUMO_TASKS[cfg.task])
    if getattr(cfg, "sumo_max_steps", None):
        task_cfg["max_steps"] = int(cfg.sumo_max_steps)
    env_impl = str(getattr(cfg, "sumo_env_impl", "default")).lower()
    if env_impl not in {"default", "fast", "rw", "realworld", "real_world"}:
        raise ValueError("sumo_env_impl must be 'default', 'fast', or 'rw'.")
    if env_impl == "fast":
        env_cls = SumoIntersectionFastEnv
    elif env_impl in {"rw", "realworld", "real_world"}:
        env_cls = SumoIntersectionRWEnv
    else:
        env_cls = SumoIntersectionEnv

    env_kwargs = dict(
        max_steps=task_cfg["max_steps"],
        desired_speed=task_cfg["desired_speed"],
        context_size=cfg.context_size,
        communication_range_m=getattr(cfg, "communication_range_m", 150.0),
        comm_topology=getattr(cfg, "comm_topology", "full"),
        lane_change_mode=getattr(cfg, "sumo_lane_change_mode", 512),
        lane_change_mode_release=getattr(cfg, "sumo_lane_change_mode_release", 1621),
        vid_cycle=cfg.vid_cycle,
        max_controllable=cfg.max_controllable,
        communication=getattr(cfg, "communication", True),
        rl_control=getattr(cfg, "rl_control", True),
        cav_control_mode=getattr(cfg, "cav_control_mode", "hybrid"),
        sumo_default_cav_behavior=getattr(cfg, "sumo_default_cav_behavior", False),
        glosa_enabled=getattr(cfg, "glosa_enabled", False),
        glosa_range=getattr(cfg, "glosa_range", 150.0),
        glosa_min_speed=getattr(cfg, "glosa_min_speed", 5.0),
        glosa_max_speedfactor=getattr(cfg, "glosa_max_speedfactor", 1.1),
        glosa_add_switchtime=getattr(cfg, "glosa_add_switchtime", 0.0),
        glosa_override_safety=getattr(cfg, "glosa_override_safety", False),
        glosa_ignore_cfmodel=getattr(cfg, "glosa_ignore_cfmodel", False),
        glosa_use_queue=getattr(cfg, "glosa_use_queue", False),
        neighbor_reward_coef=getattr(cfg, 'neighbor_reward_coef', 0.0),
        neighbor_reward_directional=getattr(cfg, 'neighbor_reward_directional', False),
        control_start_distance_m=getattr(cfg, "control_start_distance_m", 20.0),
        post_out_eval_distance_m=getattr(cfg, "post_out_eval_distance_m", 50.0),
        randomize_demand=getattr(cfg, "randomize_demand", True),
        total_flow_vph_min=getattr(cfg, "total_flow_vph_min", 1400.0),
        total_flow_vph_max=getattr(cfg, "total_flow_vph_max", 2700.0),
        cav_penetration_min=getattr(cfg, "cav_penetration_min", 0.05),
        cav_penetration_max=getattr(cfg, "cav_penetration_max", 0.95),
        controlled_edges=getattr(cfg, "controlled_edges", None),
        control_area_length_m=getattr(cfg, "control_area_length_m", 294.4),
        terminal_cycle_s=getattr(cfg, "terminal_cycle_s", 60.0),
        terminal_green_start_s=getattr(cfg, "terminal_green_start_s", 30.0),
        terminal_green_end_s=getattr(cfg, "terminal_green_end_s", 60.0),
        use_gui=bool(getattr(cfg, "use_gui", False)),
        seed=cfg.seed,
    )
    if env_impl == "fast":
        env_kwargs["max_speed_mps"] = getattr(cfg, "max_speed_mps", 18.0)
    if env_impl in {"rw", "realworld", "real_world"}:
        env_kwargs["allowed_control_lanes"] = getattr(cfg, "allowed_control_lanes", None)
        env_kwargs["sumo_gui_delay_ms"] = getattr(cfg, "sumo_gui_delay_ms", 0)
    if getattr(cfg, "sumo_cfg", None):
        env_kwargs["sumo_cfg"] = str(cfg.sumo_cfg)
    if getattr(cfg, "route_template", None):
        env_kwargs["route_template"] = str(cfg.route_template)
    env = env_cls(**env_kwargs)
    env = SumoWrapper(env, cfg)
    env = Timeout(env, max_episode_steps=task_cfg["max_steps"])

    # Expose config knobs used elsewhere
    cfg.obs_per_agent = _OBS_PER_CAV  # 35 (incl. following-mode and 4+2 signal state)
    cfg.act_per_agent = _ACT_PER_CAV  # 2

    # TD-MPC2 discount config
    cfg.episodic = True
    cfg.discount_max = 0.99
    cfg.rho = 0.7

    return env
