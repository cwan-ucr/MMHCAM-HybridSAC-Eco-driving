"""
Training script for TD-MPC2 CAV control at signalized intersection.

Usage:
    python train.py task=sumo-intersection
    python train.py task=sumo-intersection model_size=5 steps=500000
    python train.py task=sumo-intersection-dense steps=1000000
"""

import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

import torch
import hydra
from termcolor import colored

# ── Local project root takes priority so local tdmpc2.py is used ────────────
import sys
_LOCAL_ROOT = os.path.dirname(__file__)
sys.path.insert(0, _LOCAL_ROOT)

# ── Fallback: original TD-MPC2 codebase for shared utilities ────────────────
_TDMPC2_ROOT = os.path.join(_LOCAL_ROOT, '..', 'tdmpc2')
if os.path.isdir(_TDMPC2_ROOT):
    sys.path.append(_TDMPC2_ROOT)  # append so local modules always win

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.logger import Logger
from agents.tdmpc2 import TDMPC2
from agents.sac import SAC
from agents.ppo import PPO
from agents.dqn import DQN
from trainer.online_trainer import OnlineTrainer
from envs import make_env

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg):
    """
    Train a TD-MPC2 agent for CAV motion control.
    """
    # assert torch.cuda.is_available(), "CUDA is required for TD-MPC2 training."
    assert cfg.steps > 0, "Must train for at least 1 step."

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
    print(colored(f'Task: {cfg.task}', 'green', attrs=['bold']))

    # Create environment, agent, and buffer
    agent_type = str(getattr(cfg, 'agent', 'tdmpc2')).lower()
    cfg.agent = agent_type
    if agent_type == 'tdmpc2' and bool(getattr(cfg, 'lane_action_discrete', False)):
        raise NotImplementedError(
            "lane_action_discrete is for non-MPPI baselines. "
            "TD-MPC2/MPPI still uses continuous action sampling."
        )
    env = make_env(cfg)
    if agent_type in {'sac', 'dqn'}:
        # SAC/DQN are 1-step TD algorithms, so override horizon to 1 so the
        # buffer samples independent (s, a, r, s', d) pairs instead of
        # correlated multi-step windows. Also widen batch_size to keep
        # the number of gradient-updated transitions per step the same
        # as with horizon=5.
        original_horizon = max(int(cfg.horizon), 1)
        cfg.horizon = 1
        cfg.batch_size = int(cfg.batch_size) * original_horizon
        print(colored(
            f'[{agent_type.upper()}] horizon overridden to 1 for i.i.d. sampling; '
            f'batch_size scaled to {cfg.batch_size} '
            f'(= {cfg.batch_size // original_horizon} × {original_horizon})',
            'yellow',
        ))
        agent = SAC(cfg) if agent_type == 'sac' else DQN(cfg)
    elif agent_type == 'ppo':
        # PPO consumes complete per-vehicle trajectories. The replay sampler is
        # unused, but horizon=1 prevents short vehicle trajectories from being
        # dropped unnecessarily by the shared trajectory flush logic.
        cfg.horizon = 1
        print(colored(
            '[PPO] using complete per-vehicle trajectories; horizon set to 1 '
            'for trajectory recording.',
            'yellow',
        ))
        agent = PPO(cfg)
    elif agent_type == 'tdmpc2':
        agent = TDMPC2(cfg)
    else:
        raise ValueError(f'Unknown agent: {agent_type}. Expected tdmpc2, sac, ppo, or dqn.')

    # Create logger after make_env populates observation/action metadata, but
    # before checkpoint loading so the loading mode is recorded in console.log.
    logger = Logger(cfg)

    init_checkpoint = getattr(cfg, "init_checkpoint", None)
    if init_checkpoint:
        assert os.path.exists(init_checkpoint), f'Initial checkpoint {init_checkpoint} not found!'
        init_checkpoint_mode = str(getattr(cfg, "init_checkpoint_mode", "full") or "full")
        if agent_type == "sac":
            agent.load(init_checkpoint, mode=init_checkpoint_mode)
        else:
            if init_checkpoint_mode != "full":
                raise ValueError(
                    f"init_checkpoint_mode={init_checkpoint_mode} is only supported for SAC."
                )
            agent.load(init_checkpoint)
        print(colored(
            f'Loaded initial checkpoint ({init_checkpoint_mode}): {init_checkpoint}',
            'cyan',
        ))

    buffer = Buffer(cfg)

    print(colored('Environment created successfully', 'green'))
    print(f'  Observation space: {env.observation_space.shape}')
    print(f'  Action space:      {env.action_space.shape}')
    print(f'  Episode length:    {cfg.episode_length}')
    print(colored(f'  Device:            {agent.device}', 'cyan'))
    print(colored(f'  CUDA available:    {torch.cuda.is_available()}', 'cyan'))
    if torch.cuda.is_available():
        print(colored(f'  GPU:               {torch.cuda.get_device_name(0)}', 'cyan'))

    # Use the online trainer (single-task, online RL)
    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger,
    )
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
