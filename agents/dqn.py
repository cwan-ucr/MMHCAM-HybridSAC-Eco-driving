import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from agents.sac import AttentionAggregator


class DQNNet(nn.Module):
    """Q-network over 27 joint discrete actions."""

    def __init__(
        self,
        obs_dim,
        feature_dim,
        hidden,
        num_actions,
        num_heads=4,
        attention=True,
        include_self_token=False,
        fusion_gate=True,
        input_dropout=0.0,
    ):
        super().__init__()
        self.attn = AttentionAggregator(
            obs_dim,
            feature_dim,
            num_heads,
            attention=attention,
            include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, own_obs, ctx_obs, ctx_mask):
        return self.q(self.attn(own_obs, ctx_obs, ctx_mask))


class DQN(nn.Module):
    """
    DQN baseline with 27 discrete actions:
      9 longitudinal physical accelerations [-5, ..., 3] m/s^2
      × 3 lateral commands [-1:right, 0:keep, +1:left].
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.get_default_device()
        self.num_actions = 27
        self.num_accel_bins = 9
        self.num_lane_bins = 3

        obs_dim = cfg.obs_per_agent
        hidden = int(getattr(cfg, 'dqn_hidden_dim', getattr(cfg, 'sac_hidden_dim', cfg.mlp_dim)))
        feature_dim = int(getattr(cfg, 'dqn_feature_dim', getattr(cfg, 'sac_feature_dim', cfg.latent_dim)))
        use_attention = bool(getattr(cfg, 'attention', True)) and bool(getattr(cfg, 'communication', True))
        include_self_token = bool(getattr(cfg, 'sac_attention_include_self_token', False))
        fusion_gate = bool(getattr(cfg, 'attention_fusion_gate', True))
        input_dropout = float(getattr(cfg, 'dqn_input_dropout', 0.0) or 0.0)

        self.q = DQNNet(
            obs_dim,
            feature_dim,
            hidden,
            self.num_actions,
            num_heads=cfg.attention_heads,
            attention=use_attention,
            include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q_target = DQNNet(
            obs_dim,
            feature_dim,
            hidden,
            self.num_actions,
            num_heads=cfg.attention_heads,
            attention=use_attention,
            include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q_target.load_state_dict(self.q.state_dict())

        # Physical acceleration bins and their env-normalized command values.
        accel_phys = torch.arange(-5.0, 4.0, 1.0)
        accel_cmd = torch.where(accel_phys < 0, accel_phys / 5.0, accel_phys / 3.0)
        lane_cmd = torch.tensor([-1.0, 0.0, 1.0])
        actions = []
        for lane in lane_cmd:
            for accel in accel_cmd:
                actions.append(torch.stack([accel, lane]))
        self.register_buffer("_action_table", torch.stack(actions, dim=0))
        self.register_buffer("_accel_cmd_bins", accel_cmd)
        self.register_buffer("_lane_cmd_bins", lane_cmd)

        self.to(self.device)
        lr = float(getattr(cfg, 'dqn_lr', cfg.lr))
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        frac = cfg.episode_length / cfg.discount_denom
        self.discount = min(max((frac - 1) / frac, cfg.discount_min), cfg.discount_max)
        self.tau = float(getattr(cfg, 'dqn_tau', getattr(cfg, 'tau', 0.005)))
        self.target_update_freq = int(getattr(cfg, 'dqn_target_update_freq', 1))
        self.eps_start = float(getattr(cfg, 'dqn_epsilon_start', 1.0))
        self.eps_end = float(getattr(cfg, 'dqn_epsilon_end', 0.05))
        self.eps_decay_steps = max(int(getattr(cfg, 'dqn_epsilon_decay_steps', 100000)), 1)
        self.max_grad_norm = float(getattr(cfg, 'grad_clip_norm', 20.0))
        self._update_count = 0
        self._act_steps = 0
        self._model_summary = nn.ModuleDict({'q': self.q})
        print(
            f'DQN agent | device: {self.device} | discount: {self.discount:.4f} | '
            f'hidden_dim: {hidden} | feature_dim: {feature_dim} | actions: {self.num_actions}'
        )

    @property
    def model(self):
        return self._model_summary

    def _epsilon(self):
        frac = min(float(self._act_steps) / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _action_from_index(self, action_idx):
        return self._action_table[action_idx]

    def _action_index_from_env_action(self, action):
        accel = action[:, :1]
        lane = action[:, 1:2]
        accel_dist = (accel - self._accel_cmd_bins.view(1, -1)).abs()
        lane_dist = (lane - self._lane_cmd_bins.view(1, -1)).abs()
        accel_idx = accel_dist.argmin(dim=1)
        lane_idx = lane_dist.argmin(dim=1)
        return lane_idx * self.num_accel_bins + accel_idx

    @torch.no_grad()
    def act(self, obs_dict, t0=False, eval_mode=False, task=None):
        cav_ids = obs_dict['cav_ids']
        N = len(cav_ids)
        if N == 0:
            return torch.zeros((0, self.cfg.act_per_agent))

        own = obs_dict['obs'].to(self.device, non_blocking=True)
        ctx = obs_dict['ctx_obs'].to(self.device, non_blocking=True)
        mask = obs_dict['ctx_mask'].to(self.device, non_blocking=True)
        q = self.q(own, ctx, mask)
        greedy = q.argmax(dim=-1)

        if eval_mode:
            return self._action_from_index(greedy).cpu()

        eps = self._epsilon()
        random_idx = torch.randint(0, self.num_actions, (N,), device=self.device)
        explore = torch.rand(N, device=self.device) < eps
        action_idx = torch.where(explore, random_idx, greedy)
        self._act_steps += 1
        return self._action_from_index(action_idx).cpu()

    @torch.no_grad()
    def _soft_update(self):
        for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
            tp.data.lerp_(p.data, self.tau)

    def update(self, buffer):
        if buffer.num_eps == 0:
            return {}

        obs, ctx_obs, ctx_mask, _, _, action, reward, terminated = buffer.sample()
        H, B = action.shape[0], action.shape[1]
        s_own = obs[:-1].reshape(H * B, -1)
        s_ctx = ctx_obs[:-1].reshape(H * B, *ctx_obs.shape[2:])
        s_mask = ctx_mask[:-1].reshape(H * B, -1)
        ns_own = obs[1:].reshape(H * B, -1)
        ns_ctx = ctx_obs[1:].reshape(H * B, *ctx_obs.shape[2:])
        ns_mask = ctx_mask[1:].reshape(H * B, -1)
        a = action.reshape(H * B, -1)
        r = reward.reshape(H * B, 1)
        d = terminated.reshape(H * B, 1)

        action_idx = self._action_index_from_env_action(a).unsqueeze(-1)
        q_pred = self.q(s_own, s_ctx, s_mask).gather(1, action_idx)
        with torch.no_grad():
            next_q = self.q_target(ns_own, ns_ctx, ns_mask).max(dim=1, keepdim=True).values
            target = r + (1.0 - d) * self.discount * next_q

        loss = F.smooth_l1_loss(q_pred, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optim.step()
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._soft_update()

        return TensorDict({
            'dqn_loss': loss.item(),
            'dqn_q_mean': q_pred.mean().item(),
            'dqn_target_mean': target.mean().item(),
            'dqn_epsilon': self._epsilon(),
        }, batch_size=()).detach()

    def save(self, fp):
        torch.save({"model": self.state_dict()}, fp)

    def load(self, fp):
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict.get("model", state_dict))
