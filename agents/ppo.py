import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from agents.sac import AttentionAggregator


class PPOActor(nn.Module):
    """Hybrid PPO policy: continuous acceleration head + categorical lane head."""

    def __init__(
        self,
        obs_dim,
        feature_dim,
        hidden,
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
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        self.lane_logits = nn.Linear(hidden, 3)

    def forward(self, own_obs, ctx_obs, ctx_mask):
        z = self.attn(own_obs, ctx_obs, ctx_mask)
        h = self.trunk(z)
        return self.mean(h), self.log_std(h).clamp(-5, 2), self.lane_logits(h)


class PPOCritic(nn.Module):
    """State-value network with its own encoder, separate from the actor."""

    def __init__(
        self,
        obs_dim,
        feature_dim,
        hidden,
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
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
        )
        self.value = nn.Linear(hidden, 1)

    def forward(self, own_obs, ctx_obs, ctx_mask):
        z = self.attn(own_obs, ctx_obs, ctx_mask)
        h = self.trunk(z)
        return self.value(h)


class PPO(nn.Module):
    """
    Per-vehicle trajectory PPO for the SUMO DTDE environment.

    The trainer records each CAV as one trajectory. PPO consumes recently
    completed trajectories from Buffer.pop_recent_trajectories().
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.get_default_device()

        obs_dim = cfg.obs_per_agent
        hidden = int(getattr(cfg, 'ppo_hidden_dim', getattr(cfg, 'sac_hidden_dim', cfg.mlp_dim)))
        feature_dim = int(getattr(cfg, 'ppo_feature_dim', getattr(cfg, 'sac_feature_dim', cfg.latent_dim)))
        use_attention = bool(getattr(cfg, 'attention', True)) and bool(getattr(cfg, 'communication', True))
        include_self_token = bool(getattr(cfg, 'sac_attention_include_self_token', False))
        fusion_gate = bool(getattr(cfg, 'attention_fusion_gate', True))
        input_dropout = float(getattr(cfg, 'ppo_input_dropout', 0.0) or 0.0)

        self.actor = PPOActor(
            obs_dim,
            feature_dim,
            hidden,
            num_heads=cfg.attention_heads,
            attention=use_attention,
            include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.critic = PPOCritic(
            obs_dim,
            feature_dim,
            hidden,
            num_heads=cfg.attention_heads,
            attention=use_attention,
            include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.register_buffer("_lane_values", torch.tensor([-1.0, 0.0, 1.0]))
        self.to(self.device)

        lr = float(getattr(cfg, 'ppo_lr', cfg.lr))
        actor_lr = float(getattr(cfg, 'ppo_actor_lr', lr))
        critic_lr = float(getattr(cfg, 'ppo_critic_lr', lr))
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        frac = cfg.episode_length / cfg.discount_denom
        self.discount = min(max((frac - 1) / frac, cfg.discount_min), cfg.discount_max)
        self.clip_ratio = float(getattr(cfg, 'ppo_clip_ratio', 0.2))
        self.value_coef = float(getattr(cfg, 'ppo_value_coef', 0.5))
        self.entropy_coef = float(getattr(cfg, 'ppo_entropy_coef', getattr(cfg, 'entropy_coef', 1e-3)))
        self.epochs = int(getattr(cfg, 'ppo_epochs', 4))
        self.minibatch_size = int(getattr(cfg, 'ppo_minibatch_size', cfg.batch_size))
        self.min_trajectories = int(getattr(cfg, 'ppo_min_trajectories', 16))
        self.max_grad_norm = float(getattr(cfg, 'grad_clip_norm', 20.0))
        self._last_transition_info = {}

        self._model_summary = nn.ModuleDict({'actor': self.actor, 'critic': self.critic})
        print(
            f'PPO agent | device: {self.device} | discount: {self.discount:.4f} | '
            f'hidden_dim: {hidden} | feature_dim: {feature_dim} | separate actor/critic encoders'
        )

    @property
    def model(self):
        return self._model_summary

    def _lane_action_from_idx(self, lane_idx):
        return self._lane_values[lane_idx].unsqueeze(-1)

    @staticmethod
    def _atanh(x):
        x = x.clamp(-0.999999, 0.999999)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _sample_action(self, mean, log_std, lane_logits):
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        accel = torch.tanh(x)
        logp_acc = normal.log_prob(x).sum(-1, keepdim=True)
        logp_acc -= (2 * (torch.log(torch.tensor(2.0, device=x.device)) - x - F.softplus(-2 * x))).sum(-1, keepdim=True)

        lane_dist = torch.distributions.Categorical(logits=lane_logits)
        lane_idx = lane_dist.sample()
        logp_lane = lane_dist.log_prob(lane_idx).unsqueeze(-1)
        action = torch.cat([accel, self._lane_action_from_idx(lane_idx)], dim=-1)
        return action, logp_acc + logp_lane

    def _evaluate_action(self, own, ctx, mask, action):
        mean, log_std, lane_logits = self.actor(own, ctx, mask)
        value = self.critic(own, ctx, mask)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        accel = action[:, :1].clamp(-0.999999, 0.999999)
        x = self._atanh(accel)
        logp_acc = normal.log_prob(x).sum(-1, keepdim=True)
        logp_acc -= (2 * (torch.log(torch.tensor(2.0, device=x.device)) - x - F.softplus(-2 * x))).sum(-1, keepdim=True)

        lane_idx = torch.round(action[:, 1] + 1.0).long().clamp(0, 2)
        lane_dist = torch.distributions.Categorical(logits=lane_logits)
        logp_lane = lane_dist.log_prob(lane_idx).unsqueeze(-1)
        entropy = normal.entropy().sum(-1, keepdim=True) + lane_dist.entropy().unsqueeze(-1)
        return logp_acc + logp_lane, entropy, value

    @torch.no_grad()
    def act(self, obs_dict, t0=False, eval_mode=False, task=None):
        cav_ids = obs_dict['cav_ids']
        N = len(cav_ids)
        if N == 0:
            return torch.zeros((0, self.cfg.act_per_agent))

        own = obs_dict['obs'].to(self.device, non_blocking=True)
        ctx = obs_dict['ctx_obs'].to(self.device, non_blocking=True)
        mask = obs_dict['ctx_mask'].to(self.device, non_blocking=True)
        mean, log_std, lane_logits = self.actor(own, ctx, mask)
        value = self.critic(own, ctx, mask)

        if eval_mode:
            lane_idx = lane_logits.argmax(dim=-1)
            action = torch.cat([torch.tanh(mean), self._lane_action_from_idx(lane_idx)], dim=-1)
            return action.cpu()

        action, log_prob = self._sample_action(mean, log_std, lane_logits)
        self._last_transition_info = {
            cav_id: {
                'log_prob': log_prob[i, 0].detach().cpu(),
                'value': value[i, 0].detach().cpu(),
            }
            for i, cav_id in enumerate(cav_ids)
        }
        return action.cpu()

    def get_transition_info(self, cav_id):
        return self._last_transition_info.pop(cav_id, None)

    def _trajectory_training_data(self, traj):
        td = traj.to(self.device)
        obs = td['obs'][:-1]
        ctx = td['ctx_obs'][:-1]
        mask = td['ctx_mask'][:-1]
        action = td['action'][1:]
        reward = td['reward'][1:]
        done = td['terminated'][1:]
        old_logp = td['log_prob'][1:]
        old_value = td['value'][1:]

        valid = (
            torch.isfinite(action).all(dim=-1)
            & torch.isfinite(reward)
            & torch.isfinite(old_logp)
            & torch.isfinite(old_value)
        )
        if valid.sum() == 0:
            return None

        obs, ctx, mask = obs[valid], ctx[valid], mask[valid]
        action = action[valid]
        reward = reward[valid]
        done = done[valid]
        old_logp = old_logp[valid].unsqueeze(-1)
        old_value = old_value[valid].unsqueeze(-1)

        returns = torch.zeros_like(reward).unsqueeze(-1)
        running = torch.zeros(1, device=self.device)
        if done[-1] < 0.5:
            with torch.no_grad():
                bootstrap_value = self.critic(
                    td['obs'][-1:].to(self.device),
                    td['ctx_obs'][-1:].to(self.device),
                    td['ctx_mask'][-1:].to(self.device),
                )
                running = bootstrap_value[0].detach()
        for t in reversed(range(reward.shape[0])):
            running = reward[t] + self.discount * running * (1.0 - done[t])
            returns[t, 0] = running
        adv = returns - old_value
        return obs, ctx, mask, action, old_logp, returns, adv

    def update(self, buffer):
        trajectories = buffer.pop_recent_trajectories(self.min_trajectories)
        if not trajectories:
            return {}

        batches = [self._trajectory_training_data(traj) for traj in trajectories]
        batches = [b for b in batches if b is not None]
        if not batches:
            return {}

        obs = torch.cat([b[0] for b in batches], dim=0)
        ctx = torch.cat([b[1] for b in batches], dim=0)
        mask = torch.cat([b[2] for b in batches], dim=0)
        action = torch.cat([b[3] for b in batches], dim=0)
        old_logp = torch.cat([b[4] for b in batches], dim=0)
        returns = torch.cat([b[5] for b in batches], dim=0)
        adv = torch.cat([b[6] for b in batches], dim=0)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = obs.shape[0]
        last_metrics = {}
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                logp, entropy, value = self._evaluate_action(obs[idx], ctx[idx], mask[idx], action[idx])
                ratio = torch.exp(logp - old_logp[idx])
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, returns[idx])
                entropy_loss = -entropy.mean()
                actor_loss = policy_loss + self.entropy_coef * entropy_loss
                critic_loss = self.value_coef * value_loss
                loss = actor_loss + critic_loss

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                last_metrics = {
                    'ppo_loss': loss.item(),
                    'ppo_policy_loss': policy_loss.item(),
                    'ppo_value_loss': value_loss.item(),
                    'ppo_entropy': entropy.mean().item(),
                    'ppo_num_samples': float(n),
                    'ppo_num_trajectories': float(len(trajectories)),
                }

        return TensorDict(last_metrics, batch_size=()).detach()

    def save(self, fp):
        torch.save({"model": self.state_dict()}, fp)

    def load(self, fp):
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict.get("model", state_dict))
