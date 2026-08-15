"""
Standard SAC (Soft Actor-Critic) baseline for DTDE CAV control.

Each CAV is an independent agent with shared parameters. Actor and critic
are FULLY INDEPENDENT networks (no shared encoder). Each network uses its
own attention module to aggregate the K nearest-neighbor observations into
a single feature vector, which is then processed by a standard SAC MLP.

This is a clean baseline — no cycle embeddings, no target encoder, no
feature sharing. Only the Q-function has a target (standard SAC practice).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


# ──────────────────────────────────────────────────────────────────────
# Attention aggregator: own_obs + K neighbors → fixed-size feature
# ──────────────────────────────────────────────────────────────────────

class AttentionAggregator(nn.Module):
    """
    Single cross-attention block. Ego vehicle's observation is projected to
    a feature, then attends over its K neighbors (plus itself, to handle
    the no-neighbor case cleanly). Output = residual(ego_feature, attn_out).
    """

    def __init__(
        self,
        obs_dim,
        feature_dim,
        num_heads=4,
        attention=True,
        include_self_token=False,
        fusion_gate=True,
        input_dropout=0.0,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.own_proj = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LeakyReLU()
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LeakyReLU()
        )
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True) if attention else None
        self.include_self_token = include_self_token
        self.fusion_gate = fusion_gate
        self.gate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.Sigmoid(),
        ) if fusion_gate else None
        self.ln = nn.LayerNorm(feature_dim)
        self.last_gate_mean = float("nan")
        self.last_ego_attn_weight = float("nan")

    def forward(self, own_obs, ctx_obs, ctx_mask):
        """
        own_obs:  (B, obs_dim)
        ctx_obs:  (B, K, obs_dim)
        ctx_mask: (B, K)   1.0 = valid neighbor, 0.0 = padded
        returns:  (B, feature_dim)

        Attention Q = ego, K/V = neighbors ONLY (no self-token). This
        prevents the ego from dominating its own attention weights via
        the trivial self-similarity. Ego information is re-injected
        through the residual connection below.
        """
        if self.attn is None:
            self.last_gate_mean = float("nan")
            self.last_ego_attn_weight = float("nan")
            return self.own_proj(self.input_dropout(own_obs))  # no attention, just project own obs
        own_obs = self.input_dropout(own_obs)
        ctx_obs = self.input_dropout(ctx_obs)
        q  = self.own_proj(own_obs).unsqueeze(1)       # (B, 1, F)
        ctx_kv = self.ctx_proj(ctx_obs)                # (B, K, F)

        key_padding_mask = ~ctx_mask.bool()            # (B, K), True = padded
        if self.include_self_token:
            # Add ego token into K/V so attention can explicitly attend to self.
            kv = torch.cat([q, ctx_kv], dim=1)         # (B, K+1, F)
            self_mask = torch.zeros((q.shape[0], 1), dtype=torch.bool, device=q.device)
            key_padding_mask = torch.cat([self_mask, key_padding_mask], dim=1)  # (B, K+1)
        else:
            kv = ctx_kv

        # When a row has NO valid neighbors, softmax over all-True mask → NaN.
        # Detect those rows, temporarily unmask the first slot to keep the
        # softmax well-defined, then zero out their attention output.
        no_neighbor = key_padding_mask.all(dim=-1)     # (B,)
        if no_neighbor.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[no_neighbor, 0] = False

        attn_out, attn_w = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        attn_out = attn_out.squeeze(1)                 # (B, F)
        # Zero the aggregated-neighbor feature for rows with no real neighbors
        attn_out = attn_out * (~no_neighbor).float().unsqueeze(-1)

        own = q.squeeze(1)
        if self.gate is not None:
            gate = self.gate(torch.cat([own, attn_out], dim=-1))  # (B, F)
            fused = own + gate * attn_out
            self.last_gate_mean = float(gate.detach().mean().item())
        else:
            fused = own + attn_out
            self.last_gate_mean = float("nan")
        if self.include_self_token:
            # attn_w: (B, 1, S) averaged across heads in PyTorch MHA
            self.last_ego_attn_weight = float(attn_w[:, 0, 0].detach().mean().item())
        else:
            self.last_ego_attn_weight = float("nan")
        return self.ln(fused)        # residual (ego) + gated-neighbor + LN


# ──────────────────────────────────────────────────────────────────────
# Actor — squashed Gaussian policy
# ──────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(
        self,
        obs_dim,
        feature_dim,
        act_dim,
        hidden,
        num_heads=4,
        attention=True,
        include_self_token=False,
        fusion_gate=True,
        input_dropout=0.0,
        discrete_lane_action=False,
    ):
        super().__init__()
        self.discrete_lane_action = bool(discrete_lane_action)
        self.cont_dim = 1 if self.discrete_lane_action else act_dim
        self.attn = AttentionAggregator(
            obs_dim, feature_dim, num_heads, attention=attention,
            include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.trunk_input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
        )
        self.mean = nn.Linear(hidden, self.cont_dim)
        self.log_std = nn.Linear(hidden, self.cont_dim)
        self.lane_logits = nn.Linear(hidden, 3) if self.discrete_lane_action else None

    def forward(self, own_obs, ctx_obs, ctx_mask):
        z = self.attn(own_obs, ctx_obs, ctx_mask)
        z = self.trunk_input_dropout(z)
        h = self.trunk(z)
        logits = self.lane_logits(h) if self.lane_logits is not None else None
        return self.mean(h), self.log_std(h).clamp(-5, 2), logits


# ──────────────────────────────────────────────────────────────────────
# Critic — single Q-network with its own attention module
# ──────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(
        self,
        obs_dim,
        feature_dim,
        act_dim,
        hidden,
        num_heads=4,
        attention=True,
        include_self_token=False,
        fusion_gate=True,
        input_dropout=0.0,
    ):
        super().__init__()
        self.attn = AttentionAggregator(
            obs_dim, feature_dim, num_heads, attention=attention,
            include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q_input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.q = nn.Sequential(
            nn.Linear(feature_dim + act_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def encode_state(self, own_obs, ctx_obs, ctx_mask):
        z = self.attn(own_obs, ctx_obs, ctx_mask)
        return z

    def q_from_z(self, z, action):
        q_in = self.q_input_dropout(torch.cat([z, action], dim=-1))
        return self.q(q_in)

    def forward(self, own_obs, ctx_obs, ctx_mask, action):
        return self.q_from_z(self.encode_state(own_obs, ctx_obs, ctx_mask), action)


# ──────────────────────────────────────────────────────────────────────
# SAC agent
# ──────────────────────────────────────────────────────────────────────

class SAC(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.get_default_device()

        obs_dim = cfg.obs_per_agent
        act_dim = cfg.act_per_agent
        self.discrete_lane_action = bool(getattr(cfg, 'lane_action_discrete', False))
        if self.discrete_lane_action and act_dim != 2:
            raise ValueError("lane_action_discrete expects env actions [accel_cmd, lane_cmd].")
        critic_act_dim = 4 if self.discrete_lane_action else act_dim
        hidden = int(getattr(cfg, 'sac_hidden_dim', cfg.mlp_dim))
        feature_dim = int(getattr(cfg, 'sac_feature_dim', cfg.latent_dim))
        num_heads = cfg.attention_heads
        use_attention = bool(getattr(cfg, 'attention', True)) and bool(getattr(cfg, 'communication', True))
        include_self_token = bool(getattr(cfg, 'sac_attention_include_self_token', False))
        fusion_gate = bool(getattr(cfg, 'attention_fusion_gate', True))
        input_dropout = float(getattr(cfg, 'sac_input_dropout', 0.0) or 0.0)
        self.sac_log_freq = max(int(getattr(cfg, 'sac_log_freq', 50) or 1), 1)
        self._update_count = 0
        self._reuse_discrete_critic_features = self.discrete_lane_action and input_dropout <= 0.0

        # Actor + two independent critics (standard twin-Q SAC)
        self.actor = Actor(
            obs_dim, feature_dim, act_dim, hidden, num_heads,
            attention=use_attention, include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
            discrete_lane_action=self.discrete_lane_action,
        )
        self.q1 = Critic(
            obs_dim, feature_dim, critic_act_dim, hidden, num_heads,
            attention=use_attention, include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q2 = Critic(
            obs_dim, feature_dim, critic_act_dim, hidden, num_heads,
            attention=use_attention, include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q1_target = Critic(
            obs_dim, feature_dim, critic_act_dim, hidden, num_heads,
            attention=use_attention, include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q2_target = Critic(
            obs_dim, feature_dim, critic_act_dim, hidden, num_heads,
            attention=use_attention, include_self_token=include_self_token,
            fusion_gate=fusion_gate,
            input_dropout=input_dropout,
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Automatic entropy tuning. In hybrid mode, acceleration and lane
        # entropy use separate temperatures because their exploration scales
        # are different (continuous tanh-Gaussian vs 3-way categorical).
        self.log_alpha = nn.Parameter(0.12 * torch.ones(1))
        if self.discrete_lane_action:
            self.log_alpha_lane = nn.Parameter(0.12 * torch.ones(1))
            self.target_entropy_acc = float(getattr(cfg, 'sac_target_entropy_acc', -0.98))
            self.target_entropy_lane = float(getattr(cfg, 'sac_target_entropy_lane', -0.98))
            self.target_entropy = self.target_entropy_acc + self.target_entropy_lane
        else:
            self.log_alpha_lane = None
            self.target_entropy = -0.98 * float(act_dim)
        self.register_buffer("_lane_values", torch.tensor([-1.0, 0.0, 1.0]))

        self.to(self.device)

        sac_lr = getattr(cfg, 'sac_lr', cfg.lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=sac_lr, weight_decay=1e-4)
        self.critic_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=sac_lr,
            weight_decay=1e-4)
        alpha_params = [self.log_alpha]
        if self.discrete_lane_action:
            alpha_params.append(self.log_alpha_lane)
        self.alpha_optim = torch.optim.Adam(alpha_params, lr=sac_lr)

        # Discount
        frac = cfg.episode_length / cfg.discount_denom
        self.discount = min(max((frac - 1) / frac, cfg.discount_min), cfg.discount_max)
        self.tau = getattr(cfg, 'sac_tau', cfg.tau)

        # Non-recursive summary for Trainer's print(agent.model)
        self._model_summary = nn.ModuleDict({
            'actor': self.actor, 'q1': self.q1, 'q2': self.q2,
        })
        print(
            f'SAC agent | device: {self.device} | discount: {self.discount:.4f} | '
            f'hidden_dim: {hidden} | feature_dim: {feature_dim}'
            f' | lane_action: {"categorical" if self.discrete_lane_action else "continuous"}'
        )

    @property
    def model(self):
        return self._model_summary

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def alpha_acc(self):
        return self.log_alpha.exp()

    @property
    def alpha_lane(self):
        return self.log_alpha_lane.exp() if self.discrete_lane_action else self.alpha

    # ── Stochastic policy with tanh squashing ─────────────────────────

    def _sample_continuous(self, mean, log_std):
        """Reparameterized tanh-squashed Gaussian sample + log-prob."""
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        # Numerically stable log-prob correction for tanh squashing:
        # log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        log_prob = dist.log_prob(x).sum(-1, keepdim=True)
        log_prob -= (2 * (torch.log(torch.tensor(2.0, device=x.device)) - x
                          - F.softplus(-2 * x))).sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def _lane_probs(self, lane_logits):
        probs = F.softmax(lane_logits, dim=-1)
        logp = F.log_softmax(lane_logits, dim=-1)
        return probs, logp

    def _lane_one_hot(self, lane_cmd):
        lane_idx = torch.round(lane_cmd.squeeze(-1) + 1.0).long().clamp(0, 2)
        return F.one_hot(lane_idx, num_classes=3).float()

    def _critic_action(self, action):
        if not self.discrete_lane_action:
            return action
        accel = action[..., :1]
        lane_one_hot = self._lane_one_hot(action[..., 1:2])
        return torch.cat([accel, lane_one_hot], dim=-1)

    def _env_action_from_lane_idx(self, accel, lane_idx):
        lane_cmd = self._lane_values[lane_idx].unsqueeze(-1)
        return torch.cat([accel, lane_cmd], dim=-1)

    def _sample_action(self, mean, log_std, lane_logits=None):
        """Sample env action. With discrete lane mode: [accel, {-1,0,+1}]."""
        cont_action, logp_cont, cont_mean = self._sample_continuous(mean, log_std)
        if not self.discrete_lane_action:
            return cont_action, logp_cont, cont_mean

        dist = torch.distributions.Categorical(logits=lane_logits)
        lane_idx = dist.sample()
        logp_lane = dist.log_prob(lane_idx).unsqueeze(-1)
        action = self._env_action_from_lane_idx(cont_action, lane_idx)
        greedy_lane = lane_logits.argmax(dim=-1)
        mean_action = self._env_action_from_lane_idx(cont_mean, greedy_lane)
        return action, logp_cont + logp_lane, mean_action

    def _expected_next_value(self, own, ctx, mask, mean, log_std, lane_logits):
        """Exact categorical expectation over lane actions for SAC targets."""
        accel, logp_accel, _ = self._sample_continuous(mean, log_std)
        probs, logp_lane = self._lane_probs(lane_logits)
        B = accel.shape[0]
        lane_values = self._lane_values.view(1, 3, 1).expand(B, 3, 1)
        accel_all = accel.unsqueeze(1).expand(B, 3, 1)
        action_all = torch.cat([accel_all, lane_values], dim=-1).reshape(B * 3, 2)
        critic_action = self._critic_action(action_all)
        if self._reuse_discrete_critic_features:
            z1 = self.q1_target.encode_state(own, ctx, mask).repeat_interleave(3, dim=0)
            z2 = self.q2_target.encode_state(own, ctx, mask).repeat_interleave(3, dim=0)
            q1 = self.q1_target.q_from_z(z1, critic_action).view(B, 3, 1)
            q2 = self.q2_target.q_from_z(z2, critic_action).view(B, 3, 1)
        else:
            own_all = own.repeat_interleave(3, dim=0)
            ctx_all = ctx.repeat_interleave(3, dim=0)
            mask_all = mask.repeat_interleave(3, dim=0)
            q1 = self.q1_target(own_all, ctx_all, mask_all, critic_action).view(B, 3, 1)
            q2 = self.q2_target(own_all, ctx_all, mask_all, critic_action).view(B, 3, 1)
        q = torch.min(q1, q2)
        v = (
            probs.unsqueeze(-1)
            * (
                q
                - self.alpha_acc.detach() * logp_accel.unsqueeze(1)
                - self.alpha_lane.detach() * logp_lane.unsqueeze(-1)
            )
        ).sum(dim=1)
        return v

    def _actor_loss_discrete_lane(self, own, ctx, mask, mean, log_std, lane_logits):
        """Reparameterized accel + exact categorical lane-action actor loss."""
        accel, logp_accel, _ = self._sample_continuous(mean, log_std)
        probs, logp_lane = self._lane_probs(lane_logits)
        B = accel.shape[0]
        lane_values = self._lane_values.view(1, 3, 1).expand(B, 3, 1)
        accel_all = accel.unsqueeze(1).expand(B, 3, 1)
        action_all = torch.cat([accel_all, lane_values], dim=-1).reshape(B * 3, 2)
        critic_action = self._critic_action(action_all)
        if self._reuse_discrete_critic_features:
            z1 = self.q1.encode_state(own, ctx, mask).repeat_interleave(3, dim=0)
            z2 = self.q2.encode_state(own, ctx, mask).repeat_interleave(3, dim=0)
            q1 = self.q1.q_from_z(z1, critic_action).view(B, 3, 1)
            q2 = self.q2.q_from_z(z2, critic_action).view(B, 3, 1)
        else:
            own_all = own.repeat_interleave(3, dim=0)
            ctx_all = ctx.repeat_interleave(3, dim=0)
            mask_all = mask.repeat_interleave(3, dim=0)
            q1 = self.q1(own_all, ctx_all, mask_all, critic_action).view(B, 3, 1)
            q2 = self.q2(own_all, ctx_all, mask_all, critic_action).view(B, 3, 1)
        q = torch.min(q1, q2)
        expected_q = (probs.unsqueeze(-1) * q).sum(dim=1)
        expected_lane_logp = (probs * logp_lane).sum(dim=-1, keepdim=True)
        actor_loss = (
            self.alpha_acc.detach() * logp_accel
            + self.alpha_lane.detach() * expected_lane_logp
            - expected_q
        ).mean()
        total_logp = logp_accel + expected_lane_logp
        return actor_loss, logp_accel, expected_lane_logp, total_logp

    # ── Target networks soft update ───────────────────────────────────

    @torch.no_grad()
    def _soft_update(self):
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.lerp_(p.data, self.tau)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.lerp_(p.data, self.tau)

    # ── Action selection ─────────────────────────────────────────────

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
        if eval_mode:
            if self.discrete_lane_action:
                lane_idx = lane_logits.argmax(dim=-1)
                return self._env_action_from_lane_idx(torch.tanh(mean), lane_idx).cpu()
            return torch.tanh(mean).cpu()
        action, _, _ = self._sample_action(mean, log_std, lane_logits)
        return action.cpu()

    # ── Training update ──────────────────────────────────────────────

    def update(self, buffer):
        batch = buffer.sample()
        return self._update(*batch)

    def _update(self, obs, ctx_obs, ctx_mask, entry_phase_id, progress_id,
                action, reward, terminated):
        """
        Standard 1-step TD SAC update.

        Buffer returns (H+1=2, B, ...) windows with horizon=1 → B independent
        (s, a, r, s', d) transitions. entry_phase_id and progress_id are
        ignored (not needed for standard SAC).
        """
        self._update_count += 1
        H, B = action.shape[0], action.shape[1]

        # Flatten (H, B) sequence dim into a single batch
        s_own     = obs[:-1].reshape(H * B, -1)
        s_ctx     = ctx_obs[:-1].reshape(H * B, *ctx_obs.shape[2:])
        s_mask    = ctx_mask[:-1].reshape(H * B, -1)

        ns_own    = obs[1:].reshape(H * B, -1)
        ns_ctx    = ctx_obs[1:].reshape(H * B, *ctx_obs.shape[2:])
        ns_mask   = ctx_mask[1:].reshape(H * B, -1)

        a = action.reshape(H * B, -1)
        r = reward.reshape(H * B, 1)
        d = terminated.reshape(H * B, 1)

        # ── Critic update ────────────────────────────────────────────
        with torch.no_grad():
            ns_mean, ns_log_std, ns_lane_logits = self.actor(ns_own, ns_ctx, ns_mask)
            if self.discrete_lane_action:
                q_next = self._expected_next_value(
                    ns_own, ns_ctx, ns_mask, ns_mean, ns_log_std, ns_lane_logits
                )
            else:
                a_next, logp_next, _ = self._sample_action(ns_mean, ns_log_std)
                q1_next = self.q1_target(ns_own, ns_ctx, ns_mask, a_next)
                q2_next = self.q2_target(ns_own, ns_ctx, ns_mask, a_next)
                q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * logp_next
            q_target = r + (1 - d) * self.discount * q_next

        critic_a = self._critic_action(a)
        q1_pred = self.q1(s_own, s_ctx, s_mask, critic_a)
        q2_pred = self.q2(s_own, s_ctx, s_mask, critic_a)
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.cfg.grad_clip_norm,
        )
        self.critic_optim.step()

        # ── Actor update ─────────────────────────────────────────────
        mean, log_std, lane_logits = self.actor(s_own, s_ctx, s_mask)
        if self.discrete_lane_action:
            actor_loss, logp_acc_new, logp_lane_new, logp_new = self._actor_loss_discrete_lane(
                s_own, s_ctx, s_mask, mean, log_std, lane_logits
            )
        else:
            a_new, logp_new, _ = self._sample_action(mean, log_std)
            q1_new = self.q1(s_own, s_ctx, s_mask, a_new)
            q2_new = self.q2(s_own, s_ctx, s_mask, a_new)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha.detach() * logp_new - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.grad_clip_norm,
        )
        self.actor_optim.step()

        # ── Alpha update (automatic entropy tuning) ──────────────────
        if self.discrete_lane_action:
            alpha_acc_loss = -(
                self.log_alpha * (logp_acc_new.detach() + self.target_entropy_acc)
            ).mean()
            alpha_lane_loss = -(
                self.log_alpha_lane * (logp_lane_new.detach() + self.target_entropy_lane)
            ).mean()
            alpha_loss = alpha_acc_loss + alpha_lane_loss
        else:
            alpha_acc_loss = alpha_loss = -(
                self.log_alpha * (logp_new.detach() + self.target_entropy)
            ).mean()
            alpha_lane_loss = torch.zeros_like(alpha_loss)

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── Soft update target Q networks ────────────────────────────
        self._soft_update()

        if self._update_count % self.sac_log_freq != 0:
            return {}

        return TensorDict({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_acc_loss': alpha_acc_loss.item(),
            'alpha_lane_loss': alpha_lane_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_acc': self.alpha_acc.item(),
            'alpha_lane': self.alpha_lane.item(),
            'q_mean': q1_pred.mean().item(),
            'logp_mean': logp_new.mean().item(),
            'logp_acc_mean': (
                logp_acc_new.mean().item() if self.discrete_lane_action else logp_new.mean().item()
            ),
            'logp_lane_mean': (
                logp_lane_new.mean().item() if self.discrete_lane_action else 0.0
            ),
            'lane_entropy_mean': (
                (-logp_lane_new).mean().item() if self.discrete_lane_action else 0.0
            ),
            'attn_gate_mean_actor': float(self.actor.attn.last_gate_mean),
            'attn_ego_weight_actor': float(self.actor.attn.last_ego_attn_weight),
            'attn_gate_mean_q1': float(self.q1.attn.last_gate_mean),
            'attn_ego_weight_q1': float(self.q1.attn.last_ego_attn_weight),
        }).detach()

    # ── Serialization ───────────────────────────────────────────────

    def save(self, fp):
        torch.save({"model": self.state_dict()}, fp)

    def load(self, fp, mode="full"):
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=self.device, weights_only=False)
        state_dict = state_dict.get("model", state_dict)
        mode = str(mode or "full").lower().replace("-", "_")
        if (
            self.discrete_lane_action
            and "log_alpha" in state_dict
            and "log_alpha_lane" not in state_dict
        ):
            state_dict = dict(state_dict)
            state_dict["log_alpha_lane"] = state_dict["log_alpha"].clone()

        if mode == "full":
            self.load_state_dict(state_dict)
            return

        if mode != "actor":
            raise ValueError(
                f"Unknown SAC checkpoint loading mode: {mode}. "
                "Expected full or actor."
            )

        actor_state = {
            key[len("actor."):]: value
            for key, value in state_dict.items()
            if key.startswith("actor.")
        }
        if not actor_state:
            raise KeyError("No actor.* parameters found in checkpoint.")
        self.actor.load_state_dict(actor_state, strict=True)
