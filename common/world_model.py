from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from common.layers import VehicleEncoder
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	DTDE TD-MPC2 world model for multi-vehicle CAV control.

	Each vehicle is an independent agent. The only cross-vehicle coupling
	is the attention-based encoder (VehicleEncoder), which lets the ego
	vehicle attend over its co-existing neighbors to handle partial
	observability. Dynamics, reward, policy, and Q are all standard
	single-agent TD-MPC2 modules operating on a single vehicle's latent.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._communication = bool(getattr(cfg, 'communication', True))
		encoder_attention = bool(getattr(cfg, 'attention', True)) and self._communication

		# ── Attention-based per-vehicle encoder ───────────────────────
		self._encoder = VehicleEncoder(
			obs_dim=cfg.obs_per_agent,
			embed_dim=cfg.enc_dim,
			latent_dim=cfg.latent_dim,
			num_heads=cfg.attention_heads,
			vid_cycle=cfg.vid_cycle,
			simnorm_dim=cfg.simnorm_dim,
			attention=encoder_attention,
			fusion_gate=bool(getattr(cfg, 'attention_fusion_gate', True)),
		)

		# ── Single-agent dynamics / reward / policy / Q ───────────────
		self._dynamics = layers.mlp(
			cfg.latent_dim + cfg.act_per_agent,
			2 * [cfg.mlp_dim],
			cfg.latent_dim,
			act=layers.SimNorm(cfg),
		)

		# ── Optional cross-vehicle interaction layer (Level 1) ─────────
		# When enabled, every dynamics step is followed by a cross-attention
		# update where the ego latent attends to its K context vehicles.
		# Disabled by default (Level 0 backward compat).
		self._use_interaction = bool(getattr(cfg, 'use_interaction', False)) and self._communication
		if self._use_interaction:
			# Encode raw ctx_obs into latent-dim tokens (per-vehicle MLP, no attention)
			self._ctx_obs_proj = layers.mlp(
				cfg.obs_per_agent,
				[cfg.mlp_dim],
				cfg.latent_dim,
				act=layers.SimNorm(cfg),
			)
			self._inter_attn = nn.MultiheadAttention(
				cfg.latent_dim, cfg.attention_heads, batch_first=True
			)
			self._inter_norm = nn.LayerNorm(cfg.latent_dim)

		self._reward = layers.mlp(
			cfg.latent_dim + cfg.act_per_agent,
			2 * [cfg.mlp_dim],
			max(cfg.num_bins, 1),
		)
		self._pi = layers.mlp(
			cfg.latent_dim,
			2 * [cfg.mlp_dim],
			2 * cfg.act_per_agent,
		)
		self._Qs = layers.Ensemble([
			layers.mlp(
				cfg.latent_dim + cfg.act_per_agent,
				2 * [cfg.mlp_dim],
				max(cfg.num_bins, 1),
				dropout=cfg.dropout,
			)
			for _ in range(cfg.num_q)
		])

		self._termination = layers.mlp(
			cfg.latent_dim, 2 * [cfg.mlp_dim], 1
		) if cfg.episodic else None

		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Target & detach Q params (mirrors original TD-MPC2 API)
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def __repr__(self):
		repr = 'DTDE TD-MPC2 World Model\n'
		modules = ['Encoder (VehicleEncoder)', 'Dynamics', 'Reward',
				   'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward,
							   self._termination, self._pi, self._Qs]):
			if m is self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def train(self, mode=True):
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	# ── Encoder ────────────────────────────────────────────────────────

	def encode(self, own_obs, ctx_obs, ctx_mask, entry_phase_id, progress_id):
		"""
		Args:
			own_obs:         (..., obs_per_agent)
			ctx_obs:         (..., K, obs_per_agent)
			ctx_mask:        (..., K)
			entry_phase_id:  (...,) long
			progress_id:     (...,) long
		Returns:
			z: (..., latent_dim)
		"""
		return self._encoder(own_obs, ctx_obs, ctx_mask, entry_phase_id, progress_id)

	# ── Dynamics / Reward / Termination / Policy (single-agent) ───────

	def next(self, z, a, ctx_latents=None, ctx_mask=None):
		"""Predict next latent state. Single-agent dynamics by default;
		if interaction is enabled and ctx_latents+ctx_mask are provided,
		the dynamics output is fused with a cross-attention update over
		the K context vehicles (Level 1 interaction).
		"""
		tmp = self._dynamics(torch.cat([z, a], dim=-1))

		if (
			not self._use_interaction
			or ctx_latents is None
			or ctx_mask is None
		):
			return tmp

		# Cross-attention update: ego (query) attends to K context tokens
		leading = tmp.shape[:-1]
		latent_dim = tmp.shape[-1]
		K = ctx_latents.shape[-2]

		tmp_flat = tmp.reshape(-1, latent_dim)               # (B, latent)
		ctx_flat = ctx_latents.reshape(-1, K, latent_dim)     # (B, K, latent)
		mask_flat = ctx_mask.reshape(-1, K)                    # (B, K)

		q = tmp_flat.unsqueeze(1)                              # (B, 1, latent)
		kpm = ~mask_flat.bool()
		# Guard against fully-masked rows so attention doesn't return NaN.
		# We temporarily unmask the first slot to keep softmax well-defined,
		# then zero the attention output for those rows so a ghost neighbor
		# does not leak into the residual update.
		all_masked = kpm.all(dim=-1)                           # (B,)
		if all_masked.any():
			kpm = kpm.clone()
			kpm[all_masked, 0] = False

		attn_out, _ = self._inter_attn(q, ctx_flat, ctx_flat, key_padding_mask=kpm)
		attn_out = attn_out.squeeze(1)                         # (B, latent)
		# Zero out rows that actually had no valid neighbors — their attention
		# output is a spurious "ghost neighbor" value and must not be added
		# via residual.
		attn_out = attn_out * (~all_masked).float().unsqueeze(-1)

		fused = self._inter_norm(tmp_flat + attn_out)          # residual + norm
		return fused.reshape(*leading, latent_dim)

	def encode_ctx(self, ctx_obs):
		"""Encode raw context observations to latent tokens (per-vehicle MLP).

		Returns None when interaction is disabled, so callers can skip work.

		Args:
			ctx_obs: (..., K, obs_per_agent)
		Returns:
			(..., K, latent_dim) or None
		"""
		if not self._use_interaction:
			return None
		return self._ctx_obs_proj(ctx_obs)

	def roll_neighbors(self, ctx_latents):
		"""Roll neighbor latents forward one step using the shared policy
		+ dynamics. Used during MPC imagination when we don't have future
		neighbor observations.

		This reuses _pi and _dynamics directly (no new parameters). The
		policy mean is used (not a sample) to reduce variance in the
		MPC value estimate. No cross-vehicle interaction is applied to
		the neighbors themselves — each neighbor evolves with plain
		single-agent dynamics.

		Args:
			ctx_latents: (..., K, latent_dim) or None
		Returns:
			Rolled latents with the same shape, or None if input was None.
		"""
		if ctx_latents is None:
			return None
		leading = ctx_latents.shape[:-2]
		K = ctx_latents.shape[-2]
		latent_dim = ctx_latents.shape[-1]
		flat = ctx_latents.reshape(-1, latent_dim)                 # (B*K, latent)
		_, info = self.pi(flat)
		neighbor_a = info["mean"]                                   # (B*K, A)
		next_flat = self._dynamics(torch.cat([flat, neighbor_a], dim=-1))
		return next_flat.reshape(*leading, K, latent_dim)

	def interaction_parameters(self):
		"""Iterable of interaction-related params for the optimizer.
		Empty list when interaction is disabled."""
		if not self._use_interaction:
			return []
		return (
			list(self._ctx_obs_proj.parameters())
			+ list(self._inter_attn.parameters())
			+ list(self._inter_norm.parameters())
		)

	def reward(self, z, a):
		return self._reward(torch.cat([z, a], dim=-1))

	def termination(self, z, unnormalized=False):
		t = self._termination(z)
		return t if unnormalized else torch.sigmoid(t)

	def pi(self, z):
		"""Sample action from per-vehicle Gaussian policy prior."""
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		log_prob = math.gaussian_logprob(eps, log_std)
		size = eps.shape[-1]
		scaled_log_prob = log_prob * size

		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	# ── Q-function (single-agent ensemble) ────────────────────────────

	def Q(self, z, a, return_type='min', target=False, detach=False):
		"""Standard single-agent Q: (B, latent+action) -> (B, num_bins)."""
		assert return_type in {'min', 'avg', 'all'}

		za = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(za)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
