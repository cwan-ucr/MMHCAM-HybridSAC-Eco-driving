import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	DTDE TD-MPC2 agent for multi-vehicle CAV control.

	Each active vehicle is an independent agent that shares parameters
	with every other vehicle. The only cross-vehicle coupling is the
	attention-based encoder; dynamics / reward / policy / Q are all
	single-agent. At execution time, all active vehicles are processed
	in parallel (n_active as the batch dimension).
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.get_default_device()
		self.model = WorldModel(cfg).to(self.device)

		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			# Interaction layer (Level 1) — empty list when use_interaction=false
			{'params': self.model.interaction_parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
		)
		self.model.eval()
		self.scale = RunningScale(cfg)

		# Heuristic for large per-vehicle action spaces (not really needed here since act_dim=2)
		self.cfg.iterations += 2 * int(cfg.act_per_agent >= 20)

		self.discount = self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)

		# MPPI memory: a dict keyed by SUMO cav_id -> (horizon, act_per_agent).
		# Entries are removed at the top of act() when their CAV leaves the network.
		self._prev_mean: dict = {}

	def _get_discount(self, episode_length):
		frac = episode_length / self.cfg.discount_denom
		return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict.get("model", state_dict)
		self.model.load_state_dict(state_dict)

	# ── Execution: per-vehicle batched action selection ───────────────

	@torch.no_grad()
	def act(self, obs_dict, t0=False, eval_mode=False, task=None):
		"""
		Compute actions for every active CAV in parallel.

		Args:
			obs_dict: {
				'obs':            (N, 19),
				'ctx_obs':        (N, K, 19),
				'ctx_mask':       (N, K),
				'entry_phase_id': (N,),
				'progress_id':    (N,),
				'cav_ids':        List[str] length N,
			}
			t0: first timestep of episode (reset MPPI memory)
			eval_mode: deterministic action
		Returns:
			action: (N, act_per_agent) tensor aligned row-wise with cav_ids
		"""
		cav_ids = obs_dict['cav_ids']
		N = len(cav_ids)

		# Purge MPPI memory for CAVs that no longer exist (or all if t0)
		if t0:
			self._prev_mean = {}
		else:
			current_set = set(cav_ids)
			for k in list(self._prev_mean.keys()):
				if k not in current_set:
					self._prev_mean.pop(k, None)

		if N == 0:
			return torch.zeros((0, self.cfg.act_per_agent))

		own_obs = obs_dict['obs'].to(self.device, non_blocking=True)             # (N, 19)
		ctx_obs = obs_dict['ctx_obs'].to(self.device, non_blocking=True)          # (N, K, 19)
		ctx_mask = obs_dict['ctx_mask'].to(self.device, non_blocking=True)        # (N, K)
		entry_phase_id = obs_dict['entry_phase_id'].to(self.device, non_blocking=True).long()
		progress_id = obs_dict['progress_id'].to(self.device, non_blocking=True).long()

		if self.cfg.mpc:
			actions = self._plan(
				own_obs, ctx_obs, ctx_mask, entry_phase_id, progress_id,
				cav_ids, t0=t0, eval_mode=eval_mode,
			)
		else:
			z = self.model.encode(own_obs, ctx_obs, ctx_mask, entry_phase_id, progress_id)
			actions, info = self.model.pi(z)
			if eval_mode:
				actions = info["mean"]

		return actions.cpu()

	# ── MPPI planning (per-vehicle, vectorized over active vehicles) ──

	@torch.no_grad()
	def _estimate_value(self, z, actions, ctx_latents=None, ctx_mask=None):
		"""
		Single-agent MPPI value estimate, with optional Level-1 interaction
		and optional rolling-ctx imagination. When episodic, predicted
		termination gates phantom rewards and the terminal Q bootstrap.

		Args:
			z:           (V*S, latent) — V = n_active vehicles, S = num_samples
			actions:     (horizon, V*S, act_per_agent)
			ctx_latents: (V*S, K, latent) or None
			ctx_mask:    (V*S, K) or None
		Returns:
			value: (V*S, 1)
		"""
		G, discount = 0, 1
		ctx = ctx_latents   # local handle, possibly rolled each step
		roll = bool(getattr(self.cfg, 'roll_neighbors_in_mpc', False))
		episodic = bool(getattr(self.cfg, 'episodic', False))

		# Sticky termination mask: once a sample is predicted to terminate,
		# it stays terminated for the rest of the rollout.
		termination = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=z.device) \
			if episodic else None

		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t]), self.cfg)
			z = self.model.next(z, actions[t], ctx_latents=ctx, ctx_mask=ctx_mask)
			if roll and ctx is not None:
				ctx = self.model.roll_neighbors(ctx)

			if episodic:
				# Zero out rewards after the (predicted) terminal step
				G = G + discount * (1.0 - termination) * reward
				# Update sticky termination based on the NEW latent z_{t+1}
				term_prob = self.model.termination(z)           # sigmoid probability
				term_step = (term_prob > 0.5).float()
				termination = torch.clamp(termination + term_step, max=1.0)
			else:
				G = G + discount * reward

			discount = discount * self.discount

		action, _ = self.model.pi(z)
		q_term = self.model.Q(z, action, return_type='avg')
		if episodic:
			q_term = (1.0 - termination) * q_term

		return G + discount * q_term

	@torch.no_grad()
	def _plan(
		self,
		own_obs,
		ctx_obs,
		ctx_mask,
		entry_phase_id,
		progress_id,
		cav_ids,
		t0=False,
		eval_mode=False,
	):
		"""
		Vectorized MPPI: each active CAV runs an independent MPPI in parallel.

		Args:
			own_obs:        (V, 19)
			ctx_obs:        (V, K, 19)
			ctx_mask:       (V, K)
			entry_phase_id: (V,) long
			progress_id:    (V,) long
			cav_ids:        List[str] length V (for MPPI memory dict lookup)
		Returns:
			action: (V, act_per_agent)
		"""
		V = own_obs.shape[0]
		A = self.cfg.act_per_agent
		H = self.cfg.horizon
		S = self.cfg.num_samples
		P = self.cfg.num_pi_trajs

		# Encode once per CAV (attention over K-NN context applied here)
		z0 = self.model.encode(
			own_obs, ctx_obs, ctx_mask, entry_phase_id, progress_id
		)  # (V, latent)

		# Encode ctx for Level-1 interaction (held constant throughout the
		# rollout — we don't have new ctx_obs for imagined future steps).
		# Returns None when use_interaction=false.
		ctx_lat_0 = self.model.encode_ctx(ctx_obs)   # (V, K, latent) or None

		# Policy rollouts for MPPI warm-start
		if P > 0:
			pi_actions = torch.empty(H, V, P, A, device=self.device)
			_z = z0.unsqueeze(1).expand(V, P, -1).reshape(V * P, -1)  # (V*P, latent)
			# Expand ctx for V*P samples (None-safe)
			if ctx_lat_0 is not None:
				K = ctx_obs.shape[1]
				_ctx = ctx_lat_0.unsqueeze(1).expand(V, P, K, -1).reshape(V * P, K, -1)
				_cmask = ctx_mask.unsqueeze(1).expand(V, P, K).reshape(V * P, K)
			else:
				_ctx, _cmask = None, None
			roll = bool(getattr(self.cfg, 'roll_neighbors_in_mpc', False))
			for t in range(H - 1):
				a_t, _ = self.model.pi(_z)             # (V*P, A)
				pi_actions[t] = a_t.view(V, P, A)
				_z = self.model.next(_z, a_t, ctx_latents=_ctx, ctx_mask=_cmask)
				if roll and _ctx is not None:
					_ctx = self.model.roll_neighbors(_ctx)
			a_t, _ = self.model.pi(_z)
			pi_actions[-1] = a_t.view(V, P, A)

		# Initialize mean/std per CAV using stored previous means (dict lookup)
		mean = torch.zeros(V, H, A, device=self.device)
		std = torch.full((V, H, A), self.cfg.max_std, device=self.device)
		if not t0:
			# Shift previous plan forward by 1 step for CAVs that still have memory
			for v, cav_id in enumerate(cav_ids):
				prev = self._prev_mean.get(cav_id)
				if prev is not None:
					mean[v, :-1] = prev[1:]

		# Expand latent for S samples: (V, S, latent) -> (V*S, latent)
		z_rep = z0.unsqueeze(1).expand(V, S, -1).reshape(V * S, -1)

		# Expand ctx_latents/ctx_mask for the V*S MPPI samples (None-safe)
		if ctx_lat_0 is not None:
			K = ctx_obs.shape[1]
			ctx_rep = ctx_lat_0.unsqueeze(1).expand(V, S, K, -1).reshape(V * S, K, -1)
			ctx_mask_rep = ctx_mask.unsqueeze(1).expand(V, S, K).reshape(V * S, K)
		else:
			ctx_rep, ctx_mask_rep = None, None

		# Outer loop: MPPI iterations
		for _ in range(self.cfg.iterations):
			# Sample actions: (H, V, S, A)
			n_rand = S - P
			r = torch.randn(H, V, n_rand, A, device=self.device)
			sampled = mean.permute(1, 0, 2).unsqueeze(2) + std.permute(1, 0, 2).unsqueeze(2) * r
			sampled = sampled.clamp(-1, 1)  # (H, V, n_rand, A)

			actions = torch.empty(H, V, S, A, device=self.device)
			if P > 0:
				actions[:, :, :P] = pi_actions
				actions[:, :, P:] = sampled
			else:
				actions = sampled

			# Evaluate: reshape to (H, V*S, A) for single-agent rollout
			actions_flat = actions.reshape(H, V * S, A)
			value = self._estimate_value(
				z_rep, actions_flat,
				ctx_latents=ctx_rep, ctx_mask=ctx_mask_rep,
			).nan_to_num(0)                                                    # (V*S, 1)
			value = value.view(V, S, 1)                                        # (V, S, 1)

			# Elite selection per vehicle
			elite_idxs = torch.topk(value.squeeze(-1), self.cfg.num_elites, dim=1).indices  # (V, num_elites)
			elite_value = torch.gather(value.squeeze(-1), 1, elite_idxs).unsqueeze(-1)       # (V, num_elites, 1)
			# Gather elite actions: (H, V, S, A) -> (H, V, num_elites, A)
			e_idx = elite_idxs.unsqueeze(0).unsqueeze(-1).expand(H, V, self.cfg.num_elites, A)
			elite_actions = torch.gather(actions, 2, e_idx)  # (H, V, num_elites, A)

			# Softmax over elites per vehicle
			max_v = elite_value.max(dim=1, keepdim=True).values       # (V, 1, 1)
			score = torch.exp(self.cfg.temperature * (elite_value - max_v))  # (V, num_elites, 1)
			score = score / (score.sum(dim=1, keepdim=True) + 1e-9)   # (V, num_elites, 1)

			# Update mean/std per vehicle
			# elite_actions: (H, V, E, A); score: (V, E, 1)
			score_bcast = score.permute(0, 1, 2).unsqueeze(0)          # (1, V, E, 1)
			new_mean = (score_bcast * elite_actions).sum(dim=2)         # (H, V, A)
			new_mean = new_mean.permute(1, 0, 2)                        # (V, H, A)
			diff = elite_actions - new_mean.permute(1, 0, 2).unsqueeze(2)  # (H, V, E, A)
			new_std = ((score_bcast * diff ** 2).sum(dim=2)).sqrt()     # (H, V, A)
			new_std = new_std.permute(1, 0, 2).clamp(self.cfg.min_std, self.cfg.max_std)

			mean, std = new_mean, new_std

		# Sample final action per vehicle from the elite distribution
		# Use the elite with highest score (or Gumbel sample) per vehicle
		# For simplicity, sample one elite per vehicle via categorical
		probs = score.squeeze(-1)                                    # (V, num_elites)
		chosen = torch.multinomial(probs, 1).squeeze(-1)              # (V,)
		chosen_actions = elite_actions[0][torch.arange(V, device=self.device), chosen]  # (V, A)
		a_std = std[:, 0]                                             # (V, A)
		if not eval_mode:
			chosen_actions = chosen_actions + a_std * torch.randn(V, A, device=self.device)

		# Persist the plan per CAV for warm-starting the next step
		for v, cav_id in enumerate(cav_ids):
			self._prev_mean[cav_id] = mean[v].detach()
		return chosen_actions.clamp(-1, 1)

	# ── Policy update ──────────────────────────────────────────────────

	def update_pi(self, zs):
		"""
		Update per-vehicle policy using single-agent Q.

		Args:
			zs: (H+1, B, latent_dim)
		"""
		action, info = self.model.pi(zs)
		qs = self.model.Q(zs, action, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})

	# ── TD target ──────────────────────────────────────────────────────

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated):
		action, _ = self.model.pi(next_z)
		return reward + self.discount * (1 - terminated) * self.model.Q(
			next_z, action, return_type='min', target=True
		)

	# ── Main update ────────────────────────────────────────────────────

	def _update(self, obs, ctx_obs, ctx_mask, entry_phase_id, progress_id,
				action, reward, terminated):
		"""
		Args:
			obs:             (H+1, B, 19)       own obs
			ctx_obs:         (H+1, B, K, 19)    K-NN context obs
			ctx_mask:        (H+1, B, K)
			entry_phase_id:  (H+1, B)           long
			progress_id:     (H+1, B)           long
			action:          (H, B, act_per_agent)
			reward:          (H, B, 1)
			terminated:      (H, B, 1)
		"""
		H = self.cfg.horizon

		# Encode next-step latents for targets
		with torch.no_grad():
			next_obs = obs[1:]
			next_ctx = ctx_obs[1:]
			next_mask = ctx_mask[1:]
			next_ep = entry_phase_id[1:]
			next_pg = progress_id[1:]
			next_z = self.model.encode(next_obs, next_ctx, next_mask, next_ep, next_pg)  # (H, B, latent)
			td_targets = self._td_target(next_z, reward, terminated)                      # (H, B, 1)

		self.model.train()

		# Encode first step
		z = self.model.encode(
			obs[0], ctx_obs[0], ctx_mask[0], entry_phase_id[0], progress_id[0]
		)  # (B, latent)

		# Pre-encode ctx for Level-1 interaction at every rollout step.
		# Returns None when use_interaction=false; in that case the rollout
		# is identical to the original Level-0 single-agent dynamics.
		ctx_lat_all = self.model.encode_ctx(ctx_obs)   # (H+1, B, K, latent) or None

		# Latent rollout and consistency loss
		zs = torch.empty(H + 1, obs.shape[1], self.cfg.latent_dim, device=self.device)
		zs[0] = z
		consistency_loss = 0
		for t in range(H):
			# Use ctx at time t (the ego is currently at state z[t], predicting z[t+1])
			ctx_lat_t = ctx_lat_all[t] if ctx_lat_all is not None else None
			cmask_t = ctx_mask[t] if ctx_lat_all is not None else None
			z = self.model.next(z, action[t], ctx_latents=ctx_lat_t, ctx_mask=cmask_t)
			consistency_loss = consistency_loss + F.mse_loss(z, next_z[t]) * self.cfg.rho ** t
			zs[t + 1] = z

		# Predictions
		_zs = zs[:-1]                                          # (H, B, latent)
		qs = self.model.Q(_zs, action, return_type='all')      # (num_q, H, B, num_bins)
		reward_preds = self.model.reward(_zs, action)          # (H, B, num_bins)

		# Termination prediction from the rolled-out latents (z_1 .. z_H).
		# When episodic=true, we train this head with BCE against the true
		# per-transition terminated flag so MPPI can later use it to gate
		# phantom rewards past a predicted episode end.
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], unnormalized=True)  # (H, B, 1)
		else:
			termination_pred = None

		# Losses
		reward_loss, value_loss = 0, 0
		for t in range(H):
			reward_loss = reward_loss + math.soft_ce(
				reward_preds[t], reward[t], self.cfg
			).mean() * self.cfg.rho ** t
			for qi in range(qs.shape[0]):
				value_loss = value_loss + math.soft_ce(
					qs[qi, t], td_targets[t], self.cfg
				).mean() * self.cfg.rho ** t

		consistency_loss = consistency_loss / H
		reward_loss = reward_loss / H
		value_loss = value_loss / (H * self.cfg.num_q)

		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(
				termination_pred, terminated
			)
		else:
			termination_loss = 0.

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		pi_info = self.update_pi(zs.detach())
		self.model.soft_update_target_Q()

		self.model.eval()
		_term_loss = termination_loss if isinstance(termination_loss, torch.Tensor) \
			else torch.tensor(0.0, device=self.device)
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": _term_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"attn_gate_mean_enc": torch.tensor(
				float(getattr(self.model._encoder, "last_gate_mean", float("nan"))),
				device=self.device,
			),
			"attn_ego_weight_enc": torch.tensor(
				float(getattr(self.model._encoder, "last_ego_attn_weight", float("nan"))),
				device=self.device,
			),
		})
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		batch = buffer.sample()
		return self._update(*batch)
