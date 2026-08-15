import os
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""
	Online trainer for slot-free DTDE TD-MPC2 CAV control.

	Each active CAV is independently logged. Per-vehicle trajectories are
	keyed by SUMO cav_id and flushed to the replay buffer at episode end.
	Trajectories shorter than `horizon + 1` are dropped (SliceSampler with
	strict_length=True cannot use them).

	Transition recording convention:
	  - At reset we record the initial obs for every currently active CAV,
	    with NaN action/reward (like the original single-agent TD-MPC2).
	  - After each step we record the new obs for every currently active
	    CAV. If the CAV existed in the previous step we log the real action
	    and reward that moved it from t-1 -> t; otherwise NaN (new arrival).
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0          # buffer trajectory counter (from buffer.add)
		self._env_episode = 0     # environment episode counter
		self._start_time = time()
		self._vehicle_tds: Dict[str, List[Dict[str, object]]] = {}
		# Tally for trajectories that already terminated mid-episode
		# (one CAV left the control region) and were flushed early.
		self._completed_ep_reward = 0.0
		self._completed_traj_count = 0
		# Per-episode loss accumulator: every gradient update appends each
		# loss component here, and at episode end we summarise mean/std/slope
		# so train.csv reports stable, low-noise per-loss curves and we can
		# detect loss-conflict (e.g. consistency vs value).
		self._loss_history: Dict[str, List[float]] = defaultdict(list)

	def _make_transition_record(
		self,
		obs,
		ctx_obs,
		ctx_mask,
		entry_phase_id,
		progress_id,
		action,
		reward: float,
		terminated: float,
		extra: Optional[Dict[str, torch.Tensor]] = None,
	) -> Dict[str, object]:
		"""Store a lightweight per-step record; stack into TensorDict at flush."""
		log_prob = torch.tensor(float('nan'), dtype=torch.float32)
		value = torch.tensor(float('nan'), dtype=torch.float32)
		if extra is not None:
			if 'log_prob' in extra:
				log_prob = extra['log_prob'].detach().cpu().float().reshape(())
			if 'value' in extra:
				value = extra['value'].detach().cpu().float().reshape(())
		return {
			'obs': obs,
			'ctx_obs': ctx_obs,
			'ctx_mask': ctx_mask,
			'entry_phase_id': entry_phase_id,
			'progress_id': progress_id,
			'action': action,
			'reward': float(reward),
			'terminated': float(terminated),
			'log_prob': log_prob,
			'value': value,
		}

	def _trajectory_to_tensordict(self, records: List[Dict[str, object]]) -> TensorDict:
		"""Convert one vehicle's list of lightweight records into replay format."""
		traj = TensorDict(
			{
				'obs': torch.stack([r['obs'] for r in records]).float(),
				'ctx_obs': torch.stack([r['ctx_obs'] for r in records]).float(),
				'ctx_mask': torch.stack([r['ctx_mask'] for r in records]).float(),
				'entry_phase_id': torch.stack([r['entry_phase_id'] for r in records]).long(),
				'progress_id': torch.stack([r['progress_id'] for r in records]).long(),
				'action': torch.stack([r['action'] for r in records]).float(),
				'reward': torch.tensor([r['reward'] for r in records], dtype=torch.float32),
				'terminated': torch.tensor([r['terminated'] for r in records], dtype=torch.float32),
				'log_prob': torch.stack([r['log_prob'] for r in records]).float(),
				'value': torch.stack([r['value'] for r in records]).float(),
			},
			batch_size=(len(records),),
		)
		return traj

	def _agent_transition_info(self, cav_id: str) -> Optional[Dict[str, torch.Tensor]]:
		getter = getattr(self.agent, 'get_transition_info', None)
		if getter is None:
			return None
		return getter(cav_id)

	def common_metrics(self):
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._env_episode,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time,
		)

	# ── Evaluation ─────────────────────────────────────────────────────

	def eval(self):
		self.agent.eval()
		ep_rewards, ep_successes, ep_lengths = [], [], []
		ep_metrics = {
					'avg_speed': [], 'avg_jerk': [], 'total_fuel_mL': [], 'total_distance': [],
					'tet_seconds': [], 'tet_rate': [],
					'trip50_avg_time_s': [], 'trip50_avg_stop_time_s': [],
					'trip50_avg_speed_mps': [], 'trip50_num_completed': [],
					'trip50_hdv_avg_time_s': [], 'trip50_hdv_avg_stop_time_s': [],
					'trip50_hdv_avg_speed_mps': [], 'trip50_hdv_num_completed': [],
					'trip50_mix_avg_time_s': [], 'trip50_mix_avg_stop_time_s': [],
					'trip50_mix_avg_speed_mps': [], 'trip50_mix_num_completed': [],
					'r_speed_mean': [], 'r_accel_mean': [], 'r_jerk_mean': [],
					'r_safety_mean': [], 'r_lc_mean': [], 'r_idle_mean': [],
					'r_energy_mean': [], 'r_terminal_mean': [], 'num_terminals': [],
				}
		plot_ts = getattr(self.cfg, 'ts_plot_freq', 0) > 0

		for i in range(self.cfg.eval_episodes):
			# Enable FCD output for the first eval episode only
			if plot_ts and i == 0:
				fcd_dir = Path(self.logger._log_dir) / "ts_diagrams"
				fcd_dir.mkdir(parents=True, exist_ok=True)
				fcd_path = str(fcd_dir / f"ts_step{self._step:08d}.xml")
				self.env.unwrapped.set_fcd_output(fcd_path)
			else:
				self.env.unwrapped.set_fcd_output(None)

			eval_seed = self.cfg.seed + 1000 + i  # offset eval seeds from train seeds
			obs, done, ep_reward, t = self.env.reset(seed=eval_seed), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i == 0))
			while not done:
				action = self.agent.act(obs, t0=t == 0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
				n_terminals = max(int(info.get('num_terminals', 0)), 1)
				ep_reward /= n_terminals  # reward per completed terminal (safe when 0)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			for k in ep_metrics:
				ep_metrics[k].append(info.get(k, float('nan')))
			if self.cfg.save_video:
				self.logger.video.save(self._step)

		# Plot time-space diagram: run a dummy reset to flush FCD, then plot
		if plot_ts:
			self.env.unwrapped.set_fcd_output(None)
			self.env.reset()  # _close_sumo() flushes FCD, _start_sumo() with no FCD
			self._plot_time_space(fcd_path, self._step)

		self.agent.train()
		# Compute means AND stds for stability diagnosis
		out = dict(
			episode_reward=float(np.nanmean(ep_rewards)),
			episode_reward_std=float(np.nanstd(ep_rewards)),
			episode_reward_min=float(np.nanmin(ep_rewards)) if len(ep_rewards) else float('nan'),
			episode_reward_max=float(np.nanmax(ep_rewards)) if len(ep_rewards) else float('nan'),
			episode_success=float(np.nanmean(ep_successes)),
			episode_length=float(np.nanmean(ep_lengths)),
			eval_num_episodes=len(ep_rewards),
		)
		# Per-metric mean + std
		for k, v in ep_metrics.items():
			out[k] = float(np.nanmean(v)) if len(v) else float('nan')
			out[f"{k}_std"] = float(np.nanstd(v)) if len(v) else float('nan')
		return out

	# ── Per-vehicle trajectory bookkeeping ─────────────────────────────

	def _record_transition(
		self,
		obs_dict,
		prev_cav_ids: Optional[List[str]],
		prev_action: Optional[torch.Tensor],
		prev_agent_rewards: Optional[torch.Tensor],
	):
		"""
		Append one transition per CAV. Two cases handled:

		1) CAVs in current obs ("still in network"): append a regular transition
		   with the new obs and the action/reward that produced it (NaN for
		   freshly arrived CAVs that weren't in prev_cav_ids).
		2) CAVs in prev_cav_ids but no longer in current obs ("just left the
		   control region"): append ONE final terminal transition with their
		   action/reward and terminated=1.0. Without this, those CAVs lose
		   their last action's effect entirely.

		Per-vehicle done is per CAV (not per env episode): a CAV is done iff
		it left the controlled edges. Episode-end truncation does NOT mark
		any per-CAV done; those CAVs simply stop being recorded.

		Terminated trajectories are immediately flushed to the buffer (and
		removed from self._vehicle_tds) so memory doesn't grow unboundedly.
		"""
		cav_ids = obs_dict['cav_ids']
		current_set = set(cav_ids)
		A = self.cfg.act_per_agent
		min_len = self.cfg.horizon + 1

		# ── 1) Terminal transitions for CAVs that just left the control region
		if prev_cav_ids is not None and prev_action is not None and prev_agent_rewards is not None:
			for j, cav_id in enumerate(prev_cav_ids):
				if cav_id in current_set:
					continue  # still active, will get a regular td below
				if cav_id not in self._vehicle_tds:
					continue  # never started a trajectory (edge case)

				# Use the CAV's last known obs as the placeholder terminal obs.
				# Because terminated=1.0 the value bootstrap will be gated off,
				# so the actual obs values don't matter for TD targets.
				last_record = self._vehicle_tds[cav_id][-1]
				a = prev_action[j].detach().cpu().float()
				r = float(prev_agent_rewards[j].item())
				extra = self._agent_transition_info(cav_id)

				terminal_record = self._make_transition_record(
					last_record['obs'],
					last_record['ctx_obs'],
					last_record['ctx_mask'],
					last_record['entry_phase_id'],
					last_record['progress_id'],
					a,
					r,
					1.0,
					extra=extra,
				)
				self._vehicle_tds[cav_id].append(terminal_record)

				# Immediately flush this completed trajectory to the buffer
				records = self._vehicle_tds.pop(cav_id)
				if len(records) >= min_len:
					traj = self._trajectory_to_tensordict(records)
					self._ep_idx = self.buffer.add(traj)
					self._completed_ep_reward += float(traj['reward'].nansum().item())
					self._completed_traj_count += 1

		# ── 2) Regular transitions for currently active CAVs
		N = len(cav_ids)
		if N == 0:
			return

		own_obs = obs_dict['obs'].cpu()
		ctx_obs = obs_dict['ctx_obs'].cpu()
		ctx_mask = obs_dict['ctx_mask'].cpu()
		entry_phase_id = obs_dict['entry_phase_id'].cpu()
		progress_id = obs_dict['progress_id'].cpu()

		prev_idx = {cid: i for i, cid in enumerate(prev_cav_ids or [])}

		for i, cav_id in enumerate(cav_ids):
			if cav_id in prev_idx and prev_action is not None and prev_agent_rewards is not None:
				j = prev_idx[cav_id]
				a = prev_action[j].detach().cpu().float()
				r = float(prev_agent_rewards[j].item())
				extra = self._agent_transition_info(cav_id)
			else:
				a = torch.full((A,), float('nan'))
				r = float('nan')
				extra = None

			record = self._make_transition_record(
				own_obs[i],
				ctx_obs[i],
				ctx_mask[i],
				entry_phase_id[i],
				progress_id[i],
				a,
				r,
				0.0,
				extra=extra,
			)
			self._vehicle_tds.setdefault(cav_id, []).append(record)

	def _flush_trajectories(self):
		"""
		At episode end: flush any trajectories that are still open (CAVs
		still in network when the env truncated). These are not "terminated"
		so the last transition keeps terminated=0.0.
		"""
		min_len = self.cfg.horizon + 1
		ep_reward = self._completed_ep_reward
		count = self._completed_traj_count
		for cav_id, records in self._vehicle_tds.items():
			if len(records) < min_len:
				continue
			traj = self._trajectory_to_tensordict(records)
			self._ep_idx = self.buffer.add(traj)
			ep_reward += float(traj['reward'].nansum().item())
			count += 1
		self._vehicle_tds = {}
		self._completed_ep_reward = 0.0
		self._completed_traj_count = 0
		return ep_reward, count

	# ── Per-episode loss aggregation ─────────────────────────────────

	def _record_loss_step(self, metrics) -> None:
		"""Append every scalar metric from one gradient update to the
		per-episode history. Called once per agent.update() call.
		"""
		if metrics is None:
			return
		# TensorDict, dict, or anything with .items()
		try:
			items = metrics.items()
		except AttributeError:
			return
		for k, v in items:
			# Cast tensors / numpy / python scalars into a single float
			if hasattr(v, 'detach'):
				try:
					v = v.detach()
				except Exception:
					pass
			if hasattr(v, 'item') and not isinstance(v, str):
				try:
					v = v.item()
				except Exception:
					continue
			try:
				val = float(v)
			except (TypeError, ValueError):
				continue
			if not np.isfinite(val):
				continue
			self._loss_history[k].append(val)

	def _summarise_loss_history(self) -> Dict[str, float]:
		"""At episode end, summarise each loss component into:
		    <name>_mean : average over all gradient steps in the episode
		    <name>_std  : within-episode standard deviation
		    <name>_slope: linear-regression slope (per-update Δ loss),
		                  positive = increasing → useful to spot loss
		                  conflict where one loss climbs while another falls.
		"""
		summary: Dict[str, float] = {}
		for name, values in self._loss_history.items():
			if not values:
				continue
			arr = np.asarray(values, dtype=np.float64)
			summary[f"{name}_mean"] = float(arr.mean())
			summary[f"{name}_std"] = float(arr.std()) if arr.size > 1 else 0.0
			# Slope across updates within the episode (least-squares fit)
			if arr.size >= 2:
				x = np.arange(arr.size, dtype=np.float64)
				# polyfit deg=1 returns [slope, intercept]
				try:
					slope = float(np.polyfit(x, arr, 1)[0])
				except (np.linalg.LinAlgError, ValueError):
					slope = 0.0
			else:
				slope = 0.0
			summary[f"{name}_slope"] = slope
		return summary

	def _reset_loss_history(self) -> None:
		self._loss_history = defaultdict(list)

	# ── Time-space diagram ────────────────────────────────────────────

	def _plot_time_space(self, fcd_path: str, step: int):
		"""Generate time-space diagram from a completed FCD file."""
		from common.plotting import plot_time_space_diagram
		save_path = fcd_path.replace('.xml', '.png')
		plot_time_space_diagram(
			fcd_path=fcd_path,
			save_path=save_path,
			episode=step,
		)
		# Remove the XML to save disk space
		try:
			os.remove(fcd_path)
		except OSError:
			pass

	# ── Training loop ──────────────────────────────────────────────────

	def train(self):
		self.cfg.seed = 1
		train_metrics, done, eval_next = {}, True, False
		info: dict = {}
		obs = None
		prev_cav_ids: Optional[List[str]] = None
		checkpoint_freq = int(getattr(self.cfg, 'checkpoint_freq', 0) or 0)

		while self._step <= self.cfg.steps:
			# Periodic evaluation
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					ep_reward, n_traj = self._flush_trajectories()
					train_metrics.update(
						episode_reward=ep_reward / max(int(info.get('num_terminals', 0)), 1),  # reward per completed terminal
						episode_success=info.get('success', 0.0),
						episode_length=self._episode_len,
						episode_terminated=info.get('terminated', False),
						num_trajectories=n_traj,
						scenario_total_flow_vph=info.get('scenario_total_flow_vph', float('nan')),
						scenario_cav_penetration=info.get('scenario_cav_penetration', float('nan')),
						scenario_cav_flow_vph=info.get('scenario_cav_flow_vph', float('nan')),
						scenario_hdv_flow_vph=info.get('scenario_hdv_flow_vph', float('nan')),
						avg_speed=info.get('avg_speed', float('nan')),
						avg_jerk=info.get('avg_jerk', float('nan')),
						total_fuel_mL=info.get('total_fuel_mL', float('nan')),
						tet_seconds=info.get('tet_seconds', float('nan')),
						tet_rate=info.get('tet_rate', float('nan')),
							trip50_avg_time_s=info.get('trip50_avg_time_s', float('nan')),
							trip50_avg_stop_time_s=info.get('trip50_avg_stop_time_s', float('nan')),
							trip50_avg_speed_mps=info.get('trip50_avg_speed_mps', float('nan')),
							trip50_num_completed=info.get('trip50_num_completed', float('nan')),
							trip50_hdv_avg_time_s=info.get('trip50_hdv_avg_time_s', float('nan')),
							trip50_hdv_avg_stop_time_s=info.get('trip50_hdv_avg_stop_time_s', float('nan')),
							trip50_hdv_avg_speed_mps=info.get('trip50_hdv_avg_speed_mps', float('nan')),
							trip50_hdv_num_completed=info.get('trip50_hdv_num_completed', float('nan')),
							trip50_mix_avg_time_s=info.get('trip50_mix_avg_time_s', float('nan')),
							trip50_mix_avg_stop_time_s=info.get('trip50_mix_avg_stop_time_s', float('nan')),
							trip50_mix_avg_speed_mps=info.get('trip50_mix_avg_speed_mps', float('nan')),
							trip50_mix_num_completed=info.get('trip50_mix_num_completed', float('nan')),
							# Per-term rewards
							r_speed_mean=info.get('r_speed_mean', float('nan')),
						r_accel_mean=info.get('r_accel_mean', float('nan')),
						r_jerk_mean=info.get('r_jerk_mean', float('nan')),
						r_safety_mean=info.get('r_safety_mean', float('nan')),
						r_lc_mean=info.get('r_lc_mean', float('nan')),
						r_idle_mean=info.get('r_idle_mean', float('nan')),
						r_energy_mean=info.get('r_energy_mean', float('nan')),
						r_terminal_mean=info.get('r_terminal_mean', float('nan')),
						num_terminals=info.get('num_terminals', 0),
					)
					# Per-loss episode-level summary (mean / std / slope per
					# component). Overrides the noisy "last-gradient-step"
					# values that train_metrics inherited from agent.update().
					train_metrics.update(self._summarise_loss_history())
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._env_episode += 1
					self._reset_loss_history()

				train_seed = self.cfg.seed + self._env_episode  # different seed each episode
				obs = self.env.reset(seed=train_seed)
				self._vehicle_tds = {}
				self._completed_ep_reward = 0.0
				self._completed_traj_count = 0
				self._episode_len = 0
				prev_cav_ids = None
				# Record initial obs for every currently active CAV (NaN action/reward)
				self._record_transition(obs, None, None, None)
				prev_cav_ids = list(obs['cav_ids'])

			# Select action
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=self._episode_len == 0)
			else:
				action = self.env.rand_act()

			# Step env; the returned obs may have a different set of CAVs
			step_cav_ids = list(obs['cav_ids'])   # the cav_ids the action corresponds to
			obs, reward, done, info = self.env.step(action)
			self._episode_len += 1
			agent_rewards = info.get('agent_rewards', None)

			self._record_transition(
				obs,
				prev_cav_ids=step_cav_ids,
				prev_action=action,
				prev_agent_rewards=agent_rewards,
			)
			prev_cav_ids = list(obs['cav_ids'])

			# Agent update
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps and self.cfg.agent == 'tdmpc2':
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
					self._record_loss_step(_train_metrics)
				train_metrics.update(_train_metrics)

			self._step += 1
			if (
				checkpoint_freq > 0
				and self._step <= self.cfg.steps
				and self._step % checkpoint_freq == 0
			):
				self.logger.save_agent(self.agent, identifier=f'{self._step}')

		self.logger.finish(self.agent)
