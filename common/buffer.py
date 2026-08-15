import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Per-vehicle trajectory replay buffer for DTDE TD-MPC2.

	Each trajectory corresponds to one vehicle's lifetime in a single
	simulation episode (from network entry to exit). Trajectories have
	variable length. Each trajectory is stored as its own "episode"
	for the SliceSampler, so sampling draws fixed-length subsequences
	from within a single vehicle's trajectory.

	Stored fields per transition:
		obs:             (obs_per_agent,)      own observation
		ctx_obs:         (K, obs_per_agent)     K nearest neighbors' obs
		ctx_mask:        (K,)                   active mask over context
		entry_phase_id:  ()                     entry_step % vid_cycle
		progress_id:     ()                     rel_step  % vid_cycle
		action:          (act_per_agent,)
		reward:          ()                     per-vehicle scalar reward
		terminated:      ()
		episode:         ()                     unique trajectory id
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon + 1)
		self._num_eps = 0
		self._recent_trajectories = []

	@property
	def capacity(self):
		return self._capacity

	@property
	def num_eps(self):
		return self._num_eps

	def _reserve_buffer(self, storage):
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		print(f'Buffer capacity: {self._capacity:,}')
		if torch.cuda.is_available():
			mem_free, _ = torch.cuda.mem_get_info()
		else:
			mem_free = 0

		bytes_per_step = sum([
				(v.numel() * v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel() * x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step * self._capacity
		print(f'Storage required: {total_bytes / 1e9:.2f} GB')
		storage_device = 'cuda:0' if 2.5 * total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def add(self, td):
		"""
		Add one vehicle's trajectory (variable length) to the buffer.
		Assigns a unique 'episode' id so SliceSampler treats it as
		a standalone trajectory.
		"""
		td['episode'] = torch.full(
			(td.shape[0],), self._num_eps, dtype=torch.int64
		)
		if getattr(self.cfg, 'agent', '') == 'ppo':
			self._recent_trajectories.append(td.clone())
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def pop_recent_trajectories(self, min_trajectories=1):
		"""Return and clear complete per-vehicle trajectories collected for PPO."""
		if len(self._recent_trajectories) < int(min_trajectories):
			return []
		trajectories = self._recent_trajectories
		self._recent_trajectories = []
		return trajectories

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for DTDE training.

		Returns (in the order expected by TDMPC2._update):
			obs:             (H+1, B, 19)
			ctx_obs:         (H+1, B, K, 19)
			ctx_mask:        (H+1, B, K)
			entry_phase_id:  (H+1, B)
			progress_id:     (H+1, B)
			action:          (H, B, 2)
			reward:          (H, B, 1)
			terminated:      (H, B, 1)
		"""
		td = td.select(
			"obs", "ctx_obs", "ctx_mask", "entry_phase_id", "progress_id",
			"action", "reward", "terminated",
			strict=False,
		).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()                            # (H+1, B, 19)
		ctx_obs = td.get('ctx_obs').contiguous()                     # (H+1, B, K, 19)
		ctx_mask = td.get('ctx_mask').contiguous()                   # (H+1, B, K)
		entry_phase_id = td.get('entry_phase_id').contiguous()        # (H+1, B)
		progress_id = td.get('progress_id').contiguous()              # (H+1, B)
		action = td.get('action')[1:].contiguous()                    # (H, B, 2)
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()       # (H, B, 1)
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = terminated[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		return (
			obs, ctx_obs, ctx_mask, entry_phase_id, progress_id,
			action, reward, terminated,
		)

	def sample(self):
		td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
		return self._prepare_batch(td)
