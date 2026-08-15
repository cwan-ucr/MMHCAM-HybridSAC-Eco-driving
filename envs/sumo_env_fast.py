"""
SUMO Environment for CAV control at an isolated signalized intersection.

The environment controls Connected Autonomous Vehicles (CAVs) in mixed traffic
to reduce stop-and-go behavior. Each CAV takes longitudinal (acceleration) and
lateral (lane-change) actions based on surrounding vehicle states and traffic
signal information.

Observation per CAV:
    - CAV own state: speed, acceleration, lane index, position along edge,
      following-mode one-hot [head, follow-HDV, follow-CAV], last action
    - Leader vehicle: relative distance, relative speed, exists flag
    - Follower vehicle: relative distance, relative speed, exists flag
    - Left-lane leader/follower: relative distance, relative speed, exists flag
    - Right-lane leader/follower: relative distance, relative speed, exists flag
    - Traffic signal: 4 discrete G/Y flags (current+opposed edge) + [T_l, T'_l]

Action per CAV (continuous):
    - Longitudinal acceleration: [-5.0, 3.0] m/s^2
    - Lateral lane-change command: [-1, 1] (discretized: <-0.70 = right,
      >0.70 = left, else keep lane)

Reward:
    - Penalize high acceleration/deceleration (jerk proxy for stop-and-go)
    - Reward smooth speed (close to desired / free-flow speed)
    - Penalize collisions heavily
    - Penalize lane-change commands that do not result in an actual lane change
    - Penalize large speed differences with neighbors (harmonize traffic)
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# ── SUMO setup ──────────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError(
        "Please set the SUMO_HOME environment variable "
        "(e.g., export SUMO_HOME=/usr/share/sumo)"
    )

import traci
import traci.constants as tc


# ── Constants ───────────────────────────────────────────────────────────────
_SUMO_FILES_DIR = os.path.join(os.path.dirname(__file__), "sumo_files")
_DEFAULT_CFG = os.path.join(_SUMO_FILES_DIR, "intersection.sumocfg")
_DEFAULT_ROUTE_TEMPLATE = os.path.join(_SUMO_FILES_DIR, "intersection.rou.xml")

# Observation dimensions per CAV
_OBS_CAV_OWN = 9          # speed, accel, lane_idx(one-hot), pos_on_edge, following_mode(one-hot)
_OBS_NEIGHBOR = 3          # rel_dist, rel_speed, exists_flag  (×6 neighbors)
_OBS_SIGNAL = 6            # [cur_green, cur_yellow, opp_green, opp_yellow, T_l, T'_l]
_OBS_PREV_ACTION = 2       # last applied [accel_cmd, lane_change_cmd] (state-augmentation)
_OBS_PER_CAV = (
    _OBS_CAV_OWN
    + 6 * _OBS_NEIGHBOR
    + _OBS_SIGNAL
    + _OBS_PREV_ACTION
)  # 9+18+6+2 = 35

# Action dimensions per CAV
_ACT_PER_CAV = 2           # [accel, lane_change_cmd]

# Limits
_MAX_SPEED = 18.00         # ~50 km/h
_MAX_ACCEL = 3.0
_MAX_DECEL = -5.0
_MAX_DETECT_DIST = 150.0   # meters – beyond this we report default
_LANE_CHANGE_THRESH = 0.7  # lane change command magnitude above which we trigger a lane change
_TTC_THRESH = 3.0          # seconds – below this we consider a safety violation

# Reward weights
_W_SMOOTH_SPEED = 0.9
_W_ACCEL_PENALTY = 0.1
_W_JERK_PENALTY = 0.1
_W_SAFETY = 0.3
_W_LANE_CHANGE = 0.1
_W_IDLE = 0.8
_W_ENERGY = 0.4

# Terminal reward weights (applied once when a CAV exits the control area)
_W_TERMINAL_TIME = 20.0    # reward for fast travel: w · (ideal_time / actual_time - 1)
_W_TERMINAL_SPEED = 1.0   # reward for high exit speed: w · (exit_speed / max_speed - 1)
_CONTROL_AREA_LEN = 294.4  # meters — lane length of *_in edges (for ideal travel time)


class SumoIntersectionFastEnv(gym.Env):
    """
    Gymnasium environment wrapping SUMO for CAV longitudinal/lateral control
    at an isolated signalized intersection under mixed traffic.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sumo_cfg: str = _DEFAULT_CFG,
        route_template: str = _DEFAULT_ROUTE_TEMPLATE,
        max_steps: int = 3000,
        delta_t: float = 1.0,        # control timestep in seconds
        sumo_step: float = 0.2,       # SUMO simulation step
        use_gui: bool = False,
        context_size: int = 8,          # K nearest neighbors per vehicle
        communication_range_m: float = 150.0,  # max radius for KNN context candidates
        comm_topology: str = "full",   # full | front_only
        lane_change_mode: int = 512,    # SUMO TraCI lane-change mode for controlled CAVs
        lane_change_mode_release: int = 1621,  # SUMO lane-change mode outside controlled area
        vid_cycle: int = 120,            # signal cycle length in control steps
        max_controllable: int = 200,     # safety cap on CAVs per step
        communication: bool = True,      # V2X capability flag (CAV vs AV)
        rl_control: bool = True,          # false: let SUMO control CAV speed/lane changes
        cav_control_mode: str = "hybrid",  # hybrid | longitudinal_only | lane_change_only | sumo_default
        sumo_default_cav_behavior: bool = False,  # remove cav lc* overrides in generated route
        glosa_enabled: bool = False,
        glosa_range: float = 150.0,
        glosa_min_speed: float = 5.0,
        glosa_max_speedfactor: float = 1.1,
        glosa_add_switchtime: float = 0.0,
        glosa_override_safety: bool = False,
        glosa_ignore_cfmodel: bool = False,
        glosa_use_queue: bool = False,
        max_speed_mps: float = _MAX_SPEED,
        desired_speed: float = _MAX_SPEED,  # ~50 km/h target smooth speed
        neighbor_reward_coef: float = 0.0,  # weight on mean-of-K-neighbors reward (0 = selfish)
        neighbor_reward_directional: bool = False,  # share reward only with behind/parallel neighbors
        control_start_distance_m: float = 20.0,  # leave the first meters of *_in edges to SUMO
        post_out_eval_distance_m: float = 50.0,  # downstream distance after incoming edge: internal + out edge
        randomize_demand: bool = True,
        total_flow_vph_min: float = 1400.0,
        total_flow_vph_max: float = 2700.0,
        cav_penetration_min: float = 0.05,
        cav_penetration_max: float = 0.95,
        controlled_edges: Optional[object] = None,
        control_area_length_m: float = _CONTROL_AREA_LEN,
        terminal_cycle_s: float = 60.0,
        terminal_green_start_s: float = 30.0,
        terminal_green_end_s: float = 60.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.route_template = route_template
        self.max_steps = max_steps
        self.delta_t = delta_t
        self.sumo_step = sumo_step
        self.steps_per_action = min(1, int(delta_t / sumo_step))
        self.use_gui = use_gui
        self._K = context_size
        self.communication_range_m = float(communication_range_m)
        self.comm_topology = str(comm_topology).lower()
        if self.comm_topology not in {"full", "front_only"}:
            raise ValueError(
                f"Invalid comm_topology={comm_topology}. Must be 'full' or 'front_only'."
            )
        self.lane_change_mode = int(lane_change_mode)
        self.lane_change_mode_release = int(lane_change_mode_release)
        self.vid_cycle = vid_cycle
        self.max_controllable = max_controllable
        self.communication = bool(communication)
        self.rl_control = bool(rl_control)
        self.cav_control_mode = str(cav_control_mode).lower()
        if self.cav_control_mode in {"both", "full", "hybrid_sac"}:
            self.cav_control_mode = "hybrid"
        if self.cav_control_mode not in {
            "hybrid", "longitudinal_only", "lane_change_only", "sumo_default",
        }:
            raise ValueError(
                "cav_control_mode must be one of: hybrid, longitudinal_only, "
                "lane_change_only, sumo_default."
            )
        self.sumo_default_cav_behavior = bool(sumo_default_cav_behavior)
        self.glosa_enabled = bool(glosa_enabled)
        self.glosa_range = float(glosa_range)
        self.glosa_min_speed = float(glosa_min_speed)
        self.glosa_max_speedfactor = float(glosa_max_speedfactor)
        self.glosa_add_switchtime = float(glosa_add_switchtime)
        self.glosa_override_safety = bool(glosa_override_safety)
        self.glosa_ignore_cfmodel = bool(glosa_ignore_cfmodel)
        self.glosa_use_queue = bool(glosa_use_queue)
        self.max_speed_mps = float(max_speed_mps)
        self.desired_speed = desired_speed
        self.neighbor_reward_coef = float(neighbor_reward_coef)
        self.neighbor_reward_directional = bool(neighbor_reward_directional)
        self.control_start_distance_m = float(control_start_distance_m)
        self.post_out_eval_distance_m = float(post_out_eval_distance_m)
        self.control_area_length_m = float(control_area_length_m)
        self._configured_controlled_edges = self._parse_controlled_edges(controlled_edges)
        self.terminal_cycle_s = float(terminal_cycle_s)
        self.terminal_green_start_s = float(terminal_green_start_s)
        self.terminal_green_end_s = float(terminal_green_end_s)
        if self.control_start_distance_m < 0:
            raise ValueError("control_start_distance_m must be non-negative.")
        if self.post_out_eval_distance_m < 0:
            raise ValueError("post_out_eval_distance_m must be non-negative.")
        if self.control_area_length_m <= 0:
            raise ValueError("control_area_length_m must be positive.")
        if self.terminal_cycle_s <= 0:
            raise ValueError("terminal_cycle_s must be positive.")
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive.")
        self.randomize_demand = bool(randomize_demand)
        self.total_flow_vph_min = float(total_flow_vph_min)
        self.total_flow_vph_max = float(total_flow_vph_max)
        self.cav_penetration_min = float(cav_penetration_min)
        self.cav_penetration_max = float(cav_penetration_max)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        if self.total_flow_vph_min <= 0 or self.total_flow_vph_max < self.total_flow_vph_min:
            raise ValueError(
                "Invalid total flow range: expected 0 < total_flow_vph_min <= total_flow_vph_max."
            )
        if not (0.0 <= self.cav_penetration_min <= self.cav_penetration_max <= 1.0):
            raise ValueError(
                "Invalid penetration range: expected 0 <= cav_penetration_min <= cav_penetration_max <= 1."
            )
        if not os.path.exists(self.route_template):
            raise FileNotFoundError(f"Route template not found: {self.route_template}")
        
        # energy range for normalization (m/s, m/s^2)
        self._max_energy = self._get_energy(self.max_speed_mps, _MAX_ACCEL)
        self._min_energy = self._get_energy(0.0, _MAX_DECEL)

        # Dynamic observation space: variable number of active CAVs.
        # We declare a 1-CAV Box for gym compatibility; the real shape is dynamic.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, _OBS_PER_CAV), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1, _ACT_PER_CAV), dtype=np.float32
        )

        self._conn = None
        self._current_time = 0.0
        self._step_count = 0
        self._prev_accels: Dict[str, float] = {}  # for jerk computation
        self._prev_lane_idx: Dict[str, int] = {}  # for actual lane-change detection
        self._prev_actions: Dict[str, float] = {}  # for action computation
        # Records the control-step at which each CAV first appeared in the
        # network. Used to compute cycle-based identity embeddings.
        self._cav_entry_time: Dict[str, float] = {}
        # Entry speed of each CAV (m/s) when it first entered the network.
        # Used for terminal travel-time reward relative to best-possible traversal.
        self._cav_entry_speed: Dict[str, float] = {}
        self._cav_entry_lane_pos: Dict[str, float] = {}
        # Terminal rewards buffered until the exit step gets scored in _compute_reward.
        self._pending_terminal_reward: Dict[str, float] = {}
        # Most recent action command applied to each CAV (before the SUMO
        # substeps). Folded into the next observation as state augmentation.
        self._prev_action_per_cav: Dict[str, np.ndarray] = {}
        # List of active CAV ids at the most recent _get_obs() call.
        # Needed so step()/compute_reward can align actions to the same order.
        self._active_cav_ids: List[str] = []
        self._cumulative_reward = 0.0
        self.edge_controlled = set()
        # Per-vehicle holistic segment tracker:
        # control-area entry (*_in) -> post_out_eval_distance_m downstream
        # of the incoming edge, where downstream distance includes junction
        # internal lanes plus the beginning of the outgoing edge.
        self._trip50_tracker: Dict[str, Dict[str, float]] = {}
        self._tl_cycle_durations: Dict[str, float] = {}
        self._tl_program_phases: Dict[str, List[Tuple[float, str]]] = {}
        # FCD output path: when set, SUMO writes floating car data to this file
        self._fcd_output_path: Optional[str] = None
        self._episode_route_file: Optional[str] = None
        self._episode_total_flow_vph: float = 0.0
        self._episode_cav_penetration: float = 0.0
        self._episode_cav_flow_vph: float = 0.0
        self._episode_hdv_flow_vph: float = 0.0
        # K-NN neighbor indices / mask from the most recent _get_obs() call.
        # Reused by _compute_reward on the NEXT step (step_cav_ids = previous
        # _active_cav_ids) for cooperative reward augmentation.
        self._last_neighbor_idx: Optional[np.ndarray] = None
        self._last_ctx_mask: Optional[np.ndarray] = None
        # Directional reward filter: (N, K) bool — True if neighbor k is
        # "downstream" of ego i (behind or parallel in the direction of
        # motion), i.e., ego's action can physically influence it.
        self._last_downstream_mask: Optional[np.ndarray] = None
        # Per-control-step TraCI read-through caches. These are invalidated
        # after every SUMO simulationStep so cached values never cross physics
        # updates.
        self._vehicle_step_cache: Dict[str, Dict[str, object]] = {}
        self._neighbor_step_cache: Dict[Tuple[str, str, int], Optional[Tuple[str, float]]] = {}
        self._control_area_vehicle_ids_cache: Optional[List[str]] = None
        self._active_cav_ids_cache: Optional[List[str]] = None
        self._all_vehicle_ids_cache: Optional[set] = None
        self._signal_obs_step_cache: Dict[Tuple[str, int, bool], np.ndarray] = {}
        self._lane_length_cache: Dict[str, float] = {}

    # ── FCD output control ──────────────────────────────────────────────

    def set_fcd_output(self, path: Optional[str]):
        """Set FCD output path for the next episode. None disables it."""
        self._fcd_output_path = path

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _start_sumo(self):
        """Launch a SUMO (or SUMO-GUI) instance via TraCI."""
        self._episode_route_file = self._build_episode_route_file()
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--route-files", self._episode_route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--start", "true",           # auto-start in GUI mode
            "--quit-on-end", "true",
        ]
        if self._seed is not None:
            sumo_cmd += ["--seed", str(self._seed)]
        if self._fcd_output_path is not None:
            sumo_cmd += ["--fcd-output", self._fcd_output_path]

        label = f"cav_env_{id(self)}"
        traci.start(sumo_cmd, label=label)
        self._conn = traci.getConnection(label)
        # print(f"SUMO started with label: {label}")

    @staticmethod
    def _allocate_flow(total_vph: float, base_weights: List[float]) -> List[float]:
        """Allocate aggregate flow across route flows proportional to base weights."""
        n = len(base_weights)
        if n == 0:
            return []
        total = max(float(total_vph), 0.0)
        if total <= 0.0:
            return [0.0] * n
        weights = np.asarray(base_weights, dtype=np.float64)
        if np.any(weights < 0):
            weights = np.maximum(weights, 0.0)
        wsum = float(weights.sum())
        if wsum <= 0.0:
            weights = np.ones(n, dtype=np.float64)
            wsum = float(n)
        return (total * (weights / wsum)).tolist()

    def _sample_episode_demand(self) -> Tuple[float, float, float, float]:
        """Sample (total_flow, cav_penetration, cav_flow, hdv_flow) for one episode."""
        if not self.randomize_demand:
            # Keep deterministic demand from template by preserving original per-type totals.
            tree = ET.parse(self.route_template)
            root = tree.getroot()
            cav_total = 0.0
            hdv_total = 0.0
            for flow in root.findall("flow"):
                try:
                    rate = float(flow.get("vehsPerHour", "0"))
                except ValueError:
                    rate = 0.0
                if flow.get("type") == "cav":
                    cav_total += rate
                else:
                    hdv_total += rate
            total = cav_total + hdv_total
            pen = (cav_total / total) if total > 0 else 0.0
            return total, pen, cav_total, hdv_total

        low = int(np.floor(self.total_flow_vph_min))
        high = int(np.floor(self.total_flow_vph_max))
        if high < low:
            high = low
        total = int(self._rng.integers(low, high + 1))
        pen = float(self._rng.uniform(self.cav_penetration_min, self.cav_penetration_max))
        cav = int(total * pen)
        hdv = max(total - cav, 0)
        return total, pen, cav, hdv

    def _build_episode_route_file(self) -> str:
        """
        Build a route file for the current episode with sampled
        total demand and CAV penetration.
        """
        total, pen, cav_total, hdv_total = self._sample_episode_demand()
        self._episode_total_flow_vph = total
        self._episode_cav_penetration = pen
        self._episode_cav_flow_vph = cav_total
        self._episode_hdv_flow_vph = hdv_total

        tree = ET.parse(self.route_template)
        root = tree.getroot()
        if self.sumo_default_cav_behavior:
            for vtype in root.findall("vType"):
                if vtype.get("id") == "cav":
                    for attr in list(vtype.attrib):
                        if attr.startswith("lc"):
                            vtype.attrib.pop(attr, None)
        if self.glosa_enabled:
            self._equip_cav_vtypes_with_glosa(root)
        cav_flows = [f for f in root.findall("flow") if f.get("type") == "cav"]
        hdv_flows = [f for f in root.findall("flow") if f.get("type") != "cav"]

        cav_weights = []
        for flow in cav_flows:
            try:
                cav_weights.append(float(flow.get("vehsPerHour", "0")))
            except ValueError:
                cav_weights.append(0.0)
        hdv_weights = []
        for flow in hdv_flows:
            try:
                hdv_weights.append(float(flow.get("vehsPerHour", "0")))
            except ValueError:
                hdv_weights.append(0.0)

        cav_rates = self._allocate_flow(cav_total, cav_weights)
        hdv_rates = self._allocate_flow(hdv_total, hdv_weights)

        for flow, rate in zip(cav_flows, cav_rates):
            flow.set("vehsPerHour", f"{rate:.6f}")
        for flow, rate in zip(hdv_flows, hdv_rates):
            flow.set("vehsPerHour", f"{rate:.6f}")

        out_dir = os.path.join(_SUMO_FILES_DIR, "generated_routes")
        os.makedirs(out_dir, exist_ok=True)
        seed_tag = "none" if self._seed is None else str(self._seed)
        nonce = int(self._rng.integers(0, 1_000_000_000))
        route_path = os.path.join(out_dir, f"episode_routes_seed{seed_tag}_{nonce}.rou.xml")
        tree.write(route_path, encoding="UTF-8", xml_declaration=True)
        return route_path

    def _equip_cav_vtypes_with_glosa(self, root: ET.Element) -> None:
        """Equip only the CAV vType with SUMO's built-in GLOSA device."""
        glosa_params = {
            "has.glosa.device": True,
            "device.glosa.range": self.glosa_range,
            "device.glosa.min-speed": self.glosa_min_speed,
            "device.glosa.max-speedfactor": self.glosa_max_speedfactor,
            "device.glosa.add-switchtime": self.glosa_add_switchtime,
            "device.glosa.override-safety": self.glosa_override_safety,
            "device.glosa.ignore-cfmodel": self.glosa_ignore_cfmodel,
            "device.glosa.use-queue": self.glosa_use_queue,
        }

        for vtype in root.findall("vType"):
            if vtype.get("id") != "cav":
                continue
            for key, value in glosa_params.items():
                param = next(
                    (p for p in vtype.findall("param") if p.get("key") == key),
                    None,
                )
                if param is None:
                    param = ET.SubElement(vtype, "param")
                param.set("key", key)
                if isinstance(value, bool):
                    param.set("value", str(value).lower())
                else:
                    param.set("value", f"{float(value):.6g}")

    def _close_sumo(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self._conn = None
        if self._episode_route_file is not None:
            try:
                os.remove(self._episode_route_file)
            except OSError:
                pass
            self._episode_route_file = None

    def close(self):
        self._close_sumo()

    @staticmethod
    def _parse_controlled_edges(controlled_edges) -> Optional[List[str]]:
        """Normalize optional controlled-edge configuration for OSM networks."""
        if controlled_edges in (None, "", []):
            return None
        if isinstance(controlled_edges, str):
            parts = controlled_edges.replace(",", " ").split()
            return [p.strip() for p in parts if p.strip()]
        return [str(edge).strip() for edge in controlled_edges if str(edge).strip()]

    def _terminal_green_wait(self, arrival_time_s: float) -> float:
        """Return waiting time until the next configured terminal green window."""
        cycle = max(float(self.terminal_cycle_s), 1e-6)
        start = float(self.terminal_green_start_s) % cycle
        end = float(self.terminal_green_end_s) % cycle
        phase = float(arrival_time_s) % cycle
        if start < end:
            if start <= phase < end:
                return 0.0
            if phase < start:
                return start - phase
            return cycle - phase + start
        if start > end:
            if phase >= start or phase < end:
                return 0.0
            return start - phase
        return 0.0

    # ── Per-step TraCI cache ───────────────────────────────────────────────

    def _invalidate_step_caches(self):
        """Clear values that are only valid for the current SUMO physics state."""
        self._vehicle_step_cache = {}
        self._neighbor_step_cache = {}
        self._control_area_vehicle_ids_cache = None
        self._active_cav_ids_cache = None
        self._all_vehicle_ids_cache = None
        self._signal_obs_step_cache = {}

    def _cached_vehicle_value(self, veh_id: str, key: str, getter):
        veh_cache = self._vehicle_step_cache.setdefault(veh_id, {})
        if key not in veh_cache:
            veh_cache[key] = getter()
        return veh_cache[key]

    def _veh_type_id(self, veh_id: str) -> str:
        return str(self._cached_vehicle_value(
            veh_id, "type_id", lambda: self._conn.vehicle.getTypeID(veh_id)
        ))

    def _veh_speed(self, veh_id: str) -> float:
        return float(self._cached_vehicle_value(
            veh_id, "speed", lambda: self._conn.vehicle.getSpeed(veh_id)
        ))

    def _veh_accel(self, veh_id: str) -> float:
        return float(self._cached_vehicle_value(
            veh_id, "accel", lambda: self._conn.vehicle.getAcceleration(veh_id)
        ))

    def _veh_lane_idx(self, veh_id: str) -> int:
        return int(self._cached_vehicle_value(
            veh_id, "lane_idx", lambda: self._conn.vehicle.getLaneIndex(veh_id)
        ))

    def _veh_lane_id(self, veh_id: str) -> str:
        return str(self._cached_vehicle_value(
            veh_id, "lane_id", lambda: self._conn.vehicle.getLaneID(veh_id)
        ))

    def _veh_lane_pos(self, veh_id: str) -> float:
        return float(self._cached_vehicle_value(
            veh_id, "lane_pos", lambda: self._conn.vehicle.getLanePosition(veh_id)
        ))

    def _veh_road_id(self, veh_id: str) -> str:
        return str(self._cached_vehicle_value(
            veh_id, "road_id", lambda: self._conn.vehicle.getRoadID(veh_id)
        ))

    def _veh_position(self, veh_id: str) -> Tuple[float, float]:
        pos = self._cached_vehicle_value(
            veh_id, "position", lambda: self._conn.vehicle.getPosition(veh_id)
        )
        return float(pos[0]), float(pos[1])

    def _veh_next_tls(self, veh_id: str):
        return self._cached_vehicle_value(
            veh_id, "next_tls", lambda: tuple(self._conn.vehicle.getNextTLS(veh_id))
        )

    def _lane_length(self, lane_id: str) -> float:
        cached = self._lane_length_cache.get(lane_id)
        if cached is None:
            cached = float(self._conn.lane.getLength(lane_id))
            self._lane_length_cache[lane_id] = cached
        return cached

    def _get_all_vehicle_ids(self) -> set:
        if self._all_vehicle_ids_cache is None:
            self._all_vehicle_ids_cache = set(self._conn.vehicle.getIDList())
        return set(self._all_vehicle_ids_cache)

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sumo()
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        self._start_sumo()

        self._step_count = 0
        self._prev_accels = {}
        self._prev_lane_idx = {}
        self._cav_entry_time = {}
        self._cav_entry_speed = {}
        self._cav_entry_lane_pos = {}
        self._prev_action_per_cav = {}
        self._pending_terminal_reward = {}
        self._active_cav_ids = []
        self._cumulative_reward = 0.0
        self.edge_controlled = set()
        self._trip50_tracker = {}
        self._tl_cycle_durations = {}
        self._tl_program_phases = {}
        self._last_neighbor_idx = None
        self._last_ctx_mask = None
        self._last_downstream_mask = None
        self._invalidate_step_caches()

        # Per-episode performance metrics (accumulated over all CAV-steps)
        self._ep_metrics = {
            'sum_speed': 0.0,        # m/s, accumulated across CAV-steps
            'sum_jerk': 0.0,          # m/s^3, accumulated across CAV-steps
            'sum_fuel_mL': 0.0,       # mL, accumulated across CAV-steps
            'sum_co2_mg': 0.0,        # mg, accumulated across CAV-steps
            'tet_count': 0,           # count of CAV-steps with TTC < threshold
            'cav_step_count': 0,      # total CAV-steps observed
            # Per-term reward accumulators (summed across all CAV-steps)
            'sum_r_speed': 0.0,
            'sum_r_accel': 0.0,
            'sum_r_jerk': 0.0,
            'sum_r_safety': 0.0,
            'sum_r_lc': 0.0,
            'sum_r_idle': 0.0,
            'sum_r_energy': 0.0,
            'sum_r_terminal': 0.0,    # sum of terminal rewards (travel-time + exit-speed)
            'num_terminals': 0,        # number of CAVs that reached terminal (exited)
            # Holistic segment metrics: in-edge entry -> out-edge + 50m
            'trip50_sum_time_s': 0.0,
            'trip50_sum_stop_time_s': 0.0,
            'trip50_sum_speed_mps': 0.0,
            'trip50_sum_step_speed': 0.0,
            'trip50_sum_jerk': 0.0,
            'trip50_sum_fuel_mL': 0.0,
            'trip50_sum_distance_m': 0.0,
            'trip50_tet_count': 0,
            'trip50_step_count': 0,
            'trip50_num_completed': 0,
        }
        # Additional 350m-segment metrics by vehicle class and mixed traffic.
        # Naming: trip50_<group>_* where group ∈ {cav, hdv, mix}.
        for grp in ("cav", "hdv", "mix"):
            p = f"trip50_{grp}_"
            self._ep_metrics[f"{p}sum_time_s"] = 0.0
            self._ep_metrics[f"{p}sum_stop_time_s"] = 0.0
            self._ep_metrics[f"{p}sum_speed_mps"] = 0.0
            self._ep_metrics[f"{p}sum_step_speed"] = 0.0
            self._ep_metrics[f"{p}sum_jerk"] = 0.0
            self._ep_metrics[f"{p}sum_fuel_mL"] = 0.0
            self._ep_metrics[f"{p}sum_distance_m"] = 0.0
            self._ep_metrics[f"{p}tet_count"] = 0
            self._ep_metrics[f"{p}step_count"] = 0
            self._ep_metrics[f"{p}num_completed"] = 0
        
        # By default, control toy-network incoming edges. OSM scenarios can
        # provide explicit Central Ave EB edge ids through controlled_edges.
        all_edges = set(self._conn.edge.getIDList())
        if self._configured_controlled_edges is None:
            edge_list = [edge for edge in all_edges if edge.endswith("_in")]
        else:
            missing = [edge for edge in self._configured_controlled_edges if edge not in all_edges]
            if missing:
                raise ValueError(f"Controlled edges not found in SUMO network: {missing}")
            edge_list = list(self._configured_controlled_edges)
        self.edge_controlled.update(edge_list)
        
        # Warm-up: run a few steps so vehicles enter the network
        for _ in range(50):
            vehicle_ids = self._conn.vehicle.getIDList()
            if len(vehicle_ids) > 0:
                break
            self._conn.simulationStep()

        self._invalidate_step_caches()
        self._current_time = self._conn.simulation.getTime()
        self._refresh_entry_steps()
        obs_dict = self._get_obs()
        info = {
            "num_cavs": len(self._active_cav_ids),
            "scenario_total_flow_vph": self._episode_total_flow_vph,
            "scenario_cav_penetration": self._episode_cav_penetration,
            "scenario_cav_flow_vph": self._episode_cav_flow_vph,
            "scenario_hdv_flow_vph": self._episode_hdv_flow_vph,
        }
        return obs_dict, info

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """
        Args:
            action: (N_t, 2) per-active-CAV actions. N_t matches the length
                    of self._active_cav_ids (set by the previous _get_obs call).
        """
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(0, _ACT_PER_CAV) if action.size == 0 else action.reshape(-1, _ACT_PER_CAV)
        action = np.clip(action, -1.0, 1.0)

        # Align actions with the CAVs that produced the observation.
        step_cav_ids = list(self._active_cav_ids)

        # Apply actions (indexed by the observation-time cav_ids)
        self._apply_actions(action, step_cav_ids)

        # Advance SUMO by delta_t
        for _ in range(self.steps_per_action):
            self._conn.simulationStep()
            self._invalidate_step_caches()
            self._current_time = self._conn.simulation.getTime()

        self._step_count += 1
        self._refresh_entry_steps()

        # Compute per-CAV reward for the CAVs that executed the action
        agent_rewards = self._compute_reward(action, step_cav_ids)

        # Scalar reward = sum over CAVs that were present during the action
        # n = max(len(step_cav_ids), 1)
        reward = float(agent_rewards.sum()) if len(step_cav_ids) > 0 else 0.0
        self._cumulative_reward += reward

        # Build new observation (may have different set/order of CAVs)
        obs_dict = self._get_obs()

        terminated = False
        truncated = self._step_count >= self.max_steps
        done = terminated or truncated

        info = {
            "num_cavs": len(self._active_cav_ids),
            "cumulative_reward": self._cumulative_reward,
            "success": float(self._cumulative_reward > 0),
            "terminated": terminated,
            "scenario_total_flow_vph": self._episode_total_flow_vph,
            "scenario_cav_penetration": self._episode_cav_penetration,
            "scenario_cav_flow_vph": self._episode_cav_flow_vph,
            "scenario_hdv_flow_vph": self._episode_hdv_flow_vph,
            # per-CAV rewards aligned with step_cav_ids (the action-time list)
            "agent_rewards": agent_rewards,
            "step_cav_ids": step_cav_ids,
        }

        # Aggregated performance metrics at episode end
        if done:
            n = max(self._ep_metrics['cav_step_count'], 1)
            n50_steps = int(self._ep_metrics.get('trip50_step_count', 0))
            use_trip50_perf = n50_steps > 0
            perf_n = max(n50_steps, 1) if use_trip50_perf else n

            info['total_travel_time'] = perf_n * self.sumo_step                     # s
            info['total_distance'] = (
                self._ep_metrics['trip50_sum_distance_m'] if use_trip50_perf
                else self._ep_metrics['sum_speed'] * self.sumo_step
            )                                                                        # m
            info['total_fuel_mL'] = (
                self._ep_metrics['trip50_sum_fuel_mL'] * self.sumo_step if use_trip50_perf
                else self._ep_metrics['sum_fuel_mL'] * self.sumo_step
            )                                                                        # mL
            info['avg_speed'] = (
                self._ep_metrics['trip50_sum_step_speed'] / perf_n if use_trip50_perf
                else self._ep_metrics['sum_speed'] / n
            )                                                                        # m/s
            info['avg_jerk'] = (
                self._ep_metrics['trip50_sum_jerk'] / perf_n if use_trip50_perf
                else self._ep_metrics['sum_jerk'] / n
            )                                                                        # m/s^3
            info['tet_seconds'] = (
                self._ep_metrics['trip50_tet_count'] if use_trip50_perf
                else self._ep_metrics['tet_count']
            ) * self.sumo_step                                                       # CAV-seconds with TTC < threshold
            info['tet_rate'] = (
                self._ep_metrics['trip50_tet_count'] / perf_n if use_trip50_perf
                else self._ep_metrics['tet_count'] / n
            )                                                                        # fraction of CAV-steps unsafe
            info['cav_step_count'] = n
            info['trip50_step_count'] = n50_steps

            # Per-term mean rewards (per CAV-step) — for diagnosing reward balance
            info['r_speed_mean']  = self._ep_metrics['sum_r_speed'] / n
            info['r_accel_mean']  = self._ep_metrics['sum_r_accel'] / n
            info['r_jerk_mean']   = self._ep_metrics['sum_r_jerk'] / n
            info['r_safety_mean'] = self._ep_metrics['sum_r_safety'] / n
            info['r_lc_mean']     = self._ep_metrics['sum_r_lc'] / n
            info['r_idle_mean']   = self._ep_metrics['sum_r_idle'] / n
            info['r_energy_mean'] = self._ep_metrics['sum_r_energy'] / n
            # Terminal reward: per-CAV mean (over CAVs that reached terminal)
            nt = max(self._ep_metrics['num_terminals'], 1)
            info['r_terminal_mean'] = self._ep_metrics['sum_r_terminal'] / nt
            info['num_terminals']   = self._ep_metrics['num_terminals']

            n50 = max(self._ep_metrics['trip50_num_completed'], 1)
            info['trip50_avg_time_s'] = self._ep_metrics['trip50_sum_time_s'] / n50
            info['trip50_avg_stop_time_s'] = self._ep_metrics['trip50_sum_stop_time_s'] / n50
            info['trip50_avg_speed_mps'] = self._ep_metrics['trip50_sum_speed_mps'] / n50
            info['trip50_num_completed'] = self._ep_metrics['trip50_num_completed']
            # By-group and mixed 350m segment metrics.
            for grp in ("cav", "hdv", "mix"):
                p = f"trip50_{grp}_"
                n_grp = max(int(self._ep_metrics[f"{p}num_completed"]), 1)
                s_grp = max(int(self._ep_metrics[f"{p}step_count"]), 1)
                info[f"trip50_{grp}_avg_time_s"] = self._ep_metrics[f"{p}sum_time_s"] / n_grp
                info[f"trip50_{grp}_avg_stop_time_s"] = self._ep_metrics[f"{p}sum_stop_time_s"] / n_grp
                info[f"trip50_{grp}_avg_speed_mps"] = self._ep_metrics[f"{p}sum_speed_mps"] / n_grp
                info[f"trip50_{grp}_num_completed"] = self._ep_metrics[f"{p}num_completed"]
                info[f"trip50_{grp}_avg_step_speed_mps"] = self._ep_metrics[f"{p}sum_step_speed"] / s_grp
                info[f"trip50_{grp}_avg_jerk"] = self._ep_metrics[f"{p}sum_jerk"] / s_grp
                info[f"trip50_{grp}_tet_rate"] = self._ep_metrics[f"{p}tet_count"] / s_grp
                info[f"trip50_{grp}_total_fuel_mL"] = self._ep_metrics[f"{p}sum_fuel_mL"] * self.sumo_step
                info[f"trip50_{grp}_total_distance_m"] = self._ep_metrics[f"{p}sum_distance_m"]
                info[f"trip50_{grp}_step_count"] = self._ep_metrics[f"{p}step_count"]

        return obs_dict, reward, done, info

    # ── Active CAV tracking ──────────────────────────────────────────────

    def _get_control_area_vehicles(self) -> List[str]:
        """Return sorted unique vehicle IDs currently on controlled incoming edges."""
        if self._control_area_vehicle_ids_cache is not None:
            return list(self._control_area_vehicle_ids_cache)

        vehs = set()
        for edge_id in self.edge_controlled:
            vehicle_ids = self._conn.edge.getLastStepVehicleIDs(edge_id)
            for v in vehicle_ids:
                vehs.add(v)
        self._control_area_vehicle_ids_cache = sorted(vehs)
        return list(self._control_area_vehicle_ids_cache)

    def _get_active_cavs(self) -> List[str]:
        """Return CAVs past the uncontrolled entry segment of incoming edges."""
        if self._active_cav_ids_cache is not None:
            return list(self._active_cav_ids_cache)

        all_vehs = self._get_control_area_vehicles()
        cavs = []
        for veh_id in all_vehs:
            try:
                is_cav = self._veh_type_id(veh_id) == "cav"
                past_entry_segment = self._veh_lane_pos(veh_id) > self.control_start_distance_m
                if is_cav and past_entry_segment:
                    cavs.append(veh_id)
            except traci.exceptions.TraCIException:
                continue
        cavs.sort()
        if len(cavs) > self.max_controllable:
            cavs = cavs[: self.max_controllable]
        self._active_cav_ids_cache = cavs
        return list(cavs)

    def _refresh_entry_steps(self):
        """Record entry step for new CAVs, drop bookkeeping for departed ones.
        Also computes a terminal reward for each CAV that just left the
        control area, buffered in self._pending_terminal_reward[cav_id]
        so _compute_reward can attach it to the exit step's reward."""
        current_ids = self._get_active_cavs()
        current = set(current_ids)
        self._update_trip50_metrics()
        # Register all vehicles (CAV + HDV) entering the controlled area for
        # 350m-segment performance accounting.
        for veh_id in self._get_control_area_vehicles():
            if veh_id in self._trip50_tracker:
                continue
            try:
                veh_type = self._veh_type_id(veh_id).lower()
                veh_group = "cav" if "cav" in veh_type else "hdv"
                accel0 = self._veh_accel(veh_id)
            except traci.exceptions.TraCIException:
                continue
            self._trip50_tracker[veh_id] = {
                "veh_group": veh_group,
                "entry_time": float(self._current_time),
                "last_time": float(self._current_time),
                "stop_time": 0.0,
                "sum_speed": 0.0,
                "sum_jerk": 0.0,
                "sum_fuel_mL": 0.0,
                "sum_distance_m": 0.0,
                "tet_count": 0,
                "step_count": 0,
                "prev_accel": accel0,
                "downstream_distance_m": 0.0,
            }
        # Register new arrivals (record entry step AND entry speed)
        for cav_id in current:
            if cav_id not in self._cav_entry_time:
                self._cav_entry_time[cav_id] = self._current_time
                try:
                    self._cav_entry_speed[cav_id] = self._veh_speed(cav_id)
                    self._cav_entry_lane_pos[cav_id] = self._veh_lane_pos(cav_id)
                    if self.rl_control:
                        if self.cav_control_mode in {"hybrid", "lane_change_only"}:
                            # Disable autonomous lane-change motivations while keeping
                            # safety checks for explicit TraCI changeLane commands.
                            self._conn.vehicle.setLaneChangeMode(cav_id, self.lane_change_mode)
                        else:
                            self._conn.vehicle.setLaneChangeMode(cav_id, self.lane_change_mode_release)
                        if self.cav_control_mode in {"lane_change_only", "sumo_default"}:
                            self._conn.vehicle.setSpeed(cav_id, -1)
                except traci.exceptions.TraCIException:
                    self._cav_entry_speed[cav_id] = self.max_speed_mps
        # Handle departures — compute terminal reward, then clean up
        gone = [c for c in self._cav_entry_time.keys() if c not in current]
        for c in gone:
            # ── Compute terminal reward BEFORE releasing control ──
            entry_time = self._cav_entry_time[c]
            entry_speed = max(self._cav_entry_speed.get(c, self.max_speed_mps), 0.0)
            actual_time = self._current_time - entry_time  # seconds
            # estimate ideal traversal time under best-case conditions: 
            # accelerate at max until reaching max speed, then cruise
            # t = l/vmax + (vmax-v0)^2 / (2·amax·vmax)
            controlled_distance = max(
                self.control_area_length_m - self._cav_entry_lane_pos.get(c, self.control_start_distance_m),
                1e-6,
            )
            t_free = (controlled_distance - 5.1) / self.max_speed_mps + \
                    ((self.max_speed_mps - entry_speed) ** 2) / (2 * _MAX_ACCEL * self.max_speed_mps)

            arrival_abs = entry_time + t_free
            wait = self._terminal_green_wait(arrival_abs)

            ideal_time = t_free + wait
            # Reward is positive when actual ≈ ideal, approaches 0 for slow traversals
            # r_term_time = _W_TERMINAL_TIME * ((controlled_distance - 5.1) / (actual_time * self.max_speed_mps)) ** 2
            r_term_time = _W_TERMINAL_TIME * min(ideal_time / max(actual_time, 1e-6), 1.0) ** 2

            # Exit speed — higher is better (agent not decelerating before exit)
            try:
                exit_speed = self._veh_speed(c)
            except traci.exceptions.TraCIException:
                exit_speed = 0.0
            r_term_speed = _W_TERMINAL_SPEED * (exit_speed / self.max_speed_mps) ** 2

            r_terminal = float(r_term_time + r_term_speed)
            self._pending_terminal_reward[c] = r_terminal
            self._ep_metrics['sum_r_terminal'] += r_terminal
            self._ep_metrics['num_terminals'] += 1

            # ── Release control and clean up bookkeeping ──
            try:
                if self.rl_control:
                    # Restore default SUMO lane-change behavior after leaving the
                    # controlled incoming edges.
                    self._conn.vehicle.setLaneChangeMode(c, self.lane_change_mode_release)
                    self._conn.vehicle.setSpeed(c, -1)
            except traci.exceptions.TraCIException:
                pass  # vehicle may have left the simulation entirely
            self._cav_entry_time.pop(c, None)
            self._cav_entry_speed.pop(c, None)
            self._cav_entry_lane_pos.pop(c, None)
            # Keep prev_* caches for one more reward computation so the
            # final transition can use dense terms + terminal bonus.

    def _update_trip50_metrics(self):
        """
        Track each vehicle (CAV/HDV) from controlled-area entry until it has travelled
        `post_out_eval_distance_m` after leaving the incoming edge.
        The downstream portion includes SUMO internal junction lanes plus
        the beginning of the outgoing edge, so the full segment length is
        approximately control_area_length_m + post_out_eval_distance_m.
        """
        if not self._trip50_tracker:
            return

        conn = self._conn
        target_downstream_m = max(self.post_out_eval_distance_m, 0.0)
        seg_len_m = self.control_area_length_m + target_downstream_m

        for veh_id in list(self._trip50_tracker.keys()):
            rec = self._trip50_tracker.get(veh_id)
            if rec is None:
                continue
            try:
                speed = self._veh_speed(veh_id)
                accel = self._veh_accel(veh_id)
                edge_id = self._veh_road_id(veh_id)
            except traci.exceptions.TraCIException:
                # Vehicle left simulation before reaching out-distance target.
                self._trip50_tracker.pop(veh_id, None)
                continue

            dt = max(float(self._current_time) - float(rec["last_time"]), 0.0)
            if speed <= 2.0:
                rec["stop_time"] += dt
            rec["last_time"] = float(self._current_time)

            jerk = abs(accel - float(rec.get("prev_accel", accel))) / max(self.sumo_step, 1e-6)
            rec["prev_accel"] = accel
            fuel_rate_mLps = self._get_energy(speed - accel * self.sumo_step / 2.0, accel)
            ttc_below_thresh = False
            leader = self._get_neighbor(veh_id, "leader", 0)
            if leader is not None and leader[0] != "":
                leader_speed = self._veh_speed(leader[0])
                speed_diff = speed - leader_speed + 1e-6 if speed >= leader_speed else -1.0
                ttc = (leader[1] + 2.0) / speed_diff if speed_diff > 0 else float('inf')
                ttc_below_thresh = ttc < _TTC_THRESH

            rec["sum_speed"] += speed
            rec["sum_jerk"] += jerk
            rec["sum_fuel_mL"] += fuel_rate_mLps
            rec["sum_distance_m"] += speed * self.sumo_step
            rec["tet_count"] += int(ttc_below_thresh)
            rec["step_count"] += 1
            if edge_id not in self.edge_controlled:
                rec["downstream_distance_m"] += speed * self.sumo_step

            if rec["downstream_distance_m"] >= target_downstream_m:
                travel_t = max(float(self._current_time) - float(rec["entry_time"]), 0.0)
                avg_spd = seg_len_m / max(travel_t, 1e-6)
                self._ep_metrics['trip50_sum_time_s'] += travel_t
                self._ep_metrics['trip50_sum_stop_time_s'] += float(rec["stop_time"])
                self._ep_metrics['trip50_sum_speed_mps'] += avg_spd
                self._ep_metrics['trip50_sum_step_speed'] += float(rec["sum_speed"])
                self._ep_metrics['trip50_sum_jerk'] += float(rec["sum_jerk"])
                self._ep_metrics['trip50_sum_fuel_mL'] += float(rec["sum_fuel_mL"])
                self._ep_metrics['trip50_sum_distance_m'] += float(rec["sum_distance_m"])
                self._ep_metrics['trip50_tet_count'] += int(rec["tet_count"])
                self._ep_metrics['trip50_step_count'] += int(rec["step_count"])
                self._ep_metrics['trip50_num_completed'] += 1
                grp = str(rec.get("veh_group", "hdv")).lower()
                if grp not in ("cav", "hdv"):
                    grp = "hdv"
                for dst_grp in (grp, "mix"):
                    p = f"trip50_{dst_grp}_"
                    self._ep_metrics[f"{p}sum_time_s"] += travel_t
                    self._ep_metrics[f"{p}sum_stop_time_s"] += float(rec["stop_time"])
                    self._ep_metrics[f"{p}sum_speed_mps"] += avg_spd
                    self._ep_metrics[f"{p}sum_step_speed"] += float(rec["sum_speed"])
                    self._ep_metrics[f"{p}sum_jerk"] += float(rec["sum_jerk"])
                    self._ep_metrics[f"{p}sum_fuel_mL"] += float(rec["sum_fuel_mL"])
                    self._ep_metrics[f"{p}sum_distance_m"] += float(rec["sum_distance_m"])
                    self._ep_metrics[f"{p}tet_count"] += int(rec["tet_count"])
                    self._ep_metrics[f"{p}step_count"] += int(rec["step_count"])
                    self._ep_metrics[f"{p}num_completed"] += 1
                self._trip50_tracker.pop(veh_id, None)

    def _build_knn_context(
        self,
        own_obs: np.ndarray,
        positions: np.ndarray,
        edges: List[str],
        lane_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each ego CAV, select its K nearest neighbors (by Euclidean distance
        on (x, y) positions) within communication range and stack them into
        ctx_obs.

        Args:
            own_obs:   (N, obs_per_cav)
            positions: (N, 2)
        Returns:
            ctx_obs:      (N, K, obs_per_cav)
            ctx_mask:     (N, K)  1.0 valid, 0.0 padded
            neighbor_idx: (N, K)  index into cav_ids list (-1 where padded)
        """
        N = own_obs.shape[0]
        K = self._K
        ctx_obs = np.zeros((N, K, _OBS_PER_CAV), dtype=np.float32)
        ctx_mask = np.zeros((N, K), dtype=np.float32)
        neighbor_idx_out = -np.ones((N, K), dtype=np.int64)

        if N == 0:
            return ctx_obs, ctx_mask, neighbor_idx_out

        # Pairwise squared distances with self excluded (diagonal = +inf)
        diffs = positions[:, None, :] - positions[None, :, :]    # (N, N, 2)
        d2 = (diffs ** 2).sum(-1)                                 # (N, N)
        np.fill_diagonal(d2, np.inf)

        # Communication-range filter: only CAVs within radius are eligible.
        if self.communication_range_m > 0:
            range2 = self.communication_range_m ** 2
            d2[d2 > range2] = np.inf

        # Topology filter:
        # - full: no extra filtering
        # - front_only: only same-edge vehicles ahead of ego are eligible
        if self.comm_topology == "front_only":
            edge_arr = np.asarray(edges)
            same_edge = edge_arr[:, None] == edge_arr[None, :]
            ahead = lane_positions[None, :] > lane_positions[:, None]
            allowed = same_edge & ahead
            d2[~allowed] = np.inf

        # Sort neighbors by distance for every ego, then keep finite entries only.
        sort_idx = np.argsort(d2, axis=1)                          # (N, N)
        for i in range(N):
            candidates = sort_idx[i]
            finite = np.isfinite(d2[i, candidates])
            nbrs = candidates[finite][:K]
            k = len(nbrs)
            if k == 0:
                continue
            ctx_obs[i, :k] = own_obs[nbrs]
            ctx_mask[i, :k] = 1.0
            neighbor_idx_out[i, :k] = nbrs
        return ctx_obs, ctx_mask, neighbor_idx_out

    # ── Directional neighbor mask ────────────────────────────────────────

    _SIDE_THRESHOLD_M = 5.0   # parallel cars within ±5 m count as downstream

    def _compute_downstream_mask(
        self,
        neighbor_idx: np.ndarray,   # (N, K), -1 = padded
        edges: List[str],            # length N
        lane_positions: np.ndarray,  # (N,)
    ) -> np.ndarray:
        """
        Return a (N, K) float32 mask where entry (i, k) is 1.0 iff neighbor k
        is "downstream" of ego i in the traffic flow — i.e., ego's action
        can physically influence it. This holds when:
            - neighbor and ego are on the SAME edge, AND
            - neighbor's lane_pos <= ego_pos + side_threshold
              (neighbor is behind or roughly parallel to ego).

        Neighbors on different edges, strictly ahead of ego, or padded
        slots (index = -1) are marked 0 (ego cannot influence them through
        its control action).

        Vectorized implementation — O(N·K) numpy ops with no Python loops.
        """
        N, K = neighbor_idx.shape
        if N == 0:
            return np.zeros((N, K), dtype=np.float32)

        # Map edge strings to integer codes so we can broadcast-compare
        # across (N, K) without Python string operations.
        unique_edges, edge_codes = np.unique(np.asarray(edges), return_inverse=True)
        edge_codes = edge_codes.astype(np.int64)           # (N,)

        # Safe gather: replace -1 with 0 (any valid index), then filter
        # out padded entries at the end via padding_mask.
        safe_idx = np.where(neighbor_idx >= 0, neighbor_idx, 0)   # (N, K)
        padding_mask = (neighbor_idx >= 0)                        # (N, K)

        # Per-pair edge equality: ego_edge_codes[i] == edge_codes[nbr_idx[i, k]]
        same_edge = edge_codes[:, None] == edge_codes[safe_idx]   # (N, K)

        # Per-pair "behind or parallel": lane_pos[nbr] <= lane_pos[ego] + δ
        pos_diff = lane_positions[safe_idx] - lane_positions[:, None]   # (N, K)
        is_behind_or_parallel = pos_diff <= self._SIDE_THRESHOLD_M      # (N, K)

        mask = padding_mask & same_edge & is_behind_or_parallel          # (N, K) bool
        return mask.astype(np.float32)

    # ── Observations ─────────────────────────────────────────────────────

    def _get_single_cav_obs(self, veh_id: str) -> np.ndarray:
        """Build observation vector for one CAV."""
        conn = self._conn
        obs = np.zeros(_OBS_PER_CAV, dtype=np.float32)

        # -- Own state --
        speed = self._veh_speed(veh_id)
        accel = self._veh_accel(veh_id)
        lane_idx = self._veh_lane_idx(veh_id)
        lane_pos = self._veh_lane_pos(veh_id)
        current_leader = self._get_neighbor(veh_id, "leader", 0)

        obs[0] = speed / self.max_speed_mps
        obs[1] = accel / _MAX_ACCEL if accel >= 0 else -accel / _MAX_DECEL
        obs[2 + lane_idx] = 1.0        # one-hot lane index (assumes max 3 lanes per edge)
        obs[5] = np.clip(lane_pos / self.control_area_length_m, 0.0, 1.0)
        follow_mode = self._get_following_mode_onehot(
            veh_id, leader=current_leader, leader_known=True
        )
        obs[6:9] = follow_mode

        # -- Neighbors (leader, follower per lane: current, left, right) --
        idx = _OBS_CAV_OWN
        for mode in ["leader", "follower"]:
            for lane_offset in [0, 1, -1]:  # current, left, right
                target_lane = lane_idx + lane_offset
                if target_lane < 0 or target_lane > 2:
                    # Virtual neighbor: far distance, zero rel speed, exists=0.
                    obs[idx] = 1.0
                    obs[idx + 1] = 0.0
                    obs[idx + 2] = 0.0
                else:
                    if mode == "leader" and lane_offset == 0:
                        neighbor = current_leader
                    else:
                        neighbor = self._get_neighbor(veh_id, mode, lane_offset)
                    if neighbor is not None:
                        n_id, n_dist = neighbor
                        n_speed = self._veh_speed(n_id)
                        obs[idx] = min(n_dist, _MAX_DETECT_DIST) / _MAX_DETECT_DIST
                        obs[idx + 1] = (n_speed - speed) / self.max_speed_mps
                        obs[idx + 2] = 1.0
                    else:
                        obs[idx] = 1.0
                        obs[idx + 1] = 0.0
                        obs[idx + 2] = 0.0
                idx += _OBS_NEIGHBOR

        # -- Traffic signal information --
        tl_obs = self._get_traffic_signal_obs(veh_id)
        obs[idx: idx + _OBS_SIGNAL] = tl_obs
        idx += _OBS_SIGNAL

        # -- Previous action (state augmentation; zeros for fresh arrivals) --
        prev_a = self._prev_action_per_cav.get(veh_id)
        if prev_a is not None:
            obs[idx: idx + _OBS_PREV_ACTION] = prev_a
        # else: leave as zeros

        return obs

    def _get_following_mode_onehot(
        self,
        veh_id: str,
        leader: Optional[Tuple[str, float]] = None,
        leader_known: bool = False,
    ) -> np.ndarray:
        """
        Return [is_head, is_following_hdv, is_following_cav] for current lane leader.
        - is_head: no leader detected within _MAX_DETECT_DIST.
        - is_following_hdv/cav: leader exists and is that type.
        """
        mode = np.zeros(3, dtype=np.float32)
        if not leader_known:
            leader = self._get_neighbor(veh_id, mode="leader", lane_offset=0)
        if leader is None:
            mode[0] = 1.0
            return mode
        try:
            lead_id, _ = leader
            lead_type = self._veh_type_id(lead_id).lower()
            if "cav" in lead_type:
                mode[2] = 1.0
            else:
                mode[1] = 1.0
        except traci.exceptions.TraCIException:
            mode[0] = 1.0
        return mode

    def _get_neighbor(
        self, veh_id: str, mode: str, lane_offset: int
    ) -> Optional[Tuple[str, float]]:
        """
        Get leader or follower on the lane with given offset.
        Returns (neighbor_id, distance) or None.
        """
        cache_key = (veh_id, mode, int(lane_offset))
        if cache_key in self._neighbor_step_cache:
            return self._neighbor_step_cache[cache_key]

        conn = self._conn
        result = None
        try:
            if mode == "leader":
                if lane_offset == 0:
                    traci_result = conn.vehicle.getLeader(veh_id, _MAX_DETECT_DIST)
                    if traci_result is not None and traci_result[0] != "":
                        result = traci_result  # (id, dist)
                else:
                    # Use left/right leader
                    if lane_offset == 1:  # left: SUMO lane index increases to the left
                        neighbors = conn.vehicle.getLeftLeaders(veh_id, blockingOnly=False)
                    else:  # right
                        neighbors = conn.vehicle.getRightLeaders(veh_id, blockingOnly=False)
                    
                    if neighbors.__len__() > 0:
                        result = neighbors[0]  # Return the first neighbor found
            else:  # follower
                if lane_offset == 0:
                    traci_result = conn.vehicle.getFollower(veh_id, _MAX_DETECT_DIST)
                    if traci_result is not None and traci_result[0] != "":
                        result = traci_result
                else:
                    # Use left/right follower
                    if lane_offset == 1:  # left: SUMO lane index increases to the left
                        neighbors = conn.vehicle.getLeftFollowers(veh_id, blockingOnly=False)
                    else:  # right
                        neighbors = conn.vehicle.getRightFollowers(veh_id, blockingOnly=False)
                    
                    if neighbors.__len__() > 0:
                        result = neighbors[0]  # Return the first neighbor found
        except traci.exceptions.TraCIException:
            result = None

        self._neighbor_step_cache[cache_key] = result
        return result

    def _get_traffic_signal_obs(self, veh_id: str) -> np.ndarray:
        """
        Returns traffic signal observation:
            [is_green_current, is_yellow_current,
             is_green_opposed, is_yellow_opposed,
             T_l, T'_l]
        where T_l/T'_l follow the paper's cycle representation:
            T_l = cos(map)
            T'_l = -sin(map) * omega
        with piecewise map/omega for green and non-green segments.
        """
        signal_obs = np.zeros(_OBS_SIGNAL, dtype=np.float32)
        conn = self._conn

        try:
            # Get the next traffic light for this vehicle
            tls_list = self._veh_next_tls(veh_id)
            if len(tls_list) == 0:
                return signal_obs

            tl_id, tl_idx, tl_dist, tl_state = tls_list[0]
            cache_key = (str(tl_id), int(tl_idx), bool(self.communication))
            cached = self._signal_obs_step_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

            # Current SUMO phase index (0-3)
            sumo_phase = conn.trafficlight.getPhase(tl_id)

            # Time to next phase transition.
            next_switch = conn.trafficlight.getNextSwitch(tl_id)
            current_time = self._current_time
            time_to_switch = max(0.0, next_switch - current_time)

            # Current and opposed-edge signal states.
            phase_state = conn.trafficlight.getRedYellowGreenState(tl_id)
            curr_ch = tl_state
            if not curr_ch:
                try:
                    curr_ch = phase_state[int(tl_idx)]
                except Exception:
                    curr_ch = "r"

            if curr_ch in ("G", "g"):
                signal_obs[0] = 1.0
            elif curr_ch in ("y", "Y"):
                signal_obs[1] = 1.0
            opp_green, opp_yellow = self._opposed_edge_signal_flags(
                phase_state=phase_state,
                tl_idx=int(tl_idx),
            )
            signal_obs[2] = opp_green
            signal_obs[3] = opp_yellow

            # AV setting (no V2X): only 4 discrete state flags are observable.
            if not self.communication:
                self._signal_obs_step_cache[cache_key] = signal_obs.copy()
                return signal_obs

            t_l, t_l_dot = self._paper_light_cycle_terms(
                tl_id=tl_id,
                tl_idx=int(tl_idx),
                current_phase=sumo_phase,
                current_remaining=time_to_switch,
            )
            signal_obs[4] = t_l
            signal_obs[5] = t_l_dot
            self._signal_obs_step_cache[cache_key] = signal_obs.copy()
        except traci.exceptions.TraCIException:
            pass

        return signal_obs

    def _opposed_edge_signal_flags(self, phase_state: str, tl_idx: int) -> Tuple[float, float]:
        """
        Return (is_green_opposed, is_yellow_opposed) based on TLS state string.

        For the 4-approach straight-only layout, link indices are grouped as:
            N:[0..2], S:[3..5], E:[6..8], W:[9..11]
        Opposed-edge here means the crossing movement group (NS vs EW).
        """
        if not phase_state:
            return 0.0, 0.0
        state_len = len(phase_state)
        if tl_idx < 0:
            return 0.0, 0.0

        group = tl_idx // 3
        if group in (0, 1):        # current in N/S group => opposed is E/W group
            opposed_groups = (2, 3)
        elif group in (2, 3):      # current in E/W group => opposed is N/S group
            opposed_groups = (0, 1)
        else:
            return 0.0, 0.0

        opposed_chars = []
        for g in opposed_groups:
            for idx in range(g * 3, g * 3 + 3):
                if 0 <= idx < state_len:
                    opposed_chars.append(phase_state[idx])

        opp_green = 1.0 if any(ch in ("G", "g") for ch in opposed_chars) else 0.0
        opp_yellow = 1.0 if any(ch in ("y", "Y") for ch in opposed_chars) else 0.0
        return opp_green, opp_yellow

    def _paper_light_cycle_terms(
        self,
        tl_id: str,
        tl_idx: int,
        current_phase: int,
        current_remaining: float,
    ) -> Tuple[float, float]:
        """
        Paper-style light-cycle features (T_l, T'_l) for one ego-facing TLS link.
        url: https://ieeexplore.ieee.org/abstract/document/8848852
        Eq.(10)-(12) with piecewise map for green vs non-green segments.
        """
        phases = self._get_tl_program_phases(tl_id)
        if not phases:
            return 0.0, 0.0

        n = len(phases)
        phase_idx = int(current_phase) % n

        def _phase_dur(i: int) -> float:
            return max(float(phases[i][0]), 0.0)

        def _is_green(i: int) -> bool:
            state = phases[i][1]
            if 0 <= tl_idx < len(state):
                return state[tl_idx] in {"G", "g"}
            return False

        cur_is_green = _is_green(phase_idx)
        cur_dur = _phase_dur(phase_idx)
        rem = float(current_remaining)
        rem = max(0.0, min(rem, cur_dur)) if cur_dur > 0 else 0.0
        cur_elapsed = max(cur_dur - rem, 0.0)

        # Find start of the current homogeneous segment (green or non-green).
        run_start = phase_idx
        for _ in range(n - 1):
            prev_idx = (run_start - 1) % n
            if _is_green(prev_idx) != cur_is_green:
                break
            run_start = prev_idx
            if run_start == phase_idx:
                break

        run_indices: List[int] = [run_start]
        for _ in range(n - 1):
            nxt = (run_indices[-1] + 1) % n
            if nxt == run_start or _is_green(nxt) != cur_is_green:
                break
            run_indices.append(nxt)

        run_total = sum(_phase_dur(i) for i in run_indices)
        if run_total <= 1e-8:
            return 0.0, 0.0

        run_elapsed = 0.0
        for i in run_indices:
            if i == phase_idx:
                run_elapsed += cur_elapsed
                break
            run_elapsed += _phase_dur(i)
        run_elapsed = float(np.clip(run_elapsed, 0.0, run_total))

        # Paper Eq.(10)-(12): piecewise map for green vs non-green segment.
        omega = np.pi / run_total
        if cur_is_green:
            map_val = omega * run_elapsed + np.pi
        else:
            map_val = omega * run_elapsed
        t_l = float(np.cos(map_val))
        t_l_dot = float(-np.sin(map_val) * omega)
        return t_l, t_l_dot

    def _get_tl_program_phases(self, tl_id: str) -> List[Tuple[float, str]]:
        """Return [(duration_seconds, state_string), ...] for the active TLS program."""
        cached = self._tl_program_phases.get(tl_id)
        if cached is not None:
            return cached

        phases: List[Tuple[float, str]] = []
        conn = self._conn
        try:
            current_program = conn.trafficlight.getProgram(tl_id)
            logics = conn.trafficlight.getAllProgramLogics(tl_id)
            logic = next(
                (item for item in logics if getattr(item, "programID", None) == current_program),
                logics[0] if logics else None,
            )
            if logic is not None:
                phases = [
                    (float(phase.duration), str(phase.state))
                    for phase in logic.phases
                ]
        except traci.exceptions.TraCIException:
            phases = []

        self._tl_program_phases[tl_id] = phases
        return phases

    def _green_remaining_time(
        self,
        tl_id: str,
        tl_idx: int,
        current_phase: int,
        current_remaining: float,
    ) -> float:
        """Seconds of currently available green for this link (0 if not green now)."""
        phases = self._get_tl_program_phases(tl_id)
        if not phases:
            return 0.0

        green_states = {"G", "g"}
        num_phases = len(phases)
        phase_idx = current_phase % num_phases

        def link_state(idx: int) -> str:
            state = phases[idx][1]
            if 0 <= tl_idx < len(state):
                return state[tl_idx]
            return ""

        if link_state(phase_idx) not in green_states:
            return 0.0

        remaining = max(float(current_remaining), 0.0)
        for step in range(1, num_phases + 1):
            phase_idx = (current_phase + step) % num_phases
            if link_state(phase_idx) not in green_states:
                return remaining
            remaining += max(float(phases[phase_idx][0]), 0.0)

        return min(remaining, max(self._get_tl_cycle_duration(tl_id), 1.0))

    def _cycle_remaining_time(
        self,
        tl_id: str,
        current_phase: int,
        current_remaining: float,
    ) -> float:
        """Seconds until the end of current program cycle (phase index wraps to 0)."""
        phases = self._get_tl_program_phases(tl_id)
        if not phases:
            return max(float(current_remaining), 0.0)

        num_phases = len(phases)
        phase_idx = current_phase % num_phases
        remaining = max(float(current_remaining), 0.0)

        for step in range(1, num_phases + 1):
            next_idx = (phase_idx + step) % num_phases
            if next_idx == 0:
                break
            remaining += max(float(phases[next_idx][0]), 0.0)

        return min(remaining, max(self._get_tl_cycle_duration(tl_id), 1.0))

    def _get_tl_cycle_duration(self, tl_id: str) -> float:
        """Return full signal cycle duration in seconds for normalization."""
        cached = self._tl_cycle_durations.get(tl_id)
        if cached is not None:
            return cached

        conn = self._conn
        cycle_duration = sum(duration for duration, _ in self._get_tl_program_phases(tl_id))

        if cycle_duration <= 0.0:
            try:
                cycle_duration = float(conn.trafficlight.getPhaseDuration(tl_id))
            except traci.exceptions.TraCIException:
                cycle_duration = 1.0

        self._tl_cycle_durations[tl_id] = cycle_duration
        return cycle_duration

    def _get_obs(self) -> Dict:
        """Build per-CAV observations + K-NN context + cycle identity ids.

        Returns:
            Dict with keys:
              obs:            (N, _OBS_PER_CAV)   own obs per active CAV
              ctx_obs:        (N, K, _OBS_PER_CAV) K nearest neighbors' obs
              ctx_mask:       (N, K)              active mask over context
              entry_phase_id: (N,) int64          entry_step % vid_cycle
              progress_id:    (N,) int64          (now - entry) % vid_cycle
              cav_ids:        List[str] length N  SUMO CAV IDs
        """
        cav_ids = self._get_active_cavs()
        N = len(cav_ids)
        K = self._K

        if N == 0:
            self._active_cav_ids = []
            return {
                'obs': np.zeros((0, _OBS_PER_CAV), dtype=np.float32),
                'ctx_obs': np.zeros((0, K, _OBS_PER_CAV), dtype=np.float32),
                'ctx_mask': np.zeros((0, K), dtype=np.float32),
                'entry_phase_id': np.zeros(0, dtype=np.int64),
                'progress_id': np.zeros(0, dtype=np.int64),
                'cav_ids': [],
            }

        own_obs = np.zeros((N, _OBS_PER_CAV), dtype=np.float32)
        positions = np.zeros((N, 2), dtype=np.float32)
        entry_phase = np.zeros(N, dtype=np.int64)
        progress = np.zeros(N, dtype=np.int64)
        # Per-CAV edge + lane position, used to build a directional
        # (downstream-only) mask for the cooperative reward augmentation.
        edges: List[str] = [''] * N
        lane_positions = np.zeros(N, dtype=np.float32)

        valid_mask = np.ones(N, dtype=bool)
        for i, cav_id in enumerate(cav_ids):
            try:
                own_obs[i] = self._get_single_cav_obs(cav_id)
                x, y = self._veh_position(cav_id)
                positions[i] = (x, y)
                edges[i] = self._veh_road_id(cav_id)
                lane_positions[i] = self._veh_lane_pos(cav_id)
                entry_step = self._cav_entry_time.get(cav_id, self._step_count)
                entry_phase[i] = entry_step % self.vid_cycle
                progress[i] = (self._step_count - entry_step) % self.vid_cycle
            except traci.exceptions.TraCIException:
                valid_mask[i] = False

        # Drop any CAVs that failed to read (likely just left the network)
        if not valid_mask.all():
            cav_ids = [c for c, v in zip(cav_ids, valid_mask) if v]
            own_obs = own_obs[valid_mask]
            positions = positions[valid_mask]
            entry_phase = entry_phase[valid_mask]
            progress = progress[valid_mask]
            edges = [e for e, v in zip(edges, valid_mask) if v]
            lane_positions = lane_positions[valid_mask]
            N = len(cav_ids)

        ctx_obs, ctx_mask, neighbor_idx = self._build_knn_context(
            own_obs, positions, edges, lane_positions
        )
        downstream_mask = self._compute_downstream_mask(
            neighbor_idx, edges, lane_positions
        )

        self._active_cav_ids = list(cav_ids)
        # Persist K-NN indices + ctx_mask so _compute_reward of the NEXT step
        # can reuse them for the cooperative reward augmentation without
        # recomputing distances. These indices point into self._active_cav_ids
        # (which equals step_cav_ids at the next step).
        self._last_neighbor_idx = neighbor_idx
        self._last_ctx_mask = ctx_mask
        self._last_downstream_mask = downstream_mask
        return {
            'obs': own_obs,
            'ctx_obs': ctx_obs,
            'ctx_mask': ctx_mask,
            'entry_phase_id': entry_phase,
            'progress_id': progress,
            'cav_ids': list(cav_ids),
        }
        
    def _get_energy(self, speed: float, acc: float):
        """Calculate the instantanous energy consumption based on speed and acceleration
        
        Args:
            speed: current speed, m/s
            acc: acceleartion, m/s^2  
        Output:
            energy: fuel consumption in mL for this step (ml/s)
        """
        # ARRB Fuel consumption model
        # https://www.sciencedirect.com/science/article/abs/pii/0191261589900143
        # F(v, a) = alpha + max{beta1*v*Rt + beta2*m*v*a^2, 0} if a > 0 else max{beta1*v*Rt, 0}
        # R_t = b1 + b2 *v^2 + m*a + m*g*G
        m = 1600  # kg
        g = 9.81  # m/s^2
        G = 0  # Grade
        alpha = 0.666
        beta1 = 0.0717
        beta2 = 0.0344
        b1 = 0.269
        b2 = 0.0171
        b3 = 0.000672

        
        v = speed
        a = acc
        
        Rt = b1 + b2 * v  + b3 * v ** 2 + m * a / 1000 + m * g * G / 100000
        acc_Energy = beta1 * v * Rt + beta2 * m * v * a ** 2 / 1000 if a > 0 else beta1 * v * Rt
        energy= alpha + np.clip(acc_Energy, 0, 200)
        
        return energy
    
    # ── Actions ──────────────────────────────────────────────────────────

    def _apply_actions(self, action: np.ndarray, cav_ids: List[str]):
        """Apply acceleration and lane-change commands to each active CAV.

        Args:
            action: (len(cav_ids), _ACT_PER_CAV) per-CAV actions
            cav_ids: list of SUMO CAV IDs, aligned row-wise with action
        """
        if not self.rl_control or self.cav_control_mode == "sumo_default":
            for i, cav_id in enumerate(cav_ids):
                if i >= len(action):
                    continue
                self._prev_action_per_cav[cav_id] = np.asarray(
                    action[i], dtype=np.float32
                )
            return

        conn = self._conn

        for i, cav_id in enumerate(cav_ids):
            try:
                accel_cmd = action[i, 0]       # [-1, 1]
                lc_cmd = action[i, 1]          # [-1, 1]

                if self.cav_control_mode in {"hybrid", "longitudinal_only"}:
                    # ── Longitudinal: map [-1,1] → [MAX_DECEL, MAX_ACCEL] ──
                    if accel_cmd >= 0:
                        accel = accel_cmd * _MAX_ACCEL
                    else:
                        accel = -accel_cmd * _MAX_DECEL  # _MAX_DECEL is negative

                    current_speed = self._veh_speed(cav_id)
                    new_speed = max(0.0, current_speed + accel * self.sumo_step)
                    new_speed = min(new_speed, self.max_speed_mps)
                    conn.vehicle.setSpeed(cav_id, new_speed)
                else:
                    # SUMO default car-following handles longitudinal motion.
                    conn.vehicle.setSpeed(cav_id, -1)

                if self.cav_control_mode in {"hybrid", "lane_change_only"}:
                    # ── Lateral: discretize lane-change command ──
                    lane_idx = self._veh_lane_idx(cav_id)
                    if lc_cmd > _LANE_CHANGE_THRESH and lane_idx < 2: # left lane change
                        conn.vehicle.changeLane(cav_id, lane_idx + 1, self.sumo_step)
                    elif lc_cmd < -_LANE_CHANGE_THRESH and lane_idx > 0: # right lane change
                        conn.vehicle.changeLane(cav_id, lane_idx - 1, self.sumo_step)
                    else: # stay in lane
                        conn.vehicle.changeLane(cav_id, lane_idx, self.sumo_step)

                # Record this action as the CAV's latest prev-action so the
                # next _get_single_cav_obs can fold it into the observation.
                self._prev_action_per_cav[cav_id] = np.asarray(
                    (accel_cmd, lc_cmd), dtype=np.float32
                )

            except traci.exceptions.TraCIException:
                pass

    # ── Reward ───────────────────────────────────────────────────────────

    def _compute_reward(
        self, action: np.ndarray, cav_ids: List[str]
    ) -> np.ndarray:
        """
        Compute per-CAV rewards aligned with `cav_ids`.

        Args:
            action: (len(cav_ids), _ACT_PER_CAV) actions that were applied
            cav_ids: CAV IDs at the time actions were issued

        Returns:
            rewards: (len(cav_ids),) per-CAV rewards (0 for CAVs that left)
        """
        conn = self._conn
        N = len(cav_ids)
        rewards = np.zeros(N, dtype=np.float32)

        if N == 0:
            return rewards

        # # Check collisions
        # collisions = conn.simulation.getCollidingVehiclesIDList()
        current = set(self._get_active_cavs())
        all_veh_ids = None

        for i, cav_id in enumerate(cav_ids):
            terminal_bonus = float(self._pending_terminal_reward.pop(cav_id, 0.0))
            exited_control = (cav_id not in current)

            # Vehicle fully left simulation: no dense term available.
            if exited_control and all_veh_ids is None:
                all_veh_ids = self._get_all_vehicle_ids()
            if exited_control and cav_id not in all_veh_ids:
                rewards[i] = terminal_bonus
                if terminal_bonus != 0.0:
                    self._prev_accels.pop(cav_id, None)
                    self._prev_lane_idx.pop(cav_id, None)
                    self._prev_action_per_cav.pop(cav_id, None)
                continue
            try:
                speed = self._veh_speed(cav_id)
                accel = self._veh_accel(cav_id)
                curr_lane_idx = self._veh_lane_idx(cav_id)
                prev_accel = self._prev_accels.get(cav_id, accel)
                accel_desired = action[i, 0] * _MAX_ACCEL if action[i, 0] >= 0 else -action[i, 0] * _MAX_DECEL

                # 1. Speed reward: r_speed = w · (v / v_max)
                speed_error = speed / self.max_speed_mps
                r_speed = _W_SMOOTH_SPEED * speed_error - 0.05

                # 2. Acceleration penalty: r_accel = -w · (a / a_max)^2 + (a > a_actual)
                acc_penalty = ((accel / _MAX_ACCEL) ** 2 if accel >= 0 else (accel / _MAX_DECEL) ** 2)
                acc_inconsistency_penalty = ((accel_desired - accel)/(_MAX_ACCEL - _MAX_DECEL)) ** 2
                r_accel = -_W_ACCEL_PENALTY * (acc_penalty + acc_inconsistency_penalty)

                # 3. Jerk penalty (change in acceleration)
                jerk = abs(accel - prev_accel) / self.sumo_step
                r_jerk = -_W_JERK_PENALTY * (min(jerk/10.0, 1.0) if jerk > 5.0 else 0.0)
                self._prev_accels[cav_id] = accel

                # 4. Surrogate safety penalty
                leader = self._get_neighbor(cav_id, "leader", 0)
                ttc_below_thresh = False
                if leader is not None and leader[0] != "":
                    leader_speed = self._veh_speed(leader[0])
                    speed_diff = speed - leader_speed + 1e-6 if speed >= leader_speed else -1.0# Add small value to avoid division by zero
                    ttc = (leader[1] + 2.0) / speed_diff if speed_diff > 0 else float('inf')
                    ttc_below_thresh = ttc < _TTC_THRESH
                    r_safety = -_W_SAFETY * min(0.5 * max(_TTC_THRESH - ttc, 0), 1.0)
                else:
                    r_safety = 0.0

                # 5. Lane-change command conflict penalty:
                # Penalize only when the agent requested a lane change but
                # the vehicle stayed in its previous lane after SUMO applied
                # safety/boundary constraints.
                prev_lane_idx = self._prev_lane_idx.get(cav_id, curr_lane_idx)
                lane_changed = curr_lane_idx != prev_lane_idx
                lc_cmd = float(action[i, 1]) if i < len(action) else 0.0
                lane_change_requested = abs(lc_cmd) > _LANE_CHANGE_THRESH
                lane_change_conflict = lane_change_requested and not lane_changed
                r_lc = -_W_LANE_CHANGE * float(lane_change_conflict)
                self._prev_lane_idx[cav_id] = curr_lane_idx

                # 6. Idling penalty:
                r_idle = _W_IDLE * (0.0 if speed > 2.0 else -1.0)

                # 7. Energy penalty:
                energy_inst = self._get_energy(speed - accel * self.sumo_step / 2.0, accel)
                r_energy = -_W_ENERGY * ((energy_inst - self._min_energy) /
                                         (self._max_energy - self._min_energy))

                dense_reward = r_speed + r_accel + r_jerk + r_safety + r_lc + r_idle + r_energy
                rewards[i] = dense_reward + terminal_bonus

                # ── Accumulate episode performance metrics ──────────────
                self._ep_metrics['sum_speed'] += speed
                self._ep_metrics['sum_jerk'] += jerk
                self._ep_metrics['sum_fuel_mL'] += energy_inst
                self._ep_metrics['tet_count'] += int(ttc_below_thresh)
                self._ep_metrics['cav_step_count'] += 1

                # ── Accumulate per-term reward contributions ────────────
                self._ep_metrics['sum_r_speed'] += float(r_speed)
                self._ep_metrics['sum_r_accel'] += float(r_accel)
                self._ep_metrics['sum_r_jerk'] += float(r_jerk)
                self._ep_metrics['sum_r_safety'] += float(r_safety)
                self._ep_metrics['sum_r_lc'] += float(r_lc)
                self._ep_metrics['sum_r_idle'] += float(r_idle)
                self._ep_metrics['sum_r_energy'] += float(r_energy)

            except traci.exceptions.TraCIException:
                rewards[i] = terminal_bonus

            # For vehicles that just exited control area, clear history after
            # attaching dense + terminal on this final transition.
            if exited_control:
                self._prev_accels.pop(cav_id, None)
                self._prev_lane_idx.pop(cav_id, None)
                self._prev_action_per_cav.pop(cav_id, None)

        # ── K-neighbor mean reward augmentation (cooperative shaping) ──
        # r_t^i ← r_t^i + w · (1/K_i) · Σ_k r_t^{i,k}
        # Reuses the K-NN indices computed in the previous _get_obs() call
        # (they reference positions in step_cav_ids, which equals the
        # previous _active_cav_ids). Valid entries are marked by ctx_mask.
        #
        # If neighbor_reward_directional=True, we further restrict the
        # reward sharing to "downstream" neighbors only — cars that ego
        # can physically influence (behind / parallel on the same edge).
        # Preceding vehicles add non-causal noise since the ego's action
        # can't meaningfully impact them within the horizon.
        w = self.neighbor_reward_coef
        if (w > 0.0
                and self._last_neighbor_idx is not None
                and self._last_ctx_mask is not None
                and self._last_neighbor_idx.shape[0] == N
                and N > 1):
            nbr_idx = self._last_neighbor_idx                 # (N, K), -1 = padded
            nbr_mask = self._last_ctx_mask.astype(bool)       # (N, K)

            # Directional filter: only keep downstream neighbors
            if (self.neighbor_reward_directional
                    and self._last_downstream_mask is not None
                    and self._last_downstream_mask.shape == nbr_mask.shape):
                direction_mask = self._last_downstream_mask.astype(bool)
                nbr_mask = nbr_mask & direction_mask

            # Safe gather: replace -1 with 0, then zero out via mask
            safe_idx = np.where(nbr_idx >= 0, nbr_idx, 0)
            gathered = rewards[safe_idx]                       # (N, K)
            gathered = gathered * nbr_mask                     # pad → 0

            # Include ego itself in the pool: counts = behind_neighbors + 1.
            # This keeps counts ≥ 1 always (no divide-by-zero, no silent
            # down-scaling when ego has no valid neighbors) and reduces the
            # variance of the shaping signal by averaging over more samples.
            counts = nbr_mask.sum(axis=1) + 1                  # (N,) always ≥ 1
            sums = gathered.sum(axis=1)                         # (N,) sum of neighbor rewards
            mean_neighbor_r = sums / counts.astype(sums.dtype)  # share of neighbors in pool mean
            ego_share = rewards / counts.astype(rewards.dtype)  # share of ego in pool mean
            pool_mean = mean_neighbor_r + ego_share             # = (sums + r_ego) / counts

            rewards = (1 - w) * rewards + w * pool_mean.astype(rewards.dtype)

        return rewards

    # ── Rendering ────────────────────────────────────────────────────────

    def render(self, mode="human"):
        """Rendering is handled by SUMO-GUI when use_gui=True."""
        if mode == "rgb_array":
            # Could capture screenshot via traci if needed
            image = self._conn.gui.screenshot("View #0", "screenshot.png")
            
            return image
        return None

    @property
    def unwrapped(self):
        return self
