"""
Time-space diagram plotter for SUMO FCD (Floating Car Data) output.

Parses the FCD XML written by SUMO and generates a time-space diagram
with one column per active approach and one row per lane. CAVs are drawn
in blue, HDVs in black. Empty approaches (no vehicles) are skipped.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Approach definitions ─────────────────────────────────────────────────
# (incoming_edge, outgoing_edge, green_sumo_phases)
# Signal: phase 0 = N/S green (30s), 1 = N/S yellow (4s),
#         phase 2 = E/W green (30s), 3 = E/W yellow (4s)
APPROACHES = {
    'Eastbound':  ('west_in',  'east_out',  {2}),
    'Westbound':  ('east_in',  'west_out',  {2}),
    'Northbound': ('south_in', 'north_out', {0}),
    'Southbound': ('north_in', 'south_out', {0}),
}

# Signal timing (durations in seconds, matching intersection.tll.xml)
_PHASE_DURATIONS = [27, 3, 27, 3]   # phases 0..3
_CYCLE_TIME = sum(_PHASE_DURATIONS)  # 60s
_PHASE_CUMUL = np.cumsum(_PHASE_DURATIONS)

_NUM_LANES = 3  # lanes 0, 1, 2

# For this 4-way intersection net, internal edge indices 0..3 map to:
# north->south, south->north, east->west, west->east.
_INTERNAL_EDGE_INDEX_TO_APPROACH = {
    0: 'Southbound',
    1: 'Northbound',
    2: 'Westbound',
    3: 'Eastbound',
}


def _get_signal_phase(t: float) -> int:
    """Return the SUMO phase index (0-3) at simulation time t."""
    t_mod = t % _CYCLE_TIME
    for i, c in enumerate(_PHASE_CUMUL):
        if t_mod < c:
            return i
    return len(_PHASE_DURATIONS) - 1


def _edge_from_lane(lane_id: str) -> Tuple[str, int]:
    """Extract (edge_id, lane_index) from a SUMO lane ID.
    e.g. 'west_in_0' -> ('west_in', 0)
    """
    parts = lane_id.rsplit("_", 1)
    return parts[0], int(parts[1])


def _approach_from_internal_edge(edge_id: str) -> Optional[str]:
    """Best-effort mapping from SUMO internal edge id to approach name."""
    try:
        edge_idx = int(edge_id.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None
    return _INTERNAL_EDGE_INDEX_TO_APPROACH.get(edge_idx)


def parse_fcd(fcd_path: str, edge_length: float = 294.4):
    """
    Parse SUMO FCD XML and organize trajectories by (approach, lane).

    Returns
    -------
    lane_trajs : dict
        {(approach_name, lane_idx): {veh_id: [(time, y_pos), ...]}}
    veh_types : dict
        {veh_id: vtype_str}
    time_range : (float, float)
    active_approaches : set of approach names that contain data
    """
    # Raw records before converting to plotted longitudinal y:
    # {(approach, lane): {veh_id: [(t, seg_kind, lane_pos)]}}
    raw_trajs: Dict[Tuple[str, int], Dict[str, List[Tuple[float, str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Maximum observed position on internal lanes, used as in->out bridge.
    internal_len: Dict[Tuple[str, int], float] = defaultdict(float)
    # Remember each vehicle's last known approach to map internal lanes.
    last_approach: Dict[str, str] = {}
    veh_types: Dict[str, str] = {}

    # Build edge -> (approach_name, is_outgoing) lookup
    edge_to_approach: Dict[str, Tuple[str, bool]] = {}
    for name, (in_edge, out_edge, _) in APPROACHES.items():
        edge_to_approach[in_edge] = (name, False)
        edge_to_approach[out_edge] = (name, True)

    t_min, t_max = float('inf'), float('-inf')

    for event, elem in ET.iterparse(fcd_path, events=('end',)):
        if elem.tag == 'timestep':
            t = float(elem.get('time'))
            t_min = min(t_min, t)
            t_max = max(t_max, t)
            for veh in elem:
                if veh.tag != 'vehicle':
                    continue
                lane = veh.get('lane', '')
                veh_id = veh.get('id')
                pos = float(veh.get('pos'))
                vtype = veh.get('type', 'unknown')
                veh_types[veh_id] = vtype

                edge, lane_idx = _edge_from_lane(lane)
                if lane.startswith(':'):
                    # Keep internal junction lanes to avoid in->out time shift.
                    approach_name = last_approach.get(veh_id)
                    if approach_name is None:
                        approach_name = _approach_from_internal_edge(edge)
                    if approach_name is None:
                        continue
                    key = (approach_name, lane_idx)
                    raw_trajs[key][veh_id].append((t, 'internal', pos))
                    internal_len[key] = max(internal_len[key], pos)
                else:
                    if edge not in edge_to_approach:
                        continue
                    approach_name, is_outgoing = edge_to_approach[edge]
                    last_approach[veh_id] = approach_name
                    seg_kind = 'out' if is_outgoing else 'in'
                    raw_trajs[(approach_name, lane_idx)][veh_id].append((t, seg_kind, pos))

            elem.clear()

    # Convert raw (segment-local) positions into one continuous longitudinal axis:
    # in edge: [0, L_in], internal: [L_in, L_in + L_internal],
    # out edge: [L_in + L_internal, ...]
    lane_trajs: Dict[Tuple[str, int], Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, vehs in raw_trajs.items():
        bridge = internal_len.get(key, 0.0)
        for veh_id, points in vehs.items():
            for t, seg_kind, pos in points:
                if seg_kind == 'in':
                    y = pos
                elif seg_kind == 'internal':
                    y = edge_length + pos
                else:  # seg_kind == 'out'
                    y = edge_length + bridge + pos
                lane_trajs[key][veh_id].append((t, y))

    # Which approaches have at least one data point?
    active_approaches: Set[str] = set()
    for (appr, _), vehs in lane_trajs.items():
        if vehs:
            active_approaches.add(appr)

    return dict(lane_trajs), veh_types, (t_min, t_max), active_approaches


def _draw_stopbar_signal(ax, t_min, t_max, y, green_phases, height=9.0):
    """Draw signal phase colors as a compact strip at the stop bar."""
    t = t_min
    while t < t_max:
        phase = _get_signal_phase(t)
        t_mod = t % _CYCLE_TIME
        for i, c in enumerate(_PHASE_CUMUL):
            if t_mod < c:
                remaining = c - t_mod
                break
        else:
            remaining = _CYCLE_TIME - t_mod
        t_end = min(t + remaining, t_max)

        if phase in green_phases:
            color, alpha = '#4CAF50', 0.80   # green
        elif phase in {p + 1 for p in green_phases}:
            color, alpha = '#FFC107', 0.85   # yellow
        else:
            color, alpha = '#F44336', 0.70   # red
        rect = mpatches.Rectangle(
            (t, y - height / 2.0),
            t_end - t,
            height,
            facecolor=color,
            edgecolor='none',
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(rect)
        t = t_end


def plot_time_space_diagram(
    fcd_path: str,
    save_path: str,
    edge_length: float = 294.4,
    episode: int = 0,
    view: str = "all",
    trajectory_style: str = "line",
    title: Optional[str] = None,
    panel_width: float = 7.2,
    panel_height: float = 1.9,
    signal_height: float = 9.0,
) -> None:
    """
    Generate a time-space diagram from SUMO FCD output.

    Layout: columns = active approaches, rows = lanes (0, 1, 2).
    Empty approaches are skipped entirely.
    """
    fcd_file = Path(fcd_path)
    if not fcd_file.exists() or fcd_file.stat().st_size < 100:
        return

    lane_trajs, veh_types, (t_min, t_max), active_approaches = parse_fcd(
        fcd_path, edge_length=edge_length
    )
    if t_min >= t_max or not active_approaches:
        return

    # Determine subplot layout: columns=approaches, rows=lanes
    default_order = ['Eastbound', 'Westbound', 'Northbound', 'Southbound']
    view_key = str(view).strip().lower()
    if view_key in {"west_to_east", "w2e", "eastbound"}:
        approach_order = ['Eastbound'] if 'Eastbound' in active_approaches else []
    elif view_key in {"east_to_west", "wbound", "westbound"}:
        approach_order = ['Westbound'] if 'Westbound' in active_approaches else []
    else:
        approach_order = [a for a in default_order if a in active_approaches]
    if not approach_order:
        return
    n_cols = len(approach_order)
    n_rows = _NUM_LANES

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    lane_labels = ['Lane 0 (right)', 'Lane 1 (middle)', 'Lane 2 (left)']

    style_key = str(trajectory_style).strip().lower()
    use_step = style_key in {"step", "stair", "staircase"}

    for col, approach_name in enumerate(approach_order):
        _, _, green_phases = APPROACHES[approach_name]

        for row in range(n_rows):
            ax = axes[row, col]
            lane_idx = row

            # Signal phase strip at the stop bar.
            _draw_stopbar_signal(
                ax,
                t_min,
                t_max,
                edge_length,
                green_phases,
                height=signal_height,
            )

            # Stop line
            ax.axhline(y=edge_length, color='k', linestyle='--',
                       linewidth=0.8, alpha=0.65, zorder=4)

            # Vehicle trajectories for this (approach, lane)
            vehs = lane_trajs.get((approach_name, lane_idx), {})
            for veh_id, points in vehs.items():
                if len(points) < 2:
                    continue
                times, positions = zip(*sorted(points))
                vtype = veh_types.get(veh_id, 'unknown').lower()
                color = '#1976D2' if 'cav' in vtype else '#000000'
                lw = 1.0 if 'cav' in vtype else 0.5
                al = 0.85 if 'cav' in vtype else 0.35

                # Split into continuous segments: break when the time
                # gap exceeds 1s (vehicle was in another lane)
                seg_t, seg_p = [times[0]], [positions[0]]
                for k in range(1, len(times)):
                    if times[k] - times[k - 1] > 1.0:
                        # Gap detected -> plot the current segment and start a new one
                        if len(seg_t) >= 2:
                            if use_step:
                                ax.step(seg_t, seg_p, where='post',
                                        color=color, linewidth=lw, alpha=al,
                                        zorder=3)
                            else:
                                ax.plot(seg_t, seg_p, color=color,
                                        linewidth=lw, alpha=al, zorder=3)
                        seg_t, seg_p = [], []
                    seg_t.append(times[k])
                    seg_p.append(positions[k])
                if len(seg_t) >= 2:
                    if use_step:
                        ax.step(seg_t, seg_p, where='post',
                                color=color, linewidth=lw, alpha=al, zorder=3)
                    else:
                        ax.plot(seg_t, seg_p, color=color,
                                linewidth=lw, alpha=al, zorder=3)

            ax.set_xlim(t_min, t_max)
            ax.set_ylim(0, edge_length + 50)
            ax.grid(True, alpha=0.15)

            # Labels
            if row == 0:
                ax.set_title(approach_name, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{lane_labels[lane_idx]}\nPosition (m)', fontsize=9)
            else:
                ax.set_ylabel('')
            if row == n_rows - 1:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)

    # Legend
    handles = [
        mpatches.Patch(color='#1976D2', label='CAV'),
        mpatches.Patch(color='#000000', label='HDV'),
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=0.8, label='Stop bar'),
        mpatches.Patch(color='#4CAF50', alpha=0.80, label='Green'),
        mpatches.Patch(color='#FFC107', alpha=0.85, label='Yellow'),
        mpatches.Patch(color='#F44336', alpha=0.70, label='Red'),
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=6, fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
    )

    if title is None:
        title = f'Time-Space Diagram  |  Episode {episode}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
