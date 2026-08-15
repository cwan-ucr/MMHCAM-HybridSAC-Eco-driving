#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Visualize the trained CAV--V2X Hybrid SAC policy on the Sycamore/Central
# PM2026 real-parameter synthetic scenario.
#
# Optional overrides:
#   CHECKPOINT=logs/sumo-intersection/1/sycamore_pretrained_s1/models/final.pt
#   PENETRATION=0.5
#   FLOW_VPH=739
#   MAX_STEPS=2250          # 2250 * 0.2s = 450s
#   DELAY_MS=50             # SUMO-GUI delay per simulation step
#   KEEP_OPEN=true          # wait for Enter before closing SUMO-GUI

CHECKPOINT="${CHECKPOINT:-logs/sumo-intersection/1/sycamore_pretrained_s1/models/final.pt}"
PENETRATION="${PENETRATION:-0.5}"
FLOW_VPH="${FLOW_VPH:-739}"
MAX_STEPS="${MAX_STEPS:-2250}"
DELAY_MS="${DELAY_MS:-50}"
KEEP_OPEN="${KEEP_OPEN:-true}"

python3 visualize_policy.py \
  checkpoint="${CHECKPOINT}" \
  agent=sac \
  exp_name=visualize_sycamore_v2x \
  sumo_env_impl=fast \
  sumo_cfg=envs/sumo_files/sycamore_synthetic_pm2026/sycamore_synthetic_pm2026.sumocfg \
  route_template=envs/sumo_files/sycamore_synthetic_pm2026/sycamore_synthetic_pm2026.rou.xml \
  attention=true \
  communication=true \
  lane_action_discrete=true \
  comm_topology=full \
  context_size=8 \
  communication_range_m=150.0 \
  total_flow_vph_min="${FLOW_VPH}" \
  total_flow_vph_max="${FLOW_VPH}" \
  cav_penetration_min="${PENETRATION}" \
  cav_penetration_max="${PENETRATION}" \
  vid_cycle=450 \
  terminal_cycle_s=90.0 \
  terminal_green_start_s=0.0 \
  terminal_green_end_s=22.0 \
  control_area_length_m=213.6 \
  max_speed_mps=22.35 \
  control_start_distance_m=0.0 \
  post_out_eval_distance_m=50.0 \
  sumo_max_steps="${MAX_STEPS}" \
  use_gui=true \
  sumo_gui_delay_ms="${DELAY_MS}" \
  visualize_keep_open="${KEEP_OPEN}" \
  visualize_print_every=100
