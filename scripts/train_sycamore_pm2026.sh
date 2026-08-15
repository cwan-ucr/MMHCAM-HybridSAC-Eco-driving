#!/usr/bin/env bash
set -euo pipefail

# Train/fine-tune Hybrid SAC on the Central Ave EB PM-peak scenario represented
# by the original synthetic three-lane through network with real-world demand,
# approach length, and signal timing.
#
# Optional overrides:
#   EXP_NAME=... SEED=... INIT_CHECKPOINT=... INIT_CHECKPOINT_MODE=actor bash scripts/train_sycamore_pm2026.sh

cd "$(dirname "$0")/.."

SUMO_CFG="${SUMO_CFG:-envs/sumo_files/sycamore_synthetic_pm2026/sycamore_synthetic_pm2026.sumocfg}"
ROUTE_TEMPLATE="${ROUTE_TEMPLATE:-envs/sumo_files/sycamore_synthetic_pm2026/sycamore_synthetic_pm2026.rou.xml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
INIT_CHECKPOINT_MODE="${INIT_CHECKPOINT_MODE:-full}"
EXP_NAME="${EXP_NAME:-cav_control_sycamore_pm2026}"
SEED="${SEED:-1}"
STEPS="${STEPS:-300000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-50000}"

EXTRA_ARGS=()
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(init_checkpoint="${INIT_CHECKPOINT}")
  EXTRA_ARGS+=(init_checkpoint_mode="${INIT_CHECKPOINT_MODE}")
fi

python3 train.py \
  agent=sac \
  exp_name="${EXP_NAME}" \
  seed="${SEED}" \
  sumo_env_impl=fast \
  sumo_cfg="${SUMO_CFG}" \
  route_template="${ROUTE_TEMPLATE}" \
  attention=true \
  communication=true \
  lane_action_discrete=true \
  comm_topology=full \
  context_size=8 \
  communication_range_m=150.0 \
  total_flow_vph_min=739 \
  total_flow_vph_max=739 \
  cav_penetration_min=0.05 \
  cav_penetration_max=0.95 \
  vid_cycle=450 \
  terminal_cycle_s=90.0 \
  terminal_green_start_s=0.0 \
  terminal_green_end_s=22.0 \
  control_area_length_m=213.6 \
  max_speed_mps=22.35 \
  control_start_distance_m=0.0 \
  post_out_eval_distance_m=50.0 \
  steps="${STEPS}" \
  checkpoint_freq="${CHECKPOINT_FREQ}" \
  "${EXTRA_ARGS[@]}"
