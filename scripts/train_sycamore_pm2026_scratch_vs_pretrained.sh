#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Run paired Sycamore/Central real-parameter synthetic experiments:
#   1) training from scratch
#   2) fine-tuning from a pretrained synthetic CAV--V2X policy
#   3) fine-tuning from the pretrained actor only, with critics reset
#
# Optional overrides:
#   SEEDS="1 2 3"
#   STEPS=300000
#   CHECKPOINT_FREQ=50000
#   PRETRAINED_CHECKPOINT=logs/sumo-intersection/1/cav_control/models/final.pt
#   SCRATCH_PREFIX=sycamore_scratch
#   PRETRAINED_PREFIX=sycamore_pretrained
#   ACTOR_PRETRAINED_PREFIX=sycamore_actor_pretrained
#   RUN_SCRATCH=true
#   RUN_PRETRAINED=true
#   RUN_ACTOR_PRETRAINED=true

SEEDS="${SEEDS:-1}"
STEPS="${STEPS:-300000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-50000}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-logs/sumo-intersection/1/cav_control/models/final.pt}"
SCRATCH_PREFIX="${SCRATCH_PREFIX:-sycamore_scratch}"
PRETRAINED_PREFIX="${PRETRAINED_PREFIX:-sycamore_pretrained}"
ACTOR_PRETRAINED_PREFIX="${ACTOR_PRETRAINED_PREFIX:-sycamore_actor_pretrained}"
RUN_SCRATCH="${RUN_SCRATCH:-true}"
RUN_PRETRAINED="${RUN_PRETRAINED:-true}"
RUN_ACTOR_PRETRAINED="${RUN_ACTOR_PRETRAINED:-true}"
TRAIN_SCRIPT="scripts/train_sycamore_pm2026.sh"

NEEDS_PRETRAINED=false
if [[ "${RUN_PRETRAINED}" == "true" || "${RUN_ACTOR_PRETRAINED}" == "true" ]]; then
  NEEDS_PRETRAINED=true
fi

if [[ "${NEEDS_PRETRAINED}" == "true" && ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "Pretrained checkpoint not found: ${PRETRAINED_CHECKPOINT}" >&2
  exit 1
fi

for seed in ${SEEDS}; do
  if [[ "${RUN_SCRATCH}" == "true" ]]; then
    echo "=== Sycamore scratch training | seed=${seed} ==="
    INIT_CHECKPOINT= \
    EXP_NAME="${SCRATCH_PREFIX}_s${seed}" \
    SEED="${seed}" \
    STEPS="${STEPS}" \
    CHECKPOINT_FREQ="${CHECKPOINT_FREQ}" \
    bash "${TRAIN_SCRIPT}"
  fi

  if [[ "${RUN_PRETRAINED}" == "true" ]]; then
    echo "=== Sycamore pretrained fine-tuning | seed=${seed} ==="
    INIT_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
    INIT_CHECKPOINT_MODE=full \
    EXP_NAME="${PRETRAINED_PREFIX}_s${seed}" \
    SEED="${seed}" \
    STEPS="${STEPS}" \
    CHECKPOINT_FREQ="${CHECKPOINT_FREQ}" \
    bash "${TRAIN_SCRIPT}"
  fi

  if [[ "${RUN_ACTOR_PRETRAINED}" == "true" ]]; then
    echo "=== Sycamore actor-only pretrained fine-tuning | seed=${seed} ==="
    INIT_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
    INIT_CHECKPOINT_MODE=actor \
    EXP_NAME="${ACTOR_PRETRAINED_PREFIX}_s${seed}" \
    SEED="${seed}" \
    STEPS="${STEPS}" \
    CHECKPOINT_FREQ="${CHECKPOINT_FREQ}" \
    bash "${TRAIN_SCRIPT}"
  fi
done
