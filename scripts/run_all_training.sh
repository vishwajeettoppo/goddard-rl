#!/usr/bin/env bash
# Train the full R0--R6 ladder on every configured seed, N at a time.
#
# Usage: bash scripts/run_all_training.sh [UPDATES] [PARALLEL]
# Env:   SEEDS="42,101,2026"  ROLLOUT_STEPS=512
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UPDATES="${1:-120}"
PARALLEL="${2:-3}"
SEEDS="${SEEDS:-42,101,2026}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-512}"

mkdir -p results/logs results/checkpoints

JOBS=()
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
for ablation in R0 R1 R2 R3 R4 R5 R6; do
  for seed in "${SEED_ARRAY[@]}"; do
    JOBS+=("${ablation}:${seed}")
  done
done

train_one() {
  local job="$1"
  local ablation="${job%%:*}"
  local seed="${job##*:}"
  if [[ -f "results/checkpoints/${ablation}_seed${seed}.pt" ]]; then
    echo "[skip] ${ablation} seed ${seed} already trained"
    return 0
  fi
  echo "[start] ${ablation} seed ${seed}"
  OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
    .venv/bin/python src/train.py --ablation "$ablation" --seed "$seed" \
      --updates "$UPDATES" --rollout-steps "$ROLLOUT_STEPS" \
      > "results/logs/${ablation}_seed${seed}.out" 2>&1
  echo "[done] ${ablation} seed ${seed}"
}

lane() {
  for job in "$@"; do
    train_one "$job"
  done
}

n_jobs=${#JOBS[@]}
for ((lane_id = 0; lane_id < PARALLEL; lane_id++)); do
  subset=()
  for ((i = lane_id; i < n_jobs; i += PARALLEL)); do
    subset+=("${JOBS[$i]}")
  done
  lane "${subset[@]}" &
done
wait

echo "All training jobs finished."
