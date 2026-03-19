#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="${1:-flat_cv_demo}"
RUN_PREFIX="${2:-flatcv}"
N_FOLDS="${3:-5}"
FIXED_TEST_RATIO="${4:-0.2}"
LR="${5:-3e-4}"
WD="${6:-1e-4}"

for (( FOLD=0; FOLD<${N_FOLDS}; FOLD++ )); do
  python main.py \
    mode=train \
    experiment_name="${EXPERIMENT_NAME}" \
    run_name="${RUN_PREFIX}_fold${FOLD}" \
    test_after_train=true \
    datamodule.data_cfg.test_ratio="${FIXED_TEST_RATIO}" \
    model.optimizer.lr="${LR}" \
    model.optimizer.weight_decay="${WD}"
done

python scripts/aggregate_json_metrics.py \
  "outputs/${EXPERIMENT_NAME}/${RUN_PREFIX}_fold*/**/run_summary.json" \
  "val_score"

python scripts/aggregate_json_metrics.py \
  "outputs/${EXPERIMENT_NAME}/${RUN_PREFIX}_fold*/**/run_summary.json" \
  "test_score"
