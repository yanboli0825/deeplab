#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="${1:-cv_refit_demo}"
RUN_PREFIX="${2:-trial}"

python main.py -m \
  mode=cv \
  hpo=optuna \
  experiment_name="${EXPERIMENT_NAME}" \
  run_name="${RUN_PREFIX}_search"

cat <<'EOF'

[Next step]
Inspect the Hydra multirun output, identify the best trial, and then refit once
with the selected overrides. Example:

python main.py \
  mode=train \
  experiment_name=<same_experiment_name> \
  run_name=<same_run_prefix>_refit \
  test_after_train=true \
  model.optimizer.lr=3e-4 \
  model.optimizer.weight_decay=1e-4

EOF
