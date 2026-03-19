#!/usr/bin/env bash
set -euo pipefail

# Example entry for CV-based hyperparameter search.
# The search space itself is defined by the Hydra `hpo` config group.

python main.py \
  experiment_name=frozen2classes \
  run_name=hpo \
  mode=cv \
  hpo=optuna

echo "hpo finished."
