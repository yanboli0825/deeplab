#!/usr/bin/env bash
set -euo pipefail

# Prepare split manifests before running this script, for example:
#   splits/nested/outer_0/inner_0.yaml
#   splits/nested/outer_0/inner_1.yaml
#   splits/nested/outer_0/refit.yaml
#
# Each manifest is expected to contain:
#   train: [0, 1, 2]
#   val:   [3, 4]
#   test:  [5, 6]

EXPERIMENT_NAME="${1:-nested_demo}"
RUN_PREFIX="${2:-nested}"
CANDIDATES=(
  "model.optimizer.lr=3e-4 model.optimizer.weight_decay=1e-4"
  "model.optimizer.lr=1e-4 model.optimizer.weight_decay=1e-4"
)

for OUTER_DIR in splits/nested/outer_*; do
  OUTER_NAME="$(basename "${OUTER_DIR}")"
  BEST_SCORE=""
  BEST_OVERRIDE=""

  for OVERRIDE in "${CANDIDATES[@]}"; do
    SCORES=()

    for INNER_SPLIT in "${OUTER_DIR}"/inner_*.yaml; do
      RUN_NAME="${RUN_PREFIX}_${OUTER_NAME}_$(basename "${INNER_SPLIT}" .yaml)"
      python main.py \
        mode=train \
        experiment_name="${EXPERIMENT_NAME}" \
        run_name="${RUN_NAME}" \
        test_after_train=false \
        datamodule.data_cfg.split_file="${INNER_SPLIT}" \
        ${OVERRIDE}

      SCORE=$(python - <<PY
import glob, json
paths = sorted(glob.glob("outputs/${EXPERIMENT_NAME}/${RUN_NAME}/**/run_summary.json", recursive=True))
with open(paths[-1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["val_score"])
PY
)
      SCORES+=("${SCORE}")
    done

    MEAN=$(python - <<PY
vals = [float(x) for x in """${SCORES[*]}""".split()]
print(sum(vals) / len(vals))
PY
)

    echo "${OUTER_NAME} candidate='${OVERRIDE}' mean_inner=${MEAN}"

    if [[ -z "${BEST_SCORE}" ]]; then
      BEST_SCORE="${MEAN}"
      BEST_OVERRIDE="${OVERRIDE}"
    else
      if python - <<PY
best = float("${BEST_SCORE}")
cur = float("${MEAN}")
raise SystemExit(0 if cur < best else 1)
PY
      then
        BEST_SCORE="${MEAN}"
        BEST_OVERRIDE="${OVERRIDE}"
      fi
    fi
  done

  echo "[${OUTER_NAME}] selected: ${BEST_OVERRIDE}"

  python main.py \
    mode=train \
    experiment_name="${EXPERIMENT_NAME}" \
    run_name="${RUN_PREFIX}_${OUTER_NAME}_refit" \
    test_after_train=true \
    datamodule.data_cfg.split_file="${OUTER_DIR}/refit.yaml" \
    ${BEST_OVERRIDE}
done
