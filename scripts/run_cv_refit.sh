#!/usr/bin/env bash
set -euo pipefail

python -m src.workflows.hpo_refit "$@"
