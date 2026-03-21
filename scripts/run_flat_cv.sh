#!/usr/bin/env bash
set -euo pipefail

python -m src.workflows.flat_cv "$@"
