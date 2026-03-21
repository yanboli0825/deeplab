#!/usr/bin/env bash
set -euo pipefail

python -m src.workflows.nested_cv "$@"
