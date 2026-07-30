#!/bin/bash
set -euo pipefail
#
# Run the internal-dataset evaluation (thresholds + classification + retrieval).
# Edit evaluation/config/evaluate_internal.yaml first (point model.run_dir at a
# trained run). Extra CLI args are forwarded (e.g. --device cuda:0).
#
#   ./scripts/run_evaluate_internal.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EVAL_DIR}/config/evaluate_internal.yaml"
PYTHON_SCRIPT="${EVAL_DIR}/evaluate_internal.py"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

python "$PYTHON_SCRIPT" --config "$CONFIG" "$@"
