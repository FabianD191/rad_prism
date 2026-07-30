#!/bin/bash
set -euo pipefail
#
# Run the external CheXpert / CheXlocalize evaluation (classification + grounding).
# Edit evaluation/config/evaluate_chexlocalize.yaml first (model.run_dir, CheXpert
# paths, CheXlocalize GT segmentation JSONs). Extra CLI args are forwarded.
#
#   ./scripts/run_evaluate_chexlocalize.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EVAL_DIR}/config/evaluate_chexlocalize.yaml"
PYTHON_SCRIPT="${EVAL_DIR}/evaluate_chexlocalize.py"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

python "$PYTHON_SCRIPT" --config "$CONFIG" "$@"
