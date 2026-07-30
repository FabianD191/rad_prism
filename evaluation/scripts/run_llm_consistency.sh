#!/bin/bash
set -euo pipefail
#
# Optional LLM consistency check over RadPRISM per-case outputs.
# Run the inference demo first (produces classification.csv + retrieval_top_matches.csv
# per image), then run this. Edit evaluation/config/llm_consistency.yaml to set the
# LLM endpoint. Extra CLI args are forwarded (e.g. --dry-run to skip the LLM call).
#
#   ./scripts/run_llm_consistency.sh                # real run (needs an endpoint)
#   ./scripts/run_llm_consistency.sh --dry-run      # build payload/schema only
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EVAL_DIR}/config/llm_consistency.yaml"
PYTHON_SCRIPT="${EVAL_DIR}/llm_consistency_check.py"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

python "$PYTHON_SCRIPT" --config "$CONFIG" "$@"
