#!/bin/bash
set -euo pipefail
#
# Run the LLM-based report structuring and labeling pipeline.
#
# Edit config/structuring_labeling.yaml to set paths, API credentials,
# and processing parameters, then run this script.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_DIR}/config/structuring_labeling.yaml"
PYTHON_SCRIPT="${REPO_DIR}/structuring_labeling/report_structuring_labeling.py"
LOG_DIR="${REPO_DIR}/logs"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

python "$PYTHON_SCRIPT" \
  --config "$CONFIG" \
  "$@" \
  > "${LOG_DIR}/structuring_labeling.log" 2>&1 &

echo "Started (PID=$!). Check ${LOG_DIR}/structuring_labeling.log"
