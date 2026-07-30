#!/bin/bash
set -euo pipefail
#
# Extract binary labels from structured report JSON files.
#
# Edit config/binary_label_extraction.yaml to set data paths and concepts,
# then run this script.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_DIR}/config/binary_label_extraction.yaml"
PYTHON_SCRIPT="${REPO_DIR}/label_extraction/extract_binary_labels.py"
LOG_DIR="${REPO_DIR}/logs"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

python "$PYTHON_SCRIPT" \
  --config "$CONFIG" \
  "$@" \
  > "${LOG_DIR}/binary_label_extraction.log" 2>&1 &

echo "Started (PID=$!). Check ${LOG_DIR}/binary_label_extraction.log"
