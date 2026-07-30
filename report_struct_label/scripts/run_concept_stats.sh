#!/bin/bash
set -euo pipefail
#
# Extract per-concept text and label statistics from embedding/label shards.
#
# Produces concept_stats.csv for abundance-balanced imputation.
# Edit config/concept_stats.yaml to set input directories, then run this script.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_DIR}/config/concept_stats.yaml"
PYTHON_SCRIPT="${REPO_DIR}/stats/extract_concept_stats.py"
LOG_DIR="${REPO_DIR}/logs"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

python "$PYTHON_SCRIPT" \
  --config "$CONFIG" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/concept_stats.log"

echo "Done. Log: ${LOG_DIR}/concept_stats.log"
