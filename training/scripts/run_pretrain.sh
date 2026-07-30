#!/bin/bash
set -euo pipefail
#
# Launch the concept-alignment pretraining in the background.
#
# Edit config/pretrain.yaml to set data paths, the vision backbone and the
# training schedule, then run this script. Extra CLI arguments are forwarded to
# the Python script (e.g. --device cuda:0 --epochs 4).
#
#   ./scripts/run_pretrain.sh
#   ./scripts/run_pretrain.sh --device cuda:0 --batch-size 128
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_DIR}/config/pretrain.yaml"
PYTHON_SCRIPT="${REPO_DIR}/pretraining/train_run_pretrain.py"
LOG_DIR="${REPO_DIR}/logs"

# ── Activate environment (uncomment and adjust for your setup) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

# Timestamped log + PID file so concurrent runs don't clobber each other.
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_OUT="${LOG_DIR}/pretrain_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/pretrain_${TIMESTAMP}.pid"

nohup python "$PYTHON_SCRIPT" --config "$CONFIG" "$@" > "$LOG_OUT" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "============================================================"
echo "  Pretraining started in background."
echo "  PID      : $TRAIN_PID"
echo "  Log file : $LOG_OUT"
echo "  PID file : $PID_FILE"
echo "============================================================"
echo "  Monitor : tail -f \"$LOG_OUT\""
echo "  Stop    : kill \$(cat \"$PID_FILE\")"
