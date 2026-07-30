#!/bin/bash
set -euo pipefail
#
# Launch the RadPRISM classification fine-tuning in the background.
#
# Edit config/finetune_cls.yaml to point at the pretrained checkpoint/config and
# set the fine-tuning schedule, then run this script. Extra CLI arguments are
# forwarded to the Python script (e.g. --device cuda:0 --epochs 10).
#
#   ./scripts/run_finetune_cls.sh
#   ./scripts/run_finetune_cls.sh --ckpt-path /path/to/best_align.pt
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_DIR}/config/finetune_cls.yaml"
PYTHON_SCRIPT="${REPO_DIR}/finetuning/train_run_finetune_cls.py"
LOG_DIR="${REPO_DIR}/logs"

# ── Activate environment (uncomment and adjust for your setup) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_OUT="${LOG_DIR}/finetune_cls_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/finetune_cls_${TIMESTAMP}.pid"

nohup python "$PYTHON_SCRIPT" --config "$CONFIG" "$@" > "$LOG_OUT" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "============================================================"
echo "  Fine-tuning started in background."
echo "  PID      : $TRAIN_PID"
echo "  Log file : $LOG_OUT"
echo "  PID file : $PID_FILE"
echo "============================================================"
echo "  Monitor : tail -f \"$LOG_OUT\""
echo "  Stop    : kill \$(cat \"$PID_FILE\")"
