#!/bin/bash
set -euo pipefail
#
# Run text embedding with a SentenceTransformer model (e.g. Qwen3-Embedding-4B).
#
# Edit config/embedding_sentence_transformer.yaml to set model path, data paths,
# and embedding parameters, then run this script.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_DIR}/config/embedding_sentence_transformer.yaml"
PYTHON_SCRIPT="${REPO_DIR}/embedding/embed_structured_reports.py"
LOG_DIR="${REPO_DIR}/logs"

# ── Offline safety: prevent accidental model downloads ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

python "$PYTHON_SCRIPT" \
  --config "$CONFIG" \
  "$@" \
  > "${LOG_DIR}/embedding_sentence_transformer.log" 2>&1 &

echo "Started (PID=$!). Check ${LOG_DIR}/embedding_sentence_transformer.log"
