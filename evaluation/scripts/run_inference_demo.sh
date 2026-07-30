#!/bin/bash
set -euo pipefail
#
# Apply the provided RadPRISM checkpoint to images (attention overlays +
# thresholded classification + English text retrieval).
#
# Before running:
#   1. Download RAD-DINO-MAIRA-2 and set rad_dino_model_dir in the config.
#   2. Build the retrieval bank once with your Qwen model:
#        python utils/embed_radprism_text_db.py --db data/radprism_checkpoint/radprism_text_db_de.json \
#          --config data/radprism_checkpoint/radprism_config.json --model-dir /path/to/Qwen3-Embedding-4B \
#          --out-dir data/radprism_checkpoint
#
#   ./scripts/run_inference_demo.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EVAL_DIR}/config/inference_demo.yaml"
PYTHON_SCRIPT="${EVAL_DIR}/run_inference_demo.py"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

python "$PYTHON_SCRIPT" --config "$CONFIG" "$@"
