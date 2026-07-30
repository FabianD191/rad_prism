#!/bin/bash
set -euo pipefail
#
# Full pipeline with abundance-balanced imputation.
#
# Steps:
#   1. Structuring & labeling (LLM-based)
#   2. Baseline embedding (no imputation)
#   3. Baseline binary label extraction (no imputation)
#   4. Concept statistics extraction (from baseline outputs)
#   5. Final embedding with imputation + abundance balance
#   6. Final binary label extraction with imputation + abundance balance
#   7. Final concept statistics (from imputed outputs, for analysis/documentation)
#
# Usage:
#   ./scripts/run_full_pipeline.sh
#   ./scripts/run_full_pipeline.sh --embedding-backend auto_model
#   ./scripts/run_full_pipeline.sh --embedding-backend both
#   ./scripts/run_full_pipeline.sh --skip-structuring
#   ./scripts/run_full_pipeline.sh --work-dir /path/to/intermediates
#
# Intermediate (baseline) outputs are written to <work-dir>/.  Final (imputed)
# outputs go to whatever out_dir is configured in the YAML config files.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${REPO_DIR}/logs"

# ── Defaults ──
EMBEDDING_BACKEND="sentence_transformer"
WORK_DIR="${REPO_DIR}/output/pipeline_intermediates"
SKIP_STRUCTURING=false
SKIP_BASELINE=false
SKIP_STATS=false
SKIP_IMPUTED=false
SKIP_FINAL_STATS=false

# ── Parse arguments ──
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding-backend)
      EMBEDDING_BACKEND="$2"; shift 2 ;;
    --work-dir)
      WORK_DIR="$2"; shift 2 ;;
    --skip-structuring)
      SKIP_STRUCTURING=true; shift ;;
    --skip-baseline)
      SKIP_BASELINE=true; shift ;;
    --skip-stats)
      SKIP_STATS=true; shift ;;
    --skip-imputed)
      SKIP_IMPUTED=true; shift ;;
    --skip-final-stats)
      SKIP_FINAL_STATS=true; shift ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

BASELINE_EMB_DIR="${WORK_DIR}/baseline_embeddings"
BASELINE_LABEL_DIR="${WORK_DIR}/baseline_labels"
STATS_DIR="${WORK_DIR}/concept_stats"
FINAL_STATS_DIR="${WORK_DIR}/final_stats"
CONCEPT_STATS_CSV="${STATS_DIR}/concept_stats.csv"

mkdir -p "$LOG_DIR" "$WORK_DIR"
cd "$REPO_DIR"

# ── Activate environment (uncomment and adjust) ──
# source "/path/to/conda/etc/profile.d/conda.sh"
# conda activate your_env

# ── Helper: read a YAML key via Python (returns empty string if missing) ──
yaml_value() {
  local yaml_file="$1" key="$2"
  python3 -c "
import sys
try:
    import yaml
    with open('$yaml_file') as f:
        cfg = yaml.safe_load(f) or {}
    print(cfg.get('$key', ''))
except Exception:
    print('')
" 2>/dev/null
}

echo "================================================================"
echo " Full Pipeline (with abundance-balanced imputation)"
echo " Embedding backend: ${EMBEDDING_BACKEND}"
echo " Work dir:          ${WORK_DIR}"
echo " Repo dir:          ${REPO_DIR}"
echo "================================================================"
echo ""

# ── Helper: select embedding config file(s) ──
get_embedding_configs() {
  case "$EMBEDDING_BACKEND" in
    sentence_transformer)
      echo "SentenceTransformer|${REPO_DIR}/config/embedding_sentence_transformer.yaml" ;;
    auto_model)
      echo "AutoModel|${REPO_DIR}/config/embedding_auto_model.yaml" ;;
    both)
      echo "SentenceTransformer|${REPO_DIR}/config/embedding_sentence_transformer.yaml"
      echo "AutoModel|${REPO_DIR}/config/embedding_auto_model.yaml" ;;
    *)
      echo "ERROR: Unknown embedding backend '${EMBEDDING_BACKEND}'." >&2
      echo "       Use: sentence_transformer, auto_model, or both." >&2
      exit 1 ;;
  esac
}

# ─────────────────────────────────────────────────────────────────────
# Step 1/7: Structuring & Labeling (LLM-based)
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_STRUCTURING" = false ]; then
  echo "[Step 1/7] Structuring & labeling reports..."
  STRUCTURING_LOG="${LOG_DIR}/structuring_labeling.log"

  python "${REPO_DIR}/structuring_labeling/report_structuring_labeling.py" \
    --config "${REPO_DIR}/config/structuring_labeling.yaml" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "$STRUCTURING_LOG"

  echo "[Step 1/7] Done. Log: ${STRUCTURING_LOG}"
  echo ""
else
  echo "[Step 1/7] Skipped (--skip-structuring)."
  echo ""
fi

# ─────────────────────────────────────────────────────────────────────
# Step 2/7: Baseline Embedding (no imputation)
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_BASELINE" = false ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  while IFS='|' read -r backend_name config_file; do
    suffix="${backend_name// /_}"
    suffix_lower="$(echo "$suffix" | tr '[:upper:]' '[:lower:]')"
    baseline_out="${BASELINE_EMB_DIR}/${suffix_lower}"
    log_file="${LOG_DIR}/baseline_embedding_${suffix_lower}.log"

    echo "[Step 2/7] Baseline embedding (${backend_name}) -> ${baseline_out} ..."
    python "${REPO_DIR}/embedding/embed_structured_reports.py" \
      --config "$config_file" \
      --out-dir "$baseline_out" \
      --disable-support-devices-imputation \
      --disable-pathologies-imputation \
      --disable-abundance-balance \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
      2>&1 | tee "$log_file"

    echo "[Step 2/7] Done (${backend_name}). Log: ${log_file}"
    echo ""
  done < <(get_embedding_configs)

  # ─────────────────────────────────────────────────────────────────────
  # Step 3/7: Baseline Binary Label Extraction (no imputation)
  # ─────────────────────────────────────────────────────────────────────
  echo "[Step 3/7] Baseline label extraction -> ${BASELINE_LABEL_DIR} ..."
  BASELINE_LABEL_LOG="${LOG_DIR}/baseline_label_extraction.log"

  python "${REPO_DIR}/label_extraction/extract_binary_labels.py" \
    --config "${REPO_DIR}/config/binary_label_extraction.yaml" \
    --out-dir "$BASELINE_LABEL_DIR" \
    --disable-pathology-imputation \
    --disable-abundance-balance \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "$BASELINE_LABEL_LOG"

  echo "[Step 3/7] Done. Log: ${BASELINE_LABEL_LOG}"
  echo ""
else
  echo "[Step 2/7] Skipped (--skip-baseline)."
  echo "[Step 3/7] Skipped (--skip-baseline)."
  echo ""
fi

# ─────────────────────────────────────────────────────────────────────
# Step 4/7: Concept Statistics Extraction (from baseline)
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_STATS" = false ]; then
  # Pick the first available baseline embedding dir for stats
  STATS_TEXT_EMB_DIR=""
  for candidate in \
      "${BASELINE_EMB_DIR}/sentencetransformer" \
      "${BASELINE_EMB_DIR}/automodel" \
      "${BASELINE_EMB_DIR}"; do
    if [ -d "$candidate" ] && ls "$candidate"/meta_*.parquet >/dev/null 2>&1; then
      STATS_TEXT_EMB_DIR="$candidate"
      break
    fi
  done

  STATS_ARGS=()
  if [ -n "$STATS_TEXT_EMB_DIR" ]; then
    STATS_ARGS+=(--text-emb-dir "$STATS_TEXT_EMB_DIR")
  fi
  if [ -d "$BASELINE_LABEL_DIR" ] && ls "$BASELINE_LABEL_DIR"/meta_*.parquet >/dev/null 2>&1; then
    STATS_ARGS+=(--label-dir "$BASELINE_LABEL_DIR")
  fi

  if [ ${#STATS_ARGS[@]} -eq 0 ]; then
    echo "[Step 4/7] WARNING: No baseline shard directories found."
    echo "           Expected baseline outputs in ${WORK_DIR}."
    echo "           Skipping stats extraction. The imputed runs will use"
    echo "           whatever concept_stats_csv is set in the config files."
    echo ""
  else
    echo "[Step 4/7] Extracting baseline concept statistics -> ${STATS_DIR} ..."
    STATS_LOG="${LOG_DIR}/concept_stats_baseline.log"

    python "${REPO_DIR}/stats/extract_concept_stats.py" \
      --concepts-file "${REPO_DIR}/templates/concepts.txt" \
      --out-dir "$STATS_DIR" \
      "${STATS_ARGS[@]}" \
      2>&1 | tee "$STATS_LOG"

    echo "[Step 4/7] Done. Log: ${STATS_LOG}"
    echo "           concept_stats.csv: ${CONCEPT_STATS_CSV}"
    echo ""
  fi
else
  echo "[Step 4/7] Skipped (--skip-stats)."
  echo ""
fi

# ─────────────────────────────────────────────────────────────────────
# Step 5/7: Final Embedding with Imputation + Abundance Balance
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_IMPUTED" = false ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  # Use extracted stats if available, otherwise fall back to config default
  EMB_STATS_ARGS=()
  if [ -f "$CONCEPT_STATS_CSV" ]; then
    EMB_STATS_ARGS+=(--concept-stats-csv "$CONCEPT_STATS_CSV")
    echo "[Step 5/7] Using extracted concept stats: ${CONCEPT_STATS_CSV}"
  else
    echo "[Step 5/7] No extracted stats found; using concept_stats_csv from config."
  fi

  while IFS='|' read -r backend_name config_file; do
    suffix="${backend_name// /_}"
    suffix_lower="$(echo "$suffix" | tr '[:upper:]' '[:lower:]')"
    log_file="${LOG_DIR}/imputed_embedding_${suffix_lower}.log"

    echo "[Step 5/7] Imputed embedding (${backend_name})..."
    python "${REPO_DIR}/embedding/embed_structured_reports.py" \
      --config "$config_file" \
      "${EMB_STATS_ARGS[@]+"${EMB_STATS_ARGS[@]}"}" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
      2>&1 | tee "$log_file"

    echo "[Step 5/7] Done (${backend_name}). Log: ${log_file}"
    echo ""
  done < <(get_embedding_configs)

  # ─────────────────────────────────────────────────────────────────────
  # Step 6/7: Final Binary Label Extraction with Imputation + Abundance Balance
  # ─────────────────────────────────────────────────────────────────────
  LABEL_STATS_ARGS=()
  if [ -f "$CONCEPT_STATS_CSV" ]; then
    LABEL_STATS_ARGS+=(--concept-stats-csv "$CONCEPT_STATS_CSV")
  fi

  echo "[Step 6/7] Imputed label extraction..."
  IMPUTED_LABEL_LOG="${LOG_DIR}/imputed_label_extraction.log"

  python "${REPO_DIR}/label_extraction/extract_binary_labels.py" \
    --config "${REPO_DIR}/config/binary_label_extraction.yaml" \
    "${LABEL_STATS_ARGS[@]+"${LABEL_STATS_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "$IMPUTED_LABEL_LOG"

  echo "[Step 6/7] Done. Log: ${IMPUTED_LABEL_LOG}"
  echo ""
else
  echo "[Step 5/7] Skipped (--skip-imputed)."
  echo "[Step 6/7] Skipped (--skip-imputed)."
  echo ""
fi

# ─────────────────────────────────────────────────────────────────────
# Step 7/7: Final Concept Statistics (from imputed outputs)
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_FINAL_STATS" = false ]; then
  # Resolve final out_dir values from embedding and label configs
  FINAL_STATS_ARGS=()

  # Find the first embedding config's out_dir that has shards
  FINAL_EMB_DIR=""
  while IFS='|' read -r backend_name config_file; do
    candidate="$(yaml_value "$config_file" "out_dir")"
    if [ -n "$candidate" ] && [ -d "$candidate" ] && ls "$candidate"/meta_*.parquet >/dev/null 2>&1; then
      FINAL_EMB_DIR="$candidate"
      break
    fi
  done < <(get_embedding_configs)

  if [ -n "$FINAL_EMB_DIR" ]; then
    FINAL_STATS_ARGS+=(--text-emb-dir "$FINAL_EMB_DIR")
  fi

  # Resolve label out_dir from label config
  LABEL_CONFIG="${REPO_DIR}/config/binary_label_extraction.yaml"
  FINAL_LABEL_DIR="$(yaml_value "$LABEL_CONFIG" "out_dir")"
  if [ -n "$FINAL_LABEL_DIR" ] && [ -d "$FINAL_LABEL_DIR" ] && ls "$FINAL_LABEL_DIR"/meta_*.parquet >/dev/null 2>&1; then
    FINAL_STATS_ARGS+=(--label-dir "$FINAL_LABEL_DIR")
  fi

  if [ ${#FINAL_STATS_ARGS[@]} -eq 0 ]; then
    echo "[Step 7/7] WARNING: No final (imputed) shard directories found."
    echo "           Skipping final stats extraction."
    echo ""
  else
    echo "[Step 7/7] Extracting final concept statistics (with imputation) -> ${FINAL_STATS_DIR} ..."
    FINAL_STATS_LOG="${LOG_DIR}/concept_stats_final.log"

    python "${REPO_DIR}/stats/extract_concept_stats.py" \
      --concepts-file "${REPO_DIR}/templates/concepts.txt" \
      --out-dir "$FINAL_STATS_DIR" \
      "${FINAL_STATS_ARGS[@]}" \
      2>&1 | tee "$FINAL_STATS_LOG"

    echo "[Step 7/7] Done. Log: ${FINAL_STATS_LOG}"
    echo "           Output: ${FINAL_STATS_DIR}/"
    echo ""
  fi
else
  echo "[Step 7/7] Skipped (--skip-final-stats)."
  echo ""
fi

echo "================================================================"
echo " Pipeline complete."
echo " Intermediate outputs: ${WORK_DIR}/"
echo " Logs:                 ${LOG_DIR}/"
echo "================================================================"
