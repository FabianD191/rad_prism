# Structured Chest X-Ray Radiology Report Information Extraction

Extract structured information from free-text chest X-ray radiology reports using a multi-stage pipeline:

1. **Structuring & Labeling** — An LLM reads each report and produces a structured JSON with text excerpts and severity/presence labels for ~19 clinical concepts (support devices, thoracic organs, pathologies).
2. **Text Embedding** — The extracted text fields are encoded into dense vector representations using a transformer model, with rule-based imputation for missing negative entries.
3. **Binary Label Extraction** — Structured labels are mapped to binary classification targets (0/1) with validity masks, including abundance-balanced imputation for under-represented pathology concepts.
4. **Concept Statistics** *(optional)* — Extract per-concept text and label counts from the embedding/label shards to generate the `concept_stats.csv` used for abundance-balanced imputation.

The repository ships with 20 sample English CXR reports and pre-made structured outputs so you can test the embedding and label extraction steps without an LLM API.


## Repository Structure

```
public_repo/
├── config/                          # YAML configuration files
│   ├── structuring_labeling.yaml    #   LLM API settings, CSV columns, retries
│   ├── embedding_sentence_transformer.yaml  #   e.g. Qwen3-Embedding-4B
│   ├── embedding_auto_model.yaml    #   e.g. medBERT.de with mean pooling
│   ├── binary_label_extraction.yaml #   label mapping, imputation, abundance balance
│   └── concept_stats.yaml          #   concept stats extraction settings
│
├── data/                            # Input data
│   ├── sample_reports.csv           #   20 sample CXR reports (report_id, examination, report_text)
│   ├── sample_structured_jsons/     #   Pre-made structured JSON outputs (RPT001-020)
│   └── concept_stats.csv           #   Per-concept text counts for abundance-balanced imputation
│
├── templates/                       # Schema & prompt templates
│   ├── output_schema.json           #   JSON Schema for guided LLM generation (strict mode)
│   ├── system_prompt.md             #   LLM system prompt with extraction rules
│   ├── concepts.txt                 #   19 concept names (dot-separated paths)
│   └── neg_entry_sampling.json      #   Negative sentence pools per concept (for text imputation)
│
├── structuring_labeling/            # Step 1: LLM-based structuring
│   └── report_structuring_labeling.py
│
├── embedding/                       # Step 2: Text embedding
│   └── embed_structured_reports.py
│
├── label_extraction/                # Step 3: Binary label extraction
│   └── extract_binary_labels.py
│
├── stats/                           # Concept statistics extraction
│   └── extract_concept_stats.py
│
├── scripts/                         # Shell launchers
│   ├── run_full_pipeline.sh         #   Run steps 1-3 sequentially
│   ├── run_structuring_labeling.sh  #   Step 1 only
│   ├── run_embedding_sentence_transformer.sh  # Step 2 (SentenceTransformer)
│   ├── run_embedding_auto_model.sh  #   Step 2 (AutoModel)
│   ├── run_binary_label_extraction.sh  # Step 3 only
│   └── run_concept_stats.sh        #   Concept stats extraction
│
├── requirements.txt
└── logs/                            # Auto-created log directory
```


## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `openai`, `torch`, `transformers`, `sentence-transformers`, `pyyaml`, `pandas`, `numpy`, `jsonschema`.

### 2. Configure

All settings live in `config/*.yaml`. Edit these before running anything. Each config key maps to a CLI argument; CLI flags always override config file values.

**What you need to set:**

| Config file | Key settings to edit |
|---|---|
| `structuring_labeling.yaml` | `api_base`, `api_key`, `model_name`, `reports_csv_path`, `output_dir` |
| `embedding_sentence_transformer.yaml` | `model_dir` (local HuggingFace model path), `json_dir`, `out_dir`, `device` |
| `embedding_auto_model.yaml` | `model_dir`, `json_dir`, `out_dir`, `device` |
| `binary_label_extraction.yaml` | `json_dir`, `out_dir` |
| `concept_stats.yaml` | `text_emb_dir` and/or `label_dir`, `out_dir` |

For the embedding step you need a locally downloaded transformer model. The configs are pre-set for two model types:
- **SentenceTransformer** (`embedding_sentence_transformer.yaml`): e.g. Qwen3-Embedding-4B with MRL truncation
- **AutoModel** (`embedding_auto_model.yaml`): e.g. medBERT.de with manual mean-pooling and sliding-window overflow handling

### 3. Prepare your data

Your input CSV needs at minimum a `report_id` column and a `report_text` column with the free-text radiology reports. The column names are configurable in the YAML files.

```csv
report_id,examination,report_text
RPT001,chest_xray,"PA and lateral chest radiograph. Heart size is normal..."
```


## Running the Pipeline

### Full pipeline (7 steps, with abundance-balanced imputation)

The full pipeline runs the recommended workflow automatically:

1. **Structuring** — LLM-based extraction of structured JSON from reports
2. **Baseline embedding** — text embedding *without* imputation
3. **Baseline label extraction** — binary labels *without* imputation
4. **Baseline concept statistics** — extract per-concept text/label counts from baseline shards
5. **Imputed embedding** — re-run embedding *with* imputation + abundance balance (using stats from step 4)
6. **Imputed label extraction** — re-run labels *with* imputation + abundance balance (using stats from step 4)
7. **Final concept statistics** — extract stats from the imputed outputs for analysis/documentation

Intermediate outputs (baseline shards, both stats) are written to `output/pipeline_intermediates/`.
Final imputed outputs go to whatever `out_dir` is configured in the YAML config files.

```bash
# Run the full 7-step pipeline
./scripts/run_full_pipeline.sh

# Choose embedding backend
./scripts/run_full_pipeline.sh --embedding-backend auto_model
./scripts/run_full_pipeline.sh --embedding-backend both

# Skip phases you've already completed
./scripts/run_full_pipeline.sh --skip-structuring            # skip step 1
./scripts/run_full_pipeline.sh --skip-baseline               # skip steps 2-3
./scripts/run_full_pipeline.sh --skip-stats                  # skip step 4
./scripts/run_full_pipeline.sh --skip-imputed                # skip steps 5-6
./scripts/run_full_pipeline.sh --skip-final-stats            # skip step 7
./scripts/run_full_pipeline.sh --skip-structuring --skip-baseline --skip-stats  # only imputed runs

# Custom directory for intermediate outputs
./scripts/run_full_pipeline.sh --work-dir /data/my_intermediates
```

### Individual steps

Each step can also be run independently via its own shell script:

```bash
# Step 1: Structuring (requires LLM API)
./scripts/run_structuring_labeling.sh

# Step 2: Embedding (requires local transformer model + GPU)
./scripts/run_embedding_sentence_transformer.sh
# or
./scripts/run_embedding_auto_model.sh

# Step 3: Binary label extraction (CPU only)
./scripts/run_binary_label_extraction.sh

# Concept stats: extract from existing shards (CPU only)
./scripts/run_concept_stats.sh
```

All scripts accept extra CLI arguments that override config values:

```bash
./scripts/run_structuring_labeling.sh --max-to-process 10
./scripts/run_binary_label_extraction.sh --verbose-test --dry-run
```

### What each step produces

| Step | Output directory | Contents |
|---|---|---|
| Structuring | `output_dir` from structuring config | One `{report_id}_structured.json` per report |
| Embedding | `out_dir` from embedding config | Sharded `.npy` embeddings + `.parquet` metadata |
| Label extraction | `out_dir` from label extraction config | Sharded `labels_*.npy` (N,K), `masks_*.npy` (N,K), `meta_*.parquet`, `summary.json` |
| Concept stats | `out_dir` from stats config | `concept_stats.csv`, `text_stats.csv`, `label_stats.csv`, `stats_summary.json` |

When using the full pipeline, intermediate outputs are organized under `output/pipeline_intermediates/`:

```
output/pipeline_intermediates/
├── baseline_embeddings/        # Step 2: embeddings without imputation
│   └── sentencetransformer/    #   (or automodel/, depending on backend)
├── baseline_labels/            # Step 3: labels without imputation
├── concept_stats/              # Step 4: baseline statistics + concept_stats.csv
└── final_stats/                # Step 7: final statistics (with imputation)
```


## Testing with Dummy Data

The repository includes 20 sample reports and pre-made structured JSONs so you can test steps 2 and 3 without an LLM API.

### Quick test: label extraction (no model needed)

```bash
# Point json_dir at the included sample data
python label_extraction/extract_binary_labels.py \
  --config config/binary_label_extraction.yaml \
  --json-dir data/sample_structured_jsons \
  --out-dir output/test_labels \
  --verbose-test
```

This processes the 20 sample reports, applies pathology label imputation with abundance balancing from `data/concept_stats.csv`, and prints per-report imputation decisions.

### Quick test: embedding (requires a local model)

```bash
# Edit the model_dir in the config to point to your local model, then:
python embedding/embed_structured_reports.py \
  --config config/embedding_sentence_transformer.yaml \
  --json-dir data/sample_structured_jsons \
  --out-dir output/test_embeddings
```

### Quick test: structuring (requires an LLM API)

```bash
# Edit api_base, api_key, model_name in the config, then:
python structuring_labeling/report_structuring_labeling.py \
  --config config/structuring_labeling.yaml \
  --max-to-process 2
```

### Inspect a single report

```bash
# See exactly how labels are extracted for one report
python label_extraction/extract_binary_labels.py \
  --config config/binary_label_extraction.yaml \
  --json-dir data/sample_structured_jsons \
  --inspect-report-id RPT001
```


## Pipeline Details

### Structuring & Labeling

The LLM receives a system prompt (`templates/system_prompt.md`) defining extraction rules and a JSON schema (`templates/output_schema.json`) for guided generation with `strict: true`. The schema enforces `required` fields and `additionalProperties: false` at every nesting level so the LLM always produces all expected fields.

Each report is processed into a structured JSON containing:
- **Text-only fields**: `examination`, `clinical_information`, `report_date`, `comparison`, `impression`
- **Text + label fields**: `support_devices.*` (label: "present"/"not present"), `thoracic_organs.*` and `pathologies.*` (label: 0-3 severity score)

### Text Embedding with Imputation

When a concept field has no text (e.g., "chest drain" not mentioned in the report), the embedding script can impute a synthetic negative sentence sampled from `templates/neg_entry_sampling.json`. Two imputation rules:

1. **Support devices**: If label is "not present" and text is empty, sample a negative sentence (e.g., "No chest drain in place.")
2. **Pathologies**: If label is 0 (unknown) but a sufficient fraction of other pathology fields in the same report have non-empty text, sample a negative sentence

Abundance balancing (via `concept_stats.csv`) caps the number of imputed entries per concept to match naturally occurring text counts, preventing over-imputation of rare concepts.

### Binary Label Extraction with Imputation

Labels are mapped to binary targets:
- `support_devices.*`: "present" -> 1, "not present" -> 0
- `thoracic_organs.*` and `pathologies.*`: score 0 -> masked, 1 -> 0 (normal), 2-3 -> 1 (pathologic)

The same pathology imputation rule from the embedding step applies here: if a pathology label is 0 but enough other pathologies carry non-zero labels, it is imputed as 0 (negative, valid). This keeps the label set consistent with the embedding set. Abundance balancing works identically via reservoir sampling from `concept_stats.csv`.

Output arrays:
- `labels_*.npy`: (N, K) float32 binary labels
- `masks_*.npy`: (N, K) bool validity masks (False = unknown, exclude from training)
- `meta_*.parquet`: per-report metadata including imputation provenance

### Concept Statistics Extraction

The `extract_concept_stats.py` script scans existing embedding and/or label shards to produce per-concept counts that drive abundance-balanced imputation. It is meant to be run on output from a **non-imputed** (or baseline) run so the counts reflect natural text/label occurrence rates.

**Text stats** (from embedding shards): For each concept, counts how many reports have a non-imputed text entry in their embedding metadata.

**Label stats** (from label shards): For each concept, counts valid/positive/negative/masked labels plus how many were imputed.

**Split support**: An optional split file (CSV or parquet with `report_id` and `split` columns) breaks statistics down by train/val/test. Without a split file, all reports are grouped into a single "all" split -- the output `concept_stats.csv` is still usable for abundance balancing.

```bash
# From embedding shards, no splits:
python stats/extract_concept_stats.py \
  --text-emb-dir output/embeddings_baseline \
  --concepts-file templates/concepts.txt \
  --out-dir output/stats

# With both embedding + label shards and a split file:
python stats/extract_concept_stats.py \
  --text-emb-dir output/embeddings_baseline \
  --label-dir output/labels_baseline \
  --concepts-file templates/concepts.txt \
  --split-file data/splits.csv \
  --out-dir output/stats
```

The key output is `concept_stats.csv` with columns `concept_name, split, n_samples_with_text, n_samples_total` -- the same format expected by the `concept_stats_csv` setting in the embedding and label extraction configs.


## Recommended Workflow for Own Data

When working with your own dataset (beyond the demo), the full pipeline script
(`run_full_pipeline.sh`) automates the recommended 7-step workflow. This ensures
imputation quotas are derived from actual data rather than from hand-crafted
dummy counts.

If you already have a `concept_stats.csv` from a previous run, you can skip
the baseline phase and run only the imputed steps:

```bash
./scripts/run_full_pipeline.sh --skip-structuring --skip-baseline --skip-stats
```


## Compatibility

### CPU-only operation

A GPU is recommended for the embedding step but is **not required**. All scripts
work on CPU:

- **Structuring** (step 1): Calls an external LLM API — no local GPU needed.
- **Embedding** (steps 2, 5): The embedding script auto-detects whether CUDA is
  available. If `device: "cuda:0"` is configured but no GPU exists, it
  automatically falls back to CPU and switches to float32 precision. This will
  be significantly slower for large models but is fully functional.
- **Label extraction** (steps 3, 6): Pure NumPy/Pandas — CPU only, no GPU used.
- **Concept statistics** (steps 4, 7): Pure Pandas — CPU only, no GPU used.

To explicitly force CPU operation, set `device: "cpu"` in the embedding config.


## Configuration Reference

All YAML keys use underscores and map directly to CLI arguments (with dashes). For example, `json_filename_template` in YAML corresponds to `--json-filename-template` on the CLI.

CLI arguments always take precedence over config file values.

### Common patterns

```yaml
# All configs support these input settings:
reports_csv: "data/sample_reports.csv"
report_id_column: "report_id"
json_dir: "data/sample_structured_jsons"
json_filename_template: "{report_id}_structured.json"

# Imputation settings (embedding + label extraction):
pathology_valid_threshold_pct: 15.0   # min % of other pathology fields non-zero
pathology_min_other_fields: 1         # min count of other pathology fields
sampling_seed: 42                     # deterministic sampling (-1 for random)
concept_stats_csv: "data/concept_stats.csv"  # enables abundance balancing
```
