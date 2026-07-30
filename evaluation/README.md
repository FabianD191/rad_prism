# RadPRISM — Model Evaluation

Two config-driven evaluation scripts for a trained RadPRISM model, sharing the
model/metric/retrieval helpers in `eval_common.py`.

```
RadPRISM/
├── src/                                 # shared model + data code (repo root)
├── data/                                # dummy dataset (see below)
└── evaluation/
    ├── eval_common.py                   # shared: config/model/metrics/retrieval/embedders
    ├── evaluate_internal.py             # internal-dataset evaluation
    ├── evaluate_chexlocalize.py         # external CheXpert / CheXlocalize evaluation
    ├── run_inference_demo.py            # apply the provided checkpoint to images
    ├── llm_consistency_check.py         # optional LLM cls-vs-retrieval consistency check
    ├── config/
    │   ├── evaluate_internal.yaml
    │   ├── evaluate_internal_dummy.yaml
    │   ├── evaluate_chexlocalize.yaml
    │   ├── inference_demo.yaml
    │   └── llm_consistency.yaml
    ├── scripts/
    │   ├── run_evaluate_internal.sh
    │   ├── run_evaluate_chexlocalize.sh
    │   ├── run_inference_demo.sh
    │   └── run_llm_consistency.sh
    └── requirements.txt
```

Install: `pip install -r requirements.txt` (adds `tqdm` and, only for real
text-embedding backends, `sentence-transformers` / `transformers` on top of the
training requirements).


## 1. Internal evaluation (`evaluate_internal.py`)

Evaluates a trained model on an internal dataset (the bundled dummy dataset or
your own). Three independent, toggleable modes:

| Mode | Output | Notes |
|---|---|---|
| `thresholds` | `<split>_thresholds.json` | per-concept best-F1 / Youden-J thresholds on the val split |
| `classify`   | `<split>_cls_head_metrics.csv` (+ `_zero_shot_metrics.csv`) | AUROC / AUPRC / F1 per concept, for the cls heads and (optionally) zero-shot prompts |
| `retrieve`   | `<split>_retrieval.csv` | top-k text snippets per concept, per image |

`model.run_dir` points at a fine-tuning (or pretraining) run; the model
architecture and data settings are read from that run's `config.json` (the parent
pretrain config is merged in automatically for fine-tune runs).

Zero-shot and retrieval need a **text-embedding model** to encode prompts/snippets
into the model's text space. Set `text_embedding.backend` to
`sentence_transformer` / `auto_model` with the same model that produced the
training text embeddings. On the synthetic dummy dataset use `backend: hash`
(deterministic pseudo-random vectors) so everything runs without a heavy model —
those results are **not meaningful**, they only exercise the pipeline.

```bash
python evaluation/evaluate_internal.py --config evaluation/config/evaluate_internal.yaml
```


## 2. External CheXpert / CheXlocalize evaluation (`evaluate_chexlocalize.py`)

Two stages (toggle in the config; keep both on for a full run):

- **`classify`** — runs the model on CheXpert images, maps the model's concept
  predictions to the CheXpert classes, computes AUROC / AUPRC / F1, optionally
  retrieves text snippets, and **saves per-sample attention maps** (needed by
  stage 2).
- **`localize`** — visual grounding: how well the concept attention maps overlap
  the CheXlocalize ground-truth segmentations (Dice / IoU / pointing-game /
  pixel-AUPRC), for the one-to-one concept↔class targets. Supports optimizing the
  attention threshold on one split and applying it to another (e.g. val→test).

Requires the CheXpert images + label CSVs and the CheXlocalize GT segmentation
JSONs (downloaded separately — not included). The concept↔CheXpert mappings use
the English concept names and can be overridden in the config
(`chexpert_mapping`, `localize_targets`).

```bash
python evaluation/evaluate_chexlocalize.py --config evaluation/config/evaluate_chexlocalize.yaml
```


## 3. Inference demo with the provided checkpoint (`run_inference_demo.py`)

Applies the shipped RadPRISM checkpoint (`data/radprism_checkpoint/`) to chest
X-ray images and, per image, saves attention-overlay PNGs, a thresholded
classification CSV (Youden-J thresholds), and a top-3 **English** text-retrieval
CSV. 

Setup (see `data/radprism_checkpoint/README.md`): download RAD-DINO-MAIRA-2 and
build the retrieval bank once with your Qwen model
(`utils/embed_radprism_text_db.py`). Then:

```bash
python evaluation/run_inference_demo.py --config evaluation/config/inference_demo.yaml
```


## 4. Optional LLM consistency check (`llm_consistency_check.py`)

An optional post-processing step that runs *after* inference/evaluation. For each
case it sends the per-concept classification decisions and the top retrieved text
snippet to an LLM, which:

- assesses the **consistency** between the classification decision and the
  retrieved text per concept (`consistent` / `inconsistent` / `uncertain`) with a
  short feedback note (the core component), and
- optionally proposes a cleaned/corrected **final text** per concept
  (`prompting.produce_final_text_proposal`, on by default).

It consumes the per-case `classification.csv` + `retrieval_top_matches.csv`
written by the inference demo (or the evaluation scripts) and binarizes with the
same thresholds JSON. Output is validated against a strict JSON schema.

```bash
# Build payload/schema/messages WITHOUT calling an LLM (works on the dummy outputs):
python evaluation/llm_consistency_check.py --config evaluation/config/llm_consistency.yaml --dry-run

# Real run (needs an OpenAI-compatible endpoint set in the config):
python evaluation/llm_consistency_check.py --config evaluation/config/llm_consistency.yaml
```

By default it runs in **batch mode** over `output/inference_demo/`, writing a
`llm_consistency/` sub-folder in each case with `prepared_payload.json`,
`output_schema.json`, `messages.json`, and (on a real run) `final_model_output.json`.
A real run requires an OpenAI-compatible chat endpoint (`api.*` in the config, or
`OPENAI_API_KEY`); `--dry-run` requires neither.


## Running the full dummy pipeline end-to-end

From the `RadPRISM/` root (CPU-only, no external models needed):

```bash
# 1. Build the dummy dataset (image/label/text caches + master index)
python utils/make_dummy_dataset.py

# 2. Pretrain (alignment) on the dummy data
python training/pretraining/train_run_pretrain.py --config training/config/pretrain_dummy.yaml

# 3. Fine-tune the classification heads
#    (edit training/config/finetune_cls_dummy.yaml: set <PRETRAIN_RUN> to the
#     timestamped dir under output/dummy_runs/)
python training/finetuning/train_run_finetune_cls.py --config training/config/finetune_cls_dummy.yaml

# 4. Evaluate (thresholds + cls-head classification + retrieval)
#    (edit evaluation/config/evaluate_internal_dummy.yaml: set <FINETUNE_RUN>)
python evaluation/evaluate_internal.py --config evaluation/config/evaluate_internal_dummy.yaml
```

To also try the provided trained checkpoint + the optional LLM consistency check
(needs RAD-DINO-MAIRA-2 and the pre-computed text banks; see
`data/radprism_checkpoint/README.md`):

```bash
# 5. Inference demo with the provided checkpoint
python evaluation/run_inference_demo.py --config evaluation/config/inference_demo.yaml

# 6. LLM consistency check over the demo outputs (dry-run needs no LLM endpoint)
python evaluation/llm_consistency_check.py --config evaluation/config/llm_consistency.yaml --dry-run
```

The CheXlocalize evaluation cannot run on the dummy data (it needs the external
CheXpert/CheXlocalize dataset).

> Note: with only ~20 tiny dummy samples the metrics are degenerate (many NaN /
> trivial values). The dummy pipeline is for verifying the code runs and for
> illustrating the expected data structure, not for meaningful results.
