# RadPRISM — Model Training

Training code for **RadPRISM**. Training has two stages:

1. **Pretraining** (`pretraining/train_run_pretrain.py`) — concept-averaged
   InfoNCE alignment. A vision backbone (RAD-DINO-MAIRA-2 by default) encodes the
   X-ray; `K` learnable *concept tokens* (one per clinical concept) attend to the
   vision tokens to produce one image embedding per concept, which is
   contrastively matched with the concept's report-snippet text embedding.
   Concept-aware weighted sampling oversamples images covering rare concepts.

2. **Fine-tuning** (`finetuning/train_run_finetune_cls.py`) — loads a pretrained
   checkpoint, freezes the backbone and concept cross-attention, and trains only
   the per-concept **binary classification heads** (BCE), turning RadPRISM into a
   multi-label classifier.

Both stages share the model/data code in the repository-root `src/` folder and
consume the per-concept text embeddings and structured labels produced by the
[`report_struct_label`](../report_struct_label) pipeline.


## Repository Structure

```
RadPRISM/
├── report_struct_label/            # report structuring / embedding / labeling (separate pipeline)
├── src/                            # shared model + data code (used by training AND evaluation)
│   ├── model.py                    #   VisionConceptModel (backbones, cross-attention, heads)
│   ├── losses.py                   #   InfoNCE alignment loss + BCE classification loss
│   └── dataset.py                  #   datasets, splits, transforms, weighted-sampling weights
│
└── training/
    ├── config/
    │   ├── pretrain.yaml            # pretraining settings
    │   └── finetune_cls.yaml        # fine-tuning settings
    ├── pretraining/
    │   └── train_run_pretrain.py    # stage 1 entry point
    ├── finetuning/
    │   └── train_run_finetune_cls.py# stage 2 entry point
    ├── scripts/
    │   ├── run_pretrain.sh          # background launcher (logs + PID)
    │   └── run_finetune_cls.sh
    ├── requirements.txt
    └── README.md
```


## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `transformers` (for the
RAD-DINO-MAIRA-2 backbone), `pandas`, `numpy`, `scikit-learn`, `pyarrow`,
`pyyaml`. `tensorboard` is optional (training runs without it).

### 2. Download a vision backbone

The default backbone is **RAD-DINO-MAIRA-2**, loaded from a local HuggingFace
directory (offline). Set `model.rad_dino_model_dir` in `config/pretrain.yaml`.
Two other backbones are selectable via `model.vision_backbone`:

| `vision_backbone`  | Config keys to set                              |
|--------------------|-------------------------------------------------|
| `rad_dino_maira_2` | `rad_dino_model_dir` (local HF model directory) |
| `dinov2`           | `dinov2_repo_path`, `dinov2_weights_path`       |
| `resnet50`         | `vision_weights_path` (optional local `.pth`)   |

### 3. Configure

All settings live in `config/*.yaml`. Edit the placeholder paths before running.
A few high-traffic options can also be passed on the command line and take
priority over the file (`--device`, `--epochs`, `--seed`, `--run-dir`, ...).


## Expected Data Layout

This folder ships **no image data** (chest X-rays cannot be redistributed). You
provide the following, referenced from the config files:

1. **Master index** (`paths.master_index_path`) — CSV/parquet, one row per image,
   with columns `image_path`, `accession`, `SOPInstanceUID`, and `PatientID`
   (or `patient_id`, required only for `split_on: patient`).
2. **Text embeddings** (`paths.embedding_dir` / `data.text_override_dir`) — a
   sharded store of per-concept report-snippet embeddings, keyed by accession and
   `structured.<concept_name>`. Output of the `report_struct_label` embedding step.
3. **Structured labels** (`data.cls_override_dir`) — required for fine-tuning
   (the classification head); optional for pretraining.
4. **(Optional) Cached image store** (`paths.cached_dataset_path`) — sharded
   `images_*.npy` + `meta_*.parquet` for fast loading; otherwise PNGs are read on
   the fly from the master index.
5. **(Optional) Concept-coverage CSV** (`strategy.weighted_sampling_coverage_csv`,
   pretraining) and **class-statistics CSV** (`losses.class_stats_path`,
   fine-tuning) — produced by the concept-stats step of the report pipeline.

The 19 clinical concepts (`support_devices.*`, `thoracic_organs.*`,
`pathologies.*`) match the report-JSON hierarchy defined in the report pipeline.


## Running

### Stage 1 — Pretraining

```bash
# Foreground
python pretraining/train_run_pretrain.py --config config/pretrain.yaml

# Background (writes logs/pretrain_<timestamp>.log and a .pid file)
./scripts/run_pretrain.sh --device cuda:0
```

Each run creates a timestamped sub-directory under `paths.run_dir` with
`config.json`, `metrics.csv`, `best_align.pt`, periodic `epoch_*.pt`, and
(if installed) TensorBoard logs under `tb/`.

### Stage 2 — Fine-tuning

Point `config/finetune_cls.yaml` at the pretrained run's checkpoint and
`config.json` — the fine-tune **inherits** the architecture/data settings from
that config and only overrides what you specify:

```yaml
pretrain:
  ckpt_path: "/path/to/pretrain_run/best_align.pt"
  config_path: "/path/to/pretrain_run/config.json"
```

```bash
python finetuning/train_run_finetune_cls.py --config config/finetune_cls.yaml
# or
./scripts/run_finetune_cls.sh --device cuda:0
```

The fine-tune run writes `best_cls.pt` (best macro-AUPRC, with per-concept
operating thresholds), `pos_weights.pt`, `metrics.csv` and
`per_concept_metrics.csv` to a `finetune_cls_<timestamp>/` sub-directory of
`paths.run_dir`.


## Training Details

### Pretraining objective

Concept-averaged InfoNCE (`losses.use_align_loss: true`): for each concept with
at least two valid image–text pairs in the batch, a symmetric in-batch InfoNCE
loss is computed; per-concept losses are **averaged equally** so frequent
concepts do not dominate. Each concept uses a **learnable temperature**
(`losses.learnable_concept_taus: true`), clamped to `[min_tau_align, max_tau_align]`.

### Concept-aware weighted sampling

When `strategy.use_weighted_sampling: true`, a `WeightedRandomSampler` upsamples
images whose report covers rare concepts (coverage below `rare_threshold`) so
those concepts still form enough in-batch positive pairs. Weights are cached to
`strategy.weighted_sampling_cache_path`.

### Fine-tuning

Only the classification heads receive gradients; the rest of the model is frozen
and kept in eval mode. BCE positive-class weights can come from a
class-statistics CSV, a saved `pos_weights.pt`, or the training set, with optional
"tempering" (`w_c -> w_c ** gamma`, `gamma < 1`) to soften extreme weights, and an
optional concept-balanced reduction that equal-weights concepts in the loss mean.

### Reproducibility sweep

Set `train.seed_list: [1, 2, 3]` (in either config) to launch one child process
per seed automatically; each seed writes to its own run directory.


## Compatibility

- A **GPU is strongly recommended**. For a CPU smoke test set `train.device: cpu`,
  `train.bf16: false`, a small `data.batch_size`, and small
  `data.max_train`/`data.max_val`.
