# RadPRISM Trained Checkpoint

A ready-to-use RadPRISM checkpoint for inference on chest X-ray images
(classification, attention/grounding, and concept-wise text retrieval).

## Files

| File | Description |
|---|---|
| `radprism_heads.safetensors` | Trained weights only: concept tokens, cross-attention, classification heads, text projection. |
| `radprism_config.json` | Minimal architecture config + concept names (English). |
| `radprism_val_thresholds.json` | Per-concept operating thresholds: `thr_f1` (best F1) and `thr_j` (Youden-J, used by the demo). |
| `radprism_text_db_de.json` | German retrieval/zero-shot texts with English translations (source for the banks below). |
| `retrieval_bank.pt` / `zero_shot_bank.pt` | Pre-computed German Qwen embeddings (produced by `utils/embed_radprism_text_db.py`; see below). |

The RadPRISM-trained tensors distributed in this directory are covered by the
repository's MIT license. The external backbone and text-embedding models are
not included and retain their respective upstream terms.

> **Backbone not included.** The frozen vision backbone is the public
> **RAD-DINO-MAIRA-2** and is *not* redistributed here. It remained unchanged
> during training, and the exported file contains no backbone tensors. Download
> the backbone from HuggingFace and point the demo at it. Only the
> RadPRISM-trained tensors are shipped.
> RAD-DINO-MAIRA-2 is licensed under the Microsoft Research License Terms, which
> currently restrict use to non-commercial, non-revenue-generating research and
> prohibit redistribution. See the repository's `THIRD_PARTY_NOTICES.md`.

## One-time setup

1. **Download the backbone**: get `rad-dino-maira-2` (HuggingFace) locally and set
   `rad_dino_model_dir` in `evaluation/config/inference_demo.yaml`.

2. **Build the text banks** (needed for retrieval; the model was trained on German
   text embeddings, so the banks must be encoded with the same Qwen model):

   ```bash
   python utils/embed_radprism_text_db.py \
     --db data/radprism_checkpoint/radprism_text_db_de.json \
     --config data/radprism_checkpoint/radprism_config.json \
     --model-dir /path/to/Qwen3-Embedding-4B \
     --out-dir data/radprism_checkpoint
   ```

   Use the same encoding parameters as the training text embeddings
   (`truncate_dim=768`, `max_length=2048`; see the report pipeline's
   `config/embedding_sentence_transformer.yaml`).

## Run the demo

```bash
python evaluation/run_inference_demo.py --config evaluation/config/inference_demo.yaml
```

Per input image (the dummy JPGs by default) it writes to `output/inference_demo/<image>/`:
- `attention_overlays/<concept>.png` — attention heatmap over the image,
- `classification.csv` — per-concept probability, threshold, and binary prediction,
- `retrieval_top_matches.csv` — top-3 matching snippets per concept, shown in **English**.

## How it works

The model produces one image-derived embedding per concept. Classification uses the
trained heads with the Youden-J thresholds. Retrieval compares each concept
embedding against the pre-computed **German** text embeddings (projected into the
model's space) and returns the closest snippets, displayed via their **English**
translation. No text-embedding model is required at inference — only the committed
banks.

## Notes

- Concept names were translated German→English in a strictly order-preserving way,
  so they still index the same trained heads.
