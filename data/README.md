# RadPRISM Dummy Dataset

A tiny demonstration dataset that illustrates the **minimal data structure** the
training and evaluation code expects, and lets you run the whole pipeline on CPU
without external models. The reports, identifiers, labels, and text embeddings
are synthetic. The seven bundled example X-rays are not synthetic.

> This is for structure illustration and code testing only. The example JPGs are
> reused across synthetic accessions and the text embeddings are random — no
> meaningful model or metric can come out of it. The JPGs are separately
> licensed under CC BY-NC 4.0; see their
> [license and attribution notice](jpg/README.md).

## Build it

```bash
python utils/make_dummy_dataset.py
```

This generates everything below from the bundled inputs (`data/jpg/*.jpg` and the
sample structured reports in `report_struct_label/data/sample_structured_jsons/`).

## Layout

```
data/
├── jpg/                          # 7 sample chest X-ray JPGs (the only committed images)
├── dummy_master_index.csv        # one row per sample: SOPInstanceUID, accession, PatientID, image_path, view
├── dummy_image_cache/            # sharded image cache (cached_dataset_path)
│   ├── images_00000.npy          #   (N, H, W) float16
│   ├── meta_00000.parquet        #   sop_uid / SOPInstanceUID / accession / ...
│   ├── image_sharded_index.parquet
│   └── summary.json
├── dummy_text_emb_cache/         # synthetic per-concept text embeddings (text_override_dir)
│   ├── embeddings_00000.npy      #   (M, 768) random unit vectors
│   └── meta_00000.parquet        #   report_id / field=structured.<concept> / text
├── dummy_label_cache/            # binary labels from the report pipeline (cls_override_dir)
│   ├── labels_00000.npy          #   (N, 19) float
│   ├── masks_00000.npy           #   (N, 19) bool (validity)
│   ├── meta_00000.parquet        #   report_id / concepts_sha1 / K / ...
│   ├── cls_sharded_index.parquet  #   deterministic accession-to-row index
│   └── summary.json
├── dummy_retrieval_text_db.json  # English clinical sentences per concept (retrieval)
└── dummy_zero_shot_prompt_db.json# English positive/negative prompts per concept (zero-shot)
```

## Key conventions (also apply to real data)

- **Keys**: images are keyed by `SOPInstanceUID`; reports/labels/embeddings by
  `accession`. In the dummy set `accession == report_id`. The sharded stores
  accept either an `accession` or a `report_id` column in their metadata, so the
  report_struct_label pipeline output (keyed by `report_id`) works directly.
- **Concept names** are the 19 English dot-paths from
  `report_struct_label/templates/concepts.txt` (e.g. `pathologies.lung.pneumonia`),
  used identically across the report pipeline, training and evaluation.
- **Text-embedding fields** are named `structured.<concept_name>`; the embedding
  dimension must match `model.text_in_dim` (768 here).
- On real data, the text-embedding and label caches are produced by the
  `report_struct_label` pipeline (the label cache here is built by its
  `extract_binary_labels.py`; the embeddings here are synthetic stand-ins).
```
