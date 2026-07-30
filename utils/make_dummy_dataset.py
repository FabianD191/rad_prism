#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the RadPRISM dummy dataset for end-to-end training / evaluation testing.

The dummy dataset illustrates the *minimal* data structure the training and
evaluation code expects. It is NOT meant to produce meaningful models or metrics
— the images are the handful of sample JPGs in ``data/jpg`` (reused across
accessions) and the text embeddings are random vectors.

What this script produces (under ``data/``):
  1. dummy_master_index.csv     - one row per image (SOPInstanceUID, accession,
                                  PatientID, image_path, view).
  2. dummy_text_emb_cache/      - synthetic sharded per-concept text embeddings
                                  (random unit vectors) keyed by report_id/field,
                                  matching ShardedEmbeddingStore's format.
  3. dummy_image_cache/         - sharded image cache built by
                                  prepare_sharded_image_cache.py (subprocess).
  4. dummy_label_cache/         - binary label shards built by the
                                  report_struct_label pipeline (subprocess).

The label cache is created by the *real* report label-extraction script, so it
demonstrates how the report pipeline output feeds training. The text-embedding
cache is synthetic because real embeddings require a large embedding model; in a
real workflow it is produced by the report pipeline's embedding step.

Run from anywhere:
  python utils/make_dummy_dataset.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]      # RadPRISM/
DATA_DIR = REPO_ROOT / "data"
JPG_DIR = DATA_DIR / "jpg"
REPORT_DIR = REPO_ROOT / "report_struct_label"
JSON_DIR = REPORT_DIR / "data" / "sample_structured_jsons"
CONCEPTS_FILE = REPORT_DIR / "templates" / "concepts.txt"

TEXT_EMB_DIM = 768                                   # must match model.text_in_dim


def load_concepts() -> list:
    """Read the ordered concept list from the report pipeline's concepts.txt."""
    concepts = []
    for line in CONCEPTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            concepts.append(line)
    return concepts


def concept_text(report_data: dict, concept_path: str) -> str:
    """Return the free-text snippet for a concept path, or '' if absent."""
    node = report_data
    for key in concept_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return ""
        node = node[key]
    if isinstance(node, dict):
        return str(node.get("text", "") or "")
    return ""


def build_master_index(report_ids: list, rng: np.random.Generator) -> pd.DataFrame:
    """One row per (report) sample, cycling through the available JPGs."""
    jpgs = sorted(JPG_DIR.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No JPGs found in {JPG_DIR}")
    views = ["frontal", "lateral"]
    rows = []
    for i, rid in enumerate(report_ids):
        jpg = jpgs[i % len(jpgs)]
        rows.append({
            "SOPInstanceUID": f"1.2.826.0.1.{i + 1:06d}",
            "accession": rid,                     # accession == report_id in the dummy set
            "PatientID": f"DUMMYPT{i + 1:03d}",   # unique patient per sample
            "image_path": str(Path("data") / "jpg" / jpg.name),   # repo-root-relative
            "view": views[i % len(views)],
        })
    return pd.DataFrame(rows)


def build_text_embeddings(report_ids: list, concepts: list, rng: np.random.Generator, out_dir: Path):
    """
    Write a synthetic sharded text-embedding cache.

    For every concept that has a non-empty text snippet in a report, emit one
    random unit vector. The layout (embeddings_00000.npy + meta_00000.parquet with
    report_id/field/text) is exactly what ShardedEmbeddingStore consumes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = []
    meta_rows = []
    for rid in report_ids:
        jpath = JSON_DIR / f"{rid}_structured.json"
        report = json.loads(jpath.read_text(encoding="utf-8"))
        data = report.get("data", report)
        for cname in concepts:
            text = concept_text(data, cname)
            if not text.strip():
                continue
            v = rng.standard_normal(TEXT_EMB_DIM).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-8)
            vectors.append(v)
            meta_rows.append({"report_id": rid, "field": f"structured.{cname}", "text": text})

    embeddings = np.stack(vectors, axis=0).astype(np.float32)
    np.save(out_dir / "embeddings_00000.npy", embeddings)
    pd.DataFrame(meta_rows).to_parquet(out_dir / "meta_00000.parquet", index=False)
    print(f"[text-emb] wrote {len(meta_rows)} synthetic embeddings ({TEXT_EMB_DIM}-d) -> {out_dir}")


def run_image_cache(master_index: Path, out_dir: Path, image_size: int):
    """Build the sharded image cache via prepare_sharded_image_cache.py."""
    if out_dir.exists() and any(out_dir.glob("images_*.npy")):
        print(f"[image-cache] {out_dir} already populated, skipping.")
        return
    cmd = [
        sys.executable, str(REPO_ROOT / "utils" / "prepare_sharded_image_cache.py"),
        "--master_index", str(master_index.relative_to(REPO_ROOT)),
        "--out_dir", str(out_dir.relative_to(REPO_ROOT)),
        "--image_column", "image_path",
        "--accession_column", "accession",
        "--sop_uid_column", "SOPInstanceUID",
        "--image_size", str(image_size),
        "--keep_aspect_ratio",
        "--shard_size", "4096",
        "--storage_dtype", "float16",
    ]
    print(f"[image-cache] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def run_label_cache(out_dir: Path):
    """Build the binary label cache via the report_struct_label pipeline."""
    if out_dir.exists() and any(out_dir.glob("labels_*.npy")):
        print(f"[label-cache] {out_dir} already populated, skipping.")
        return
    cmd = [
        sys.executable, str(REPORT_DIR / "label_extraction" / "extract_binary_labels.py"),
        "--reports-csv", str((REPORT_DIR / "data" / "sample_reports.csv").relative_to(REPO_ROOT)),
        "--json-dir", str(JSON_DIR.relative_to(REPO_ROOT)),
        "--concepts-file", str(CONCEPTS_FILE.relative_to(REPO_ROOT)),
        "--out-dir", str(out_dir.relative_to(REPO_ROOT)),
        "--disable-abundance-balance",
    ]
    print(f"[label-cache] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    ap = argparse.ArgumentParser(description="Generate the RadPRISM dummy dataset.")
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-image-cache", action="store_true")
    ap.add_argument("--skip-label-cache", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use every sample report that has a structured JSON.
    report_ids = sorted(p.stem.replace("_structured", "")
                        for p in JSON_DIR.glob("*_structured.json"))
    concepts = load_concepts()
    print(f"[info] {len(report_ids)} reports, {len(concepts)} concepts.")

    # 1. Master index
    master = build_master_index(report_ids, rng)
    master_path = DATA_DIR / "dummy_master_index.csv"
    master.to_csv(master_path, index=False)
    print(f"[master-index] wrote {len(master)} rows -> {master_path}")

    # 2. Synthetic text-embedding cache
    build_text_embeddings(report_ids, concepts, rng, DATA_DIR / "dummy_text_emb_cache")

    # 3. Sharded image cache
    if not args.skip_image_cache:
        run_image_cache(master_path, DATA_DIR / "dummy_image_cache", args.image_size)

    # 4. Binary label cache (via the report pipeline)
    if not args.skip_label_cache:
        run_label_cache(DATA_DIR / "dummy_label_cache")

    print("\n[done] Dummy dataset ready under", DATA_DIR)


if __name__ == "__main__":
    main()
