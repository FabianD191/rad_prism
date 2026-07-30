#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-concept text and label statistics from embedding and label shards.

Produces a concept_stats.csv that can be used for abundance-balanced imputation
in the embedding and label extraction scripts.

Supports an optional split file (CSV/parquet with report_id + split columns)
to break statistics down by train/val/test.  Without a split file, all reports
are grouped into a single "all" split.

Usage:
  # From embedding shards only (text stats):
  python extract_concept_stats.py \\
    --text-emb-dir output/embeddings_qwen3 \\
    --concepts-file templates/concepts.txt \\
    --out-dir output/stats

  # With split information:
  python extract_concept_stats.py \\
    --text-emb-dir output/embeddings_qwen3 \\
    --label-dir output/binary_labels \\
    --concepts-file templates/concepts.txt \\
    --split-file data/splits.csv \\
    --out-dir output/stats

  # With YAML config:
  python extract_concept_stats.py --config config/concept_stats.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()


def normalize_rel_field(path: str) -> str:
    p = clean_text(path)
    if p.startswith("structured."):
        p = p[len("structured."):]
    return p


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for --config. Install: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg else {}


def load_concept_names(path: Optional[Path], inline: Optional[str]) -> List[str]:
    names: List[str] = []
    if inline:
        names.extend(p.strip() for p in inline.split(",") if p.strip())
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    names.append(s)
    seen: set = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def canonical_split_order(splits: Sequence[str]) -> List[str]:
    base = ["train", "val", "test"]
    seen: set = set()
    uniq: List[str] = []
    for s in splits:
        ss = str(s)
        if ss not in seen:
            uniq.append(ss)
            seen.add(ss)
    ordered = [s for s in base if s in seen]
    ordered.extend(s for s in uniq if s not in set(base))
    return ordered


def load_split_map(
    split_path: Path,
    report_id_column: str = "report_id",
    split_column: str = "split",
) -> Dict[str, str]:
    """Load a split file and return {report_id -> split} mapping."""
    suffix = split_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(split_path)
    else:
        df = pd.read_csv(split_path, dtype=str)

    rid_col = None
    for cand in [report_id_column, "report_id", "accession", "group_id"]:
        if cand in df.columns:
            rid_col = cand
            break
    if rid_col is None:
        raise KeyError(
            f"No report ID column found in {split_path}. "
            f"Tried: {report_id_column}, report_id, accession, group_id. "
            f"Available: {df.columns.tolist()}"
        )

    split_col = None
    for cand in [split_column, "split"]:
        if cand in df.columns:
            split_col = cand
            break
    if split_col is None:
        raise KeyError(
            f"No split column found in {split_path}. "
            f"Available: {df.columns.tolist()}"
        )

    df[rid_col] = df[rid_col].astype(str).str.strip()
    df[split_col] = df[split_col].astype(str).str.strip()
    df = df[df[rid_col] != ""].drop_duplicates(subset=[rid_col], keep="first")
    return dict(zip(df[rid_col], df[split_col]))


# ═══════════════════════════════════════════════════════════════════════════════
# Text statistics from embedding shards
# ═══════════════════════════════════════════════════════════════════════════════

def compute_text_stats(
    text_emb_dir: Path,
    concept_names: List[str],
    split_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Scan embedding meta parquets and count per-concept text occurrences.

    For each concept, counts report_ids that have a non-imputed text entry.
    Returns a DataFrame with columns:
      concept_name, split, n_samples_with_text, n_samples_without_text, n_samples_total
    """
    concept_fields = {f"structured.{c}": c for c in concept_names}
    wanted_fields = set(concept_fields.keys())

    meta_files = sorted(text_emb_dir.glob("meta_*.parquet"))
    if not meta_files:
        print(f"[text] No meta_*.parquet files found in {text_emb_dir}")
        return pd.DataFrame()

    chunks: List[pd.DataFrame] = []
    for mp in meta_files:
        try:
            cols_available = pd.read_parquet(mp, columns=[]).columns.tolist()
            read_cols = ["report_id", "field"]
            if "is_imputed" in cols_available:
                read_cols.append("is_imputed")
            df = pd.read_parquet(mp, columns=read_cols)
        except Exception as e:
            print(f"[text] Warning: could not read {mp}: {e}")
            continue

        df["field"] = df["field"].astype(str).str.strip()
        df = df[df["field"].isin(wanted_fields)].copy()
        if df.empty:
            continue

        if "is_imputed" in df.columns:
            df = df[~df["is_imputed"].fillna(False).astype(bool)].copy()

        df["report_id"] = df["report_id"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["report_id", "field"], keep="first")
        chunks.append(df[["report_id", "field"]])

    if not chunks:
        print("[text] No matching text entries found in embedding shards.")
        return pd.DataFrame()

    all_text = pd.concat(chunks, ignore_index=True)
    all_text = all_text.drop_duplicates(subset=["report_id", "field"], keep="first")

    all_report_ids = set(all_text["report_id"].unique())
    for mp in meta_files:
        try:
            df = pd.read_parquet(mp, columns=["report_id"])
            all_report_ids.update(df["report_id"].astype(str).str.strip().unique())
        except Exception:
            continue

    if split_map:
        rid_to_split = split_map
        all_text["split"] = all_text["report_id"].map(rid_to_split)
        all_text = all_text[all_text["split"].notna()].copy()

        rid_splits = {rid: rid_to_split[rid] for rid in all_report_ids if rid in rid_to_split}
        splits = canonical_split_order(set(rid_splits.values()))
        split_totals = {}
        for s in splits:
            split_totals[s] = sum(1 for v in rid_splits.values() if v == s)
    else:
        all_text["split"] = "all"
        splits = ["all"]
        split_totals = {"all": len(all_report_ids)}

    all_text["concept_name"] = all_text["field"].map(concept_fields)

    rows: List[Dict[str, Any]] = []
    for split in splits:
        total = split_totals.get(split, 0)
        split_df = all_text[all_text["split"] == split]

        for concept in concept_names:
            n_with = int(split_df[split_df["concept_name"] == concept]["report_id"].nunique())
            rows.append({
                "concept_name": concept,
                "split": split,
                "n_samples_with_text": n_with,
                "n_samples_without_text": max(0, total - n_with),
                "n_samples_total": total,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Label statistics from label shards
# ═══════════════════════════════════════════════════════════════════════════════

def compute_label_stats(
    label_dir: Path,
    concept_names: List[str],
    split_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Scan label shards and compute per-concept label distributions.

    Returns a DataFrame with columns:
      concept_name, split, n_valid, n_positive, n_negative, n_masked, n_imputed, n_total
    """
    meta_files = sorted(label_dir.glob("meta_*.parquet"))
    label_files = sorted(label_dir.glob("labels_*.npy"))
    mask_files = sorted(label_dir.glob("masks_*.npy"))

    if not meta_files or not label_files or not mask_files:
        print(f"[label] Incomplete shard files in {label_dir}")
        return pd.DataFrame()

    k = len(concept_names)

    shard_indices = sorted(set(
        int(p.stem.split("_")[-1]) for p in meta_files
    ) & set(
        int(p.stem.split("_")[-1]) for p in label_files
    ) & set(
        int(p.stem.split("_")[-1]) for p in mask_files
    ))

    if not shard_indices:
        print("[label] No complete shard triplets found.")
        return pd.DataFrame()

    per_split: Dict[str, Dict[str, np.ndarray]] = {}

    def ensure_split(s: str) -> None:
        if s not in per_split:
            per_split[s] = {
                "valid": np.zeros(k, dtype=np.int64),
                "pos": np.zeros(k, dtype=np.int64),
                "neg": np.zeros(k, dtype=np.int64),
                "imputed": np.zeros(k, dtype=np.int64),
                "total": np.zeros(k, dtype=np.int64),
            }

    for shard_idx in shard_indices:
        meta_path = label_dir / f"meta_{shard_idx:05d}.parquet"
        lab_path = label_dir / f"labels_{shard_idx:05d}.npy"
        msk_path = label_dir / f"masks_{shard_idx:05d}.npy"

        try:
            meta_df = pd.read_parquet(meta_path)
            labels = np.load(lab_path)
            masks = np.load(msk_path)
        except Exception as e:
            print(f"[label] Warning: could not read shard {shard_idx}: {e}")
            continue

        if labels.shape[1] != k:
            print(
                f"[label] Warning: shard {shard_idx} has K={labels.shape[1]} "
                f"but expected K={k}. Skipping."
            )
            continue

        meta_df["report_id"] = meta_df["report_id"].astype(str).str.strip()

        has_imputed_col = "imputed_targets" in meta_df.columns

        if split_map:
            meta_df["split"] = meta_df["report_id"].map(split_map)
            meta_df["split"] = meta_df["split"].fillna("_unassigned_")
        else:
            meta_df["split"] = "all"

        for split_name, group in meta_df.groupby("split", sort=False):
            if split_name == "_unassigned_":
                continue
            s = str(split_name)
            ensure_split(s)

            idx = group.index.to_numpy()
            row_positions = np.arange(len(meta_df))[np.isin(np.arange(len(meta_df)), idx)]

            lab_slice = labels[row_positions].astype(np.float32)
            msk_slice = masks[row_positions].astype(bool)

            n = len(row_positions)
            per_split[s]["total"] += n
            per_split[s]["valid"] += msk_slice.sum(axis=0).astype(np.int64)
            per_split[s]["pos"] += (msk_slice & np.isclose(lab_slice, 1.0)).sum(axis=0).astype(np.int64)
            per_split[s]["neg"] += (msk_slice & np.isclose(lab_slice, 0.0)).sum(axis=0).astype(np.int64)

            if has_imputed_col:
                for _, row in group.iterrows():
                    imp_str = str(row.get("imputed_targets", "") or "")
                    if not imp_str:
                        continue
                    for target in imp_str.split("|"):
                        target = target.strip()
                        if target in concept_names:
                            ci = concept_names.index(target)
                            per_split[s]["imputed"][ci] += 1

    if not per_split:
        print("[label] No label data found.")
        return pd.DataFrame()

    splits = canonical_split_order(per_split.keys())

    rows: List[Dict[str, Any]] = []
    for split in splits:
        d = per_split[split]
        total_reports = int(d["total"][0]) if k > 0 else 0
        for i, concept in enumerate(concept_names):
            n_valid = int(d["valid"][i])
            n_pos = int(d["pos"][i])
            n_neg = int(d["neg"][i])
            n_masked = total_reports - n_valid
            n_imputed = int(d["imputed"][i])
            rows.append({
                "concept_name": concept,
                "split": split,
                "n_valid_label": n_valid,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_masked": n_masked,
                "n_imputed_labels": n_imputed,
                "n_total": total_reports,
                "pos_rate_pct": (100.0 * n_pos / n_valid) if n_valid > 0 else 0.0,
                "masked_rate_pct": (100.0 * n_masked / total_reports) if total_reports > 0 else 0.0,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract per-concept statistics from embedding and label shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--config", type=str, default=None,
                    help="YAML config file (values serve as defaults; CLI overrides)")

    ap.add_argument("--text-emb-dir", type=str, default=None,
                    help="Directory with text embedding shards (meta_*.parquet)")
    ap.add_argument("--label-dir", type=str, default=None,
                    help="Directory with binary label shards (labels/masks/meta_*.npy/.parquet)")

    ap.add_argument("--concepts-file", type=str, default=None,
                    help="Text file with one concept name per line")
    ap.add_argument("--concepts", type=str, default=None,
                    help="Comma-separated concept names")

    ap.add_argument("--split-file", type=str, default=None,
                    help="Optional CSV/parquet with report_id and split columns")
    ap.add_argument("--report-id-column", type=str, default="report_id",
                    help="Report ID column name in the split file")
    ap.add_argument("--split-column", type=str, default="split",
                    help="Split column name in the split file")

    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory for statistics CSVs")

    # Preliminary parse for --config
    preliminary, _ = ap.parse_known_args()
    if preliminary.config:
        cfg = load_yaml_config(preliminary.config)
        defaults = {}
        for key, value in cfg.items():
            arg_key = key.replace("-", "_")
            if hasattr(preliminary, arg_key):
                defaults[arg_key] = value
        ap.set_defaults(**defaults)

    args = ap.parse_args()

    if not args.text_emb_dir and not args.label_dir:
        ap.error("At least one of --text-emb-dir or --label-dir is required.")
    if not args.out_dir:
        ap.error("--out-dir is required.")

    concept_names = load_concept_names(
        Path(args.concepts_file) if args.concepts_file else None,
        args.concepts,
    )
    if not concept_names:
        ap.error("No concepts provided. Use --concepts-file and/or --concepts.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load optional split mapping
    split_map: Optional[Dict[str, str]] = None
    if args.split_file:
        split_path = Path(args.split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split_map = load_split_map(
            split_path,
            report_id_column=args.report_id_column,
            split_column=args.split_column,
        )
        print(f"[splits] Loaded {len(split_map)} report-to-split mappings from {split_path}")
        split_counts = {}
        for v in split_map.values():
            split_counts[v] = split_counts.get(v, 0) + 1
        for s in canonical_split_order(split_counts.keys()):
            print(f"[splits]   {s}: {split_counts[s]} reports")
    else:
        print("[splits] No split file provided. All reports will be grouped as 'all'.")

    text_stats_df = None
    label_stats_df = None

    # ── Text statistics ──
    if args.text_emb_dir:
        text_emb_dir = Path(args.text_emb_dir)
        if not text_emb_dir.is_dir():
            raise NotADirectoryError(f"Text embedding dir not found: {text_emb_dir}")

        print(f"\n[text] Scanning embedding shards in {text_emb_dir} ...")
        text_stats_df = compute_text_stats(text_emb_dir, concept_names, split_map)

        if not text_stats_df.empty:
            text_out = out_dir / "text_stats.csv"
            text_stats_df.to_csv(text_out, index=False)
            print(f"[text] Wrote {text_out} ({len(text_stats_df)} rows)")

            # Also write the abundance-balance-compatible concept_stats.csv
            concept_stats_df = text_stats_df[
                ["concept_name", "split", "n_samples_with_text", "n_samples_total"]
            ].copy()
            concept_stats_out = out_dir / "concept_stats.csv"
            concept_stats_df.to_csv(concept_stats_out, index=False)
            print(f"[text] Wrote {concept_stats_out} (for abundance balancing)")

    # ── Label statistics ──
    if args.label_dir:
        label_dir_path = Path(args.label_dir)
        if not label_dir_path.is_dir():
            raise NotADirectoryError(f"Label dir not found: {label_dir_path}")

        print(f"\n[label] Scanning label shards in {label_dir_path} ...")
        label_stats_df = compute_label_stats(label_dir_path, concept_names, split_map)

        if not label_stats_df.empty:
            label_out = out_dir / "label_stats.csv"
            label_stats_df.to_csv(label_out, index=False)
            print(f"[label] Wrote {label_out} ({len(label_stats_df)} rows)")

    # ── Summary ──
    summary: Dict[str, Any] = {
        "concepts": concept_names,
        "n_concepts": len(concept_names),
        "text_emb_dir": args.text_emb_dir,
        "label_dir": args.label_dir,
        "split_file": args.split_file,
        "has_splits": split_map is not None,
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "stats_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console output
    print(f"\n{'='*70}")
    print("Concept Statistics Summary")
    print(f"{'='*70}")

    if text_stats_df is not None and not text_stats_df.empty:
        print("\nText coverage per concept:")
        for split in text_stats_df["split"].unique():
            sdf = text_stats_df[text_stats_df["split"] == split]
            total = sdf["n_samples_total"].iloc[0] if len(sdf) > 0 else 0
            print(f"\n  Split: {split} (n={total})")
            for _, row in sdf.iterrows():
                n_with = row["n_samples_with_text"]
                pct = (100.0 * n_with / total) if total > 0 else 0.0
                print(f"    {row['concept_name']:40s}  {n_with:>6d} / {total:>6d}  ({pct:5.1f}%)")

    if label_stats_df is not None and not label_stats_df.empty:
        print("\nLabel distribution per concept:")
        for split in label_stats_df["split"].unique():
            sdf = label_stats_df[label_stats_df["split"] == split]
            total = sdf["n_total"].iloc[0] if len(sdf) > 0 else 0
            print(f"\n  Split: {split} (n={total})")
            for _, row in sdf.iterrows():
                print(
                    f"    {row['concept_name']:40s}  "
                    f"valid={int(row['n_valid_label']):>5d}  "
                    f"pos={int(row['n_positive']):>5d}  "
                    f"neg={int(row['n_negative']):>5d}  "
                    f"masked={int(row['n_masked']):>5d}  "
                    f"imputed={int(row['n_imputed_labels']):>5d}"
                )

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
