#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary Label Extraction from Structured Report JSON (with Imputation)
=====================================================================

Reads structured radiology report JSON files and extracts binary classification
labels for a configurable set of medical concepts.

Label mapping rules:
  - support_devices.* (string labels):
      "present"     -> 1 (valid)
      "not present" -> 0 (valid)
      other/missing -> 0 (masked out)

  - thoracic_organs.*, pathologies.* (numeric score labels 0-3):
      0 -> not mentioned / unknown -> masked out (not valid)
      1 -> normal / excluded       -> 0 (valid)
      2 -> uncertain / suspicious  -> 1 (valid, pathologic)
      3 -> clearly pathological    -> 1 (valid, pathologic)

Pathology label imputation:
  When a pathologies field has label 0 (unknown/not mentioned), but a sufficient
  fraction of *other* pathologies fields in the same report carry non-zero labels,
  the missing field is imputed as label=0, mask=True (negative).  This mirrors the
  text imputation logic in the embedding script.

  Abundance-balanced mode (enabled via --concept-stats-csv) uses reservoir sampling
  to cap the number of imputed labels per concept, matching naturally occurring
  label counts to avoid over-imputation.

Output:
  Sharded .npy arrays for labels and masks, plus .parquet metadata.

Usage:
  python extract_binary_labels.py --config config.yaml
"""

import argparse
import json
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Sequence, Tuple

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np
import pandas as pd
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def clean_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()


def normalize_rel_field(path: str) -> str:
    p = clean_text(path)
    if p.startswith("structured."):
        p = p[len("structured."):]
    return p


def parse_csv_list(raw: str, *, normalize_paths: bool = False) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for part in raw.split(","):
        p = clean_text(part)
        if normalize_paths:
            p = normalize_rel_field(p)
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError(
            "PyYAML is required to use --config. Install it with: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg else {}


def append_log(log_path: Path, report_id: str, reason: str, detail: str = "") -> None:
    exists = log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["report_id", "reason", "detail"])
        w.writerow([report_id, reason, detail])


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def json_path_for_report_id(
    report_id: str,
    json_dir: Path,
    filename_template: str,
    fallback_glob: Optional[str] = None,
) -> Optional[Path]:
    p = json_dir / filename_template.format(report_id=report_id)
    if p.exists():
        return p
    if fallback_glob:
        matches = sorted(json_dir.glob(fallback_glob.format(report_id=report_id)))
        if matches:
            return matches[0]
    return None


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
    if not out:
        raise ValueError("No concept names provided. Use --concepts-file and/or --concepts.")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Abundance-balanced imputation quotas
# ═══════════════════════════════════════════════════════════════════════════════

def load_concept_valid_text_quotas(
    path: Path,
    *,
    count_column: str = "n_samples_with_text",
) -> Tuple[Dict[str, int], str]:
    """
    Load per-concept valid counts from a statistics CSV and aggregate
    across all data splits.  Returns (concept -> summed count, column used).
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Concept stats CSV is empty: {path}")

    used_count_col = count_column
    if used_count_col not in df.columns:
        for fallback in ("n_samples_with_text", "n_accessions_with_text"):
            if fallback in df.columns:
                used_count_col = fallback
                break
        else:
            raise KeyError(
                f"Count column '{count_column}' not found in {path}. "
                f"Columns: {df.columns.tolist()}"
            )

    if "concept_name" in df.columns:
        concept_series = df["concept_name"].astype(str).map(normalize_rel_field)
    elif "text_field" in df.columns:
        concept_series = df["text_field"].astype(str).map(normalize_rel_field)
    else:
        raise KeyError(f"Need concept_name or text_field column in {path}")

    counts = pd.to_numeric(df[used_count_col], errors="coerce").fillna(0.0)
    tmp = pd.DataFrame({"concept": concept_series, "valid_count": counts})
    tmp = tmp[tmp["concept"].astype(str) != ""]

    grouped = tmp.groupby("concept", as_index=False)["valid_count"].sum()
    quotas = {
        str(r["concept"]): max(0, int(round(float(r["valid_count"]))))
        for _, r in grouped.iterrows()
    }
    return quotas, used_count_col


# ═══════════════════════════════════════════════════════════════════════════════
# Label extraction logic
# ═══════════════════════════════════════════════════════════════════════════════

PATHOLOGY_GROUP_PREFIXES = (
    "pathologies.lung.",
    "pathologies.pleura.",
    "pathologies.vessels.",
    "pathologies.bones.",
)


@dataclass
class PathologyImputationConfig:
    enabled: bool
    targets: List[str]
    threshold_pct: float
    min_other_fields: int


def _get_node_value(report: Optional[dict], concept_name: str) -> Tuple[Any, bool]:
    """Traverse report["data"] by dot path; returns (value, found)."""
    if report is None:
        return None, False

    data = report.get("data", report)
    node: Any = data
    for k in concept_name.split("."):
        if not isinstance(node, dict) or k not in node:
            return None, False
        node = node[k]

    if isinstance(node, dict) and "label" in node:
        return node["label"], True

    return node, True


def _parse_numeric_label(val: Any) -> Optional[int]:
    if isinstance(val, dict) and "label" in val:
        val = val.get("label")
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if float(val).is_integer():
            return int(val)
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            fv = float(s)
            if fv.is_integer():
                return int(fv)
        except Exception:
            return None
    return None


def extract_binary_label(report: Optional[dict], concept_name: str) -> Tuple[float, bool]:
    """
    Map a structured JSON concept value to a binary label.

    Returns (label, valid) where valid=False means masked/unknown.
    """
    val, ok = _get_node_value(report, concept_name)
    if not ok:
        return 0.0, False

    if isinstance(val, dict) and "label" in val:
        val = val["label"]

    # String labels: "present" / "not present"
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "present":
            return 1.0, True
        if v == "not present":
            return 0.0, True
        if v.isdigit():
            return extract_binary_label({"data": {"_": int(v)}}, "_")
        return 0.0, False

    # Numeric labels: score 0-3
    if isinstance(val, (int, float, np.integer, np.floating)):
        if val == 0:
            return 0.0, False
        if val == 1:
            return 0.0, True
        if val in (2, 3):
            return 1.0, True
        return 0.0, False

    return 0.0, False


# ═══════════════════════════════════════════════════════════════════════════════
# Pathology imputation decision logic
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_pathology_label_map(report: Optional[dict]) -> Dict[str, Optional[int]]:
    """
    Walk the report and collect all pathology leaf labels as
    {"pathologies.lung.pneumonia": numeric_label_or_None, ...}.
    """
    out: Dict[str, Optional[int]] = {}
    if report is None:
        return out

    data = report.get("data", report)

    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            if prefix and prefix.startswith(PATHOLOGY_GROUP_PREFIXES) and "label" in node:
                out[prefix] = _parse_numeric_label(node.get("label"))
            for k, v in node.items():
                if k in ("label", "text"):
                    continue
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                walk(v, new_prefix)
            return

        if isinstance(node, list):
            for i, v in enumerate(node):
                new_prefix = f"{prefix}.[{i}]" if prefix else f"[{i}]"
                walk(v, new_prefix)
            return

        if prefix.startswith(PATHOLOGY_GROUP_PREFIXES):
            parsed = _parse_numeric_label(node)
            if parsed is not None:
                out[prefix] = parsed

    walk(data)
    return out


def _decide_pathology_imputations(
    report: Optional[dict],
    cfg: PathologyImputationConfig,
    *,
    sample_pos: Optional[int] = None,
    selected_positions_by_concept: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[Dict[str, bool], Dict[str, Dict[str, Any]]]:
    """
    For each imputation target, decide if it's eligible and (quota-)selected.

    Returns:
      eligible_by_target: {target -> bool}
      decision_details:   {target -> rich metadata dict}
    """
    eligible_by_target: Dict[str, bool] = {}
    decision_details: Dict[str, Dict[str, Any]] = {}

    if report is None or not cfg.enabled or not cfg.targets:
        return eligible_by_target, decision_details

    label_map = _collect_pathology_label_map(report)

    for target_raw in cfg.targets:
        target = normalize_rel_field(target_raw)
        target_val, target_exists = _get_node_value(report, target)
        target_num = _parse_numeric_label(target_val) if target_exists else None
        target_is_zero = target_exists and (target_num == 0)

        other_fields = [p for p in label_map if p != target]
        other_total_count = len(other_fields)
        other_valid_count = sum(
            1 for p in other_fields
            if label_map.get(p) is not None and int(label_map[p]) != 0
        )
        ratio_pct = 0.0 if other_total_count == 0 else (100.0 * other_valid_count / float(other_total_count))

        enough_other = other_total_count >= cfg.min_other_fields
        threshold_ok = enough_other and (ratio_pct >= cfg.threshold_pct)
        base_eligible = bool(target_is_zero and threshold_ok)

        selected_by_quota = True
        if selected_positions_by_concept is not None:
            selected_set = selected_positions_by_concept.get(target, set())
            selected_by_quota = (sample_pos is not None) and (sample_pos in selected_set)

        eligible = bool(base_eligible and selected_by_quota)

        eligible_by_target[target] = eligible
        decision_details[target] = {
            "target_field": target,
            "target_exists": bool(target_exists),
            "target_raw_label": target_val,
            "target_numeric_label": target_num,
            "target_is_zero": bool(target_is_zero),
            "other_total_count": other_total_count,
            "other_valid_count": other_valid_count,
            "other_valid_ratio_pct": float(ratio_pct),
            "threshold_pct": float(cfg.threshold_pct),
            "min_other_fields": cfg.min_other_fields,
            "threshold_ok": bool(threshold_ok),
            "base_eligible": bool(base_eligible),
            "selected_by_quota": bool(selected_by_quota),
            "eligible": bool(eligible),
        }

    return eligible_by_target, decision_details


def collect_eligible_pathology_targets(
    report: Optional[dict],
    cfg: PathologyImputationConfig,
) -> Set[str]:
    """Return targets that are rule-eligible (ignoring quota selection)."""
    _, decisions = _decide_pathology_imputations(report, cfg)
    return {c for c, d in decisions.items() if d.get("base_eligible", False)}


def select_positions_for_abundance_balanced_imputation(
    *,
    report_id_list: List[str],
    json_dir: Path,
    json_filename_template: str,
    json_fallback_glob: Optional[str],
    imputation_cfg: PathologyImputationConfig,
    quotas_by_concept: Dict[str, int],
    seed: Optional[int],
) -> Tuple[Dict[str, Set[int]], Dict[str, int], Dict[str, int]]:
    """
    Reservoir-sample report positions per concept so selections are uniformly
    random over all eligible candidates, capped by quota.
    """
    active_concepts = {c for c, q in quotas_by_concept.items() if int(q) > 0}
    reservoirs: Dict[str, List[int]] = {c: [] for c in active_concepts}
    eligible_counts: Dict[str, int] = {c: 0 for c in quotas_by_concept}

    rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed) + 100_003)

    for pos in tqdm(range(len(report_id_list)), desc="Pre-scan pathology imputation candidates"):
        rid = report_id_list[pos]
        jp = json_path_for_report_id(rid, json_dir, json_filename_template, json_fallback_glob)
        if jp is None:
            continue

        try:
            with open(jp, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception:
            continue

        eligible_for_report = collect_eligible_pathology_targets(report, imputation_cfg)
        if not eligible_for_report:
            continue

        for concept in eligible_for_report:
            if concept not in quotas_by_concept:
                continue
            eligible_counts[concept] = eligible_counts.get(concept, 0) + 1

            k = int(quotas_by_concept.get(concept, 0))
            if k <= 0:
                continue

            res = reservoirs.setdefault(concept, [])
            seen = eligible_counts[concept]

            if len(res) < k:
                res.append(pos)
            else:
                j = int(rng.integers(1, seen + 1))
                if j <= k:
                    res[j - 1] = pos

    selected_positions = {c: set(v) for c, v in reservoirs.items()}
    selected_counts = {c: len(selected_positions.get(c, set())) for c in quotas_by_concept}
    return selected_positions, eligible_counts, selected_counts


# ═══════════════════════════════════════════════════════════════════════════════
# Label array builder (with imputation)
# ═══════════════════════════════════════════════════════════════════════════════

def build_label_arrays(
    report: Optional[dict],
    concept_names: List[str],
    *,
    imputation_cfg: PathologyImputationConfig,
    sample_pos: Optional[int] = None,
    selected_positions_by_concept: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Dict[str, Any]]]:
    """
    Extract binary labels and validity masks for all concepts from a report,
    applying pathology imputation where eligible.

    Returns:
      labels:              (K,) float32
      mask:                (K,) bool
      imputed_targets:     list of concept paths where imputation was applied
      imputation_decisions: per-target decision metadata
    """
    K = len(concept_names)
    labels = np.zeros((K,), dtype=np.float32)
    mask = np.zeros((K,), dtype=bool)
    imputed_targets: List[str] = []

    if report is None:
        return labels, mask, imputed_targets, {}

    eligible_by_target, decision_details = _decide_pathology_imputations(
        report,
        imputation_cfg,
        sample_pos=sample_pos,
        selected_positions_by_concept=selected_positions_by_concept,
    )

    for i, cname in enumerate(concept_names):
        rel = normalize_rel_field(cname)
        y, valid = extract_binary_label(report, rel)

        if valid:
            labels[i] = float(y)
            mask[i] = True
            continue

        if eligible_by_target.get(rel, False):
            labels[i] = 0.0
            mask[i] = True
            imputed_targets.append(rel)

    return labels, mask, imputed_targets, decision_details


# ═══════════════════════════════════════════════════════════════════════════════
# Inspection utility
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_report(
    report_id: str,
    json_dir: Path,
    json_filename_template: str,
    json_fallback_glob: Optional[str],
    concept_names: List[str],
    imputation_cfg: PathologyImputationConfig,
) -> None:
    jp = json_path_for_report_id(report_id, json_dir, json_filename_template, json_fallback_glob)
    print(f"report_id={report_id}")
    print(f"json_path={jp}")
    if jp is None:
        print("No JSON found.")
        return

    try:
        with open(jp, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        return

    labels, masks, imputed_targets, decision_details = build_label_arrays(
        report, concept_names, imputation_cfg=imputation_cfg,
    )
    imputed_set = set(imputed_targets)

    print("\nPer-concept mapping:")
    for i, cname in enumerate(concept_names):
        rel = normalize_rel_field(cname)
        raw_val, found = _get_node_value(report, rel)
        y, valid = extract_binary_label(report, rel)
        was_imputed = rel in imputed_set
        print(
            f"  [{cname}] raw={raw_val!r} found={found} -> label={float(labels[i])} "
            f"mask={bool(masks[i])} base_valid={bool(valid)} imputed={was_imputed}"
        )

    print("\nPathology imputation decisions:")
    if not decision_details:
        print("  (none — disabled, no targets, or no JSON)")
    else:
        for t in sorted(decision_details):
            d = decision_details[t]
            print(
                f"  {t}: eligible={d['eligible']} target_label={d['target_numeric_label']} "
                f"other_valid={d['other_valid_count']}/{d['other_total_count']} "
                f"ratio={d['other_valid_ratio_pct']:.2f}% threshold={d['threshold_pct']}"
            )

    print(f"\nvalid_concepts={int(masks.sum())}/{len(concept_names)}")
    print(f"imputed_targets_count={len(imputed_targets)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sharded label writer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LabelShardWriter:
    out_dir: Path
    shard_size: int
    label_dtype: str = "float32"
    shard_idx: int = 0
    meta_rows: Optional[List[Dict[str, Any]]] = None
    label_rows: Optional[List[np.ndarray]] = None
    mask_rows: Optional[List[np.ndarray]] = None

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_rows = []
        self.label_rows = []
        self.mask_rows = []

    def add(self, meta: Dict[str, Any], labels: np.ndarray, mask: np.ndarray) -> None:
        self.meta_rows.append(meta)
        self.label_rows.append(labels)
        self.mask_rows.append(mask)
        if len(self.meta_rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.meta_rows:
            return

        meta_df = pd.DataFrame(self.meta_rows)
        labels_arr = np.stack(self.label_rows, axis=0)
        masks_arr = np.stack(self.mask_rows, axis=0)

        if self.label_dtype == "float16":
            labels_arr = labels_arr.astype(np.float16, copy=False)
        else:
            labels_arr = labels_arr.astype(np.float32, copy=False)

        np.save(self.out_dir / f"labels_{self.shard_idx:05d}.npy", labels_arr)
        np.save(self.out_dir / f"masks_{self.shard_idx:05d}.npy", masks_arr)
        meta_df.to_parquet(self.out_dir / f"meta_{self.shard_idx:05d}.parquet", index=False)

        self.shard_idx += 1
        self.meta_rows.clear()
        self.label_rows.clear()
        self.mask_rows.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract binary labels from structured report JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config ──
    ap.add_argument("--config", type=str, default=None,
                    help="Path to a YAML config file. Values serve as defaults; CLI args override.")

    # ── Inputs ──
    ap.add_argument("--reports-csv", type=str, default=None,
                    help="CSV file containing report IDs")
    ap.add_argument("--report-id-column", type=str, default="report_id")
    ap.add_argument("--json-dir", type=str, default=None,
                    help="Directory containing structured JSON report files")

    # ── JSON naming ──
    ap.add_argument("--json-filename-template", type=str, default="{report_id}_structured.json")
    ap.add_argument("--json-fallback-glob", type=str, default=None)

    # ── Concepts ──
    ap.add_argument("--concepts-file", type=str, default=None)
    ap.add_argument("--concepts", type=str, default=None,
                    help="Comma-separated concept names")

    # ── Output ──
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--shard-size", type=int, default=200_000)
    ap.add_argument("--label-dtype", type=str, default="float32",
                    choices=["float32", "float16"])
    ap.add_argument("--log-path", type=str, default=None)

    # ── Pathology label imputation ──
    ap.add_argument("--disable-pathology-imputation", action="store_true",
                    help="Disable pathology label imputation entirely.")
    ap.add_argument("--impute-pathology-targets", type=str, default="",
                    help="Comma-separated pathology target fields. "
                         "If empty, all pathologies.* concepts from the concept list are used.")
    ap.add_argument("--pathology-valid-threshold-pct", type=float, default=70.0,
                    help="Imputation requires at least this %% of other pathology fields to be non-zero.")
    ap.add_argument("--pathology-min-other-fields", type=int, default=1,
                    help="Minimum number of other pathology fields required before threshold applies.")
    ap.add_argument("--sampling-seed", type=int, default=42,
                    help="Seed for deterministic abundance-balanced selection. Negative = random.")

    # ── Abundance balance ──
    ap.add_argument("--concept-stats-csv", type=str, default=None,
                    help="CSV with per-split concept statistics for abundance-balanced quotas.")
    ap.add_argument("--abundance-count-column", type=str, default="n_samples_with_text",
                    help="Column in --concept-stats-csv used as valid count.")
    ap.add_argument("--disable-abundance-balance", action="store_true",
                    help="Disable quota balancing; impute all eligible entries.")

    # ── Limiting / inspection / testing ──
    ap.add_argument("--process-max", type=int, default=None,
                    help="Process only first N report IDs")
    ap.add_argument("--inspect-report-id", type=str, default=None,
                    help="Print label details for one report ID and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute labels/imputations without writing shards")
    ap.add_argument("--verbose-test", action="store_true",
                    help="Emit detailed per-report imputation decisions")
    ap.add_argument("--verbose-test-limit", type=int, default=10,
                    help="Max reports printed in verbose-test mode")
    ap.add_argument("--verbose-test-out", type=str, default=None,
                    help="JSONL path for verbose decision records")
    ap.add_argument("--verbose-test-only-imputed", action="store_true",
                    help="Only emit records with >=1 imputation")

    # ── Preliminary parse to detect --config ──
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

    # Validate required arguments
    if not args.reports_csv:
        ap.error("--reports-csv is required (or set reports_csv in config)")
    if not args.json_dir:
        ap.error("--json-dir is required (or set json_dir in config)")
    if not args.out_dir:
        ap.error("--out-dir is required (or set out_dir in config)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else (out_dir / "processing_log.csv")

    concept_names = load_concept_names(
        Path(args.concepts_file) if args.concepts_file else None,
        args.concepts,
    )
    concept_rel = [normalize_rel_field(c) for c in concept_names]

    json_dir = Path(args.json_dir)

    # ── Build pathology imputation config ──
    pathology_targets = parse_csv_list(
        args.impute_pathology_targets or "", normalize_paths=True
    )
    if not pathology_targets and not args.disable_pathology_imputation:
        pathology_targets = sorted({c for c in concept_rel if c.startswith("pathologies.")})

    if not args.disable_pathology_imputation and not pathology_targets:
        raise ValueError(
            "Pathology imputation is enabled but no targets were found. "
            "Set --impute-pathology-targets or disable with --disable-pathology-imputation."
        )

    imputation_cfg = PathologyImputationConfig(
        enabled=not bool(args.disable_pathology_imputation),
        targets=pathology_targets,
        threshold_pct=float(args.pathology_valid_threshold_pct),
        min_other_fields=max(0, int(args.pathology_min_other_fields)),
    )

    sampling_seed: Optional[int] = None
    if int(args.sampling_seed) >= 0:
        sampling_seed = int(args.sampling_seed)

    # ── Abundance-balanced quotas ──
    abundance_balance_enabled = (
        not bool(args.disable_abundance_balance) and bool(args.concept_stats_csv)
    )
    concept_valid_counts: Dict[str, int] = {}
    concept_quotas: Dict[str, int] = {}
    used_abundance_count_col = str(args.abundance_count_column)
    target_concepts_for_quota = sorted(
        {normalize_rel_field(c) for c in imputation_cfg.targets}
    )

    if abundance_balance_enabled:
        concept_valid_counts, used_abundance_count_col = load_concept_valid_text_quotas(
            Path(args.concept_stats_csv),
            count_column=str(args.abundance_count_column),
        )
        concept_quotas = {
            c: int(concept_valid_counts.get(c, 0)) for c in target_concepts_for_quota
        }
        print(
            f"[abundance] Loaded concept stats: {args.concept_stats_csv} "
            f"(count_column={used_abundance_count_col}, aggregated over all splits)"
        )
        for c in target_concepts_for_quota:
            q = concept_quotas.get(c, 0)
            print(f"[abundance] concept={c} valid_count={q} -> max_imputations={q}")
    elif not args.disable_abundance_balance:
        print(
            "[abundance] --concept-stats-csv not provided. "
            "Falling back to old behavior (impute all eligible entries)."
        )

    # ── Verbose test output setup ──
    verbose_test_path: Optional[Path] = None
    if args.verbose_test:
        if args.verbose_test_out:
            verbose_test_path = Path(args.verbose_test_out)
        else:
            verbose_test_path = out_dir / "verbose_test_debug.jsonl"
        if verbose_test_path.exists():
            verbose_test_path.unlink()

    # ── Inspect single report and exit ──
    if args.inspect_report_id:
        inspect_report(
            report_id=str(args.inspect_report_id).strip(),
            json_dir=json_dir,
            json_filename_template=args.json_filename_template,
            json_fallback_glob=args.json_fallback_glob,
            concept_names=concept_names,
            imputation_cfg=imputation_cfg,
        )
        return

    # ── Build report ID list ──
    df_reports = pd.read_csv(args.reports_csv, dtype=str)
    if args.report_id_column not in df_reports.columns:
        raise KeyError(
            f"Column '{args.report_id_column}' not found in {args.reports_csv}. "
            f"Available columns: {df_reports.columns.tolist()}"
        )
    report_id_list = (
        df_reports[args.report_id_column].dropna().astype(str).str.strip().tolist()
    )

    if args.process_max:
        report_id_list = report_id_list[: int(args.process_max)]

    # ── Abundance-balanced pre-scan ──
    selected_positions_by_concept: Optional[Dict[str, Set[int]]] = None
    eligible_candidates_by_concept: Dict[str, int] = {}
    selected_candidates_by_concept: Dict[str, int] = {}
    effective_concept_quotas: Dict[str, int] = dict(concept_quotas)

    if abundance_balance_enabled and target_concepts_for_quota:
        (
            selected_positions_by_concept,
            eligible_candidates_by_concept,
            selected_candidates_by_concept,
        ) = select_positions_for_abundance_balanced_imputation(
            report_id_list=report_id_list,
            json_dir=json_dir,
            json_filename_template=str(args.json_filename_template),
            json_fallback_glob=args.json_fallback_glob,
            imputation_cfg=imputation_cfg,
            quotas_by_concept=effective_concept_quotas,
            seed=sampling_seed,
        )

        print("[abundance] Candidate prescan completed.")
        for c in target_concepts_for_quota:
            quota_total = concept_quotas.get(c, 0)
            eligible_n = eligible_candidates_by_concept.get(c, 0)
            selected_n = selected_candidates_by_concept.get(c, 0)
            print(
                f"[abundance] concept={c} quota_total={quota_total} "
                f"eligible_candidates={eligible_n} selected_for_imputation={selected_n}"
            )
    elif abundance_balance_enabled:
        print("[abundance] Enabled, but no active pathology targets. Skipping prescan.")

    # ── Main processing loop ──
    writer: Optional[LabelShardWriter] = None
    if not args.dry_run:
        writer = LabelShardWriter(
            out_dir=out_dir,
            shard_size=int(args.shard_size),
            label_dtype=str(args.label_dtype),
        )

    n_reports = 0
    n_has_json = 0
    n_no_json = 0
    n_error = 0
    n_imputed_total = 0
    n_reports_with_imputation = 0
    imputed_by_target: Dict[str, int] = {t: 0 for t in sorted(imputation_cfg.targets)}

    verbose_printed = 0

    for pos in tqdm(range(len(report_id_list)), desc="Reports"):
        report_id = report_id_list[pos]
        n_reports += 1

        jp = json_path_for_report_id(
            report_id, json_dir, args.json_filename_template, args.json_fallback_glob
        )

        report = None
        has_json = False
        error = ""
        if jp is None:
            n_no_json += 1
        else:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    report = json.load(f)
                has_json = True
                n_has_json += 1
            except Exception as e:
                error = f"json_read_error: {type(e).__name__}"
                n_error += 1
                append_log(log_path, report_id, "json_read_error", str(e))

        labels, masks, imputed_targets, imputation_decisions = build_label_arrays(
            report,
            concept_names,
            imputation_cfg=imputation_cfg,
            sample_pos=pos,
            selected_positions_by_concept=selected_positions_by_concept,
        )

        n_imputed = len(imputed_targets)
        n_imputed_total += n_imputed
        if n_imputed > 0:
            n_reports_with_imputation += 1
            for t in imputed_targets:
                imputed_by_target[t] = imputed_by_target.get(t, 0) + 1

        if args.verbose_test:
            debug = {
                "report_id": report_id,
                "json_path": str(jp) if jp else "",
                "has_json": bool(has_json),
                "error": error,
                "n_valid": int(masks.sum()),
                "imputed_targets": sorted(imputed_targets),
                "imputed_count": n_imputed,
                "pathology_decisions": [
                    imputation_decisions[k] for k in sorted(imputation_decisions)
                ],
            }
            should_emit = True
            if args.verbose_test_only_imputed and n_imputed == 0:
                should_emit = False

            if should_emit:
                if verbose_test_path is not None:
                    append_jsonl(verbose_test_path, debug)
                if verbose_printed < int(args.verbose_test_limit):
                    print(json.dumps(debug, ensure_ascii=False, indent=2))
                    verbose_printed += 1

        meta = {
            "report_id": report_id,
            "json_path": str(jp) if jp else "",
            "has_json": bool(has_json),
            "error": error,
            "concepts_sha1": sha1("|".join(concept_names)),
            "K": len(concept_names),
            "n_valid": int(masks.sum()),
            "n_imputed": n_imputed,
            "imputed_targets": "|".join(sorted(imputed_targets)),
        }

        if writer is not None:
            writer.add(meta, labels, masks)

    if writer is not None:
        writer.flush()

    # ── Summary ──
    summary = {
        "dry_run": bool(args.dry_run),
        "reports_seen": n_reports,
        "reports_with_json": n_has_json,
        "reports_no_json": n_no_json,
        "reports_with_error": n_error,
        "reports_with_imputation": n_reports_with_imputation,
        "imputed_labels_total": n_imputed_total,
        "imputed_by_target": imputed_by_target,
        "pathology_imputation_enabled": imputation_cfg.enabled,
        "pathology_targets": sorted(imputation_cfg.targets),
        "pathology_threshold_pct": imputation_cfg.threshold_pct,
        "pathology_min_other_fields": imputation_cfg.min_other_fields,
        "sampling_seed": sampling_seed,
        "abundance_balance_enabled": abundance_balance_enabled,
        "concept_stats_csv": str(args.concept_stats_csv) if args.concept_stats_csv else "",
        "abundance_count_column": used_abundance_count_col,
        "abundance_target_concepts": target_concepts_for_quota,
        "concept_quotas_total": {k: int(v) for k, v in concept_quotas.items()},
        "concept_quotas_effective": {k: int(v) for k, v in effective_concept_quotas.items()},
        "eligible_candidates_by_concept": {k: int(v) for k, v in eligible_candidates_by_concept.items()},
        "selected_candidates_by_concept": {k: int(v) for k, v in selected_candidates_by_concept.items()},
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "shards_written": writer.shard_idx if writer is not None else 0,
        "K": len(concept_names),
        "label_dtype": str(args.label_dtype),
        "verbose_test_out": str(verbose_test_path) if verbose_test_path else "",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
