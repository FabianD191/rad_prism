#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Report Text Embedding with Optional Imputation
==========================================================

Generates text embeddings from structured radiology report JSON files using
either a SentenceTransformer model (e.g. Qwen3-Embedding) or a HuggingFace
AutoModel with manual pooling (e.g. medBERT.de).

Key features:
  - Unified script supporting two model backends via --model-backend:
      "sentence_transformer" (e.g. Qwen3-Embedding-4B)
      "auto_model"           (e.g. medBERT.de with mean/CLS pooling)
  - Extracts text from structured JSON leaves and optionally from
    unstructured report text.
  - Rule-based text imputation for missing negative entries: samples
    synthetic negative sentences from a configurable template pool for
    specified concept fields.
  - Abundance-balanced imputation: limits imputed samples per concept
    to match natural text occurrence counts (from a concept stats CSV).
  - Sharded output: embeddings (.npy) + metadata (.parquet) per shard.
  - Resume support: can continue from the last completed shard.
  - Overflow handling: truncation or sliding-window mean pooling for
    texts exceeding the model's token limit.
  - Optional YAML config file for default parameters.

Usage:
  python embed_structured_reports.py \\
    --config config.yaml \\
    --model-backend sentence_transformer \\
    --model-dir /path/to/Qwen3-Embedding-4B \\
    --reports-csv reports.csv \\
    --json-dir /path/to/structured_jsons \\
    --neg-sampling-path templates/neg_entry_sampling.json \\
    --out-dir /path/to/output
"""

import os

# Migrate deprecated TRANSFORMERS_CACHE to HF_HOME before importing transformers
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ.pop("TRANSFORMERS_CACHE")
elif "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]

import re
import json
import csv
import hashlib
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ===============================================================================
# YAML config loading
# ===============================================================================

def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a flat dict.

    Keys use the same names as CLI arguments (with hyphens replaced by
    underscores).  Nested YAML structures are not supported; the file
    should be a flat key-value mapping.

    Raises ImportError if PyYAML is not installed.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required when using --config. "
            "Install it with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data).__name__}")
    # Normalize keys: YAML may use hyphens or underscores
    return {k.replace("-", "_"): v for k, v in data.items()}


def apply_yaml_defaults(parser: argparse.ArgumentParser, yaml_cfg: Dict[str, Any]) -> None:
    """
    Set parser defaults from a YAML config dict.

    CLI arguments still override these defaults because argparse applies
    defaults only when the user does not supply the argument explicitly.
    """
    parser.set_defaults(**yaml_cfg)


# ===============================================================================
# Text utilities
# ===============================================================================

def clean_text(s: Any) -> str:
    """Normalize whitespace and strip a string; return '' for non-strings."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_rel_field(path: str) -> str:
    """Strip the 'structured.' prefix from a field path if present."""
    p = clean_text(path)
    if p.startswith("structured."):
        p = p[len("structured."):]
    return p


def sha1(s: str) -> str:
    """Compute SHA-1 hex digest of a string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def parse_csv_list(raw: str) -> List[str]:
    """Parse a comma-separated string into unique normalized field names."""
    out: List[str] = []
    seen = set()
    for part in raw.split(","):
        p = normalize_rel_field(part)
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def stringify_label(label: Any) -> str:
    """Convert a label value to a clean string representation."""
    if label is None:
        return ""
    if isinstance(label, str):
        return clean_text(label)
    return str(label)


def label_is_not_present(label: Any) -> bool:
    """Check if a label value represents 'not present'."""
    if not isinstance(label, str):
        return False
    return clean_text(label).lower() == "not present"


def label_is_binary_01(label: Any) -> bool:
    """Check if a label value is binary (0 or 1) in any supported type."""
    if isinstance(label, bool):
        return int(label) in (0, 1)
    if isinstance(label, int):
        return label in (0, 1)
    if isinstance(label, float):
        return label in (0.0, 1.0)
    if isinstance(label, str):
        return clean_text(label) in ("0", "1")
    return False


# ===============================================================================
# Data I/O utilities
# ===============================================================================

def read_reports(csv_path: Path, report_id_column: str = "report_id") -> pd.DataFrame:
    """
    Read the reports CSV and return a DataFrame indexed by the report ID column.

    Expected columns: report_id, examination, report_text
    (The report_id column name is configurable via report_id_column.)
    """
    df = pd.read_csv(csv_path, dtype={report_id_column: str})
    if report_id_column not in df.columns:
        raise KeyError(
            f"Report ID column '{report_id_column}' not found in CSV. "
            f"Columns: {df.columns.tolist()}"
        )
    df[report_id_column] = df[report_id_column].astype(str).str.strip()
    need_cols = [report_id_column] + [
        c for c in ["examination", "report_text"] if c in df.columns
    ]
    df = df[need_cols].drop_duplicates(report_id_column)
    return df.set_index(report_id_column, drop=True)


def append_log(log_path: Path, report_id: str, reason: str, detail: str = "") -> None:
    """Append a log entry (CSV format) for processing events."""
    exists = log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["report_id", "reason", "detail"])
        w.writerow([report_id, reason, detail])


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a JSON object as a single line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ===============================================================================
# Resume / shard discovery
# ===============================================================================

def discover_completed_shards(out_dir: Path) -> List[int]:
    """Find shard indices that have both a meta parquet and an embeddings npy file."""
    meta = {int(p.stem.split("_")[1]) for p in out_dir.glob("meta_*.parquet")}
    emb = {int(p.stem.split("_")[1]) for p in out_dir.glob("embeddings_*.npy")}
    common = sorted(meta & emb)

    ok = []
    for idx in common:
        mp = out_dir / f"meta_{idx:05d}.parquet"
        ep = out_dir / f"embeddings_{idx:05d}.npy"
        try:
            _ = pd.read_parquet(mp, columns=["report_id"])
            _ = np.load(ep, mmap_mode="r")
            ok.append(idx)
        except Exception:
            pass
    return ok


def last_report_id_in_shard(meta_path: Path) -> str:
    """Return the last report ID stored in a shard's metadata parquet."""
    df = pd.read_parquet(meta_path, columns=["report_id"])
    return str(df["report_id"].iloc[-1]) if len(df) > 0 else ""


def find_last_index(id_list: List[str], report_id: str) -> int:
    """Find the last index of a report ID in the list (reverse search)."""
    for i in range(len(id_list) - 1, -1, -1):
        if id_list[i] == report_id:
            return i
    return -1


def load_processed_report_ids(out_dir: Path) -> set:
    """Load the set of all report IDs already written to any shard."""
    processed = set()
    for mp in sorted(out_dir.glob("meta_*.parquet")):
        try:
            df = pd.read_parquet(mp, columns=["report_id"])
            processed.update(df["report_id"].astype(str).unique().tolist())
        except Exception:
            pass
    return processed


# ===============================================================================
# JSON extraction helpers
# ===============================================================================

def should_exclude(field_path: str, exclude_fields: set, exclude_prefixes: set) -> bool:
    """Check if a field path should be excluded based on exact match or prefix."""
    if field_path in exclude_fields:
        return True
    return any(field_path.startswith(p) for p in exclude_prefixes)


def extract_text_leaves(
    node: Any,
    prefix: str = "",
    *,
    text_key: str = "text",
    label_key: str = "label",
) -> List[Tuple[str, str]]:
    """
    Recursively extract (dot.path, text) pairs from a JSON structure.

    Handles:
      - Dict leaves like {"text": "...", "label": ...} -> extracts non-empty text
      - Plain strings/scalars
      - Nested dicts and lists
    """
    out: List[Tuple[str, str]] = []

    if isinstance(node, dict):
        if text_key in node:
            txt = clean_text(node.get(text_key))
            if txt:
                out.append((prefix, txt))
        for k, v in node.items():
            if k in (label_key, text_key):
                continue
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.extend(extract_text_leaves(v, new_prefix, text_key=text_key, label_key=label_key))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            new_prefix = f"{prefix}.[{i}]" if prefix else f"[{i}]"
            out.extend(extract_text_leaves(v, new_prefix, text_key=text_key, label_key=label_key))
    else:
        txt = clean_text("" if node is None else str(node))
        if txt:
            out.append((prefix, txt))

    return [(p if p else "_root", t) for p, t in out]


def extract_text_label_leaves(
    node: Any,
    prefix: str = "",
    *,
    text_key: str = "text",
    label_key: str = "label",
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Like extract_text_leaves, but returns both text AND label for each leaf,
    including leaves with empty text (needed for imputation eligibility checks).
    """
    out: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(node, dict):
        if text_key in node:
            out.append((
                prefix if prefix else "_root",
                {"text": clean_text(node.get(text_key)), "label": node.get(label_key)},
            ))
        for k, v in node.items():
            if k in (label_key, text_key):
                continue
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.extend(extract_text_label_leaves(v, new_prefix, text_key=text_key, label_key=label_key))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            new_prefix = f"{prefix}.[{i}]" if prefix else f"[{i}]"
            out.extend(extract_text_label_leaves(v, new_prefix, text_key=text_key, label_key=label_key))

    return out


def build_grouped_texts(
    leaf_pairs: List[Tuple[str, str]],
    group_paths: Sequence[str],
    delimiter: str,
) -> Dict[str, str]:
    """
    Concatenate text from multiple leaf paths that share a common group prefix.

    E.g., group_path="support_devices" collects all support_devices.* leaf texts
    joined by the delimiter.
    """
    grouped: Dict[str, str] = {}
    for gp in group_paths:
        texts = [txt.strip() for path, txt in leaf_pairs
                 if (path == gp or path.startswith(gp + ".")) and txt.strip()]
        if texts:
            grouped[gp] = clean_text(delimiter.join(texts))
    return grouped


def json_path_for_report_id(
    report_id: str,
    json_dir: Path,
    filename_template: str,
    fallback_glob: Optional[str] = None,
) -> Optional[Path]:
    """Locate the structured JSON file for a given report ID."""
    p = json_dir / filename_template.format(report_id=report_id)
    if p.exists():
        return p
    if fallback_glob:
        matches = sorted(json_dir.glob(fallback_glob.format(report_id=report_id)))
        if matches:
            return matches[0]
    return None


# ===============================================================================
# Negative-sentence sampling templates (for imputation)
# ===============================================================================

def load_neg_sampling_templates(path: Path) -> Dict[str, List[str]]:
    """
    Load a JSON file mapping concept field names to lists of negative sentence
    templates (e.g., "No tube present." for support_devices.airway).

    These are sampled to fill in text for fields where the label indicates
    absence but no text was originally present.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Negative sampling file must contain a JSON object: {path}")

    templates: Dict[str, List[str]] = {}
    for k, v in data.items():
        key = normalize_rel_field(str(k))
        if not key:
            continue
        if isinstance(v, list):
            choices = [clean_text(x) for x in v if clean_text(x)]
            if choices:
                templates[key] = choices

    if not templates:
        raise ValueError(f"No valid sampling templates found in: {path}")
    return templates


def get_sampling_pool(templates: Dict[str, List[str]], rel_field: str) -> List[str]:
    """Get the list of candidate negative sentences for a specific field."""
    return templates.get(normalize_rel_field(rel_field), [])


def deterministic_sample(
    pool: List[str], seed: Optional[int], report_id: str, rel_field: str,
) -> Tuple[str, int]:
    """
    Pick one sentence from the pool, deterministically if seed is provided.

    Uses SHA-1 of (seed|report_id|field) to select an index, ensuring the
    same sample is always chosen for the same combination.
    """
    if not pool:
        raise ValueError("pool must be non-empty")

    if seed is None:
        idx = np.random.randint(0, len(pool))
        return pool[idx], int(idx)

    token = f"{seed}|{report_id}|{rel_field}"
    h = sha1(token)
    idx = int(h[:8], 16) % len(pool)
    return pool[idx], int(idx)


# ===============================================================================
# Abundance-balanced imputation quotas
# ===============================================================================

def load_concept_valid_text_quotas(
    path: Path,
    *,
    count_column: str = "n_samples_with_text",
) -> Tuple[Dict[str, int], str]:
    """
    Load per-concept valid text counts from a statistics CSV and aggregate
    across all data splits.

    Returns:
      - dict mapping concept name -> total valid text count (used as max quota)
      - the actual count column name used
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
    quotas = {str(r["concept"]): max(0, int(round(float(r["valid_count"])))) for _, r in grouped.iterrows()}
    return quotas, used_count_col


def load_existing_imputed_counts_by_concept(out_dir: Path) -> Dict[str, int]:
    """Count already-written imputed rows per concept from existing shards (for resume)."""
    counts: Dict[str, int] = defaultdict(int)
    for mp in sorted(out_dir.glob("meta_*.parquet")):
        try:
            df = pd.read_parquet(mp, columns=["is_imputed", "imputation_sample_key"])
        except Exception:
            continue
        if "is_imputed" not in df.columns or "imputation_sample_key" not in df.columns:
            continue
        m = df["is_imputed"] == True  # noqa: E712
        if not m.any():
            continue
        vc = df.loc[m, "imputation_sample_key"].dropna().astype(str).map(normalize_rel_field).value_counts()
        for concept, n in vc.items():
            counts[str(concept)] += int(n)
    return dict(counts)


def extract_text_label_map_from_json_path(json_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a structured JSON file and extract the text+label map for all leaf fields."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return {}
    data = j.get("data", j)
    if not isinstance(data, (dict, list)):
        return {}
    return {p: info for p, info in extract_text_label_leaves(data, prefix="")}


# Pathology group prefixes used for context-ratio checks during imputation
PATHOLOGY_GROUP_PREFIXES = (
    "pathologies.lung.",
    "pathologies.pleura.",
    "pathologies.vessels.",
    "pathologies.bones.",
)


@dataclass
class ImputationConfig:
    """Configuration for rule-based text imputation."""
    enabled_support_devices: bool       # Impute missing support_devices text
    enabled_pathologies: bool           # Impute missing pathologies text
    templates: Dict[str, List[str]]     # Negative sentence pools per concept
    pathologies_targets: List[str]      # Which pathologies fields to impute
    pathologies_threshold_pct: float    # Min % of other pathology fields with text
    pathologies_min_other_fields: int   # Min number of other pathology fields present
    sampling_seed: Optional[int]        # Seed for deterministic sampling


def collect_eligible_imputation_concepts(
    text_label_map: Dict[str, Dict[str, Any]],
    *,
    imputation_cfg: ImputationConfig,
    exclude_fields: set,
    exclude_prefixes: set,
) -> Set[str]:
    """
    Identify which concept fields in a report are eligible for imputation.

    Eligibility rules:
      - support_devices: empty text + label "not present" + has sampling pool
      - pathologies targets: empty text + binary label (0/1) + sufficient other
        pathologies fields have non-empty text (context ratio threshold)
    """
    eligible: Set[str] = set()
    if not text_label_map:
        return eligible

    # Rule 1: support_devices with empty text and "not present" label
    if imputation_cfg.enabled_support_devices:
        for rel_path in (p for p in text_label_map if p.startswith("support_devices.")):
            info = text_label_map[rel_path]
            if clean_text(info.get("text")) != "":
                continue
            if not label_is_not_present(info.get("label")):
                continue
            if should_exclude(f"structured.{rel_path}", exclude_fields, exclude_prefixes):
                continue
            if not get_sampling_pool(imputation_cfg.templates, rel_path):
                continue
            eligible.add(rel_path)

    # Rule 2: specified pathologies targets with empty text + context threshold
    if imputation_cfg.enabled_pathologies and imputation_cfg.pathologies_targets:
        patho_leaf_paths = sorted(
            p for p in text_label_map if any(p.startswith(pref) for pref in PATHOLOGY_GROUP_PREFIXES)
        )
        valid_text_flags = {p: clean_text(text_label_map[p].get("text")) != "" for p in patho_leaf_paths}

        for target in imputation_cfg.pathologies_targets:
            rel_target = normalize_rel_field(target)
            info = text_label_map.get(rel_target)
            if info is None or clean_text(info.get("text")) != "":
                continue
            if not label_is_binary_01(info.get("label")):
                continue

            # Check context ratio: enough other pathologies fields must have text
            other_fields = [p for p in patho_leaf_paths if p != rel_target]
            other_total = len(other_fields)
            other_valid = sum(1 for p in other_fields if valid_text_flags.get(p, False))
            ratio_pct = 0.0 if other_total == 0 else (100.0 * other_valid / float(other_total))

            if other_total < imputation_cfg.pathologies_min_other_fields:
                continue
            if ratio_pct < imputation_cfg.pathologies_threshold_pct:
                continue
            if should_exclude(f"structured.{rel_target}", exclude_fields, exclude_prefixes):
                continue
            if not get_sampling_pool(imputation_cfg.templates, rel_target):
                continue
            eligible.add(rel_target)

    return eligible


def select_positions_for_abundance_balanced_imputation(
    *,
    report_id_list: List[str],
    resume_start_idx: int,
    processed_report_ids: Optional[set],
    json_dir: Path,
    json_filename_template: str,
    json_fallback_glob: Optional[str],
    imputation_cfg: ImputationConfig,
    exclude_fields: set,
    exclude_prefixes: set,
    quotas_by_concept: Dict[str, int],
    seed: Optional[int],
) -> Tuple[Dict[str, Set[int]], Dict[str, int], Dict[str, int]]:
    """
    Pre-scan all reports and use reservoir sampling to select which positions
    get imputation for each concept, ensuring uniform random selection up to
    the quota limit.

    Returns:
      - selected_positions_by_concept: concept -> set of report_id_list indices
      - eligible_counts: concept -> total eligible candidates found
      - selected_counts: concept -> number actually selected
    """
    active_concepts = {c for c, q in quotas_by_concept.items() if int(q) > 0}
    reservoirs: Dict[str, List[int]] = {c: [] for c in active_concepts}
    eligible_counts: Dict[str, int] = {c: 0 for c in quotas_by_concept}

    rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed) + 100_003)

    for pos in tqdm(range(resume_start_idx, len(report_id_list)), desc="Pre-scan imputation candidates"):
        report_id = report_id_list[pos]
        if processed_report_ids is not None and report_id in processed_report_ids:
            continue

        jp = json_path_for_report_id(report_id, json_dir, json_filename_template, json_fallback_glob)
        if jp is None:
            continue

        text_label_map = extract_text_label_map_from_json_path(jp)
        if not text_label_map:
            continue

        eligible_for_report = collect_eligible_imputation_concepts(
            text_label_map,
            imputation_cfg=imputation_cfg,
            exclude_fields=exclude_fields,
            exclude_prefixes=exclude_prefixes,
        )

        for concept in eligible_for_report:
            if concept not in quotas_by_concept:
                continue
            eligible_counts[concept] = eligible_counts.get(concept, 0) + 1

            k = quotas_by_concept.get(concept, 0)
            if k <= 0:
                continue

            res = reservoirs.setdefault(concept, [])
            seen = eligible_counts[concept]

            # Reservoir sampling: keep first k, then replace with probability k/seen
            if len(res) < k:
                res.append(pos)
            else:
                j = int(rng.integers(1, seen + 1))
                if j <= k:
                    res[j - 1] = pos

    selected_positions = {c: set(v) for c, v in reservoirs.items()}
    selected_counts = {c: len(selected_positions.get(c, set())) for c in quotas_by_concept}
    return selected_positions, eligible_counts, selected_counts


# ===============================================================================
# Imputation unit builder
# ===============================================================================

def make_imputed_structured_unit(
    *,
    report_id: str,
    rel_field: str,
    sampled_sentence: str,
    source_json_path: str,
    append_field_prefix: bool,
    rule: str,
    sample_index: int,
    original_text: str,
    original_label: Any,
    patho_other_valid_count: Optional[int] = None,
    patho_other_total_count: Optional[int] = None,
    patho_other_valid_ratio_pct: Optional[float] = None,
    patho_threshold_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create an embedding unit dict for an imputed (synthetic) text entry.

    The unit includes full imputation provenance metadata for traceability.
    """
    text_for_embedding = sampled_sentence
    if append_field_prefix:
        text_for_embedding = f"{rel_field}: {sampled_sentence}"

    return {
        "report_id": report_id,
        "field": f"structured.{rel_field}",
        "text": text_for_embedding,
        "source": "structured",
        "json_path": source_json_path,
        "is_imputed": True,
        "imputation_rule": rule,
        "imputation_sample_key": rel_field,
        "imputation_sampled_sentence": sampled_sentence,
        "imputation_sample_index": sample_index,
        "imputation_original_text": original_text,
        "imputation_original_label": stringify_label(original_label),
        "pathologies_other_valid_count": patho_other_valid_count,
        "pathologies_other_total_count": patho_other_total_count,
        "pathologies_other_valid_ratio_pct": patho_other_valid_ratio_pct,
        "pathologies_threshold_pct": patho_threshold_pct,
    }


# ===============================================================================
# Per-report unit builder (extraction + imputation)
# ===============================================================================

def build_units_for_report(
    report_id: str,
    csv_row: pd.Series,
    json_dir: Path,
    *,
    json_filename_template: str,
    json_fallback_glob: Optional[str],
    embed_level: str,
    append_field_prefix: bool,
    enable_grouped: bool,
    grouped_fields: Sequence[str],
    grouped_delimiter: str,
    exclude_fields: set,
    exclude_prefixes: set,
    dedup_within_report: bool,
    imputation_cfg: ImputationConfig,
    verbose_test: bool,
    sample_pos: Optional[int],
    selected_positions_by_concept: Optional[Dict[str, Set[int]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    """
    Build all embedding units for one report.

    Steps:
      1. Optionally create an unstructured full-text unit
      2. Extract non-empty text leaves from the structured JSON
      3. Apply imputation rules for missing text (support_devices, pathologies)
      4. Optionally create grouped concatenated text units
      5. Optionally create full-structured and full-report units
      6. Deduplicate if enabled

    Returns:
      - units: list of embedding unit dicts
      - stats: counters for logging
      - debug: detailed decision info (only if verbose_test=True)
    """
    stats = {
        "has_json": 0, "no_json": 0, "empty_units": 0,
        "imputed_total": 0, "imputed_support_devices": 0, "imputed_pathologies": 0,
    }
    units: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {
        "report_id": report_id, "json_path": "",
        "support_devices_decisions": [], "pathologies_decisions": [],
        "imputed_units": [], "embedded_units": [],
        "json_support_devices": None, "json_pathologies": None,
    }

    # --- Unstructured full text (from CSV columns) ---
    parts = []
    exam_text = csv_row.get("examination")
    report_text = csv_row.get("report_text")
    if isinstance(exam_text, str) and exam_text.strip():
        parts.append(clean_text(exam_text))
    if isinstance(report_text, str) and report_text.strip():
        parts.append(clean_text(report_text))
    unstructured_full = " ".join(p for p in parts if p)

    if unstructured_full and embed_level in ("full", "both"):
        if not should_exclude("unstructured.full", exclude_fields, exclude_prefixes):
            units.append({
                "report_id": report_id, "field": "unstructured.full",
                "text": unstructured_full, "source": "unstructured",
                "json_path": "", "is_imputed": False,
            })

    # --- Structured JSON ---
    jp = json_path_for_report_id(report_id, json_dir, json_filename_template, json_fallback_glob)
    if jp is None:
        stats["no_json"] += 1
        return units, stats, (debug if verbose_test else {})

    stats["has_json"] += 1
    debug["json_path"] = str(jp)

    try:
        with open(jp, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return units, stats, (debug if verbose_test else {})

    data = j.get("data", j)
    if not isinstance(data, (dict, list)):
        return units, stats, (debug if verbose_test else {})

    if verbose_test and isinstance(data, dict):
        debug["json_support_devices"] = data.get("support_devices")
        debug["json_pathologies"] = data.get("pathologies")

    # Standard non-empty leaf text units
    leaf_pairs = extract_text_leaves(data, prefix="")
    text_label_leaves = extract_text_label_leaves(data, prefix="")
    text_label_map = {p: info for p, info in text_label_leaves}

    for rel_path, txt in leaf_pairs:
        full_path = f"structured.{rel_path}"
        if should_exclude(full_path, exclude_fields, exclude_prefixes):
            continue
        text_for_embedding = f"{rel_path}: {txt}" if append_field_prefix else txt
        units.append({
            "report_id": report_id, "field": full_path,
            "text": text_for_embedding, "source": "structured",
            "json_path": str(jp), "is_imputed": False,
        })

    imputed_leaf_pairs: List[Tuple[str, str]] = []

    # --- Imputation Rule 1: support_devices with empty text + "not present" ---
    if imputation_cfg.enabled_support_devices:
        for rel_path in sorted(p for p in text_label_map if p.startswith("support_devices.")):
            info = text_label_map[rel_path]
            original_text = clean_text(info.get("text"))
            original_label = info.get("label")
            has_empty = original_text == ""
            label_ok = label_is_not_present(original_label)
            excluded = should_exclude(f"structured.{rel_path}", exclude_fields, exclude_prefixes)
            pool = get_sampling_pool(imputation_cfg.templates, rel_path)
            has_pool = len(pool) > 0

            base_eligible = has_empty and label_ok and has_pool and not excluded
            selected_by_quota = True
            if selected_positions_by_concept is not None:
                selected_by_quota = sample_pos is not None and sample_pos in selected_positions_by_concept.get(rel_path, set())
            eligible = base_eligible and selected_by_quota

            if verbose_test:
                debug["support_devices_decisions"].append({
                    "field": rel_path, "original_text": original_text,
                    "original_label": original_label, "text_empty": has_empty,
                    "label_is_not_present": label_ok, "has_sampling_pool": has_pool,
                    "excluded": excluded, "selected_by_quota": selected_by_quota,
                    "eligible": eligible,
                })

            if not eligible:
                continue

            sampled, sample_idx = deterministic_sample(pool, imputation_cfg.sampling_seed, report_id, rel_path)
            unit = make_imputed_structured_unit(
                report_id=report_id, rel_field=rel_path, sampled_sentence=sampled,
                source_json_path=str(jp), append_field_prefix=append_field_prefix,
                rule="support_devices_empty_text_label_not_present",
                sample_index=sample_idx, original_text=original_text,
                original_label=original_label,
            )
            units.append(unit)
            imputed_leaf_pairs.append((rel_path, sampled))
            stats["imputed_total"] += 1
            stats["imputed_support_devices"] += 1

    # --- Imputation Rule 2: pathologies targets with context ratio threshold ---
    if imputation_cfg.enabled_pathologies and imputation_cfg.pathologies_targets:
        patho_leaf_paths = sorted(
            p for p in text_label_map if any(p.startswith(pref) for pref in PATHOLOGY_GROUP_PREFIXES)
        )
        valid_text_flags = {p: clean_text(text_label_map[p].get("text")) != "" for p in patho_leaf_paths}

        for target in imputation_cfg.pathologies_targets:
            rel_target = normalize_rel_field(target)
            info = text_label_map.get(rel_target)
            if info is None:
                continue

            original_text = clean_text(info.get("text"))
            original_label = info.get("label")
            target_empty = original_text == ""
            target_label_binary = label_is_binary_01(original_label)

            other_fields = [p for p in patho_leaf_paths if p != rel_target]
            other_total = len(other_fields)
            other_valid = sum(1 for p in other_fields if valid_text_flags.get(p, False))
            ratio_pct = 0.0 if other_total == 0 else (100.0 * other_valid / float(other_total))

            threshold_ok = (
                other_total >= imputation_cfg.pathologies_min_other_fields
                and ratio_pct >= imputation_cfg.pathologies_threshold_pct
            )
            excluded = should_exclude(f"structured.{rel_target}", exclude_fields, exclude_prefixes)
            pool = get_sampling_pool(imputation_cfg.templates, rel_target)
            has_pool = len(pool) > 0

            base_eligible = target_empty and target_label_binary and threshold_ok and has_pool and not excluded
            selected_by_quota = True
            if selected_positions_by_concept is not None:
                selected_by_quota = sample_pos is not None and sample_pos in selected_positions_by_concept.get(rel_target, set())
            eligible = base_eligible and selected_by_quota

            if verbose_test:
                debug["pathologies_decisions"].append({
                    "target_field": rel_target, "original_text": original_text,
                    "original_label": original_label, "target_text_empty": target_empty,
                    "target_label_is_binary": target_label_binary,
                    "other_total_count": other_total, "other_valid_count": other_valid,
                    "other_valid_ratio_pct": ratio_pct,
                    "threshold_pct": imputation_cfg.pathologies_threshold_pct,
                    "threshold_ok": threshold_ok, "has_sampling_pool": has_pool,
                    "excluded": excluded, "selected_by_quota": selected_by_quota,
                    "eligible": eligible,
                })

            if not eligible:
                continue

            sampled, sample_idx = deterministic_sample(pool, imputation_cfg.sampling_seed, report_id, rel_target)
            unit = make_imputed_structured_unit(
                report_id=report_id, rel_field=rel_target, sampled_sentence=sampled,
                source_json_path=str(jp), append_field_prefix=append_field_prefix,
                rule="pathologies_empty_text_with_other_valid_ratio",
                sample_index=sample_idx, original_text=original_text,
                original_label=original_label, patho_other_valid_count=other_valid,
                patho_other_total_count=other_total,
                patho_other_valid_ratio_pct=ratio_pct,
                patho_threshold_pct=imputation_cfg.pathologies_threshold_pct,
            )
            units.append(unit)
            imputed_leaf_pairs.append((rel_target, sampled))
            stats["imputed_total"] += 1
            stats["imputed_pathologies"] += 1

    # --- Grouped text units (optional) ---
    if enable_grouped and grouped_fields:
        grouped_input = list(leaf_pairs) + imputed_leaf_pairs
        grouped_texts = build_grouped_texts(grouped_input, grouped_fields, grouped_delimiter)
        for gp, gtext in grouped_texts.items():
            field_name = f"structured.grouped.{gp}"
            if should_exclude(field_name, exclude_fields, exclude_prefixes) or not gtext:
                continue
            units.append({
                "report_id": report_id, "field": field_name, "text": gtext,
                "source": "structured_grouped", "json_path": str(jp), "is_imputed": False,
            })

    # --- Full structured / full report (optional) ---
    if embed_level in ("full", "both"):
        all_structured = clean_text(" ".join(
            u["text"] for u in units
            if u.get("source") == "structured" and str(u.get("field", "")).startswith("structured.")
        ))
        if all_structured and not should_exclude("structured.full", exclude_fields, exclude_prefixes):
            units.append({
                "report_id": report_id, "field": "structured.full", "text": all_structured,
                "source": "structured", "json_path": str(jp), "is_imputed": False,
            })
        full_report = clean_text(f"{unstructured_full} {all_structured}") if unstructured_full else all_structured
        if full_report and not should_exclude("report.full", exclude_fields, exclude_prefixes):
            units.append({
                "report_id": report_id, "field": "report.full", "text": full_report,
                "source": "combined", "json_path": str(jp), "is_imputed": False,
            })

    # --- Deduplication ---
    if dedup_within_report:
        seen = set()
        deduped = []
        for u in units:
            key = (u["field"], sha1(u["text"]))
            if key not in seen:
                seen.add(key)
                deduped.append(u)
        units = deduped

    if verbose_test:
        debug["embedded_units"] = [
            {"field": u.get("field"), "source": u.get("source"),
             "is_imputed": bool(u.get("is_imputed", False)), "text": u.get("text", "")}
            for u in units
        ]
    else:
        debug = {}

    if not units:
        stats["empty_units"] += 1

    return units, stats, debug


# ===============================================================================
# Embedding backends
# ===============================================================================

# --- Backend 1: SentenceTransformer (e.g., Qwen3-Embedding) ---

def encode_sentence_transformer(
    model,
    texts: List[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
    truncate_dim: Optional[int],
    overflow_strategy: str,
    max_length: int,
    stride: int,
    log_long_text: bool,
    log_path: Path,
    report_ids: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Encode texts using a SentenceTransformer model.

    Supports two overflow strategies:
      - truncate: standard model truncation at max_seq_length
      - window_mean: split into overlapping windows and average embeddings
    """
    def finalize_embeddings(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if truncate_dim is not None:
            arr = arr[..., :truncate_dim]
        if normalize_embeddings:
            arr = arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8)
        return arr.astype(np.float32, copy=False)

    if overflow_strategy == "truncate":
        values = model.encode(
            texts, batch_size=batch_size, normalize_embeddings=False,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return finalize_embeddings(values)

    tok = model.tokenizer
    out: List[np.ndarray] = []

    for i, text in enumerate(texts):
        if not text.strip():
            dim = truncate_dim or model.get_sentence_embedding_dimension()
            out.append(np.zeros((dim,), dtype=np.float32))
            continue

        enc = tok(
            text, truncation=True, max_length=max_length,
            return_overflowing_tokens=True, stride=stride,
            padding=False, add_special_tokens=True,
        )
        windows_ids = enc["input_ids"]
        if isinstance(windows_ids[0], int):
            windows_ids = [windows_ids]

        n_windows = len(windows_ids)
        if log_long_text and n_windows > 1 and report_ids and fields:
            append_log(log_path, report_ids[i], "long_text_windowed",
                       f"{fields[i]}: windows={n_windows}, max_length={max_length}, stride={stride}")

        window_texts = [clean_text(tok.decode(ids, skip_special_tokens=True)) for ids in windows_ids]
        window_texts = [w for w in window_texts if w]
        if not window_texts:
            dim = truncate_dim or model.get_sentence_embedding_dimension()
            out.append(np.zeros((dim,), dtype=np.float32))
            continue

        w_emb = model.encode(
            window_texts, batch_size=min(batch_size, len(window_texts)),
            normalize_embeddings=False,
            convert_to_numpy=True, show_progress_bar=False,
        )
        w_emb = finalize_embeddings(w_emb)
        vec = w_emb.mean(axis=0)
        if normalize_embeddings:
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        out.append(vec.astype(np.float32, copy=False))

    return np.vstack(out)


# --- Backend 2: AutoModel with manual pooling (e.g., medBERT.de) ---

_ALLOWED_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def _select_model_inputs(enc: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Filter tokenizer output to only keys the model accepts and move to device."""
    return {k: v.to(device) for k, v in enc.items() if k in _ALLOWED_KEYS}


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    """CLS token pooling (take first token representation)."""
    return last_hidden[:, 0, :]


def _pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Apply the specified pooling strategy."""
    return _cls_pool(last_hidden) if pooling == "cls" else _mean_pool(last_hidden, attention_mask)


def encode_auto_model(
    model,
    tokenizer,
    texts: List[str],
    *,
    batch_size: int,
    pooling: str,
    normalize_embeddings: bool,
    overflow_strategy: str,
    max_length: int,
    stride: int,
    log_long_text: bool,
    log_path: Path,
    report_ids: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Encode texts using a HuggingFace AutoModel with manual pooling.

    Supports mean pooling and CLS pooling, with truncation or
    sliding-window overflow handling.
    """
    device = next(model.parameters()).device
    dim = int(model.config.hidden_size)

    if overflow_strategy == "truncate":
        out_batches: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length, add_special_tokens=True)
                inputs = _select_model_inputs(enc, device)
                hidden = model(**inputs).last_hidden_state
                pooled = _pool_hidden(hidden, inputs["attention_mask"], pooling)
                if normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)
                out_batches.append(pooled.detach().cpu().float().numpy())
        return np.vstack(out_batches) if out_batches else np.zeros((0, dim), dtype=np.float32)

    # Window-mean overflow strategy
    out_rows: List[np.ndarray] = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            if not text.strip():
                out_rows.append(np.zeros((dim,), dtype=np.float32))
                continue

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length, return_overflowing_tokens=True,
                            stride=stride, padding="longest", add_special_tokens=True)
            n_windows = int(enc["input_ids"].shape[0])

            if log_long_text and n_windows > 1 and report_ids and fields:
                append_log(log_path, report_ids[i], "long_text_windowed",
                           f"{fields[i]}: windows={n_windows}, max_length={max_length}, stride={stride}")

            inputs = _select_model_inputs(enc, device)
            hidden = model(**inputs).last_hidden_state
            pooled = _pool_hidden(hidden, inputs["attention_mask"], pooling)

            if pooled.ndim == 1:
                vec = pooled
            else:
                if normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)
                vec = pooled.mean(dim=0)

            if normalize_embeddings:
                vec = F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)
            out_rows.append(vec.detach().cpu().float().numpy())

    return np.vstack(out_rows) if out_rows else np.zeros((0, dim), dtype=np.float32)


# ===============================================================================
# Sharded writer (npy + parquet)
# ===============================================================================

@dataclass
class ShardWriter:
    """Writes embedding vectors and metadata in shards to disk."""
    out_dir: Path
    shard_size: int
    save_dtype: str           # "float16" or "float32"
    store_text: bool
    shard_idx: int = 0
    meta_rows: Optional[List[Dict[str, Any]]] = None
    emb_rows: Optional[List[np.ndarray]] = None

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_rows = []
        self.emb_rows = []

    def add(self, meta: Dict[str, Any], emb: np.ndarray) -> None:
        self.meta_rows.append(meta)
        self.emb_rows.append(emb)
        if len(self.meta_rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.meta_rows:
            return
        meta_df = pd.DataFrame(self.meta_rows)
        emb = np.vstack(self.emb_rows)
        emb = emb.astype(np.float16 if self.save_dtype == "float16" else np.float32, copy=False)

        np.save(self.out_dir / f"embeddings_{self.shard_idx:05d}.npy", emb)
        meta_df.to_parquet(self.out_dir / f"meta_{self.shard_idx:05d}.parquet", index=False)

        self.shard_idx += 1
        self.meta_rows.clear()
        self.emb_rows.clear()


# ===============================================================================
# Main
# ===============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Embed structured report text fields with optional imputation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Config file --
    ap.add_argument("--config", type=str, default=None,
                    help="Path to a YAML config file. Values serve as defaults; "
                         "CLI arguments override config values.")

    # -- Model backend --
    ap.add_argument("--model-backend", type=str, default=None,
                    choices=["sentence_transformer", "auto_model"],
                    help="Embedding backend: 'sentence_transformer' (e.g. Qwen3-Embedding) "
                         "or 'auto_model' (e.g. medBERT.de with manual pooling)")
    ap.add_argument("--model-dir", type=str, default=None,
                    help="Local path to the model directory")
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="Torch device (e.g. 'cuda:0', 'cpu')")
    ap.add_argument("--precision", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])

    # -- Inputs --
    ap.add_argument("--reports-csv", type=str, default=None,
                    help="CSV with report texts (columns: report_id, examination, report_text)")
    ap.add_argument("--report-id-column", type=str, default="report_id",
                    help="Name of the report ID column in the reports CSV")
    ap.add_argument("--json-dir", type=str, default=None,
                    help="Directory containing structured JSON report files")
    ap.add_argument("--neg-sampling-path", type=str, default=None,
                    help="Path to negative entry sampling templates (JSON)")

    # -- JSON file naming --
    ap.add_argument("--json-filename-template", type=str, default="{report_id}_structured.json",
                    help="Template for structured JSON filenames ({report_id} is replaced)")
    ap.add_argument("--json-fallback-glob", type=str, default=None,
                    help="Fallback glob pattern for JSON lookup (e.g. '{report_id}*.json')")

    # -- Output --
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--log-path", type=str, default=None)
    ap.add_argument("--shard-size", type=int, default=200_000,
                    help="Number of embedding units per shard")

    # -- What to embed --
    ap.add_argument("--embed-level", type=str, default="leaf",
                    choices=["leaf", "full", "both"],
                    help="'leaf': individual fields only; 'full': full concatenated; 'both': both")
    ap.add_argument("--append-field-prefix", action="store_true",
                    help="Prefix each text with its field path (e.g. 'pathologies.lung.pneumonia: ...')")
    ap.add_argument("--dedup-within-report", action="store_true",
                    help="Deduplicate identical texts within the same report")
    ap.add_argument("--enable-grouped", action="store_true",
                    help="Create grouped text embeddings from subfield concatenation")
    ap.add_argument("--grouped-fields", type=str,
                    default="support_devices,thoracic_organs,pathologies.lung,pathologies.pleura,pathologies.vessels,pathologies.bones")
    ap.add_argument("--grouped-delimiter", type=str, default=" ; ")

    # -- Exclusions --
    ap.add_argument("--exclude-fields", type=str, default="structured.report_date,structured.comparison",
                    help="Comma-separated field paths to exclude from embedding")
    ap.add_argument("--exclude-prefixes", type=str, default="")

    # -- Imputation --
    ap.add_argument("--disable-support-devices-imputation", action="store_true")
    ap.add_argument("--disable-pathologies-imputation", action="store_true")
    ap.add_argument("--impute-pathologies-targets", type=str, default="",
                    help="Comma-separated pathologies target paths for imputation")
    ap.add_argument("--pathologies-valid-threshold-pct", type=float, default=70.0,
                    help="Min percent of other pathologies fields with text for imputation eligibility")
    ap.add_argument("--pathologies-min-other-fields", type=int, default=1)
    ap.add_argument("--sampling-seed", type=int, default=42,
                    help="Seed for deterministic imputation sampling (-1 = random)")
    ap.add_argument("--concept-stats-csv", type=str, default=None,
                    help="CSV with per-concept text counts for abundance-balanced imputation")
    ap.add_argument("--abundance-count-column", type=str, default="n_samples_with_text")
    ap.add_argument("--disable-abundance-balance", action="store_true")

    # -- Embedding parameters --
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"],
                    help="Pooling strategy (auto_model backend only)")
    ap.add_argument("--normalize-embeddings", dest="normalize_embeddings", action="store_true")
    ap.add_argument("--no-normalize-embeddings", dest="normalize_embeddings", action="store_false")
    ap.set_defaults(normalize_embeddings=True)
    ap.add_argument("--truncate-dim", type=int, default=None,
                    help="Truncate embedding dimension (sentence_transformer backend only)")
    ap.add_argument("--max-length", type=int, default=512,
                    help="Maximum token length for truncation/windowing")
    ap.add_argument("--overflow-strategy", type=str, default="truncate",
                    choices=["truncate", "window_mean"])
    ap.add_argument("--stride", type=int, default=128,
                    help="Stride for sliding window overlap")
    ap.add_argument("--log-long-text", action="store_true")
    ap.add_argument("--store-text", dest="store_text", action="store_true")
    ap.add_argument("--no-store-text", dest="store_text", action="store_false")
    ap.set_defaults(store_text=True)
    ap.add_argument("--process-max", type=int, default=None,
                    help="Limit number of reports to process")
    ap.add_argument("--only-report-ids", type=str, default="",
                    help="Comma-separated list of specific report IDs to process")

    # -- Resume --
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-mode", type=str, default="last_shard",
                    choices=["last_shard", "skip_processed"])

    # -- Debug / test --
    ap.add_argument("--dry-run", action="store_true",
                    help="Run extraction logic without embedding or writing")
    ap.add_argument("--verbose-test", action="store_true")
    ap.add_argument("--verbose-test-limit", type=int, default=10)
    ap.add_argument("--verbose-test-out", type=str, default=None)
    ap.add_argument("--verbose-test-only-imputed", action="store_true")

    # --- Load YAML config as defaults (before parsing) ---
    # We do a preliminary parse to check for --config, then apply YAML defaults
    preliminary, _ = ap.parse_known_args()
    if preliminary.config:
        yaml_cfg = load_yaml_config(preliminary.config)
        apply_yaml_defaults(ap, yaml_cfg)

    args = ap.parse_args()

    # Validate required arguments (may come from config or CLI)
    missing = []
    for req in ("model_backend", "model_dir", "reports_csv", "json_dir", "neg_sampling_path", "out_dir"):
        if getattr(args, req, None) is None:
            missing.append(f"--{req.replace('_', '-')}")
    if missing:
        ap.error(f"The following arguments are required: {', '.join(missing)} "
                 f"(set via CLI or in the YAML config file)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else (out_dir / "processing_log.csv")

    # Offline safety: prevent accidental model downloads
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # -- Load imputation templates --
    templates = load_neg_sampling_templates(Path(args.neg_sampling_path))

    pathologies_targets = parse_csv_list(args.impute_pathologies_targets)
    if not pathologies_targets and not args.disable_pathologies_imputation:
        pathologies_targets = sorted(k for k in templates if k.startswith("pathologies."))

    if not args.disable_pathologies_imputation and not pathologies_targets:
        raise ValueError(
            "Pathologies imputation enabled but no targets available. "
            "Provide --impute-pathologies-targets or add pathologies.* keys to --neg-sampling-path."
        )

    sampling_seed: Optional[int] = None if args.sampling_seed < 0 else args.sampling_seed

    imputation_cfg = ImputationConfig(
        enabled_support_devices=not args.disable_support_devices_imputation,
        enabled_pathologies=not args.disable_pathologies_imputation,
        templates=templates,
        pathologies_targets=pathologies_targets,
        pathologies_threshold_pct=args.pathologies_valid_threshold_pct,
        pathologies_min_other_fields=max(0, args.pathologies_min_other_fields),
        sampling_seed=sampling_seed,
    )

    # -- Abundance-balanced quotas --
    abundance_enabled = not args.disable_abundance_balance and bool(args.concept_stats_csv)
    concept_quotas: Dict[str, int] = {}
    effective_quotas: Dict[str, int] = {}
    already_written: Dict[str, int] = {}

    support_device_concepts = sorted(k for k in templates if k.startswith("support_devices.")) if imputation_cfg.enabled_support_devices else []
    patho_concepts = [normalize_rel_field(x) for x in imputation_cfg.pathologies_targets] if imputation_cfg.enabled_pathologies else []
    target_concepts = list(dict.fromkeys(support_device_concepts + patho_concepts))  # unique, ordered

    if abundance_enabled:
        concept_counts, used_col = load_concept_valid_text_quotas(
            Path(args.concept_stats_csv), count_column=args.abundance_count_column)
        concept_quotas = {c: concept_counts.get(c, 0) for c in target_concepts}
        print(f"[abundance] Loaded concept stats from {args.concept_stats_csv} (column={used_col})")
        for c in target_concepts:
            print(f"[abundance] concept={c} valid_text_count={concept_quotas.get(c, 0)}")

    effective_quotas = dict(concept_quotas)

    # -- Load model --
    model = None
    tokenizer = None
    truncate_dim = args.truncate_dim

    if not args.dry_run:
        device_str = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map[args.precision]

        if args.model_backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                str(Path(args.model_dir)), device=device_str,
                model_kwargs={"torch_dtype": torch_dtype},
                tokenizer_kwargs={"padding_side": "left"},
            )
            model.max_seq_length = args.max_length

        elif args.model_backend == "auto_model":
            from transformers import AutoModel, AutoTokenizer
            if device_str == "cpu" and torch_dtype != torch.float32:
                torch_dtype = torch.float32  # CPU does not support half precision well
            tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_dir)), local_files_only=True)
            model = AutoModel.from_pretrained(
                str(Path(args.model_dir)), add_pooling_layer=False,
                torch_dtype=torch_dtype, local_files_only=True,
            ).to(torch.device(device_str))
            model.eval()

    # -- Build report ID list from CSV --
    df_reports_idx = read_reports(Path(args.reports_csv), report_id_column=args.report_id_column)
    report_id_list = df_reports_idx.index.astype(str).tolist()

    if args.only_report_ids.strip():
        report_id_list = parse_csv_list(args.only_report_ids)
    if args.process_max:
        report_id_list = report_id_list[:args.process_max]

    exclude_fields = {s.strip() for s in args.exclude_fields.split(",") if s.strip()}
    exclude_prefixes = {s.strip() for s in args.exclude_prefixes.split(",") if s.strip()}
    grouped_fields = [s.strip() for s in args.grouped_fields.split(",") if s.strip()]

    # -- Verbose test output setup --
    verbose_test_path: Optional[Path] = None
    if args.verbose_test:
        verbose_test_path = Path(args.verbose_test_out) if args.verbose_test_out else out_dir / "verbose_test_debug.jsonl"
        if verbose_test_path.exists() and not args.resume:
            verbose_test_path.unlink()

    # -- Resume / shard writer setup --
    writer: Optional[ShardWriter] = None
    resume_start_idx = 0
    processed_report_ids = None

    if not args.dry_run:
        save_dtype = "float16" if args.precision in ("float16", "bfloat16") else "float32"
        if args.resume:
            completed = discover_completed_shards(out_dir)
            if completed:
                last_shard = completed[-1]
                writer = ShardWriter(out_dir=out_dir, shard_size=args.shard_size,
                                     save_dtype=save_dtype, store_text=args.store_text)
                writer.shard_idx = last_shard + 1
                if args.resume_mode == "last_shard":
                    last_rid = last_report_id_in_shard(out_dir / f"meta_{last_shard:05d}.parquet")
                    if last_rid:
                        pos = find_last_index(report_id_list, last_rid)
                        resume_start_idx = pos + 1 if pos >= 0 else 0
                elif args.resume_mode == "skip_processed":
                    processed_report_ids = load_processed_report_ids(out_dir)
            else:
                writer = ShardWriter(out_dir=out_dir, shard_size=args.shard_size,
                                     save_dtype=save_dtype, store_text=args.store_text)
        else:
            writer = ShardWriter(out_dir=out_dir, shard_size=args.shard_size,
                                 save_dtype=save_dtype, store_text=args.store_text)

    # -- Abundance-balanced pre-scan --
    selected_positions_by_concept: Optional[Dict[str, Set[int]]] = None
    eligible_by_concept: Dict[str, int] = {}
    selected_by_concept: Dict[str, int] = {}

    if abundance_enabled and target_concepts:
        if args.resume and not args.dry_run:
            already_written = load_existing_imputed_counts_by_concept(out_dir)
            for c in effective_quotas:
                effective_quotas[c] = max(0, effective_quotas[c] - already_written.get(c, 0))

        selected_positions_by_concept, eligible_by_concept, selected_by_concept = (
            select_positions_for_abundance_balanced_imputation(
                report_id_list=report_id_list, resume_start_idx=resume_start_idx,
                processed_report_ids=processed_report_ids, json_dir=Path(args.json_dir),
                json_filename_template=args.json_filename_template,
                json_fallback_glob=args.json_fallback_glob,
                imputation_cfg=imputation_cfg, exclude_fields=exclude_fields,
                exclude_prefixes=exclude_prefixes, quotas_by_concept=effective_quotas,
                seed=sampling_seed,
            )
        )
        print("[abundance] Pre-scan completed.")
        for c in target_concepts:
            print(f"[abundance] concept={c} quota={concept_quotas.get(c, 0)} "
                  f"eligible={eligible_by_concept.get(c, 0)} selected={selected_by_concept.get(c, 0)}")

    # -- Processing counters --
    n_units_total = 0
    n_reports = 0
    n_no_json = 0
    n_empty = 0
    n_imputed_total = 0
    n_imputed_support_devices = 0
    n_imputed_pathologies = 0
    n_reports_with_imputation = 0

    buf_units: List[Dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal buf_units, n_units_total

        if args.dry_run:
            n_units_total += len(buf_units)
            buf_units = []
            return
        if not buf_units:
            return

        assert model is not None and writer is not None

        texts = [u["text"] for u in buf_units]
        ids_buf = [u["report_id"] for u in buf_units]
        fields_buf = [u["field"] for u in buf_units]

        # Dispatch to the appropriate backend
        if args.model_backend == "sentence_transformer":
            emb = encode_sentence_transformer(
                model, texts, batch_size=args.batch_size,
                normalize_embeddings=args.normalize_embeddings,
                truncate_dim=truncate_dim, overflow_strategy=args.overflow_strategy,
                max_length=args.max_length, stride=args.stride,
                log_long_text=args.log_long_text, log_path=log_path,
                report_ids=ids_buf, fields=fields_buf,
            )
        else:
            assert tokenizer is not None
            emb = encode_auto_model(
                model, tokenizer, texts, batch_size=args.batch_size,
                pooling=args.pooling, normalize_embeddings=args.normalize_embeddings,
                overflow_strategy=args.overflow_strategy, max_length=args.max_length,
                stride=args.stride, log_long_text=args.log_long_text,
                log_path=log_path, report_ids=ids_buf, fields=fields_buf,
            )

        for u, e in zip(buf_units, emb):
            meta = {
                "report_id": u["report_id"], "field": u["field"],
                "source": u["source"], "json_path": u.get("json_path", ""),
                "text_sha1": sha1(u["text"]), "n_chars": len(u["text"]),
                "model": str(args.model_dir),
                "normalized": args.normalize_embeddings,
                "max_length": args.max_length,
                "overflow_strategy": args.overflow_strategy,
                "is_imputed": bool(u.get("is_imputed", False)),
                "imputation_rule": u.get("imputation_rule"),
                "imputation_sample_key": u.get("imputation_sample_key"),
                "imputation_sampled_sentence": u.get("imputation_sampled_sentence"),
                "imputation_sample_index": u.get("imputation_sample_index"),
                "imputation_original_text": u.get("imputation_original_text"),
                "imputation_original_label": u.get("imputation_original_label"),
                "pathologies_other_valid_count": u.get("pathologies_other_valid_count"),
                "pathologies_other_total_count": u.get("pathologies_other_total_count"),
                "pathologies_other_valid_ratio_pct": u.get("pathologies_other_valid_ratio_pct"),
                "pathologies_threshold_pct": u.get("pathologies_threshold_pct"),
            }
            # Backend-specific metadata
            if args.model_backend == "auto_model":
                meta["pooling"] = args.pooling
                meta["dim"] = int(e.shape[0])
            else:
                meta["truncate_dim"] = truncate_dim or int(e.shape[0])

            if writer.store_text:
                meta["text"] = u["text"]
            writer.add(meta, e)

        n_units_total += len(buf_units)
        buf_units = []

    verbose_printed = 0

    # -- Main loop --
    for pos in tqdm(range(resume_start_idx, len(report_id_list)), desc="Reports"):
        report_id = report_id_list[pos]
        if processed_report_ids is not None and report_id in processed_report_ids:
            continue

        n_reports += 1
        csv_row = df_reports_idx.loc[report_id] if report_id in df_reports_idx.index else pd.Series({}, dtype=object)

        units, stats, debug = build_units_for_report(
            report_id, csv_row, Path(args.json_dir),
            json_filename_template=args.json_filename_template,
            json_fallback_glob=args.json_fallback_glob,
            embed_level=args.embed_level,
            append_field_prefix=args.append_field_prefix,
            enable_grouped=args.enable_grouped,
            grouped_fields=grouped_fields,
            grouped_delimiter=args.grouped_delimiter,
            exclude_fields=exclude_fields,
            exclude_prefixes=exclude_prefixes,
            dedup_within_report=args.dedup_within_report,
            imputation_cfg=imputation_cfg,
            verbose_test=args.verbose_test,
            sample_pos=pos,
            selected_positions_by_concept=selected_positions_by_concept,
        )

        if stats["no_json"] > 0:
            n_no_json += 1
            if not units:
                append_log(log_path, report_id, "skip_no_json", "no structured json and no unstructured text")
                continue

        if stats["empty_units"] > 0:
            n_empty += 1
            append_log(log_path, report_id, "skip_empty_units", "json present but produced no units")
            continue

        n_imputed_total += stats["imputed_total"]
        n_imputed_support_devices += stats["imputed_support_devices"]
        n_imputed_pathologies += stats["imputed_pathologies"]
        if stats["imputed_total"] > 0:
            n_reports_with_imputation += 1

        if args.verbose_test:
            debug["imputed_total"] = stats["imputed_total"]
            should_emit = not args.verbose_test_only_imputed or stats["imputed_total"] > 0
            if should_emit:
                if verbose_test_path:
                    append_jsonl(verbose_test_path, debug)
                if verbose_printed < args.verbose_test_limit:
                    print(json.dumps(debug, ensure_ascii=False, indent=2))
                    verbose_printed += 1

        buf_units.extend(units)
        if len(buf_units) >= args.batch_size * 8:
            flush_buffer()

    flush_buffer()
    if writer is not None:
        writer.flush()

    # -- Summary --
    summary = {
        "dry_run": args.dry_run,
        "model_backend": args.model_backend,
        "reports_seen": n_reports,
        "units_written": n_units_total,
        "reports_with_no_json": n_no_json,
        "reports_with_empty_units": n_empty,
        "imputed_units_total": n_imputed_total,
        "imputed_units_support_devices": n_imputed_support_devices,
        "imputed_units_pathologies": n_imputed_pathologies,
        "reports_with_imputation": n_reports_with_imputation,
        "abundance_balance_enabled": abundance_enabled,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
