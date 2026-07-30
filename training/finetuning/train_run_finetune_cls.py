#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RadPRISM Classification Fine-tuning
===================================

Loads an alignment-pretrained :class:`VisionConceptModel` and trains **only the
per-concept binary classification heads** on top of the frozen backbone and
frozen concept cross-attention. This turns the contrastively-pretrained model
into a multi-label chest X-ray classifier without disturbing the learned
representations.

Design
------
The fine-tune run *inherits* the architecture and data configuration from the
pretrained run: it reads that run's ``config.json`` (concept lists, model dims,
data settings) and only overrides what the fine-tuning YAML specifies. This
guarantees the classification model matches the pretrained weights it loads.

    python train_run_finetune_cls.py --config config/finetune_cls.yaml

Only the classification heads receive gradients; everything else is frozen and
kept in eval mode (so BatchNorm/dropout in the backbone stay fixed).

Positive-class weighting for the BCE loss can be derived from a pre-computed
class-statistics CSV, from a saved ``pos_weights.pt``, or from the training set
directly, with optional "tempering" (a power < 1 that softens extreme weights).
"""

import os
import sys
import json
import math
import time
import random
import argparse
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler

    def make_grad_scaler(enabled: bool):
        return _GradScaler("cuda", enabled=enabled)

    def bf16_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        return _autocast("cuda", dtype=torch.bfloat16)
except ImportError:  # PyTorch 2.0-2.2 compatibility
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler

    def make_grad_scaler(enabled: bool):
        return _GradScaler(enabled=enabled)

    def bf16_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        return _autocast(dtype=torch.bfloat16)
from sklearn.metrics import average_precision_score, precision_recall_curve

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Shared source modules live at the repository root:
#   RadPRISM/src/  <-  RadPRISM/training/finetuning/train_run_finetune_cls.py
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from model import ModelConfig, VisionConceptModel                       # noqa: E402
from losses import LossConfig, compute_losses, build_tempered_pos_weight_from_counts  # noqa: E402
from dataset import (                                                    # noqa: E402
    CXRMultimodalDataset, PTCachedCXRDataset, make_dataloader,
    create_group_splits, apply_group_splits,
)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class PretrainCfg:
    """Which pretrained run to fine-tune from."""
    ckpt_path: str = ""           # pretrained checkpoint (e.g. best_align.pt)
    config_path: str = ""         # config.json written by the pretraining run
    reset_cls_heads: bool = True  # re-initialize cls heads (True if they were untrained)


@dataclass
class PathsOverride:
    """
    Output directory plus optional overrides of the inherited pretrain paths.
    Any field left as None is inherited from the pretrained run's config.json.
    """
    run_dir: str = "output/finetune_runs"
    master_index_path: Optional[str] = None
    embedding_dir: Optional[str] = None
    label_dir: Optional[str] = None
    split_path: Optional[str] = None
    cached_dataset_path: Optional[str] = None


@dataclass
class DataOverride:
    """
    Optional overrides of the inherited data config. None = inherit from the
    pretrained run's config.json.
    """
    image_size: Optional[int] = None
    augment: Optional[bool] = None
    norm_mode: Optional[str] = None
    batch_size: Optional[int] = None
    effective_batch: Optional[int] = None
    num_workers: Optional[int] = None
    max_train: Optional[int] = None
    max_val: Optional[int] = None
    subsample_seed: Optional[int] = None
    val_frac: Optional[float] = None
    test_frac: Optional[float] = None
    split_on: Optional[str] = None
    accession_column: Optional[str] = None
    sop_uid_column: Optional[str] = None

    # Text-embedding override store
    text_override_dir: Optional[str] = None
    use_text_override: Optional[bool] = None
    text_override_cache_size: Optional[int] = None
    text_override_strict: Optional[bool] = None
    text_override_mode: Optional[str] = None
    text_override_index_cache_name: Optional[str] = None
    text_override_shard_memmap_cache_size: Optional[int] = None

    # Classification-label override store
    cls_override_dir: Optional[str] = None
    use_cls_override: Optional[bool] = None
    cls_override_cache_size: Optional[int] = None
    cls_override_strict: Optional[bool] = None
    cls_override_mode: Optional[str] = None
    cls_override_index_cache_name: Optional[str] = None
    cls_override_shard_memmap_cache_size: Optional[int] = None

    # Dataloader
    dataloader_prefetch_factor: Optional[int] = None
    dataloader_persistent_workers: Optional[bool] = None
    dataloader_pin_memory: Optional[bool] = None

    # Transforms
    affine_degrees: Optional[float] = None
    affine_translate: Optional[tuple] = None
    cornercutout_size: Optional[tuple] = None
    cornercutout_probs: Optional[dict] = None
    cornercutout_fill: Optional[float] = None


@dataclass
class TrainCfg:
    """Optimization schedule for the classification heads."""
    epochs: int = 14
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    log_every: int = 10
    save_every_epochs: int = 100
    bf16: bool = True
    device: str = "cuda:0"
    clip_grad_norm: float = 1.0
    seed: int = 1337
    seed_list: Optional[List[int]] = field(default_factory=list)   # non-empty -> one process per seed


@dataclass
class LossSettings:
    """BCE classification loss and positive-weight configuration."""
    w_cls: float = 1.0
    use_concept_balanced_reduction: bool = True   # equal-weight the concepts in the mean

    # Positive-class weighting sources (checked in this order):
    #   1. class_stats_path (per-concept pos/neg counts CSV)
    #   2. pos_weight_path  (a saved [K] tensor)
    #   3. computed from the training set
    class_stats_path: Optional[str] = None
    class_stats_split: str = "train"
    class_stats_use_accessions: bool = False
    pos_weight_path: Optional[str] = None

    # Optional tempering of the pos-weights: w_c -> w_c ** gamma (gamma < 1 softens).
    use_tempered_pos_weight: bool = True
    tempered_pos_weight_gamma: float = 0.7
    tempered_prior_smoothing_alpha: float = 0.0
    tempered_global_pos_prior: Optional[float] = None

    max_pos_weight: float = 0.0   # clip pos-weights to this max (0 = no clipping)


# =============================================================================
# 2. CONCEPT NAMES + HEAD MAPPING
# =============================================================================

# The 19 fine-grained clinical concepts (fallback when the pretrained config
# does not list them explicitly). Must match the report_struct_label pipeline.
FINE_CONCEPT_NAMES = [
    "support_devices.airway", "support_devices.gastric_tube", "support_devices.central_line",
    "support_devices.chest_drain", "support_devices.pacemaker", "support_devices.other",
    "thoracic_organs.heart", "thoracic_organs.mediastinum",
    "pathologies.lung.pneumonia", "pathologies.lung.atelectasis", "pathologies.lung.emphysema",
    "pathologies.lung.fibrosis", "pathologies.lung.mass",
    "pathologies.pleura.pneumothorax", "pathologies.pleura.pleural_effusion",
    "pathologies.vessels.congestion", "pathologies.vessels.pulmonary_edema",
    "pathologies.bones.fracture", "pathologies.bones.other",
]


def _ensure_unique(names: List[str], what: str) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"{what} contains duplicates, which is not supported: {names}")


def build_name_based_cls_head_map(cls_names: List[str], align_names: List[str]) -> List[int]:
    """Map each classification concept to the index of its alignment concept token."""
    _ensure_unique(cls_names, "cls_concept_names")
    _ensure_unique(align_names, "align_concept_names")
    align_idx = {name: i for i, name in enumerate(align_names)}
    out: List[int] = []
    for cname in cls_names:
        if cname not in align_idx:
            raise ValueError(
                f"CLS concept '{cname}' not found in align_concept_names. "
                "CLS concepts must be a subset of alignment concepts."
            )
        out.append(align_idx[cname])
    return out


def _resolve_concepts(pre_cfg: dict) -> Tuple[list, list, Optional[list]]:
    """Read the concept lists (and optional head map) from the pretrained config."""
    align = pre_cfg.get("align_concept_names") or pre_cfg.get("concept_names") or list(FINE_CONCEPT_NAMES)
    cls = pre_cfg.get("cls_concept_names") or align
    cls_head_map = pre_cfg.get("cls_head_map")
    return align, cls, cls_head_map


def _infer_cls_head_map(align_names: list, cls_names: list) -> Optional[list]:
    """None when cls == align (identity), otherwise a name-based mapping."""
    if len(align_names) == len(cls_names) and align_names == cls_names:
        return None
    return build_name_based_cls_head_map(cls_names, align_names)


# =============================================================================
# 3. LOGGING + GENERAL HELPERS
# =============================================================================

class CSVScalarLogger:
    """Append-only CSV logger for scalar metrics."""
    def __init__(self, path: str):
        import csv
        self.path = path
        new = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.wr = csv.writer(self.fh)
        if new:
            self.wr.writerow(["step", "epoch", "split", "metric", "value", "walltime"])
            self.fh.flush()

    def log(self, step, epoch, split, metric, value):
        self.wr.writerow([step, epoch, split, metric, float(value), time.time()])
        self.fh.flush()

    def log_many(self, step, epoch, split, kv):
        for k, v in kv.items():
            self.log(step, epoch, split, k, v)

    def close(self):
        self.fh.close()


class CSVPerConceptLogger:
    """Append-only CSV logger for per-concept classification metrics."""
    def __init__(self, path: str):
        import csv
        self.path = path
        new = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.wr = csv.writer(self.fh)
        if new:
            self.wr.writerow(["epoch", "concept", "metric", "value", "optimal_threshold"])
            self.fh.flush()

    def log_concept(self, epoch, concept, metric, value, threshold=None):
        t_val = threshold if threshold is not None else -1.0
        self.wr.writerow([epoch, concept, metric, float(value), float(t_val)])
        self.fh.flush()

    def close(self):
        self.fh.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_seed_list(seed_list: Optional[List[int]]) -> List[int]:
    if not seed_list:
        return []
    return [int(s) for s in seed_list]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def apply_overrides(base: dict, overrides) -> dict:
    """Return a copy of ``base`` with every non-None field of ``overrides`` applied."""
    out = dict(base)
    for f in fields(overrides):
        val = getattr(overrides, f.name)
        if val is not None:
            out[f.name] = val
    return out


def _coerce_mode(val, default: str) -> str:
    if val is None:
        return default
    if isinstance(val, list):
        return val[0] if val else default
    return val


def _set_default(d: dict, key: str, default):
    if key not in d or d[key] is None:
        d[key] = default
    return d[key]


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)


# =============================================================================
# 4. POSITIVE-WEIGHT COMPUTATION
# =============================================================================

def compute_pos_neg_counts_for_dataset(ds, concept_names) -> Tuple[torch.Tensor, torch.Tensor]:
    """Count valid positive/negative labels per concept across a dataset."""
    K = len(concept_names)
    pos = np.zeros(K, dtype=np.float64)
    neg = np.zeros(K, dtype=np.float64)

    if hasattr(ds, "_build_cls_tensors"):
        acc_col = "accession" if "accession" in ds.df.columns else getattr(ds, "acc_col", "accession")
        for i in range(len(ds)):
            acc = str(ds.df.iloc[i][acc_col])
            labels, mask = ds._build_cls_tensors(acc)
            y = labels.numpy(); m = mask.numpy().astype(bool)
            if y.shape[0] != K or m.shape[0] != K:
                raise ValueError(
                    f"CLS tensor length mismatch while computing pos_weight. "
                    f"Expected K={K}, got labels={y.shape[0]}, mask={m.shape[0]}."
                )
            pos += ((y == 1.0) & m)
            neg += ((y == 0.0) & m)
        return (torch.from_numpy(pos.astype(np.float32)), torch.from_numpy(neg.astype(np.float32)))

    use_cls_override = bool(getattr(ds, "use_cls_override", False) and getattr(ds, "cls_override_dir", None))
    if hasattr(ds, "preproc_dir") and not use_cls_override:
        sop_col = getattr(ds, "sop_col", "SOPInstanceUID")
        for i in range(len(ds)):
            sop = str(ds.df.iloc[i][sop_col])
            samp = torch.load(
                os.path.join(ds.preproc_dir, f"{sop}.pt"),
                map_location="cpu",
                weights_only=True,
            )
            if ("cls_labels" not in samp) or ("cls_mask" not in samp):
                continue
            y = samp["cls_labels"].numpy(); m = samp["cls_mask"].numpy().astype(bool)
            if y.shape[0] != K or m.shape[0] != K:
                raise ValueError(
                    f"Cached CLS tensor length mismatch while computing pos_weight. "
                    f"Expected K={K}, got labels={y.shape[0]}, mask={m.shape[0]}. "
                    "Regenerate cache or enable matching cls_override."
                )
            pos += ((y == 1.0) & m)
            neg += ((y == 0.0) & m)
        return (torch.from_numpy(pos.astype(np.float32)), torch.from_numpy(neg.astype(np.float32)))

    # Keep this compatibility fallback single-process: it is a one-time scan and
    # must also work in containers or systems without shared-memory workers.
    tmp_loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    for batch in tmp_loader:
        if ("cls_labels" not in batch) or ("cls_mask" not in batch):
            continue
        y = batch["cls_labels"].numpy(); m = batch["cls_mask"].numpy().astype(bool)
        if y.shape[1] != K or m.shape[1] != K:
            raise ValueError(
                f"Batched CLS tensor length mismatch while computing pos_weight. "
                f"Expected K={K}, got labels={y.shape[1]}, mask={m.shape[1]}."
            )
        pos += ((y == 1.0) & m).sum(axis=0)
        neg += ((y == 0.0) & m).sum(axis=0)
    return (torch.from_numpy(pos.astype(np.float32)), torch.from_numpy(neg.astype(np.float32)))


def load_pos_neg_counts_from_stats(
    stats_path: str,
    concept_names: List[str],
    split: str = "train",
    use_accessions: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read per-concept positive/negative counts from a class-statistics table."""
    df = _read_table(stats_path)
    if "split" in df.columns:
        # Fall back to all rows if the requested split is absent (the report
        # pipeline emits a single "all" split when no split file is provided).
        sel = df[df["split"].astype(str) == str(split)]
        df = sel if len(sel) > 0 else df
    if "has_cls_head" in df.columns:
        df = df[df["has_cls_head"].fillna(True).astype(bool)]

    concept_col = "concept_name" if "concept_name" in df.columns else None
    source_col = "cls_source_concept" if "cls_source_concept" in df.columns else None
    if concept_col is None and source_col is None:
        raise ValueError("Class-stats table needs either 'concept_name' or 'cls_source_concept'.")

    # Accept the training-style columns (n_samples_positive/negative or
    # n_accessions_*) or the report pipeline's label_stats columns (n_positive/n_negative).
    def _pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None
    if use_accessions:
        pos_col = _pick("n_accessions_positive", "n_positive")
        neg_col = _pick("n_accessions_negative", "n_negative")
    else:
        pos_col = _pick("n_samples_positive", "n_positive")
        neg_col = _pick("n_samples_negative", "n_negative")
    if pos_col is None or neg_col is None:
        raise ValueError(
            "Class-stats table missing positive/negative count columns "
            "(expected n_samples_positive/negative, n_accessions_*, or n_positive/n_negative). "
            f"Available columns: {list(df.columns)}"
        )

    pos_vals: List[float] = []
    neg_vals: List[float] = []
    for cname in concept_names:
        rows = pd.DataFrame()
        if concept_col is not None:
            rows = df[df[concept_col].astype(str) == str(cname)]
        if rows.empty and source_col is not None:
            rows = df[df[source_col].astype(str) == str(cname)]
        if rows.empty:
            raise ValueError(f"Concept '{cname}' not found in class-stats table: {stats_path}")
        if len(rows) > 1:
            rows = rows.iloc[:1]
        pos_vals.append(float(rows.iloc[0][pos_col]))
        neg_vals.append(float(rows.iloc[0][neg_col]))

    return (torch.tensor(pos_vals, dtype=torch.float32), torch.tensor(neg_vals, dtype=torch.float32))


# =============================================================================
# 5. CHECKPOINT LOADING
# =============================================================================

def _load_checkpoint(path: str) -> dict:
    """Return the model state dict from a checkpoint file (accepts a few formats)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def _safe_load_state(model: nn.Module, state: dict, reset_cls_heads: bool = False) -> None:
    """
    Load pretrained weights, tolerating (only) the classification-head keys being
    absent, reset, or shape-mismatched — everything else must match exactly.
    """
    model_state = model.state_dict()
    filtered, mismatched, skipped = {}, [], []

    for k, v in state.items():
        if reset_cls_heads and (k.startswith("cls_heads") or k.startswith("cls_head_map_idx")):
            skipped.append(k)
            continue
        if k not in model_state:
            skipped.append(k)
            continue
        if model_state[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    allowed_prefixes = ("cls_heads", "cls_head_map_idx")
    bad = [k for k, _, _ in mismatched if not k.startswith(allowed_prefixes)]
    if bad:
        raise ValueError(f"Checkpoint shape mismatch for non-cls keys: {bad}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if mismatched:
        print("Skipped mismatched keys (cls heads only):")
        for k, a, b in mismatched:
            print(f"  - {k}: ckpt {a} -> model {b}")
    if skipped:
        print(f"Skipped {len(skipped)} keys not loaded (missing in model or reset).")
    if missing:
        print(f"Missing keys after load (expected for new cls heads): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys after load: {len(unexpected)}")


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _count_cls_params(model: nn.Module) -> int:
    return 0 if model.cls_heads is None else sum(p.numel() for p in model.cls_heads.parameters())


def _list_non_cls_trainables(model: nn.Module) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls_heads")]


# =============================================================================
# 6. EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_losses_cls(model, dl, device, lcfg):
    """Average validation losses over the dataloader (classification only)."""
    model.eval()
    model.cfg.record_attn_maps = False
    sums = {"total": 0.0, "cls": 0.0, "align": 0.0}
    n = 0
    for batch in dl:
        imgs = batch["images"].to(device)
        if batch.get("cls_labels") is not None:
            batch["cls_labels"] = batch["cls_labels"].to(device)
        if batch.get("cls_mask") is not None:
            batch["cls_mask"] = batch["cls_mask"].to(device)
        outputs = model(images=imgs, need_attn=False)
        loss_dict = compute_losses(batch, outputs, model, lcfg)
        for k in sums:
            sums[k] += float(loss_dict.get(k, 0.0))
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def calibrate_and_evaluate(model, dl, device, concept_names):
    """Per-concept AUROC/AUPRC and best-F1 operating threshold."""
    model.eval()
    model.cfg.record_attn_maps = False
    K = len(concept_names)
    all_probs = [[] for _ in range(K)]
    all_targets = [[] for _ in range(K)]

    for batch in dl:
        imgs = batch["images"].to(device)
        out = model(images=imgs, need_attn=False)
        probs = torch.sigmoid(out["concept_logits"]).cpu().numpy()
        targets = batch["cls_labels"].numpy()
        mask = batch["cls_mask"].numpy().astype(bool)
        for k in range(K):
            m = mask[:, k]
            if m.any():
                all_probs[k].append(probs[m, k])
                all_targets[k].append(targets[m, k])

    results = {"auroc_per_concept": [], "auprc_per_concept": [],
               "best_f1_per_concept": [], "best_thresh_per_concept": []}
    for k in range(K):
        if not all_probs[k]:
            results["auroc_per_concept"].append(np.nan)
            results["auprc_per_concept"].append(np.nan)
            results["best_f1_per_concept"].append(0.0)
            results["best_thresh_per_concept"].append(0.5)
            continue
        y_score = np.concatenate(all_probs[k])
        y_true = np.concatenate(all_targets[k]).astype(int)
        if len(np.unique(y_true)) < 2:
            results["auroc_per_concept"].append(np.nan)
            results["auprc_per_concept"].append(np.nan)
            results["best_f1_per_concept"].append(0.0)
            results["best_thresh_per_concept"].append(0.5)
            continue
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(y_true, y_score)
        except Exception:
            auroc = np.nan
        auprc = average_precision_score(y_true, y_score)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        results["auroc_per_concept"].append(auroc)
        results["auprc_per_concept"].append(auprc)
        results["best_f1_per_concept"].append(best_f1)
        results["best_thresh_per_concept"].append(best_thresh)
    return results


# =============================================================================
# 7. CONFIG LOADING (YAML -> dataclasses)
# =============================================================================

def load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _apply_section(obj, section: Optional[dict], section_name: str):
    """Override dataclass fields from a config sub-dict (lists coerced to tuples where needed)."""
    if not section:
        return
    valid = {f.name: f for f in fields(obj)}
    for key, value in section.items():
        if key not in valid:
            print(f"[config] WARNING: unknown key '{section_name}.{key}' ignored.")
            continue
        if isinstance(getattr(obj, key), tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)


def build_configs(config_path: Optional[str], cli_overrides: dict):
    pre = PretrainCfg()
    paths_override = PathsOverride()
    data_override = DataOverride()
    train_cfg = TrainCfg()
    loss_settings = LossSettings()

    if config_path:
        cfg = load_yaml_config(config_path)
        _apply_section(pre, cfg.get("pretrain"), "pretrain")
        _apply_section(paths_override, cfg.get("paths"), "paths")
        _apply_section(data_override, cfg.get("data"), "data")
        _apply_section(train_cfg, cfg.get("train"), "train")
        _apply_section(loss_settings, cfg.get("losses"), "losses")

    if cli_overrides.get("device") is not None:
        train_cfg.device = cli_overrides["device"]
    if cli_overrides.get("epochs") is not None:
        train_cfg.epochs = cli_overrides["epochs"]
    if cli_overrides.get("seed") is not None:
        train_cfg.seed = cli_overrides["seed"]
    if cli_overrides.get("run_dir") is not None:
        paths_override.run_dir = cli_overrides["run_dir"]
    if cli_overrides.get("ckpt_path") is not None:
        pre.ckpt_path = cli_overrides["ckpt_path"]
    if cli_overrides.get("config_path") is not None:
        pre.config_path = cli_overrides["config_path"]

    return pre, paths_override, data_override, train_cfg, loss_settings


def parse_args():
    ap = argparse.ArgumentParser(description="RadPRISM classification fine-tuning.")
    ap.add_argument("--config", type=str, default=None, help="Path to config/finetune_cls.yaml.")
    ap.add_argument("--device", type=str, default=None, help="Override train.device.")
    ap.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    ap.add_argument("--seed", type=int, default=None, help="Override train.seed.")
    ap.add_argument("--run-dir", type=str, default=None, help="Override paths.run_dir.")
    ap.add_argument("--ckpt-path", type=str, default=None, help="Override pretrain.ckpt_path.")
    ap.add_argument("--config-path", type=str, default=None, help="Override pretrain.config_path.")
    return ap.parse_args()


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    args = parse_args()
    cli_overrides = {
        "device": args.device, "epochs": args.epochs, "seed": args.seed,
        "run_dir": args.run_dir, "ckpt_path": args.ckpt_path, "config_path": args.config_path,
    }
    pre, paths_override, data_override, train_cfg, loss_settings = build_configs(args.config, cli_overrides)

    if not pre.ckpt_path or not pre.config_path:
        raise ValueError("pretrain.ckpt_path and pretrain.config_path must be set (in the YAML or via CLI).")

    # ── Multi-seed sweep: launch one child process per seed ─────────────────────
    seed_list = _normalize_seed_list(train_cfg.seed_list)
    is_seed_worker = os.environ.get("VCM_SINGLE_SEED") is not None
    if seed_list and not is_seed_worker:
        script_path = os.path.abspath(__file__)
        for idx, seed in enumerate(seed_list, start=1):
            print(f"[Seed sweep] Launching run {idx}/{len(seed_list)} with train.seed={seed}")
            env = os.environ.copy()
            env["VCM_SINGLE_SEED"] = str(seed)
            cmd = [sys.executable, script_path]
            if args.config:
                cmd += ["--config", args.config]
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                raise RuntimeError(f"Seed sweep failed for seed={seed} (return code {proc.returncode})")
        print("[Seed sweep] All runs completed successfully.")
        return
    if is_seed_worker:
        train_cfg.seed = int(os.environ["VCM_SINGLE_SEED"])
        print(f"[Seed sweep] Worker run with train.seed={train_cfg.seed}")

    # ── Inherit configuration from the pretrained run ───────────────────────────
    with open(pre.config_path, "r") as f:
        pre_cfg = json.load(f)

    align_concept_names, cls_concept_names, cls_head_map = _resolve_concepts(pre_cfg)
    if cls_head_map is None:
        cls_head_map = _infer_cls_head_map(align_concept_names, cls_concept_names)

    # Merge inherited paths/data with the YAML overrides.
    paths = apply_overrides(pre_cfg.get("paths", {}), paths_override)
    data = apply_overrides(pre_cfg.get("data", {}), data_override)

    # Fill any gaps for older configs.
    _set_default(paths, "run_dir", paths_override.run_dir)
    _set_default(data, "batch_size", 128)
    _set_default(data, "effective_batch", data["batch_size"])
    _set_default(data, "num_workers", 8)
    _set_default(data, "image_size", 518)
    _set_default(data, "augment", True)
    _set_default(data, "norm_mode", "rad_dino_maira_2")
    _set_default(data, "val_frac", 0.1)
    _set_default(data, "test_frac", 0.1)
    _set_default(data, "subsample_seed", 1337)
    _set_default(data, "split_on", "patient")
    _set_default(data, "accession_column", "accession")
    _set_default(data, "sop_uid_column", "SOPInstanceUID")

    model_cfg = dict(pre_cfg.get("model", {}))
    # Fine-tuning always trains the classification heads; persist that in the run
    # config so downstream evaluation scripts know the heads are present.
    model_cfg["use_cls_heads"] = True

    # ── Run directory & logging ─────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_label = f"finetune_cls_{timestamp}_seed{train_cfg.seed}" if is_seed_worker else f"finetune_cls_{timestamp}"
    run_dir = os.path.join(paths["run_dir"], run_label)
    ensure_dir(run_dir)

    full_cfg = {
        "pretrain": asdict(pre),
        "paths": paths,
        "data": data,
        "model": model_cfg,
        "train": asdict(train_cfg),
        "losses": asdict(loss_settings),
        "align_concept_names": align_concept_names,
        "cls_concept_names": cls_concept_names,
        "cls_head_map": cls_head_map,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(full_cfg, f, indent=2)

    metrics_logger = CSVScalarLogger(os.path.join(run_dir, "metrics.csv"))
    concept_logger = CSVPerConceptLogger(os.path.join(run_dir, "per_concept_metrics.csv"))
    tb_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb")) if SummaryWriter else None

    # ── Seed & device ───────────────────────────────────────────────────────────
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ── Data ────────────────────────────────────────────────────────────────────
    print(f"Loading master index: {paths['master_index_path']}")
    master = (pd.read_csv(paths["master_index_path"])
              if paths["master_index_path"].endswith(".csv")
              else pd.read_parquet(paths["master_index_path"]))

    if paths.get("split_path") is None:
        paths["split_path"] = os.path.join(run_dir, "splits.parquet")
        create_group_splits(master, paths["split_path"], val_frac=data["val_frac"],
                            test_frac=data["test_frac"], split_on=data["split_on"])

    train_df = apply_group_splits(master, paths["split_path"], split_on=data["split_on"], split="train")
    val_df = apply_group_splits(master, paths["split_path"], split_on=data["split_on"], split="val")

    if data.get("max_train"):
        train_df = train_df.sample(n=data["max_train"], random_state=data["subsample_seed"])
    if data.get("max_val"):
        val_df = val_df.sample(n=data["max_val"], random_state=data["subsample_seed"])

    cached_path = paths.get("cached_dataset_path")
    if cached_path is not None and os.path.exists(cached_path):
        print(f"Using cached (sharded) image store from {cached_path}")
        text_override_mode = _coerce_mode(data.get("text_override_mode"), "sharded")
        cls_override_mode = _coerce_mode(data.get("cls_override_mode"), "sharded")
        common_kwargs = dict(
            text_concept_names=align_concept_names,
            cls_concept_names=cls_concept_names,
            cached_dataset_path=cached_path,
            image_size=data["image_size"],
            norm_mode=data["norm_mode"],
            include_cls_labels=True,
            accession_column=data["accession_column"],
            sop_uid_column=data["sop_uid_column"],
            affine_degrees=data.get("affine_degrees", 5),
            affine_translate=data.get("affine_translate", (0.02, 0.02)),
            cornercutout_size=data.get("cornercutout_size", (0.2, 0.25)),
            cornercutout_probs=data.get("cornercutout_probs", None),
            cornercutout_fill=data.get("cornercutout_fill", 0.0),
            text_override_dir=data.get("text_override_dir"),
            use_text_override=data.get("use_text_override", False),
            text_override_cache_size=data.get("text_override_cache_size", 64),
            text_override_strict=data.get("text_override_strict", True),
            text_override_mode=text_override_mode,
            text_override_index_cache_name=data.get("text_override_index_cache_name", "sharded_index.parquet"),
            text_override_shard_memmap_cache_size=data.get("text_override_shard_memmap_cache_size", 4),
            cls_override_dir=data.get("cls_override_dir"),
            use_cls_override=data.get("use_cls_override", False),
            cls_override_cache_size=data.get("cls_override_cache_size", 64),
            cls_override_strict=data.get("cls_override_strict", True),
            cls_override_mode=cls_override_mode,
            cls_override_index_cache_name=data.get("cls_override_index_cache_name", "cls_sharded_index.parquet"),
            cls_override_shard_memmap_cache_size=data.get("cls_override_shard_memmap_cache_size", 4),
        )
        cols = [data["accession_column"], data["sop_uid_column"]]
        ds_train = PTCachedCXRDataset(train_df[cols].copy(), align_concept_names, augment=data["augment"], **common_kwargs)
        ds_val = PTCachedCXRDataset(val_df[cols].copy(), align_concept_names, augment=False, **common_kwargs)
    else:
        print("Building datasets from scratch (reading PNGs + embeddings on the fly)...")
        common_kwargs = dict(
            text_concept_names=align_concept_names, cls_concept_names=cls_concept_names,
            embedding_dir=paths["embedding_dir"], label_dir=paths["label_dir"],
            include_cls_labels=True, image_size=data["image_size"], norm_mode=data["norm_mode"],
            affine_degrees=data.get("affine_degrees", 5),
            affine_translate=data.get("affine_translate", (0.02, 0.02)),
            cornercutout_size=data.get("cornercutout_size", (0.2, 0.25)),
            cornercutout_probs=data.get("cornercutout_probs", None),
            cornercutout_fill=data.get("cornercutout_fill", 0.0),
        )
        ds_train = CXRMultimodalDataset(train_df, align_concept_names, augment=data["augment"], **common_kwargs)
        ds_val = CXRMultimodalDataset(val_df, align_concept_names, augment=False, **common_kwargs)

    dl_train = make_dataloader(
        ds_train, data["batch_size"], shuffle=True, num_workers=data["num_workers"],
        prefetch_factor=data.get("dataloader_prefetch_factor", 8),
        persistent_workers=data.get("dataloader_persistent_workers", False),
        pin_memory=data.get("dataloader_pin_memory", True), drop_last=True,
    )
    dl_val = make_dataloader(
        ds_val, max(32, data["batch_size"] // 2), shuffle=False, num_workers=data["num_workers"],
        prefetch_factor=data.get("dataloader_prefetch_factor", 8),
        persistent_workers=data.get("dataloader_persistent_workers", False),
        pin_memory=data.get("dataloader_pin_memory", True), drop_last=False,
    )
    print(f"Training samples:   {len(ds_train)}")
    print(f"Validation samples: {len(ds_val)}")

    # ── Model (architecture inherited from the pretrained config) ───────────────
    vision_weights_path = model_cfg.get("vision_weights_path", None) or None
    mcfg = ModelConfig(
        concept_names=align_concept_names,
        cls_concept_names=cls_concept_names,
        cls_head_map=cls_head_map,
        use_cls_heads=True,
        d_model=model_cfg.get("d_model", 540),
        vision_backbone=model_cfg.get("vision_backbone", "rad_dino_maira_2"),
        pretrained_vision=model_cfg.get("pretrained_vision", False),
        vision_weights_path=vision_weights_path,
        project_text=model_cfg.get("project_text", True),
        text_in_dim=model_cfg.get("text_in_dim", 768),
        n_heads=model_cfg.get("n_heads", 12),
        dropout=model_cfg.get("dropout", 0.1),
        dinov2_repo_path=model_cfg.get("dinov2_repo_path", None),
        dinov2_weights_path=model_cfg.get("dinov2_weights_path", None),
        rad_dino_model_dir=model_cfg.get("rad_dino_model_dir", None),
    )
    model = VisionConceptModel(mcfg).to(device)

    # Load pretrained weights (cls heads may be reset).
    state = _load_checkpoint(pre.ckpt_path)
    _safe_load_state(model, state, reset_cls_heads=pre.reset_cls_heads)

    # Freeze everything except the classification heads.
    for p in model.parameters():
        p.requires_grad = False
    if model.cls_heads is None:
        raise ValueError("cls_heads missing. Ensure use_cls_heads=True and cls_concept_names are set.")
    for p in model.cls_heads.parameters():
        p.requires_grad = True

    total_params, trainable_params = _count_params(model)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"CLS head parameters:  {_count_cls_params(model):,}")
    non_cls_trainables = _list_non_cls_trainables(model)
    if non_cls_trainables:
        print("WARNING: Non-CLS parameters are trainable:")
        for n in non_cls_trainables:
            print(f"  - {n}")
    else:
        print("Trainable parameters restricted to cls_heads only.")

    # Keep the frozen backbone in eval mode (no BatchNorm updates / dropout noise);
    # only the classification heads train.
    model.eval()
    model.cls_heads.train()

    optim = torch.optim.AdamW(model.cls_heads.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scaler = make_grad_scaler(train_cfg.bf16)

    # ── Positive-class weights for BCE ──────────────────────────────────────────
    pos_weight_tensor = None
    pos_counts_tensor = None
    neg_counts_tensor = None
    expected_k = len(cls_concept_names)

    if loss_settings.class_stats_path:
        if not os.path.exists(loss_settings.class_stats_path):
            raise FileNotFoundError(f"class_stats_path not found: {loss_settings.class_stats_path}")
        pos_counts_tensor, neg_counts_tensor = load_pos_neg_counts_from_stats(
            stats_path=loss_settings.class_stats_path,
            concept_names=cls_concept_names,
            split=loss_settings.class_stats_split,
            use_accessions=bool(loss_settings.class_stats_use_accessions),
        )
        if pos_counts_tensor.numel() != expected_k or neg_counts_tensor.numel() != expected_k:
            raise ValueError(
                f"class-stats counts have wrong length: pos={pos_counts_tensor.numel()}, "
                f"neg={neg_counts_tensor.numel()}, expected={expected_k}"
            )
        pos_weight_tensor = (neg_counts_tensor + 1.0) / (pos_counts_tensor + 1.0)
        print(f"Loaded class counts from {loss_settings.class_stats_path} "
              f"(split={loss_settings.class_stats_split}, use_accessions={bool(loss_settings.class_stats_use_accessions)})")

    if pos_weight_tensor is None and loss_settings.pos_weight_path and os.path.exists(loss_settings.pos_weight_path):
        print(f"Loading pos_weights from {loss_settings.pos_weight_path}")
        pos_weight_tensor = torch.load(
            loss_settings.pos_weight_path,
            map_location=device,
            weights_only=True,
        ).float().reshape(-1)
        if pos_weight_tensor.numel() != expected_k:
            print(f"[WARN] Loaded pos_weight length {pos_weight_tensor.numel()} != expected {expected_k}. Recomputing.")
            pos_weight_tensor = None

    if pos_weight_tensor is None:
        pos_counts_tensor, neg_counts_tensor = compute_pos_neg_counts_for_dataset(ds_train, cls_concept_names)
        pos_weight_tensor = (neg_counts_tensor + 1.0) / (pos_counts_tensor + 1.0)
        save_path = os.path.join(run_dir, "pos_weights.pt")
        torch.save(pos_weight_tensor, save_path)
        print(f"Computed and saved pos_weights to {save_path}")

    if loss_settings.use_tempered_pos_weight:
        if (pos_counts_tensor is None) or (neg_counts_tensor is None):
            if loss_settings.tempered_prior_smoothing_alpha > 0:
                print("[WARN] Tempered prior smoothing requested but counts are unavailable from the loaded "
                      "pos_weight. Falling back to power tempering on the loaded weights only.")
            pos_weight_tensor = pos_weight_tensor.float().pow(float(loss_settings.tempered_pos_weight_gamma))
        else:
            pos_weight_tensor = build_tempered_pos_weight_from_counts(
                pos_counts=pos_counts_tensor,
                neg_counts=neg_counts_tensor,
                gamma=float(loss_settings.tempered_pos_weight_gamma),
                prior_smoothing_alpha=float(loss_settings.tempered_prior_smoothing_alpha),
                global_pos_prior=loss_settings.tempered_global_pos_prior,
            )
        print(f"Applied tempered pos weighting: gamma={loss_settings.tempered_pos_weight_gamma}, "
              f"prior_alpha={loss_settings.tempered_prior_smoothing_alpha}, "
              f"global_prior={loss_settings.tempered_global_pos_prior}")

    if loss_settings.max_pos_weight > 0:
        print(f"Clipping pos_weights to max {loss_settings.max_pos_weight}")
        pos_weight_tensor = torch.clamp(pos_weight_tensor, max=loss_settings.max_pos_weight)

    print("Effective pos weights per concept:")
    for cname, w in zip(cls_concept_names, pos_weight_tensor):
        print(f"  - {cname}: {float(w):.4f}")
    pos_weight_tensor = pos_weight_tensor.to(device)

    lcfg = LossConfig(
        use_cls_loss=True,
        use_align_loss=False,
        w_cls=loss_settings.w_cls,
        pos_weight=pos_weight_tensor,
        cls_concept_balanced_reduction=loss_settings.use_concept_balanced_reduction,
    )

    # ── Initial evaluation ──────────────────────────────────────────────────────
    print("Running initial evaluation (epoch 0)...")
    init_val_losses = evaluate_losses_cls(model, dl_val, device, lcfg)
    metrics_logger.log_many(0, 0, "val", {f"loss/{k}": v for k, v in init_val_losses.items()})
    init_calib_res = calibrate_and_evaluate(model, dl_val, device, cls_concept_names)
    init_macro_auroc = np.nanmean(init_calib_res["auroc_per_concept"])
    init_macro_auprc = np.nanmean(init_calib_res["auprc_per_concept"])
    metrics_logger.log(0, 0, "val", "macro_auroc", init_macro_auroc)
    metrics_logger.log(0, 0, "val", "macro_auprc", init_macro_auprc)
    if tb_writer:
        for k, v in init_val_losses.items():
            tb_writer.add_scalar(f"val/loss/{k}", v, 0)
        tb_writer.add_scalar("val/macro_auroc", init_macro_auroc, 0)
        tb_writer.add_scalar("val/macro_auprc", init_macro_auprc, 0)
    print(f"Initial eval | Loss: {init_val_losses['total']:.4f} | "
          f"AUROC: {init_macro_auroc:.4f} | AUPRC: {init_macro_auprc:.4f}")

    # =========================================================================
    # TRAINING LOOP (classification heads only)
    # =========================================================================
    best_scores = {"cls": -1.0}
    accum = max(1, data["effective_batch"] // data["batch_size"])
    total_steps = 0
    start_time = time.time()
    print(f"\nStarting fine-tuning at {time.ctime(start_time)}")

    for epoch in range(1, train_cfg.epochs + 1):
        # Backbone stays frozen/eval; only the cls heads train.
        model.eval()
        model.cls_heads.train()
        running_loss = {}
        t0 = time.time()

        for i, batch in enumerate(dl_train):
            total_steps += 1

            # LR schedule (linear warmup then cosine decay)
            warmup = train_cfg.warmup_steps
            if total_steps < warmup:
                lr = train_cfg.lr * (total_steps / warmup)
            else:
                progress = (total_steps - warmup) / max(1, (len(dl_train) * train_cfg.epochs) - warmup)
                lr = train_cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optim.param_groups:
                pg["lr"] = lr

            imgs = batch["images"].to(device, non_blocking=True)
            if batch.get("cls_labels") is not None:
                batch["cls_labels"] = batch["cls_labels"].to(device, non_blocking=True)
            if batch.get("cls_mask") is not None:
                batch["cls_mask"] = batch["cls_mask"].to(device, non_blocking=True)

            with bf16_autocast(train_cfg.bf16):
                outputs = model(images=imgs, need_attn=False)
                loss_dict = compute_losses(batch, outputs, model, lcfg)
                loss = loss_dict["total"] / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                if train_cfg.clip_grad_norm:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.cls_heads.parameters(), train_cfg.clip_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            for k, v in loss_dict.items():
                running_loss[k] = running_loss.get(k, 0.0) + float(v)

            if total_steps % train_cfg.log_every == 0:
                avg = {k: v / train_cfg.log_every for k, v in running_loss.items()}
                metrics_logger.log_many(total_steps, epoch, "train", avg)
                metrics_logger.log(total_steps, epoch, "train", "lr", lr)
                if tb_writer:
                    tb_writer.add_scalar("train/lr", lr, total_steps)
                    for k, v in avg.items():
                        tb_writer.add_scalar(f"train/loss/{k}", v, total_steps)
                running_loss = {}

        # ── Validation & checkpointing ──────────────────────────────────────────
        val_losses = evaluate_losses_cls(model, dl_val, device, lcfg)
        metrics_logger.log_many(total_steps, epoch, "val", {f"loss/{k}": v for k, v in val_losses.items()})

        calib_res = calibrate_and_evaluate(model, dl_val, device, cls_concept_names)
        macro_auroc = np.nanmean(calib_res["auroc_per_concept"])
        macro_auprc = np.nanmean(calib_res["auprc_per_concept"])
        macro_f1 = np.nanmean(calib_res["best_f1_per_concept"])
        metrics_logger.log(total_steps, epoch, "val", "macro_auroc", macro_auroc)
        metrics_logger.log(total_steps, epoch, "val", "macro_auprc", macro_auprc)
        metrics_logger.log(total_steps, epoch, "val", "macro_f1_opt", macro_f1)

        if tb_writer:
            for k, v in val_losses.items():
                tb_writer.add_scalar(f"val/loss/{k}", v, total_steps)
            tb_writer.add_scalar("val/macro_auroc", macro_auroc, total_steps)
            tb_writer.add_scalar("val/macro_auprc", macro_auprc, total_steps)
            tb_writer.add_scalar("val/macro_f1_opt", macro_f1, total_steps)
            for k, name in enumerate(cls_concept_names):
                auprc_val = calib_res["auprc_per_concept"][k]
                if not np.isnan(auprc_val):
                    tb_writer.add_scalar(f"val/concept_auprc/{name}", auprc_val, total_steps)

        print(f"Epoch {epoch} | Val Loss: {val_losses['total']:.4f} | "
              f"Macro AUROC: {macro_auroc:.4f} | Macro AUPRC: {macro_auprc:.4f} | Macro F1: {macro_f1:.4f}")
        for k, name in enumerate(cls_concept_names):
            concept_logger.log_concept(epoch, name, "auroc", calib_res["auroc_per_concept"][k])
            concept_logger.log_concept(epoch, name, "auprc", calib_res["auprc_per_concept"][k])
            concept_logger.log_concept(epoch, name, "best_f1", calib_res["best_f1_per_concept"][k],
                                       threshold=calib_res["best_thresh_per_concept"][k])
        print(f"Epoch duration: {(time.time() - t0) / 60:.2f} minutes")

        if macro_auprc > best_scores["cls"]:
            best_scores["cls"] = macro_auprc
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "config": full_cfg,
                "best_metric": "Macro AUPRC", "best_score": float(macro_auprc),
                "thresholds": {n: float(t) for n, t in zip(cls_concept_names, calib_res["best_thresh_per_concept"])},
            }, os.path.join(run_dir, "best_cls.pt"))
            print(f"  [+] Saved best_cls.pt (AUPRC: {macro_auprc:.4f})")

        if train_cfg.save_every_epochs > 0 and (epoch % train_cfg.save_every_epochs) == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": full_cfg},
                       os.path.join(run_dir, f"epoch_{epoch:03d}.pt"))

    metrics_logger.close()
    concept_logger.close()
    if tb_writer:
        tb_writer.close()
    print("\nFine-tuning complete.")
    print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
