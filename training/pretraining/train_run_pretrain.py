#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chest X-ray Vision-Language Concept-Alignment Pretraining
=========================================================

Pretrains the :class:`VisionConceptModel` with a concept-averaged InfoNCE
objective: for every clinical concept the model learns to align its
image-derived concept embedding with the pre-computed text embedding of the
matching radiology-report snippet.

Because the different clinical concepts appear in the reports with very
different frequencies, the training loop supports *concept-aware weighted
sampling* (referred to as "Strategy B"): images whose report covers rare
concepts are oversampled so that those concepts still form enough positive
pairs per batch. Weights are computed once from a concept text-coverage table
and cached to disk.

Configuration
-------------
All settings live in a single YAML file (see ``config/pretrain.yaml``). The
defaults below (the ``@dataclass`` blocks) document every option; the YAML file
overrides them per section. A few high-traffic options can additionally be
overridden on the command line:

    python train_run_pretrain.py --config config/pretrain.yaml
    python train_run_pretrain.py --config config/pretrain.yaml --device cuda:0 --epochs 4

Outputs (checkpoints, metrics CSVs, TensorBoard logs, resolved config.json) are
written to a timestamped sub-directory of ``paths.run_dir``.
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
from typing import Optional, List, Dict

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

# Make the shared source modules (model/losses/dataset) importable regardless of
# the current working directory. src/ lives at the repository root:
#   RadPRISM/src/  <-  RadPRISM/training/pretraining/train_run_pretrain.py
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from model import ModelConfig, VisionConceptModel               # noqa: E402
from losses import LossConfig, compute_losses                   # noqa: E402
from dataset import (                                            # noqa: E402
    CXRMultimodalDataset, PTCachedCXRDataset, make_dataloader,
    create_group_splits, apply_group_splits,
    compute_concept_sample_weights,
)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class Paths:
    """Filesystem locations for inputs and outputs."""
    master_index_path: str = ""            # CSV/parquet master index (one row per image)
    embedding_dir: str = ""                # pre-computed report text embeddings
    label_dir: Optional[str] = None        # structured report labels (only for the cls head)
    run_dir: str = "output/runs"           # parent directory for run outputs
    split_path: Optional[str] = None       # parquet with train/val/test assignment; created if None
    cached_dataset_path: Optional[str] = None  # sharded/pre-cached image tensors (fast path)


@dataclass
class DataCfg:
    """Dataset, augmentation and dataloader settings."""
    image_size: int = 518
    augment: bool = True
    norm_mode: str = "rad_dino_maira_2"
    batch_size: int = 256
    effective_batch: int = 256             # gradient-accumulated batch size
    num_workers: int = 4
    max_train: Optional[int] = None        # subsample the train split (debugging)
    max_val: Optional[int] = None          # subsample the val split (debugging)
    subsample_seed: int = 1337

    # Split creation (used only when paths.split_path is None)
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_on: str = "patient"              # "patient" | "accession" | "image"
    accession_column: str = "accession"
    sop_uid_column: str = "SOPInstanceUID"

    # Text-embedding override store (sharded reader keyed by accession)
    text_override_dir: Optional[str] = None
    use_text_override: bool = True
    text_override_cache_size: int = 512
    text_override_strict: bool = True
    text_override_mode: str = "sharded"

    # Classification-label override store (only needed when use_cls_loss=True)
    cls_override_dir: Optional[str] = None
    use_cls_override: bool = True
    cls_override_cache_size: int = 1024
    cls_override_strict: bool = True
    cls_override_mode: str = "sharded"
    cls_override_index_cache_name: str = "cls_sharded_index.parquet"
    cls_override_shard_memmap_cache_size: int = 12

    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True

    # Augmentation parameters
    affine_degrees: float = 5
    affine_translate: tuple = (0.02, 0.02)
    cornercutout_size: tuple = (0.2, 0.25)
    cornercutout_probs: dict = field(default_factory=lambda: {
        "tl": 0.7, "tr": 0.85, "bl": 0.0, "br": 0.0
    })
    cornercutout_fill: float = 0.0

    # Concept selection. By default the model is trained on all fine concepts.
    # Optionally provide an explicit subset for the classification head and/or
    # extra alignment-only concepts.
    cls_concept_names: Optional[List[str]] = None       # None -> all FINE_CONCEPT_NAMES
    align_extra_concepts: List[str] = field(default_factory=lambda: [])


@dataclass
class ModelCfg:
    """Vision backbone and concept-model architecture."""
    vision_backbone: str = "rad_dino_maira_2"   # "resnet50" | "dinov2" | "rad_dino_maira_2"
    rad_dino_model_dir: Optional[str] = None    # local HF dir for rad_dino_maira_2
    dinov2_repo_path: Optional[str] = None       # local repo for dinov2
    dinov2_weights_path: Optional[str] = None    # local weights for dinov2
    d_model: int = 540
    n_heads: int = 12
    dropout: float = 0.1
    text_in_dim: int = 768
    project_text: bool = True
    pretrained_vision: bool = False
    vision_weights_path: Optional[str] = None
    use_cls_heads: bool = False                  # enable per-concept classification heads


@dataclass
class TrainCfg:
    """Optimization schedule and runtime settings."""
    epochs: int = 8
    freeze_backbone_epochs: int = 8              # keep the backbone frozen for N epochs
    lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    log_every: int = 10
    save_every_epochs: int = 5
    bf16: bool = True
    device: str = "cuda:0"
    clip_grad_norm: float = 1.0
    seed: int = 1337
    # If non-empty, launch one child process per seed (reproducibility sweep).
    seed_list: Optional[List[int]] = field(default_factory=lambda: [])


@dataclass
class LossSettings:
    """Which losses are active and their hyper-parameters."""
    use_cls_loss: bool = False                   # auxiliary BCE classification head
    use_align_loss: bool = True                  # concept alignment (InfoNCE), the main objective

    # Classification (BCE) — only used when use_cls_loss=True
    pos_weight_path: Optional[str] = None        # optional pre-computed [K] pos-weights
    max_pos_weight: float = 40.0

    # Loss weights
    w_cls: float = 0.1
    w_align: float = 1.0

    # Concept alignment
    concept_averaged_align: bool = True          # macro-average over concepts (recommended)
    tau_align: float = 0.07                       # temperature (init value if learnable)
    learnable_concept_taus: bool = True          # learn one temperature per concept
    min_tau_align: Optional[float] = 0.02
    max_tau_align: Optional[float] = 0.12
    symmetric_align: bool = True


@dataclass
class StrategyCfg:
    """
    Concept-aware weighted sampling ("Strategy B").

    When enabled, the training DataLoader uses a WeightedRandomSampler instead of
    a uniform shuffle. Images whose report covers rare concepts (text coverage
    below ``rare_threshold``) receive an upsampling bonus so those concepts form
    enough in-batch positive pairs for the InfoNCE loss.
    """
    use_weighted_sampling: bool = True

    # CSV with columns [split, concept_name, pct_samples_with_text]. When present
    # the per-concept coverage is read directly (fast); otherwise it is estimated
    # by scanning the sharded embedding index.
    weighted_sampling_coverage_csv: Optional[str] = None
    weighted_sampling_split: str = "train"

    # Concepts with coverage < rare_threshold get an upsampling bonus.
    weighted_sampling_rare_threshold: float = 0.10
    # rarity_mult_c = max(0, target_coverage / coverage_c - 1).
    weighted_sampling_target_coverage: float = 0.10

    # Where to cache the computed per-sample weights (.npy). Reused on later runs.
    weighted_sampling_cache_path: Optional[str] = None


# =============================================================================
# 2. CONCEPT NAMES
# =============================================================================

# The 19 fine-grained clinical concepts (dot-separated report-JSON paths).
# These must match the concept names produced by the report_struct_label pipeline.
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
        raise ValueError(f"{what} contains duplicates: {names}")


def _merge_unique_preserve_order(base: List[str], extras: List[str]) -> List[str]:
    out, seen = [], set()
    for name in list(base) + list(extras):
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def build_name_based_cls_head_map(cls_names: List[str], align_names: List[str]) -> List[int]:
    """Map each classification concept to the index of its alignment concept token."""
    _ensure_unique(cls_names, "cls_concept_names")
    _ensure_unique(align_names, "align_concept_names")
    align_idx = {name: i for i, name in enumerate(align_names)}
    out = []
    for cname in cls_names:
        if cname not in align_idx:
            raise ValueError(f"CLS concept '{cname}' not found in align_concept_names.")
        out.append(align_idx[cname])
    return out


# =============================================================================
# 3. LOGGING HELPERS
# =============================================================================

class CSVScalarLogger:
    """Append-only CSV logger for scalar metrics (step/epoch/split/metric/value)."""
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


def make_optimizer(model, lr, weight_decay, extra_params: Optional[List[torch.nn.Parameter]] = None):
    """AdamW with weight decay only on >=2D tensors (no decay on biases / norms / taus)."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    if extra_params:
        nodecay_params.extend([p for p in extra_params if p is not None and p.requires_grad])
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr)


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


def compute_pos_weight_for_dataset(ds, concept_names) -> torch.Tensor:
    """Compute per-concept BCE pos-weights ((neg+1)/(pos+1)) over a dataset."""
    K = len(concept_names)
    pos = np.zeros(K, dtype=np.float64)
    neg = np.zeros(K, dtype=np.float64)

    if hasattr(ds, "_build_cls_tensors"):
        acc_col = "accession" if "accession" in ds.df.columns else getattr(ds, "acc_col", "accession")
        for i in range(len(ds)):
            acc = str(ds.df.iloc[i][acc_col])
            labels, mask = ds._build_cls_tensors(acc)
            y = labels.numpy(); m = mask.numpy().astype(bool)
            pos += ((y == 1.0) & m)
            neg += ((y == 0.0) & m)
        return torch.from_numpy(((neg + 1.0) / (pos + 1.0)).astype(np.float32))

    use_cls_override = bool(getattr(ds, "use_cls_override", False) and getattr(ds, "cls_override_dir", None))
    if hasattr(ds, "preproc_dir") and not use_cls_override:
        for i in range(len(ds)):
            sop = str(ds.df.iloc[i][getattr(ds, "sop_col", "SOPInstanceUID")])
            p = os.path.join(ds.preproc_dir, f"{sop}.pt")
            samp = torch.load(p, map_location="cpu", weights_only=True)
            if "cls_labels" not in samp or "cls_mask" not in samp:
                continue
            y = samp["cls_labels"].numpy(); m = samp["cls_mask"].numpy().astype(bool)
            pos += ((y == 1.0) & m); neg += ((y == 0.0) & m)
        return torch.from_numpy(((neg + 1.0) / (pos + 1.0)).astype(np.float32))

    # Keep this compatibility fallback single-process: it is a one-time scan and
    # must also work in containers or systems without shared-memory workers.
    tmp_loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    for batch in tmp_loader:
        if "cls_labels" not in batch or "cls_mask" not in batch:
            continue
        y = batch["cls_labels"].numpy(); m = batch["cls_mask"].numpy().astype(bool)
        pos += ((y == 1.0) & m).sum(axis=0)
        neg += ((y == 0.0) & m).sum(axis=0)
    return torch.from_numpy(((neg + 1.0) / (pos + 1.0)).astype(np.float32))


# =============================================================================
# 4. EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_losses(model, dl, device, lcfg):
    """Average validation losses over the whole dataloader."""
    model.eval()
    model.cfg.record_attn_maps = False
    sums = {"total": 0.0, "cls": 0.0, "align": 0.0}
    n = 0
    for batch in dl:
        imgs = batch["images"].to(device)
        text_emb = batch.get("text_emb"); text_mask = batch.get("text_mask")
        if text_emb is not None:
            text_emb = text_emb.to(device)
        if text_mask is not None:
            text_mask = text_mask.to(device)
        if batch.get("cls_labels") is not None:
            batch["cls_labels"] = batch["cls_labels"].to(device)
        if batch.get("cls_mask") is not None:
            batch["cls_mask"] = batch["cls_mask"].to(device)
        outputs = model(images=imgs, concept_text_emb=text_emb, concept_text_mask=text_mask,
                        need_attn=False)
        loss_dict = compute_losses(batch, outputs, model, lcfg)
        for k in sums:
            sums[k] += float(loss_dict.get(k, 0.0))
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def calibrate_and_evaluate(model, dl, device, concept_names):
    """Per-concept AUROC/AUPRC and best-F1 threshold (only meaningful with cls heads)."""
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
# 5. CONFIG LOADING (YAML -> dataclasses, with a few CLI overrides)
# =============================================================================

def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config file into a nested dict. Raises if PyYAML is missing."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _apply_section(obj, section: Optional[dict], section_name: str):
    """Override dataclass fields from a config sub-dict, tolerating tuples for tuple fields."""
    if not section:
        return
    valid = {f.name: f for f in fields(obj)}
    for key, value in section.items():
        if key not in valid:
            print(f"[config] WARNING: unknown key '{section_name}.{key}' ignored.")
            continue
        # YAML has no tuple type; coerce lists into tuples where the default is a tuple.
        if isinstance(getattr(obj, key), tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)


def build_configs(config_path: Optional[str], cli_overrides: dict):
    """Instantiate all config dataclasses, apply the YAML file, then CLI overrides."""
    paths = Paths()
    data = DataCfg()
    model_cfg = ModelCfg()
    train_cfg = TrainCfg()
    loss_settings = LossSettings()
    strategy = StrategyCfg()

    if config_path:
        cfg = load_yaml_config(config_path)
        _apply_section(paths, cfg.get("paths"), "paths")
        _apply_section(data, cfg.get("data"), "data")
        _apply_section(model_cfg, cfg.get("model"), "model")
        _apply_section(train_cfg, cfg.get("train"), "train")
        _apply_section(loss_settings, cfg.get("losses"), "losses")
        _apply_section(strategy, cfg.get("strategy"), "strategy")

    # A handful of convenience CLI overrides (highest priority).
    if cli_overrides.get("device") is not None:
        train_cfg.device = cli_overrides["device"]
    if cli_overrides.get("epochs") is not None:
        train_cfg.epochs = cli_overrides["epochs"]
    if cli_overrides.get("batch_size") is not None:
        data.batch_size = cli_overrides["batch_size"]
        data.effective_batch = max(data.effective_batch, data.batch_size)
    if cli_overrides.get("seed") is not None:
        train_cfg.seed = cli_overrides["seed"]
    if cli_overrides.get("run_dir") is not None:
        paths.run_dir = cli_overrides["run_dir"]

    return paths, data, model_cfg, train_cfg, loss_settings, strategy


def parse_args():
    ap = argparse.ArgumentParser(description="CXR vision-language concept-alignment pretraining.")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to a YAML config file (see config/pretrain.yaml).")
    ap.add_argument("--device", type=str, default=None, help="Override train.device, e.g. cuda:0.")
    ap.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    ap.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size.")
    ap.add_argument("--seed", type=int, default=None, help="Override train.seed.")
    ap.add_argument("--run-dir", type=str, default=None, help="Override paths.run_dir.")
    return ap.parse_args()


# =============================================================================
# 6. MAIN RUNNER
# =============================================================================

def main():
    args = parse_args()
    cli_overrides = {
        "device": args.device, "epochs": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "run_dir": args.run_dir,
    }
    paths, data, model_cfg, train_cfg, loss_settings, strategy = build_configs(
        args.config, cli_overrides
    )

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
                raise RuntimeError(f"Seed sweep failed for seed={seed}")
        print("[Seed sweep] All runs completed successfully.")
        return
    if is_seed_worker:
        train_cfg.seed = int(os.environ["VCM_SINGLE_SEED"])
        print(f"[Seed sweep] Worker run with train.seed={train_cfg.seed}")

    # ── Concept name setup (all fine concepts by default) ───────────────────────
    cls_concept_names = (
        list(data.cls_concept_names)
        if data.cls_concept_names is not None
        else list(FINE_CONCEPT_NAMES)
    )
    align_concept_names = _merge_unique_preserve_order(
        cls_concept_names, list(data.align_extra_concepts)
    )
    cls_head_map = None
    if cls_concept_names != align_concept_names:
        cls_head_map = build_name_based_cls_head_map(cls_concept_names, align_concept_names)

    if loss_settings.use_cls_loss and not model_cfg.use_cls_heads:
        raise ValueError("use_cls_loss=True requires model.use_cls_heads=True.")

    # ── Run directory & logging ─────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_label = timestamp if not is_seed_worker else f"{timestamp}_seed{train_cfg.seed}"
    run_dir = os.path.join(paths.run_dir, run_label)
    ensure_dir(run_dir)

    full_cfg = {
        "paths": asdict(paths),
        "data": asdict(data),
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "losses": asdict(loss_settings),
        "strategy": asdict(strategy),
        "align_concept_names": align_concept_names,
        "cls_concept_names": cls_concept_names,
        "cls_head_map": cls_head_map,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(full_cfg, f, indent=2)

    metrics_logger = CSVScalarLogger(os.path.join(run_dir, "metrics.csv"))
    concept_logger = CSVPerConceptLogger(os.path.join(run_dir, "per_concept_metrics.csv"))
    tb_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb")) if SummaryWriter is not None else None

    # ── Seed & device ───────────────────────────────────────────────────────────
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ── Data preparation ────────────────────────────────────────────────────────
    print(f"Loading master index: {paths.master_index_path}")
    master = (pd.read_csv(paths.master_index_path)
              if paths.master_index_path.endswith(".csv")
              else pd.read_parquet(paths.master_index_path))

    if paths.split_path is None:
        paths.split_path = os.path.join(run_dir, f"splits_{data.split_on}.parquet")
        print(f"Creating new splits on: {data.split_on}")
        create_group_splits(master, paths.split_path,
                            val_frac=data.val_frac, test_frac=data.test_frac,
                            split_on=data.split_on)

    train_df = apply_group_splits(master, paths.split_path, split_on=data.split_on, split="train")
    val_df = apply_group_splits(master, paths.split_path, split_on=data.split_on, split="val")
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Empty split detected (split_on='{data.split_on}', split_path='{paths.split_path}'). "
            "Regenerate splits for the current split_on setting.")

    if data.max_train:
        train_df = train_df.sample(n=data.max_train, random_state=data.subsample_seed)
    if data.max_val:
        val_df = val_df.sample(n=data.max_val, random_state=data.subsample_seed)

    # ── Dataset construction ────────────────────────────────────────────────────
    if paths.cached_dataset_path is not None and os.path.exists(paths.cached_dataset_path):
        print("Building datasets from the cached (sharded) image store...")
        common_kwargs = dict(
            cached_dataset_path=paths.cached_dataset_path,
            image_size=data.image_size, norm_mode=data.norm_mode,
            include_cls_labels=loss_settings.use_cls_loss,
            accession_column=data.accession_column,
            sop_uid_column=data.sop_uid_column,
            affine_degrees=data.affine_degrees,
            affine_translate=data.affine_translate,
            cornercutout_size=data.cornercutout_size,
            cornercutout_probs=data.cornercutout_probs,
            cornercutout_fill=data.cornercutout_fill,
            text_override_dir=data.text_override_dir,
            use_text_override=data.use_text_override,
            text_override_cache_size=data.text_override_cache_size,
            text_override_strict=data.text_override_strict,
            text_override_mode=data.text_override_mode,
            cls_override_dir=data.cls_override_dir,
            use_cls_override=data.use_cls_override,
            cls_override_cache_size=data.cls_override_cache_size,
            cls_override_strict=data.cls_override_strict,
            cls_override_mode=data.cls_override_mode,
            cls_override_index_cache_name=data.cls_override_index_cache_name,
            cls_override_shard_memmap_cache_size=data.cls_override_shard_memmap_cache_size,
            text_concept_names=align_concept_names,
            cls_concept_names=cls_concept_names,
        )
        ds_train = PTCachedCXRDataset(train_df, align_concept_names, augment=data.augment, **common_kwargs)
        ds_val = PTCachedCXRDataset(val_df, align_concept_names, augment=False, **common_kwargs)
    else:
        print("Building datasets from scratch (reading PNGs + embeddings on the fly)...")
        common_kwargs = dict(
            text_concept_names=align_concept_names, cls_concept_names=cls_concept_names,
            embedding_dir=paths.embedding_dir, label_dir=paths.label_dir,
            include_cls_labels=loss_settings.use_cls_loss, image_size=data.image_size,
            norm_mode=data.norm_mode,
            affine_degrees=data.affine_degrees,
            affine_translate=data.affine_translate,
            cornercutout_size=data.cornercutout_size,
            cornercutout_probs=data.cornercutout_probs,
            cornercutout_fill=data.cornercutout_fill,
        )
        ds_train = CXRMultimodalDataset(train_df, align_concept_names, augment=data.augment, **common_kwargs)
        ds_val = CXRMultimodalDataset(val_df, align_concept_names, augment=False, **common_kwargs)

    print(f"Training samples:   {len(ds_train)}")
    print(f"Validation samples: {len(ds_val)}")

    # ── Strategy B: concept-aware sample weights ────────────────────────────────
    train_sample_weights = None
    if strategy.use_weighted_sampling:
        if data.text_override_dir and data.use_text_override:
            print("[Strategy B] Computing concept-aware sample weights...")
            train_sample_weights = compute_concept_sample_weights(
                df=ds_train.df,
                concept_names=align_concept_names,
                text_override_dir=data.text_override_dir,
                acc_col=data.accession_column,
                coverage_csv_path=strategy.weighted_sampling_coverage_csv,
                rare_threshold=strategy.weighted_sampling_rare_threshold,
                target_coverage=strategy.weighted_sampling_target_coverage,
                weights_cache_path=strategy.weighted_sampling_cache_path,
                split_filter=strategy.weighted_sampling_split,
            )
        else:
            print("[Strategy B] WARNING: use_weighted_sampling=True but text_override_dir "
                  "is not set. Falling back to uniform sampling.")

    # ── DataLoaders ─────────────────────────────────────────────────────────────
    dl_train = make_dataloader(
        ds_train, data.batch_size,
        shuffle=True,                       # ignored when sample_weights is provided
        num_workers=data.num_workers,
        prefetch_factor=data.dataloader_prefetch_factor,
        persistent_workers=data.dataloader_persistent_workers,
        pin_memory=data.dataloader_pin_memory,
        drop_last=True,
        sample_weights=train_sample_weights,   # None -> plain shuffle
    )
    dl_val = make_dataloader(
        ds_val, max(32, data.batch_size // 2),
        shuffle=False,
        num_workers=data.num_workers,
        prefetch_factor=data.dataloader_prefetch_factor,
        persistent_workers=data.dataloader_persistent_workers,
        pin_memory=data.dataloader_pin_memory,
        drop_last=False,
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    mcfg = ModelConfig(
        concept_names=align_concept_names,
        cls_concept_names=cls_concept_names if model_cfg.use_cls_heads else None,
        cls_head_map=cls_head_map,
        use_cls_heads=model_cfg.use_cls_heads,
        d_model=model_cfg.d_model,
        vision_backbone=model_cfg.vision_backbone,
        rad_dino_model_dir=model_cfg.rad_dino_model_dir,
        dinov2_repo_path=model_cfg.dinov2_repo_path,
        dinov2_weights_path=model_cfg.dinov2_weights_path,
        pretrained_vision=model_cfg.pretrained_vision,
        vision_weights_path=model_cfg.vision_weights_path,
        project_text=model_cfg.project_text,
        text_in_dim=model_cfg.text_in_dim,
        n_heads=model_cfg.n_heads,
        dropout=model_cfg.dropout,
    )
    model = VisionConceptModel(mcfg).to(device)

    # ── Learnable per-concept temperatures ──────────────────────────────────────
    use_learnable_concept_taus = (
        loss_settings.learnable_concept_taus
        and loss_settings.use_align_loss
        and loss_settings.concept_averaged_align
    )
    if loss_settings.learnable_concept_taus and not use_learnable_concept_taus:
        print("[WARN] learnable_concept_taus=True only applies to concept_averaged_align=True. "
              "Falling back to fixed tau_align.")

    tau_min = loss_settings.min_tau_align if use_learnable_concept_taus else None
    tau_max = loss_settings.max_tau_align if use_learnable_concept_taus else None
    if tau_min is not None and tau_min <= 0:
        raise ValueError(f"min_tau_align must be > 0, got {tau_min}")
    if tau_max is not None and tau_max <= 0:
        raise ValueError(f"max_tau_align must be > 0, got {tau_max}")
    if (tau_min is not None) and (tau_max is not None) and (tau_min >= tau_max):
        raise ValueError(f"min_tau_align must be < max_tau_align, got {tau_min} >= {tau_max}")
    log_tau_min = math.log(tau_min) if tau_min is not None else None
    log_tau_max = math.log(tau_max) if tau_max is not None else None

    align_log_tau: Optional[torch.nn.Parameter] = None
    if use_learnable_concept_taus:
        if loss_settings.tau_align <= 0:
            raise ValueError(f"tau_align must be > 0, got {loss_settings.tau_align}")
        init_log_tau = math.log(loss_settings.tau_align)
        align_log_tau = torch.nn.Parameter(
            torch.full((len(align_concept_names),), init_log_tau,
                       device=device, dtype=torch.float32)
        )

    def clamp_align_log_tau_():
        if align_log_tau is None:
            return
        if log_tau_min is None and log_tau_max is None:
            return
        lo = log_tau_min if log_tau_min is not None else -float("inf")
        hi = log_tau_max if log_tau_max is not None else float("inf")
        with torch.no_grad():
            align_log_tau.clamp_(min=lo, max=hi)

    clamp_align_log_tau_()

    def set_backbone_trainable(flag: bool):
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad = flag

    optim = make_optimizer(
        model, train_cfg.lr, train_cfg.weight_decay,
        extra_params=[align_log_tau] if align_log_tau is not None else None,
    )
    params_for_grad_clip = list(model.parameters())
    if align_log_tau is not None:
        params_for_grad_clip.append(align_log_tau)
    scaler = make_grad_scaler(train_cfg.bf16)

    # ── Positive weights for BCE (only when the cls head is enabled) ─────────────
    pos_weight_tensor = None
    if loss_settings.use_cls_loss:
        expected_k = len(cls_concept_names)
        if loss_settings.pos_weight_path and os.path.exists(loss_settings.pos_weight_path):
            print(f"Loading pos_weights from {loss_settings.pos_weight_path}")
            pos_weight_tensor = torch.load(
                loss_settings.pos_weight_path,
                map_location=device,
                weights_only=True,
            )
            pos_weight_tensor = pos_weight_tensor.float().reshape(-1)
            if pos_weight_tensor.numel() != expected_k:
                print(f"[WARN] Loaded pos_weight length {pos_weight_tensor.numel()} != {expected_k}. Recomputing.")
                pos_weight_tensor = None
        if pos_weight_tensor is None:
            pos_weight_tensor = compute_pos_weight_for_dataset(ds_val, cls_concept_names)
            save_path = os.path.join(run_dir, "pos_weights.pt")
            torch.save(pos_weight_tensor, save_path)
            print(f"Computed and saved pos_weights to {save_path}")
        pos_weight_tensor = pos_weight_tensor.to(device)
        if loss_settings.max_pos_weight > 0:
            pos_weight_tensor = torch.clamp(pos_weight_tensor, max=loss_settings.max_pos_weight)
        print(f"Final pos_weights: {pos_weight_tensor.cpu().numpy().round(2)}")

    # ── LossConfig ──────────────────────────────────────────────────────────────
    lcfg = LossConfig(
        use_cls_loss=loss_settings.use_cls_loss,
        use_align_loss=loss_settings.use_align_loss,
        w_cls=loss_settings.w_cls,
        pos_weight=pos_weight_tensor,
        w_align=loss_settings.w_align,
        concept_averaged_align=loss_settings.concept_averaged_align,
        tau_align=loss_settings.tau_align,
        learnable_concept_taus=use_learnable_concept_taus,
        align_log_tau_per_concept=align_log_tau,
        symmetric_align=loss_settings.symmetric_align,
    )

    # ── Print active configuration ──────────────────────────────────────────────
    print("\nActive losses:")
    if loss_settings.use_cls_loss:
        print("  - Classification loss (BCE)")
    if loss_settings.use_align_loss:
        print("  - Concept alignment loss (InfoNCE)")
        if use_learnable_concept_taus:
            print(f"    -> learnable tau per concept (init={loss_settings.tau_align}, "
                  f"min={tau_min}, max={tau_max})")
        else:
            print(f"    -> fixed tau_align={loss_settings.tau_align}")
    print(f"Weighted sampling (Strategy B): {strategy.use_weighted_sampling} "
          f"(rare_threshold={strategy.weighted_sampling_rare_threshold:.2f}, "
          f"target_coverage={strategy.weighted_sampling_target_coverage:.2f})")

    # ── Utility functions ───────────────────────────────────────────────────────
    def get_current_align_tau_values() -> Optional[torch.Tensor]:
        if align_log_tau is None:
            return None
        tau_vals = torch.exp(align_log_tau.detach().cpu())
        if tau_min is not None or tau_max is not None:
            lo = tau_min if tau_min is not None else 0.0
            hi = tau_max if tau_max is not None else float("inf")
            tau_vals = tau_vals.clamp(min=lo, max=hi)
        return tau_vals

    def get_align_tau_metrics() -> Dict[str, float]:
        tau_vals = get_current_align_tau_values()
        if tau_vals is None:
            return {}
        return {f"tau_align/{name}": float(tau_vals[i].item())
                for i, name in enumerate(align_concept_names)}

    def save_checkpoint(filename, metric_name, score, thresholds=None):
        save_dict = {
            "model": model.state_dict(), "epoch": epoch,
            "config": full_cfg, "best_metric": metric_name, "best_score": float(score),
        }
        if thresholds is not None:
            save_dict["thresholds"] = {n: float(t) for n, t in zip(cls_concept_names, thresholds)}
        if align_log_tau is not None:
            tau_vals = get_current_align_tau_values()
            save_dict["align_log_tau_per_concept"] = align_log_tau.detach().cpu()
            save_dict["align_tau_per_concept"] = {
                name: float(tau_vals[i].item()) for i, name in enumerate(align_concept_names)
            }
        torch.save(save_dict, os.path.join(run_dir, filename))

    best_scores = {"cls": -1.0, "align": float('inf')}

    # ── Initial evaluation ──────────────────────────────────────────────────────
    print("\nRunning initial evaluation (epoch 0)...")
    init_val_losses = evaluate_losses(model, dl_val, device, lcfg)
    metrics_logger.log_many(0, 0, "val", {f"loss/{k}": v for k, v in init_val_losses.items()})
    if tb_writer:
        for k, v in init_val_losses.items():
            tb_writer.add_scalar(f"val/loss/{k}", v, 0)
    if loss_settings.use_cls_loss:
        init_calib_res = calibrate_and_evaluate(model, dl_val, device, cls_concept_names)
        metrics_logger.log(0, 0, "val", "macro_auroc", np.nanmean(init_calib_res["auroc_per_concept"]))
        metrics_logger.log(0, 0, "val", "macro_auprc", np.nanmean(init_calib_res["auprc_per_concept"]))

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    accum = max(1, data.effective_batch // data.batch_size)
    total_steps = 0
    start_time = time.time()
    print(f"\nStarting training at {time.ctime(start_time)}")

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()

        # Backbone freeze / unfreeze schedule
        set_backbone_trainable(epoch > train_cfg.freeze_backbone_epochs)

        running_loss = {}
        t0 = time.time()

        for i, batch in enumerate(dl_train):
            total_steps += 1

            # LR schedule (cosine with linear warmup)
            warmup = train_cfg.warmup_steps
            if total_steps < warmup:
                lr = train_cfg.lr * (total_steps / warmup)
            else:
                progress = (total_steps - warmup) / max(1, (len(dl_train) * train_cfg.epochs) - warmup)
                lr = train_cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optim.param_groups:
                pg["lr"] = lr

            # Move to device
            imgs = batch["images"].to(device, non_blocking=True)
            text_emb = batch.get("text_emb"); text_mask = batch.get("text_mask")
            if text_emb is not None:
                text_emb = text_emb.to(device, non_blocking=True)
            if text_mask is not None:
                text_mask = text_mask.to(device, non_blocking=True)
            if batch.get("cls_labels") is not None:
                batch["cls_labels"] = batch["cls_labels"].to(device, non_blocking=True)
            if batch.get("cls_mask") is not None:
                batch["cls_mask"] = batch["cls_mask"].to(device, non_blocking=True)

            batch["text_mask"] = text_mask
            batch["text_emb"] = text_emb

            # Forward + loss
            with bf16_autocast(train_cfg.bf16):
                outputs = model(
                    images=imgs,
                    concept_text_emb=text_emb, concept_text_mask=text_mask,
                    need_attn=False,
                )
                loss_dict = compute_losses(batch, outputs, model, lcfg)
                loss = loss_dict["total"] / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                if train_cfg.clip_grad_norm:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params_for_grad_clip, train_cfg.clip_grad_norm)
                scaler.step(optim)
                scaler.update()
                clamp_align_log_tau_()
                optim.zero_grad()

            for k, v in loss_dict.items():
                running_loss[k] = running_loss.get(k, 0.0) + float(v)

            if total_steps % train_cfg.log_every == 0:
                avg = {k: v / train_cfg.log_every for k, v in running_loss.items()}
                metrics_logger.log_many(total_steps, epoch, "train", avg)
                metrics_logger.log(total_steps, epoch, "train", "lr", lr)
                tau_metrics = get_align_tau_metrics()
                if tau_metrics:
                    metrics_logger.log_many(total_steps, epoch, "train", tau_metrics)
                if tb_writer:
                    tb_writer.add_scalar("train/lr", lr, total_steps)
                    for k, v in avg.items():
                        tb_writer.add_scalar(f"train/loss/{k}", v, total_steps)
                    for k, v in tau_metrics.items():
                        tb_writer.add_scalar(f"train/{k}", v, total_steps)
                running_loss = {}

        # ── Validation & checkpointing ──────────────────────────────────────────
        val_losses = evaluate_losses(model, dl_val, device, lcfg)
        metrics_logger.log_many(total_steps, epoch, "val", {f"loss/{k}": v for k, v in val_losses.items()})
        if tb_writer:
            for k, v in val_losses.items():
                tb_writer.add_scalar(f"val/loss/{k}", v, total_steps)

        ckpt_thresholds = None
        if loss_settings.use_cls_loss:
            calib_res = calibrate_and_evaluate(model, dl_val, device, cls_concept_names)
            ckpt_thresholds = calib_res["best_thresh_per_concept"]
            macro_auroc = np.nanmean(calib_res["auroc_per_concept"])
            macro_auprc = np.nanmean(calib_res["auprc_per_concept"])
            macro_f1 = np.nanmean(calib_res["best_f1_per_concept"])
            metrics_logger.log(total_steps, epoch, "val", "macro_auroc", macro_auroc)
            metrics_logger.log(total_steps, epoch, "val", "macro_auprc", macro_auprc)
            metrics_logger.log(total_steps, epoch, "val", "macro_f1_opt", macro_f1)
            if tb_writer:
                tb_writer.add_scalar("val/macro_auroc", macro_auroc, total_steps)
                tb_writer.add_scalar("val/macro_auprc", macro_auprc, total_steps)
                tb_writer.add_scalar("val/macro_f1_opt", macro_f1, total_steps)
                for k, name in enumerate(cls_concept_names):
                    auprc_val = calib_res["auprc_per_concept"][k]
                    if not np.isnan(auprc_val):
                        tb_writer.add_scalar(f"val/concept_auprc/{name}", auprc_val, total_steps)
            print(f"Epoch {epoch} | Val Loss: {val_losses['total']:.4f} | "
                  f"Macro AUROC: {macro_auroc:.4f} | Macro AUPRC: {macro_auprc:.4f} | "
                  f"Macro F1: {macro_f1:.4f}")
            for k, name in enumerate(cls_concept_names):
                concept_logger.log_concept(epoch, name, "auroc", calib_res["auroc_per_concept"][k])
                concept_logger.log_concept(epoch, name, "auprc", calib_res["auprc_per_concept"][k])
                concept_logger.log_concept(epoch, name, "best_f1", calib_res["best_f1_per_concept"][k],
                                           threshold=calib_res["best_thresh_per_concept"][k])
        else:
            print(f"Epoch {epoch} | Val Loss: {val_losses['total']:.4f} | "
                  f"(classification metrics skipped: cls loss disabled)")

        current_taus = get_current_align_tau_values()
        if current_taus is not None:
            tau_str = " | ".join(f"{name}: {current_taus[i].item():.5f}"
                                 for i, name in enumerate(align_concept_names))
            print(f"Epoch {epoch} | Alignment taus | {tau_str}")

        print(f"Epoch duration: {(time.time() - t0) / 60:.2f} minutes")

        # ── Multi-track model selection ─────────────────────────────────────────
        if loss_settings.use_cls_loss:
            current_cls = macro_auprc
            if current_cls > best_scores["cls"]:
                best_scores["cls"] = current_cls
                save_checkpoint("best_cls.pt", "Macro AUPRC", current_cls, thresholds=ckpt_thresholds)
                print(f"  [+] Saved best_cls.pt   (AUPRC: {current_cls:.4f})")

        if loss_settings.use_align_loss:
            current_align = val_losses.get("align", float('inf'))
            if current_align < best_scores["align"]:
                best_scores["align"] = current_align
                save_checkpoint("best_align.pt", "Align Loss", current_align, thresholds=ckpt_thresholds)
                print(f"  [+] Saved best_align.pt (Loss: {current_align:.4f})")

        # ── Periodic checkpoint ─────────────────────────────────────────────────
        if train_cfg.save_every_epochs > 0 and epoch % train_cfg.save_every_epochs == 0:
            save_checkpoint(f"epoch_{epoch:04d}.pt", "periodic", epoch)

    metrics_logger.close()
    concept_logger.close()
    if tb_writer:
        tb_writer.close()
    print("\nTraining complete.")
    print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
