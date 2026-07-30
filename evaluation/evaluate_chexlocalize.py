#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RadPRISM CheXpert / CheXlocalize External Evaluation
====================================================

One config-driven script for evaluating a trained RadPRISM model on the external
CheXpert / CheXlocalize datasets, combining two stages (toggle in the YAML):

  1. classify : run the model on CheXpert images, map the model's concept
                predictions to the CheXpert classes, and compute classification
                metrics (AUROC / AUPRC / F1). Optionally retrieves top-k text
                snippets per concept and saves per-sample attention maps (needed
                for stage 2).
  2. localize : evaluate visual grounding — how well the concept attention maps
                overlap the CheXlocalize ground-truth segmentations (Dice / IoU /
                pointing-game / pixel-AUPRC), for the one-to-one concept<->class
                targets. Supports optimizing the attention threshold on one split
                and applying it to another (e.g. val -> test).

Stage 2 consumes the attention maps saved by stage 1, so a typical run enables
both (classify first, then localize).

Usage:
  python evaluate_chexlocalize.py --config config/evaluate_chexlocalize.yaml
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

import eval_common as ec  # adds RadPRISM/src to sys.path on import
from dataset import IMAGENET_MEAN, IMAGENET_STD, RAD_DINO_MAIRA_2_MEAN, RAD_DINO_MAIRA_2_STD, NormalizeCXR  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Concept <-> CheXpert mappings (English concept names)
# ─────────────────────────────────────────────────────────────────────────────

# Many-to-one mapping used for classification (grouped by CheXpert class).
DEFAULT_CHEXPERT_MAPPING: "OrderedDict[str, List[str]]" = OrderedDict({
    "Support Devices": ["support_devices.airway", "support_devices.gastric_tube",
                        "support_devices.central_line", "support_devices.chest_drain",
                        "support_devices.pacemaker", "support_devices.other"],
    "Enlarged Cardiomediastinum": ["thoracic_organs.mediastinum"],
    "Cardiomegaly": ["thoracic_organs.heart"],
    "Edema": ["pathologies.vessels.congestion", "pathologies.vessels.pulmonary_edema"],
    "Pneumonia": ["pathologies.lung.pneumonia"],
    "Atelectasis": ["pathologies.lung.atelectasis"],
    "Pneumothorax": ["pathologies.pleura.pneumothorax"],
    "Pleural Effusion": ["pathologies.pleura.pleural_effusion"],
    "Fracture": ["pathologies.bones.fracture"],
    "Lung Lesion": ["pathologies.lung.mass"],
})

# One-to-one targets used for the localization (grounding) evaluation.
DEFAULT_LOCALIZE_TARGETS: List[Dict[str, str]] = [
    {"concept_name": "thoracic_organs.heart", "chexpert_class": "Cardiomegaly"},
    {"concept_name": "thoracic_organs.mediastinum", "chexpert_class": "Enlarged Cardiomediastinum"},
    {"concept_name": "pathologies.lung.atelectasis", "chexpert_class": "Atelectasis"},
    {"concept_name": "pathologies.pleura.pneumothorax", "chexpert_class": "Pneumothorax"},
    {"concept_name": "pathologies.pleura.pleural_effusion", "chexpert_class": "Pleural Effusion"},
    {"concept_name": "pathologies.lung.mass", "chexpert_class": "Lung Lesion"},
]

RESAMPLE_MAP = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(p) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


def json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_fname(name: str) -> str:
    for a, b in [(" ", "_"), ("/", "_"), ("\\", "_"), (":", "_"), (".", "_")]:
        name = name.replace(a, b)
    return name


def parse_binary_label(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    return 0 if fv == 0.0 else (1 if fv == 1.0 else None)


def infer_view(path_str: str, cell=None) -> str:
    if isinstance(cell, str) and cell.strip():
        v = cell.strip().lower()
        if "frontal" in v:
            return "frontal"
        if "lateral" in v:
            return "lateral"
    return "lateral" if "lateral" in str(path_str).lower() else "frontal"


def resolve_chexpert_image_path(chexpert_root: str, split: str, csv_path: str) -> Path:
    root, raw = Path(chexpert_root), Path(csv_path)
    if raw.is_absolute() and raw.exists():
        return raw
    if (root / raw).exists():
        return root / raw
    s = str(csv_path).replace("\\", "/")
    for c in [s.replace("CheXpert-v1.0/valid/", "val/"), s.replace("CheXpert-v1.0/valid", "val"),
              s.replace("CheXpert-v1.0/test/", "test/"), s.replace("CheXpert-v1.0/test", "test"),
              s.replace("valid/", "val/")]:
        if (root / c).exists():
            return root / c
    m = re.search(r"(patient\d+/study\d+/view\d+_[^/]+\.(jpg|jpeg|png))$", s, re.IGNORECASE)
    if m and (root / split / m.group(1)).exists():
        return root / split / m.group(1)
    raise FileNotFoundError(f"Could not resolve image path for split='{split}': {csv_path}")


def csv_path_to_image_id(csv_path: str) -> str:
    m = re.search(r"(patient\d+/study\d+/view\d+_[^/]+)\.(jpg|jpeg|png)$",
                  str(csv_path).replace("\\", "/"), re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse image_id from csv path: {csv_path}")
    return m.group(1).replace("/", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_image_float01(path: str) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if np.issubdtype(arr.dtype, np.integer):
        a, b = float(arr.min()), float(arr.max())
        return (arr.astype(np.float32) - a) / max(b - a, 1.0)
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def resize_to_square(x: np.ndarray, out_size: int, keep_aspect: bool, pad_value: float, resample: str) -> np.ndarray:
    h, w = x.shape
    pil = Image.fromarray(x.astype(np.float32))
    r = RESAMPLE_MAP[resample]
    if not keep_aspect:
        return np.asarray(pil.resize((out_size, out_size), resample=r), dtype=np.float32)
    scale = out_size / float(max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(pil.resize((nw, nh), resample=r), dtype=np.float32)
    canvas = np.full((out_size, out_size), float(pad_value), dtype=np.float32)
    y0, x0 = (out_size - nh) // 2, (out_size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def prepare_input_tensor(path, size, keep_aspect, pad_value, resample, normalizer) -> torch.Tensor:
    x = resize_to_square(load_image_float01(path), size, keep_aspect, pad_value, resample)
    t3 = torch.from_numpy(x).float().unsqueeze(0).repeat(3, 1, 1)
    return normalizer(t3)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers (classification)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def binary_metrics(y_true, y_pred) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1))); tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1))); fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec, rec = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "specificity": _safe_div(tn, tn + fp),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn), "accuracy": _safe_div(tp + tn, tp + tn + fp + fn)}


def auc_metrics(y_true, y_score) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int); y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score)
    y_true, y_score = y_true[finite], y_score[finite]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float("nan")
    return {"auroc": auroc, "auprc": float(average_precision_score(y_true, y_score))}


def load_concept_thresholds(thr_path: Optional[str], key: str, default: float, cls_names) -> Dict[str, float]:
    out = {c: default for c in cls_names}
    if thr_path and os.path.exists(thr_path):
        data = ec.load_json(thr_path)
        for c in cls_names:
            item = data.get(c)
            if isinstance(item, dict) and key in item and item[key] is not None:
                try:
                    out[c] = float(item[key])
                except Exception:
                    pass
    return out


def map_concepts_to_chexpert(concept_bin, concept_prob, mapping) -> Tuple[dict, dict]:
    pred_map, prob_map = {}, {}
    for target, srcs in mapping.items():
        pred_map[target] = int(any(int(concept_bin.get(c, 0)) == 1 for c in srcs))
        probs = [float(concept_prob.get(c, float("nan"))) for c in srcs]
        probs = [p for p in probs if not math.isnan(p)]
        prob_map[target] = max(probs) if probs else float("nan")
    return pred_map, prob_map


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: classification (+ retrieval, + attention-map saving)
# ─────────────────────────────────────────────────────────────────────────────

def build_split_rows(chexpert_root, split, labels_df, mapping, view_filter, max_images) -> pd.DataFrame:
    rows = []
    for i, row in labels_df.iterrows():
        csv_path = str(row["Path"])
        try:
            image_path = resolve_chexpert_image_path(chexpert_root, split, csv_path)
        except Exception:
            continue
        view = infer_view(csv_path, row.get("Frontal/Lateral"))
        if view_filter in ("frontal", "lateral") and view != view_filter:
            continue
        r = {"split": split, "csv_path": csv_path, "image_path": str(image_path), "view": view}
        for cls in mapping.keys():
            r[f"gt::{cls}"] = parse_binary_label(row.get(cls))
        rows.append(r)
    df = pd.DataFrame(rows)
    if max_images:
        df = df.head(int(max_images)).copy()
    return df.reset_index(drop=True)


def make_normalizer(norm_mode, cxr_mean, cxr_std) -> NormalizeCXR:
    return NormalizeCXR(mode=norm_mode, cxr_mean=cxr_mean, cxr_std=cxr_std)


@torch.no_grad()
def run_classification(cfg, model, align_names, cls_names, cls_head_map, device, out_dir):
    """Classify each CheXpert split, save attention maps + mapped predictions + metrics."""
    from tqdm import tqdm
    ccfg = cfg["classify"]
    mapping = OrderedDict(cfg.get("chexpert_mapping") or DEFAULT_CHEXPERT_MAPPING)
    norm_mode = cfg["model"].get("norm_mode") or cfg["_run_cfg"]["data"].get("norm_mode", "rad_dino_maira_2")
    normalizer = make_normalizer(norm_mode, None, None)
    thresholds = load_concept_thresholds(ccfg.get("concept_thresholds_json"),
                                         ccfg.get("threshold_key", "thr_f1"),
                                         float(ccfg.get("default_threshold", 0.5)), cls_names)
    cls_to_align = {c: (cls_head_map[i] if cls_head_map is not None else i) for i, c in enumerate(cls_names)}

    # Retrieval DB (optional, shared across splits).
    retrieval_db = None
    if ccfg.get("retrieve", False):
        rpath = ccfg.get("custom_text_db_path")
        if rpath and os.path.exists(str(rpath)):
            embedder = ec.TextEmbedder(
                backend=(cfg.get("text_embedding") or {}).get("backend", "hash"),
                dim=int(cfg["_run_cfg"]["model"].get("text_in_dim", 768)),
                model_dir=(cfg.get("text_embedding") or {}).get("model_dir"),
                device=(cfg.get("text_embedding") or {}).get("device", device),
                precision=(cfg.get("text_embedding") or {}).get("precision", "float32"),
                truncate_dim=(cfg.get("text_embedding") or {}).get("truncate_dim"),
            )
            custom = ec.load_json(rpath)
            custom = custom.get("concept_texts", custom)
            retrieval_db = ec.build_retrieval_db_from_custom(custom, align_names, embedder)
            print(f"[classify] retrieval DB over {len(retrieval_db)} concepts.")

    size = int(ccfg.get("image_size", 518))
    top_k = int(ccfg.get("retrieval_top_k", 3))
    for split in ccfg["splits"]:
        labels_csv = ccfg["labels_csv_by_split"][split]
        labels_df = pd.read_csv(labels_csv)
        split_df = build_split_rows(cfg["chexpert_root"], split, labels_df, mapping,
                                    ccfg.get("view_filter", "all"), ccfg.get("max_images_per_split"))
        split_dir = ensure_dir(out_dir / split)
        samples_dir = ensure_dir(split_dir / "samples")
        print(f"[classify] split '{split}': {len(split_df)} images")

        mapped_rows, retr_rows = [], []
        for i, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"[{split}] classify"):
            try:
                img_t = prepare_input_tensor(row["image_path"], size, ccfg.get("keep_aspect_ratio", True),
                                             ccfg.get("pad_value", 0.0), ccfg.get("resize_resample", "bilinear"),
                                             normalizer)
                out = model(images=img_t.unsqueeze(0).to(device), need_attn=True)
                probs = torch.sigmoid(out["concept_logits"][0]).cpu().numpy()
                v_concepts = out["v_concepts"][0].detach().cpu()
                attn = out.get("concept_attn")
                sample_id = f"{split}__{safe_fname(row['csv_path'])}"
                s_dir = ensure_dir(samples_dir / sample_id)

                # Save attention maps [K_align, H', W'] for the localization stage.
                if attn is not None:
                    np.save(s_dir / "attention_maps.npy", attn[0].cpu().numpy().astype(np.float32))
                    json_dump({"align_concept_names": align_names, "shape": list(attn[0].shape)},
                              s_dir / "attention_maps_meta.json")

                concept_prob = {c: float(probs[k]) for k, c in enumerate(cls_names)}
                concept_bin = {c: int(concept_prob[c] >= thresholds[c]) for c in cls_names}
                pred_map, prob_map = map_concepts_to_chexpert(concept_bin, concept_prob, mapping)

                mrow = {"split": split, "sample_id": sample_id, "csv_path": row["csv_path"], "view": row["view"]}
                for cls in mapping.keys():
                    gt = row.get(f"gt::{cls}")
                    valid = int(gt in (0, 1))
                    mrow[f"gt::{cls}"] = int(gt) if valid else np.nan
                    mrow[f"gt_valid::{cls}"] = valid
                    mrow[f"pred::{cls}"] = int(pred_map[cls])
                    mrow[f"pred_prob::{cls}"] = float(prob_map[cls])
                mapped_rows.append(mrow)

                if retrieval_db is not None:
                    for m in ec.query_retrieval_db(v_concepts, retrieval_db, model, align_names, top_k=top_k):
                        retr_rows.append({"split": split, "sample_id": sample_id, **m})
            except Exception as ex:
                print(f"[classify] WARN sample {i}: {ex}")

        mapped_df = pd.DataFrame(mapped_rows)
        mapped_df.to_csv(split_dir / "chexpert_mapped_predictions.csv", index=False)
        if retr_rows:
            pd.DataFrame(retr_rows).to_csv(split_dir / "retrieval_top_matches.csv", index=False)

        # Per-class classification metrics.
        metric_rows = []
        for cls in mapping.keys():
            vcol, gcol, pcol, scol = f"gt_valid::{cls}", f"gt::{cls}", f"pred::{cls}", f"pred_prob::{cls}"
            if mapped_df.empty or gcol not in mapped_df:
                metric_rows.append({"class": cls, "n": 0}); continue
            vmask = mapped_df[vcol].fillna(0).astype(int).to_numpy() == 1
            yt = mapped_df.loc[vmask, gcol].astype(int).to_numpy()
            yp = mapped_df.loc[vmask, pcol].astype(int).to_numpy()
            ys = mapped_df.loc[vmask, scol].astype(float).to_numpy()
            row_m = {"class": cls}
            if len(yt):
                row_m.update(binary_metrics(yt, yp)); row_m.update(auc_metrics(yt, ys))
            else:
                row_m["n"] = 0
            metric_rows.append(row_m)
        mdf = pd.DataFrame(metric_rows)
        mdf.to_csv(split_dir / "chexpert_mapped_metrics.csv", index=False)
        macro = {"macro_auroc": float(np.nanmean(mdf.get("auroc", pd.Series([np.nan])))),
                 "macro_auprc": float(np.nanmean(mdf.get("auprc", pd.Series([np.nan]))))}
        json_dump({"split": split, **macro, "n_images": int(len(split_df))}, split_dir / "summary.json")
        print(f"[classify] {split} macro AUROC={macro['macro_auroc']:.4f} AUPRC={macro['macro_auprc']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: localization (grounding) — attention vs GT segmentation
# ─────────────────────────────────────────────────────────────────────────────

def coco_rle_counts_to_list(counts) -> List[int]:
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    cnts, p, m, n = [], 0, 0, len(counts)
    while p < n:
        x, k, more = 0, 0, 1
        while more:
            c = ord(counts[p]) - 48; p += 1
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20; k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(int(x)); m += 1
    return cnts


def decode_coco_rle_mask(rle) -> np.ndarray:
    h, w = map(int, rle["size"])
    counts = coco_rle_counts_to_list(rle["counts"])
    total = h * w
    flat = np.zeros(total, dtype=np.uint8)
    idx, val = 0, 0
    for c in counts:
        if idx >= total:
            break
        end = min(total, idx + int(c))
        if val == 1 and end > idx:
            flat[idx:end] = 1
        idx, val = end, 1 - val
    return flat.reshape((w, h)).T.astype(bool)  # COCO is column-major


def letterbox_params(h, w, out_size):
    scale = out_size / float(max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return scale, nh, nw, (out_size - nh) // 2, (out_size - nw) // 2


def resize_and_pad_mask(mask_bool, out_size):
    h, w = mask_bool.shape
    _, nh, nw, y0, x0 = letterbox_params(h, w, out_size)
    m = np.array(Image.fromarray((mask_bool.astype(np.uint8) * 255)).resize((nw, nh), Image.Resampling.NEAREST)) > 0
    canvas = np.zeros((out_size, out_size), dtype=bool); canvas[y0:y0 + nh, x0:x0 + nw] = m
    valid = np.zeros((out_size, out_size), dtype=bool); valid[y0:y0 + nh, x0:x0 + nw] = True
    return canvas, valid


def upsample_attention(attn_small, out_size):
    t = torch.tensor(attn_small, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0].numpy()


def get_eval_mask(valid, mode):
    if mode == "full_canvas":
        return np.ones_like(valid, dtype=bool)
    if mode == "non_padded":
        return valid.astype(bool)
    raise ValueError(f"Unsupported eval_region: {mode}")


def normalize_attention(attn, eval_mask, mode):
    if mode == "none":
        out = attn.astype(np.float32, copy=True)
    elif mode == "sum_to_one_eval_region":
        out = attn.astype(np.float32, copy=True)
        s = float(out[eval_mask].sum())
        out = out / s if s > 0 else out
    elif mode == "minmax_eval_region":
        x = np.where(eval_mask, attn, np.nan)
        mn, mx = np.nanmin(x), np.nanmax(x)
        out = (attn - mn) / (mx - mn) if (np.isfinite(mn) and np.isfinite(mx) and mx > mn) else np.zeros_like(attn, np.float32)
    else:
        raise ValueError(f"Unsupported attention_normalization: {mode}")
    return np.where(eval_mask, out, 0.0).astype(np.float32)


def overlap_stats(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum()); union = int(np.logical_or(pred, gt).sum())
    return {"dice": (2 * inter) / (int(pred.sum()) + int(gt.sum()) + 1e-8), "iou": inter / (union + 1e-8)}


def load_case_attention(chexpert_eval_dir, split, sample_id, concept_name):
    s_dir = Path(chexpert_eval_dir) / split / "samples" / sample_id
    meta = ec.load_json(str(s_dir / "attention_maps_meta.json"))
    names = meta.get("align_concept_names", [])
    if concept_name not in names:
        raise KeyError(f"concept '{concept_name}' not in saved attention maps")
    attn_all = np.load(s_dir / "attention_maps.npy")
    return attn_all[int(names.index(concept_name))]


def filtered_target_df(pred_df, chex_class, gt_positive_only, pred_positive_only, max_cases, seed):
    cols = {f"gt::{chex_class}": "gt", f"gt_valid::{chex_class}": "gt_valid",
            f"pred::{chex_class}": "pred", f"pred_prob::{chex_class}": "pred_prob"}
    df = pred_df[["sample_id", "csv_path"] + list(cols.keys())].rename(columns=cols)
    df = df[df["gt_valid"] == 1]
    if gt_positive_only:
        df = df[df["gt"] == 1]
    if pred_positive_only:
        df = df[df["pred"] == 1]
    df = df.reset_index(drop=True)
    if max_cases and len(df) > int(max_cases):
        df = df.sample(n=int(max_cases), random_state=int(seed)).reset_index(drop=True)
    return df


def optimize_thresholds(cases, grid) -> Dict[str, float]:
    """Sweep thresholds over collected (attn_norm, gt, eval_mask) cases; return best Dice/IoU thresholds."""
    n = len(grid)
    dice_sums, iou_sums, n_ok = np.zeros(n), np.zeros(n), 0
    for attn_norm, gt, eval_mask in cases:
        yt = gt[eval_mask].astype(np.int64)
        ys = attn_norm[eval_mask].astype(np.float32)
        gt_total = int(yt.sum())
        if gt_total <= 0:
            continue
        order = np.argsort(ys)[::-1]
        cum_true = np.cumsum(yt[order])
        pred_tot = np.searchsorted(-ys[order], -grid, side="right").astype(np.int64)
        inter = np.where(pred_tot > 0, cum_true[np.clip(pred_tot - 1, 0, len(cum_true) - 1)], 0)
        union = pred_tot + gt_total - inter
        dice_sums += (2.0 * inter) / (pred_tot + gt_total + 1e-8)
        iou_sums += inter / (union + 1e-8)
        n_ok += 1
    if n_ok == 0:
        return {"dice": 0.5, "iou": 0.5}
    return {"dice": float(grid[int(np.argmax(dice_sums))]), "iou": float(grid[int(np.argmax(iou_sums))])}


def case_metrics(attn_norm, gt, eval_mask, mask_method, fixed_threshold, percentile, opt_thr):
    yt = gt[eval_mask].astype(np.uint8); ys = attn_norm[eval_mask].astype(np.float32)
    energy = float((attn_norm[eval_mask] * gt[eval_mask]).sum() / (attn_norm[eval_mask].sum() + 1e-8))
    flat = np.flatnonzero(eval_mask.ravel())
    py, px = np.unravel_index(int(flat[int(np.argmax(ys))]), gt.shape)
    pointing = int(gt[py, px])
    k = max(1, int(gt.sum()))
    topk = np.zeros(eval_mask.size, bool); topk[flat[np.argsort(ys)[::-1][:k]]] = True
    topk = topk.reshape(eval_mask.shape)
    topk_st = overlap_stats(topk, gt)

    if opt_thr is not None:
        pred = np.logical_and(attn_norm >= opt_thr["iou"], eval_mask)
        sel = overlap_stats(pred, gt)
        dice_sel = overlap_stats(np.logical_and(attn_norm >= opt_thr["dice"], eval_mask), gt)["dice"]
        iou_sel = sel["iou"]; thr_used = opt_thr["iou"]
    else:
        if mask_method == "topk_gt_area":
            pred, thr_used = topk, float("nan")
        elif mask_method == "fixed_threshold":
            thr_used = float(fixed_threshold); pred = np.logical_and(attn_norm >= thr_used, eval_mask)
        elif mask_method == "percentile":
            thr_used = float(np.percentile(ys, percentile)); pred = np.logical_and(attn_norm >= thr_used, eval_mask)
        else:
            raise ValueError(f"Unsupported attention_mask_method: {mask_method}")
        sel = overlap_stats(pred, gt); dice_sel, iou_sel = sel["dice"], sel["iou"]

    if len(np.unique(yt)) < 2:
        aur, aup = float("nan"), float("nan")
    else:
        aur, aup = float(roc_auc_score(yt, ys)), float(average_precision_score(yt, ys))
    return {"energy_inside": energy, "pointing_hit": pointing,
            "dice_topk": float(topk_st["dice"]), "iou_topk": float(topk_st["iou"]),
            "dice_selected": float(dice_sel), "iou_selected": float(iou_sel),
            "selected_threshold": float(thr_used), "auroc_pixel": aur, "auprc_pixel": aup}


def run_localization(cfg, out_dir):
    lcfg = cfg["localize"]
    targets = cfg.get("localize_targets") or DEFAULT_LOCALIZE_TARGETS
    eval_dir = lcfg.get("chexpert_eval_dir") or str(out_dir)   # where classify saved attention maps
    size = int(lcfg.get("model_input_size", 518))
    eval_region = lcfg.get("eval_region", "full_canvas")
    norm_mode = lcfg.get("attention_normalization", "minmax_eval_region")
    mask_method = lcfg.get("attention_mask_method", "percentile")
    fixed_thr = float(lcfg.get("attention_fixed_threshold", 0.5))
    percentile = float(lcfg.get("attention_percentile", 98.0))
    gt_pos = bool(lcfg.get("filter_gt_positive_only", True))
    pred_pos = bool(lcfg.get("filter_pred_positive_only", False))
    max_cases = lcfg.get("max_cases_per_target_per_split")
    seed = int(lcfg.get("seed", 42))
    loc_dir = ensure_dir(out_dir / "localization")

    opt_mode = bool(lcfg.get("optimized_threshold_mode", False))
    src_split = lcfg.get("optimized_threshold_source_split", "val")
    apply_split = lcfg.get("optimized_threshold_apply_split", "test")
    grid = np.round(np.arange(float(lcfg.get("optimized_threshold_search_min", 0.0)),
                              float(lcfg.get("optimized_threshold_search_max", 1.0)) + 1e-9,
                              float(lcfg.get("optimized_threshold_search_step", 0.01))), 6)

    gt_seg_by_split = {s: ec.load_json(p) for s, p in lcfg["gt_seg_json_by_split"].items()}
    summary_rows = []

    for tgt in targets:
        concept, chex = tgt["concept_name"], tgt["chexpert_class"]
        # Optionally optimize thresholds on the source split.
        opt_thr = None
        if opt_mode:
            src_pred = pd.read_csv(Path(eval_dir) / src_split / "chexpert_mapped_predictions.csv")
            src_df = filtered_target_df(src_pred, chex, gt_pos, pred_pos, max_cases, seed)
            cases = []
            for _, r in src_df.iterrows():
                try:
                    gt_orig = decode_coco_rle_mask(gt_seg_by_split[src_split][csv_path_to_image_id(r["csv_path"])][chex])
                    gt_m, valid = resize_and_pad_mask(gt_orig, size)
                    em = get_eval_mask(valid, eval_region)
                    if int(gt_m.sum()) <= 0:
                        continue
                    an = normalize_attention(upsample_attention(load_case_attention(eval_dir, src_split, r["sample_id"], concept), size), em, norm_mode)
                    cases.append((an, gt_m, em))
                except Exception:
                    continue
            opt_thr = optimize_thresholds(cases, grid)
            print(f"[localize] {concept}->{chex} optimized thresholds on '{src_split}': {opt_thr}")

        eval_splits = [apply_split] if opt_mode else lcfg["splits"]
        for split in eval_splits:
            pred_df = pd.read_csv(Path(eval_dir) / split / "chexpert_mapped_predictions.csv")
            df = filtered_target_df(pred_df, chex, gt_pos, pred_pos, max_cases, seed)
            rows = []
            for _, r in df.iterrows():
                try:
                    gt_orig = decode_coco_rle_mask(gt_seg_by_split[split][csv_path_to_image_id(r["csv_path"])][chex])
                    gt_m, valid = resize_and_pad_mask(gt_orig, size)
                    em = get_eval_mask(valid, eval_region)
                    if int(gt_m.sum()) <= 0:
                        continue
                    an = normalize_attention(upsample_attention(load_case_attention(eval_dir, split, r["sample_id"], concept), size), em, norm_mode)
                    m = case_metrics(an, gt_m, em, mask_method, fixed_thr, percentile, opt_thr)
                    rows.append({"split": split, "concept_name": concept, "chexpert_class": chex,
                                 "sample_id": r["sample_id"], **m})
                except Exception:
                    continue
            case_df = pd.DataFrame(rows)
            tag = safe_fname(f"{concept}__{chex}__{split}")
            case_df.to_csv(loc_dir / f"cases_{tag}.csv", index=False)
            if len(case_df):
                summ = {"concept_name": concept, "chexpert_class": chex, "split": split,
                        "n_cases": int(len(case_df))}
                for col in ["energy_inside", "pointing_hit", "dice_topk", "iou_topk",
                            "dice_selected", "iou_selected", "auprc_pixel"]:
                    summ[f"mean_{col}"] = float(np.nanmean(case_df[col]))
                if opt_thr is not None:
                    summ["opt_threshold_dice"] = opt_thr["dice"]; summ["opt_threshold_iou"] = opt_thr["iou"]
                summary_rows.append(summ)
                print(f"[localize] {concept}->{chex} [{split}] n={summ['n_cases']} "
                      f"IoU={summ['mean_iou_selected']:.3f} Dice={summ['mean_dice_selected']:.3f} "
                      f"pointing={summ['mean_pointing_hit']:.3f}")

    pd.DataFrame(summary_rows).to_csv(loc_dir / "localization_summary.csv", index=False)
    print(f"[localize] wrote {loc_dir / 'localization_summary.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RadPRISM CheXpert / CheXlocalize evaluation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = ec.load_yaml(args.config)
    mcfg = cfg.get("model", {})
    run_dir = mcfg["run_dir"]
    device = args.device or mcfg.get("device", "cpu")
    modes = cfg.get("modes", {})

    run_cfg = ec.resolve_effective_run_config(
        run_dir, use_parent_pretrain_config=mcfg.get("use_parent_pretrain_config", True),
        parent_pretrain_config_path=mcfg.get("parent_pretrain_config_path"),
        auto_parent_from_finetune_config=mcfg.get("auto_parent_from_finetune_config", True))
    cfg["_run_cfg"] = run_cfg

    out_dir = ensure_dir(cfg.get("output_dir") or os.path.join(run_dir, "eval_chexlocalize", time.strftime("%Y%m%d-%H%M%S")))
    print(f"[eval] output dir: {out_dir}")

    if modes.get("classify", False):
        model, align_names, cls_names, cls_head_map = ec.build_model_from_config(run_cfg, device, record_attn_maps=True)
        ec.load_model_weights(model, os.path.join(run_dir, mcfg.get("checkpoint_name", "best_cls.pt")))
        run_classification(cfg, model, align_names, cls_names, cls_head_map, device, out_dir)

    if modes.get("localize", False):
        run_localization(cfg, out_dir)

    print("\n[eval] done.")


if __name__ == "__main__":
    main()
