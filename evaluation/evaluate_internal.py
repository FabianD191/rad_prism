#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RadPRISM Internal-Dataset Evaluation
====================================

One config-driven script that evaluates a trained RadPRISM model on an internal
dataset (the bundled dummy dataset, or your own), combining three modes that can
be toggled independently in the YAML config:

  1. thresholds : per-concept operating thresholds (best-F1 and Youden-J) on the
                  validation split, saved to val_thresholds.json. These thresholds
                  are reused by the classification metrics and by the CheXlocalize
                  evaluation.
  2. classify   : classification metrics (AUROC / AUPRC / F1) per concept, for
                  - the fine-tuned classification heads ("cls_head"), and
                  - optionally zero-shot prompting ("zero_shot"), which scores each
                    image concept embedding against positive/negative text prompts.
  3. retrieve   : concept-wise text retrieval — for each image, the top-k most
                  similar text snippets per concept, drawn from a custom text DB or
                  from a "global" DB built from the dataset's own report snippets.

Zero-shot and retrieval need a text-embedding model to encode prompts/snippets
into the model's text space. On the synthetic dummy dataset use the "hash"
backend (deterministic pseudo-random vectors) so everything runs without a heavy
model; on real data use "sentence_transformer" / "auto_model" with the same model
that produced the training text embeddings.

Usage:
  python evaluate_internal.py --config config/evaluate_internal.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import eval_common as ec  # adds RadPRISM/src to sys.path on import
from dataset import (  # noqa: E402
    CXRMultimodalDataset, PTCachedCXRDataset, make_dataloader, apply_group_splits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_split_path(eval_cfg: dict, run_dir: str, paths_cfg: dict, split_on: str) -> str:
    """Prefer an explicit override, then the run's own split file, then the config's."""
    override = (eval_cfg.get("data") or {}).get("split_path")
    candidates = [override,
                  os.path.join(run_dir, f"splits_{split_on}.parquet"),
                  os.path.join(run_dir, "splits.parquet"),
                  paths_cfg.get("split_path")]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(f"No split file found (checked: {candidates}).")


def build_split_dataset(cfg: dict, split_df: pd.DataFrame, align_names: List[str], cls_names: List[str]):
    """Rebuild an evaluation dataset for one split, mirroring the training setup."""
    paths, data = cfg["paths"], cfg["data"]
    cached = paths.get("cached_dataset_path") or data.get("preprocessed_dir")
    common = dict(text_concept_names=align_names, cls_concept_names=cls_names,
                  include_cls_labels=True, image_size=data.get("image_size", 518),
                  augment=False, norm_mode=data.get("norm_mode", "rad_dino_maira_2"))

    if cached and os.path.exists(cached):
        return PTCachedCXRDataset(
            split_df, align_names,
            cached_dataset_path=cached,
            accession_column=data.get("accession_column", "accession"),
            sop_uid_column=data.get("sop_uid_column", "SOPInstanceUID"),
            text_override_dir=data.get("text_override_dir"),
            use_text_override=data.get("use_text_override", False),
            text_override_mode=ec_coerce(data.get("text_override_mode"), "sharded"),
            text_override_strict=data.get("text_override_strict", False),
            cls_override_dir=data.get("cls_override_dir"),
            use_cls_override=data.get("use_cls_override", False),
            cls_override_mode=ec_coerce(data.get("cls_override_mode"), "sharded"),
            cls_override_strict=data.get("cls_override_strict", False),
            **common,
        )
    return CXRMultimodalDataset(
        split_df, align_names,
        embedding_dir=paths.get("embedding_dir"), label_dir=paths.get("label_dir"),
        **common,
    )


def ec_coerce(val, default):
    if val is None:
        return default
    return val[0] if isinstance(val, list) and val else (val if not isinstance(val, list) else default)


# ─────────────────────────────────────────────────────────────────────────────
# Score collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_scores(model, dl, device, cls_align_indices, zs_banks=None, zs_temp=1.0):
    """
    Run the model over a split and collect cls-head probabilities, optional
    zero-shot probabilities, and labels/masks.
    """
    model.eval()
    model.cfg.record_attn_maps = False
    idx_t = torch.tensor(list(cls_align_indices), dtype=torch.long, device=device)
    cls_c, zs_c, lab_c, mask_c = [], [], [], []
    for batch in dl:
        imgs = batch["images"].to(device)
        out = model(images=imgs, need_attn=False)
        logits = out.get("concept_logits")
        if logits is None:
            raise ValueError("Model has no concept_logits — classification heads are disabled.")
        cls_c.append(torch.sigmoid(logits).float().cpu().numpy())
        if zs_banks is not None:
            v = F.normalize(out["v_concepts"].index_select(1, idx_t), dim=-1)
            sim_pos = torch.einsum("bkd,kd->bk", v, zs_banks[0])
            sim_neg = torch.einsum("bkd,kd->bk", v, zs_banks[1])
            zs_c.append(torch.sigmoid((sim_pos - sim_neg) / max(zs_temp, 1e-6)).float().cpu().numpy())
        lab_c.append(batch["cls_labels"].numpy().astype(np.float32))
        mask_c.append(batch["cls_mask"].numpy().astype(bool))
    cls = np.concatenate(cls_c, 0)
    zs = np.concatenate(zs_c, 0) if zs_c else None
    return cls, zs, np.concatenate(lab_c, 0), np.concatenate(mask_c, 0)


def metrics_table(scores, labels, mask, concept_names) -> pd.DataFrame:
    """Per-concept AUROC/AUPRC/best-F1 over valid entries."""
    rows = []
    for k, cname in enumerate(concept_names):
        m = mask[:, k]
        if m.any():
            r = ec.per_concept_ranking_metrics(labels[m, k], scores[m, k])
        else:
            r = {"auroc": float("nan"), "auprc": float("nan"), "best_f1": 0.0, "thr_f1": 0.5}
        rows.append({"concept": cname, "n": int(m.sum()), **r})
    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot prompt banks
# ─────────────────────────────────────────────────────────────────────────────

def build_zero_shot_banks(model, cls_names, prompt_db, embedder, device, pooling="mean"):
    """Encode positive/negative prompts per concept into pooled query-space banks."""
    pos_bank, neg_bank = [], []
    for cname in cls_names:
        item = prompt_db.get(cname)
        if not item or not item.get("positive") or not item.get("negative"):
            raise ValueError(f"Prompt DB missing positive/negative prompts for '{cname}'.")
        pos_raw = embedder.encode(item["positive"])
        neg_raw = embedder.encode(item["negative"])
        pos = ec.project_text_to_query_space(pos_raw, model)
        neg = ec.project_text_to_query_space(neg_raw, model)
        pool = (lambda t: t.mean(0) if pooling == "mean" else t.max(0).values)
        pos_bank.append(F.normalize(pool(pos), dim=0))
        neg_bank.append(F.normalize(pool(neg), dim=0))
    return torch.stack(pos_bank).to(device), torch.stack(neg_bank).to(device)


def load_prompt_db(path: str, cls_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
    data = ec.load_json(path)
    if "concept_prompts" in data:
        data = data["concept_prompts"]
    out = {}
    for cname in cls_names:
        item = data.get(cname)
        if not isinstance(item, dict):
            raise KeyError(f"Prompt DB missing concept '{cname}'.")
        out[cname] = {"positive": [str(x) for x in (item.get("positive") or [])],
                      "negative": [str(x) for x in (item.get("negative") or [])]}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RadPRISM internal-dataset evaluation.")
    ap.add_argument("--config", required=True, help="Path to config/evaluate_internal.yaml")
    ap.add_argument("--device", default=None, help="Override model.device.")
    args = ap.parse_args()

    cfg_yaml = ec.load_yaml(args.config)
    mcfg = cfg_yaml.get("model", {})
    run_dir = mcfg["run_dir"]
    device = args.device or mcfg.get("device", "cpu")
    modes = cfg_yaml.get("modes", {})

    # Resolve the trained run config (merging the parent pretrain config for fine-tune runs).
    cfg = ec.resolve_effective_run_config(
        run_dir,
        use_parent_pretrain_config=mcfg.get("use_parent_pretrain_config", True),
        parent_pretrain_config_path=mcfg.get("parent_pretrain_config_path"),
        auto_parent_from_finetune_config=mcfg.get("auto_parent_from_finetune_config", True),
    )
    # Apply eval-side data overrides (e.g. point at a different split/cache).
    cfg.setdefault("paths", {})
    cfg.setdefault("data", {})
    for k, v in (cfg_yaml.get("data") or {}).items():
        if v is not None:
            cfg["data"][k] = v
            if k in ("cached_dataset_path", "master_index_path", "split_path", "embedding_dir", "label_dir"):
                cfg["paths"][k] = v

    out_dir = Path(cfg_yaml.get("output_dir") or os.path.join(run_dir, "eval_internal", time.strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] output dir: {out_dir}")

    # Build model + load weights.
    model, align_names, cls_names, cls_head_map = ec.build_model_from_config(cfg, device)
    ckpt_path = os.path.join(run_dir, mcfg.get("checkpoint_name", "best_cls.pt"))
    ec.load_model_weights(model, ckpt_path)
    cls_align_indices = cls_head_map if cls_head_map is not None else list(range(len(cls_names)))

    # Load master index + splits.
    master_path = cfg["paths"]["master_index_path"]
    master = pd.read_csv(master_path) if master_path.endswith(".csv") else pd.read_parquet(master_path)
    split_on = cfg["data"].get("split_on", "patient")
    split_path = _resolve_split_path(cfg_yaml, run_dir, cfg["paths"], split_on)
    splits = cfg_yaml.get("data", {}).get("splits", ["val"])
    print(f"[eval] splits={splits} | split_on={split_on} | split_file={split_path}")

    # ── Text embedder for zero-shot / retrieval (built lazily) ──
    embedder = None
    def get_embedder():
        nonlocal embedder
        if embedder is None:
            ez = cfg_yaml.get("text_embedding", {})
            embedder = ec.TextEmbedder(
                backend=ez.get("backend", "hash"),
                dim=int(cfg["model"].get("text_in_dim", 768)),
                model_dir=ez.get("model_dir"),
                device=ez.get("device", device),
                precision=ez.get("precision", "float32"),
                pooling=ez.get("pooling", "mean"),
                max_length=ez.get("max_length", 512),
                batch_size=ez.get("batch_size", 32),
                truncate_dim=ez.get("truncate_dim"),
            )
        return embedder

    # ── Zero-shot banks (optional) ──
    zs_banks = None
    if modes.get("zero_shot", False):
        zs_path = cfg_yaml.get("zero_shot", {}).get("prompt_db_path")
        if not zs_path or not os.path.exists(str(zs_path)):
            print("[zero_shot] SKIPPED: prompt_db_path not set or missing.")
        else:
            prompt_db = load_prompt_db(zs_path, cls_names)
            zs_banks = build_zero_shot_banks(
                model, cls_names, prompt_db, get_embedder(), device,
                pooling=cfg_yaml.get("zero_shot", {}).get("prompt_pooling", "mean"),
            )
            print(f"[zero_shot] built prompt banks for {len(cls_names)} concepts.")

    # ── Retrieval DB (optional) ──
    retrieval_db = None
    if modes.get("retrieve", False):
        rcfg = cfg_yaml.get("retrieve", {})
        custom_path = rcfg.get("custom_text_db_path")
        if custom_path and os.path.exists(str(custom_path)):
            custom = ec.load_json(custom_path)
            if "concept_texts" in custom:
                custom = custom["concept_texts"]
            retrieval_db = ec.build_retrieval_db_from_custom(custom, align_names, get_embedder())
            print(f"[retrieve] built custom-text retrieval DB over {len(retrieval_db)} concepts.")
        else:
            print("[retrieve] SKIPPED: custom_text_db_path not set/missing "
                  "(global-DB-from-dataset construction is documented for real datasets).")

    zs_temp = float(cfg_yaml.get("zero_shot", {}).get("temperature", 1.0))
    bs = int(cfg_yaml.get("data", {}).get("batch_size", 32))
    nw = int(cfg_yaml.get("data", {}).get("num_workers", 0))
    max_samples = cfg_yaml.get("data", {}).get("max_samples")

    all_val_thresholds = {}
    for split in splits:
        split_df = apply_group_splits(master, split_path, split_on=split_on, split=split)
        if max_samples:
            split_df = split_df.sample(n=min(int(max_samples), len(split_df)), random_state=1337)
        if len(split_df) == 0:
            print(f"[eval] split '{split}' empty, skipping."); continue
        ds = build_split_dataset(cfg, split_df, align_names, cls_names)
        dl = make_dataloader(ds, max(8, bs), shuffle=False, num_workers=nw, drop_last=False)
        print(f"[eval] split '{split}': {len(ds)} samples")

        # Classification / thresholds / zero-shot all need cls-head scores.
        if modes.get("classify", False) or modes.get("thresholds", False) or zs_banks is not None:
            cls_scores, zs_scores, labels, mask = collect_scores(
                model, dl, device, cls_align_indices, zs_banks=zs_banks, zs_temp=zs_temp)

            if modes.get("thresholds", False):
                thr = {}
                for k, cname in enumerate(cls_names):
                    m = mask[:, k]
                    thr[cname] = ec.compute_best_thresholds(cls_scores[m, k], labels[m, k]) if m.any() else None
                (out_dir / f"{split}_thresholds.json").write_text(json.dumps(thr, indent=2))
                if split == "val":
                    all_val_thresholds = thr
                print(f"[thresholds] wrote {split}_thresholds.json")

            if modes.get("classify", False):
                df = metrics_table(cls_scores, labels, mask, cls_names)
                df.to_csv(out_dir / f"{split}_cls_head_metrics.csv", index=False)
                print(f"[classify] {split} cls_head macro AUROC={np.nanmean(df['auroc']):.4f} "
                      f"AUPRC={np.nanmean(df['auprc']):.4f} -> {split}_cls_head_metrics.csv")
                if zs_scores is not None:
                    zdf = metrics_table(zs_scores, labels, mask, cls_names)
                    zdf.to_csv(out_dir / f"{split}_zero_shot_metrics.csv", index=False)
                    print(f"[classify] {split} zero_shot macro AUROC={np.nanmean(zdf['auroc']):.4f} "
                          f"AUPRC={np.nanmean(zdf['auprc']):.4f} -> {split}_zero_shot_metrics.csv")

        # Retrieval: top-k text matches per concept for each image.
        if retrieval_db is not None:
            rows = run_retrieval(model, dl, device, retrieval_db, align_names,
                                 top_k=int(cfg_yaml.get("retrieve", {}).get("top_k", 3)))
            pd.DataFrame(rows).to_csv(out_dir / f"{split}_retrieval.csv", index=False)
            print(f"[retrieve] wrote {split}_retrieval.csv ({len(rows)} rows)")

    print("\n[eval] done.")


@torch.no_grad()
def run_retrieval(model, dl, device, retrieval_db, align_names, top_k=3):
    """Collect per-image top-k retrieval matches over a split."""
    model.eval()
    model.cfg.record_attn_maps = False
    rows = []
    sample_idx = 0
    for batch in dl:
        imgs = batch["images"].to(device)
        out = model(images=imgs, need_attn=False)
        v = out["v_concepts"].detach().cpu()
        accs = batch.get("accession", [None] * v.shape[0])
        for b in range(v.shape[0]):
            matches = ec.query_retrieval_db(v[b], retrieval_db, model, align_names, top_k=top_k)
            for m in matches:
                m2 = {"sample_idx": sample_idx, "accession": str(accs[b]) if accs is not None else "", **m}
                rows.append(m2)
            sample_idx += 1
    return rows


if __name__ == "__main__":
    main()
