#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RadPRISM Inference Demo
=======================

Apply the provided RadPRISM checkpoint to chest X-ray images and save, per image:

  - attention-overlay PNGs (one per concept, or a selected subset),
  - a classification CSV: per-concept probability + binary prediction using the
    provided validation thresholds (Youden-J by default),
  - a top-3 concept-wise text-retrieval CSV, showing the ENGLISH translation of
    the best-matching snippets.

The model was trained with German text embeddings, so retrieval/zero-shot use
pre-computed German Qwen embeddings (``retrieval_bank.pt`` / ``zero_shot_bank.pt``,
produced by utils/embed_radprism_text_db.py); the displayed retrieval text is the
English translation. No text-embedding model is needed at inference time.

The checkpoint ships as trained weights only (``radprism_heads.safetensors``); the
frozen RAD-DINO-MAIRA-2 backbone is loaded from a local HuggingFace directory.

Usage:
  python evaluation/run_inference_demo.py --config config/inference_demo.yaml
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import eval_common as ec  # adds RadPRISM/src to sys.path on import
from dataset import NormalizeCXR  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing + attention overlay
# ─────────────────────────────────────────────────────────────────────────────

def load_image_float01(path: str) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    a, b = float(arr.min()), float(arr.max())
    return (arr.astype(np.float32) - a) / max(b - a, 1.0)


def letterbox(x: np.ndarray, size: int, pad_value: float = 0.0) -> np.ndarray:
    h, w = x.shape
    scale = size / float(max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(Image.fromarray(x).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32)
    canvas = np.full((size, size), float(pad_value), dtype=np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def window_for_display(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Percentile contrast-stretch a [0,1] image so it is reliably visible as a
    grayscale background (display only — never applied to the model input).

    Raw CXR JPEGs vary widely in intensity/contrast; without this, dark or
    low-contrast images render nearly black and disappear under the attention
    colormap. Percentiles are computed over the non-padding pixels so the
    letterbox border does not skew the window.
    """
    valid = img[img > 0]                    # ignore the zero-valued letterbox padding
    if valid.size == 0:
        valid = img.reshape(-1)
    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))
    if hi <= lo:
        return np.clip(img, 0.0, 1.0)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def overlay_attention(img01: np.ndarray, attn_small: np.ndarray, out_path: str,
                      alpha: float = 0.5, vmax: Optional[float] = None,
                      cmap: Optional[str] = None):
    """
    Overlay an attention heatmap on a grayscale CXR, matching the renderer used by
    the other evaluation scripts (matplotlib imshow compositing).

    The background image is percentile-windowed for robust visibility, and the
    attention map is upsampled to the image size and clamped to [0, 1]. ``vmax``
    fixes the colour scale (e.g. 0.1) so maps are comparable across concepts/images;
    when None, matplotlib autoscales per map (higher contrast, not comparable).
    ``cmap`` None uses matplotlib's default colormap (viridis).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = img01.shape[0]
    a = torch.tensor(attn_small, dtype=torch.float32)[None, None]
    a = F.interpolate(a, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    a = np.clip(a, 0.0, 1.0)

    bg = window_for_display(img01)           # display-only contrast stretch
    fig = plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
    if vmax is not None:
        plt.imshow(a, alpha=alpha, vmin=0.0, vmax=vmax, cmap=cmap)
    else:
        plt.imshow(a, alpha=alpha, cmap=cmap)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (heads-only + HF backbone)
# ─────────────────────────────────────────────────────────────────────────────

def build_radprism_model(cfg: dict, rad_dino_dir: str, heads_path: str, device: str):
    m = dict(cfg["model"])
    m["rad_dino_model_dir"] = rad_dino_dir
    run_cfg = {"model": m, "align_concept_names": cfg["align_concept_names"],
               "cls_concept_names": cfg["cls_concept_names"], "cls_head_map": cfg.get("cls_head_map")}
    model, align_names, cls_names, cls_head_map = ec.build_model_from_config(run_cfg, device, record_attn_maps=True)

    # Load the trained (non-backbone) weights on top of the HF backbone.
    if heads_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(heads_path)
    else:
        state = torch.load(heads_path, map_location="cpu", weights_only=True)
        state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_backbone_missing = [k for k in missing if not k.startswith("backbone")]
    if non_backbone_missing:
        print(f"[demo] WARNING: {len(non_backbone_missing)} trained tensors missing from heads file.")
    if unexpected:
        print(f"[demo] WARNING: {len(unexpected)} unexpected tensors in heads file.")
    model.eval()
    return model, align_names, cls_names, cls_head_map


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RadPRISM inference demo.")
    ap.add_argument("--config", required=True, help="config/inference_demo.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    C = ec.load_yaml(args.config)
    device = args.device or C.get("device", "cpu")
    ckpt_dir = Path(C["checkpoint_dir"])
    cfg = ec.load_json(str(ckpt_dir / C.get("config_name", "radprism_config.json")))
    heads_path = str(ckpt_dir / C.get("heads_name", "radprism_heads.safetensors"))
    rad_dino_dir = C["rad_dino_model_dir"]
    out_dir = Path(C.get("output_dir", "output/inference_demo"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, align_names, cls_names, cls_head_map = build_radprism_model(cfg, rad_dino_dir, heads_path, device)
    cls_align_idx = cls_head_map if cls_head_map is not None else list(range(len(cls_names)))

    # Thresholds (Youden-J by default).
    thr_key = C.get("threshold_key", "thr_j")
    thr_json = ec.load_json(str(ckpt_dir / C.get("thresholds_name", "radprism_val_thresholds.json")))
    thresholds = {c: float(thr_json.get(c, {}).get(thr_key, 0.5)) for c in cls_names}

    # Retrieval bank (raw German Qwen embeddings + English display text).
    retrieval_db = None
    rb_path = ckpt_dir / C.get("retrieval_bank_name", "retrieval_bank.pt")
    if C.get("do_retrieval", True) and rb_path.exists():
        bank = torch.load(rb_path, map_location="cpu", weights_only=True)
        retrieval_db = {int(k): {"raw": v["raw"].numpy() if torch.is_tensor(v["raw"]) else np.asarray(v["raw"]),
                                 "meta": v["meta"]} for k, v in bank.items()}
        print(f"[demo] loaded retrieval bank ({len(retrieval_db)} concepts).")
    elif C.get("do_retrieval", True):
        print(f"[demo] retrieval bank not found at {rb_path} — run utils/embed_radprism_text_db.py first. "
              "Skipping retrieval.")

    # Input images.
    images = []
    for pat in (C.get("image_glob") or ["data/jpg/*.jpg"]):
        images.extend(sorted(glob.glob(pat)))
    if C.get("max_images"):
        images = images[: int(C["max_images"])]
    if not images:
        raise FileNotFoundError("No input images matched image_glob.")
    print(f"[demo] {len(images)} input images -> {out_dir}")

    size = int(cfg.get("image_size", 518))
    normalizer = NormalizeCXR(mode=cfg.get("norm_mode", "rad_dino_maira_2"))
    overlay_concepts = C.get("overlay_concepts")   # None -> all
    top_k = int(C.get("retrieval_top_k", 3))
    overlay_alpha = float(C.get("overlay_alpha", 0.5))
    overlay_vmax = C.get("overlay_vmax")           # None -> per-map autoscale
    overlay_cmap = C.get("overlay_cmap")           # None -> matplotlib default (viridis)

    for img_path in images:
        stem = Path(img_path).stem
        s_dir = out_dir / stem
        s_dir.mkdir(parents=True, exist_ok=True)

        img01 = letterbox(load_image_float01(img_path), size)
        x = normalizer(torch.from_numpy(img01).unsqueeze(0).repeat(3, 1, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(images=x, need_attn=True)
        probs = torch.sigmoid(out["concept_logits"][0]).cpu().numpy()
        v_concepts = out["v_concepts"][0].detach().cpu()
        attn = out.get("concept_attn")

        # Classification CSV.
        pd.DataFrame([{
            "concept": c, "probability": float(probs[k]),
            "threshold": thresholds[c], "prediction": int(probs[k] >= thresholds[c]),
        } for k, c in enumerate(cls_names)]).to_csv(s_dir / "classification.csv", index=False)

        # Attention overlays.
        if attn is not None and C.get("do_overlays", True):
            ov_dir = s_dir / "attention_overlays"; ov_dir.mkdir(exist_ok=True)
            names = overlay_concepts if overlay_concepts else align_names
            name_to_aidx = {n: i for i, n in enumerate(align_names)}
            for n in names:
                if n in name_to_aidx:
                    overlay_attention(img01, attn[0][name_to_aidx[n]].cpu().numpy(),
                                      str(ov_dir / f"{n.replace('.', '_')}.png"),
                                      alpha=overlay_alpha, vmax=overlay_vmax, cmap=overlay_cmap)

        # Retrieval (English display text).
        if retrieval_db is not None:
            matches = ec.query_retrieval_db(v_concepts, retrieval_db, model, align_names, top_k=top_k)
            pd.DataFrame(matches).to_csv(s_dir / "retrieval_top_matches.csv", index=False)

        print(f"[demo] {stem}: wrote classification.csv"
              + (", overlays" if attn is not None and C.get('do_overlays', True) else "")
              + (", retrieval_top_matches.csv" if retrieval_db is not None else ""))

    print("\n[demo] done.")


if __name__ == "__main__":
    main()
