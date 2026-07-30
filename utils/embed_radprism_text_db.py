#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-compute the German text embeddings for the RadPRISM checkpoint demo.

The provided RadPRISM checkpoint was trained on German Qwen3-Embedding-4B text
vectors, so the zero-shot prompts and retrieval snippets must be embedded with
the SAME model to land in the model's text space. This script encodes the German
texts in ``radprism_text_db_de.json`` once and writes two small banks that the
inference demo consumes — so the demo itself needs no embedding model.

Run this once with your local Qwen3-Embedding-4B, using the SAME encoding
parameters as the training text embeddings (see the report pipeline's
config/embedding_sentence_transformer.yaml): truncate_dim=768, max_length=2048.

  python utils/embed_radprism_text_db.py \\
    --db data/radprism_checkpoint/radprism_text_db_de.json \\
    --config data/radprism_checkpoint/radprism_config.json \\
    --model-dir /path/to/Qwen3-Embedding-4B \\
    --out-dir data/radprism_checkpoint

Outputs (commit these so the demo runs out of the box):
  retrieval_bank.pt   {align_idx: {"raw": [N,768] float32, "meta": [{"text": <EN>, "de": <DE>}]}}
  zero_shot_bank.pt   {align_idx: {"pos_raw": [P,768], "neg_raw": [Q,768]}}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="Embed the German RadPRISM text DB with Qwen.")
    ap.add_argument("--db", required=True, help="radprism_text_db_de.json")
    ap.add_argument("--config", required=True, help="radprism_config.json (for concept order)")
    ap.add_argument("--model-dir", required=True, help="Local Qwen3-Embedding-4B directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--truncate-dim", type=int, default=768)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-normalize", action="store_true",
                    help="Disable L2 normalization (match your training embedding config).")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    db = json.load(open(args.db, encoding="utf-8"))["concepts"]
    cfg = json.load(open(args.config))
    align_names = cfg["align_concept_names"]
    name_to_idx = {n: i for i, n in enumerate(align_names)}
    normalize = not args.no_normalize

    print(f"[embed] loading {args.model_dir} on {args.device}")
    model = SentenceTransformer(args.model_dir, device=args.device)
    try:
        model.max_seq_length = int(args.max_length)
    except Exception:
        pass

    def encode(texts):
        vecs = model.encode(list(texts), batch_size=args.batch_size, convert_to_numpy=True,
                            normalize_embeddings=False,
                            show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)[:, :args.truncate_dim]
        if normalize:
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return vecs.astype(np.float32, copy=False)

    retrieval_bank, zero_shot_bank = {}, {}
    for cname, entry in db.items():
        if cname not in name_to_idx:
            print(f"[embed] WARNING: '{cname}' not in checkpoint concepts, skipped.")
            continue
        aidx = name_to_idx[cname]

        # Retrieval: embed the German text, keep the English translation for display.
        retr = entry.get("retrieval", [])
        if retr:
            raw = encode([r["de"] for r in retr])
            retrieval_bank[aidx] = {
                "raw": torch.from_numpy(raw),
                "meta": [{"text": r.get("en", r["de"]), "de": r["de"], "concept": cname} for r in retr],
            }

        # Zero-shot: embed German positive/negative prompts.
        zs = entry.get("zero_shot", {})
        pos, neg = zs.get("positive", []), zs.get("negative", [])
        if pos and neg:
            zero_shot_bank[aidx] = {
                "pos_raw": torch.from_numpy(encode(pos)),
                "neg_raw": torch.from_numpy(encode(neg)),
            }

    out = Path(args.out_dir)
    torch.save(retrieval_bank, out / "retrieval_bank.pt")
    torch.save(zero_shot_bank, out / "zero_shot_bank.pt")
    print(f"[embed] wrote retrieval_bank.pt ({len(retrieval_bank)} concepts) and "
          f"zero_shot_bank.pt ({len(zero_shot_bank)} concepts) -> {out}")
    print(f"[embed] embedding dim={args.truncate_dim}, normalize={normalize}")


if __name__ == "__main__":
    main()
