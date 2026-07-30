#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a sharded image cache for fast CXR training / evaluation.

Reads the images listed in a master index (CSV/parquet), applies the same
grayscale -> float[0,1] -> (optional contrast) -> letterbox-resize preprocessing
used at training time, and writes them to disk as shards:

  images_00000.npy            # (N, H, W) float16/float32/uint16
  meta_00000.parquet          # per-row: sop_uid, SOPInstanceUID, accession, ...
  image_sharded_index.parquet # sop_uid/accession -> (shard, row) lookup
  summary.json                # preprocessing parameters

This is the ``cached_dataset_path`` / ``preprocessed_dir`` consumed by
``PTCachedCXRDataset`` (see src/dataset.py). Any PIL-readable image format works
(PNG, JPEG, ...).

Example (dummy dataset):
  python prepare_sharded_image_cache.py \\
    --master_index ../data/dummy_master_index.csv \\
    --out_dir ../data/dummy_image_cache \\
    --image_column image_path --sop_uid_column SOPInstanceUID \\
    --accession_column accession --image_size 518 --keep_aspect_ratio \\
    --shard_size 4096 --storage_dtype float16
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


try:
    _RESAMPLE = Image.Resampling  # Pillow >= 9
except AttributeError:  # pragma: no cover
    _RESAMPLE = Image


RESAMPLE_MAP = {
    "nearest": _RESAMPLE.NEAREST,
    "bilinear": _RESAMPLE.BILINEAR,
    "bicubic": _RESAMPLE.BICUBIC,
    "lanczos": _RESAMPLE.LANCZOS,
}


def load_png_to_float01_1ch(path: str) -> np.ndarray:
    """
    Load PNG as single-channel float32 in [0, 1].
    Supports 8-bit and 16-bit PNG.
    """
    with Image.open(path) as img:
        arr = np.array(img)

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.integer):
        a = float(arr.min())
        b = float(arr.max())
        x = (arr.astype(np.float32) - a) / max(b - a, 1.0)
    else:
        x = np.asarray(arr, dtype=np.float32)
        x = np.clip(x, 0.0, 1.0)

    return x


def apply_hist_eq_01(x: np.ndarray, bins: int = 2048) -> np.ndarray:
    """
    Global histogram equalization on [0,1] image.
    """
    v = np.clip(x, 0.0, 1.0).ravel()
    hist, bin_edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] <= 0:
        return x.astype(np.float32, copy=False)
    cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1e-12)
    out = np.interp(v, bin_edges[:-1], cdf).reshape(x.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def apply_percentile_contrast_01(
    x: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    lo = float(np.percentile(x, low_pct))
    hi = float(np.percentile(x, high_pct))
    if hi <= lo:
        return x.astype(np.float32, copy=False)
    y = (x.astype(np.float32) - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def apply_contrast(
    x: np.ndarray,
    mode: str,
    low_pct: float,
    high_pct: float,
) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "hist_eq":
        return apply_hist_eq_01(x)
    if mode == "percentile":
        return apply_percentile_contrast_01(x, low_pct=low_pct, high_pct=high_pct)
    raise ValueError(f"Unsupported contrast mode: {mode}")


def resize_to_square(
    x: np.ndarray,
    out_size: int,
    keep_aspect_ratio: bool,
    pad_value: float,
    resample_name: str,
) -> np.ndarray:
    """
    Resize image to out_size x out_size.
    If keep_aspect_ratio=True, uses letterbox padding.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {x.shape}")

    h, w = x.shape
    pil = Image.fromarray(x.astype(np.float32), mode="F")
    resample = RESAMPLE_MAP[resample_name]

    if not keep_aspect_ratio:
        out = pil.resize((out_size, out_size), resample=resample)
        return np.asarray(out, dtype=np.float32)

    scale = float(out_size) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = pil.resize((new_w, new_h), resample=resample)
    resized_arr = np.asarray(resized, dtype=np.float32)

    canvas = np.full((out_size, out_size), float(pad_value), dtype=np.float32)
    y0 = (out_size - new_h) // 2
    x0 = (out_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized_arr
    return canvas


def detect_existing_shards(out_dir: Path) -> List[int]:
    meta_idxs = {
        int(p.stem.split("_")[1])
        for p in out_dir.glob("meta_*.parquet")
        if "_" in p.stem
    }
    img_idxs = {
        int(p.stem.split("_")[1])
        for p in out_dir.glob("images_*.npy")
        if "_" in p.stem
    }
    return sorted(meta_idxs & img_idxs)


@dataclass
class ShardWriter:
    out_dir: Path
    shard_size: int
    storage_dtype: str
    start_shard_idx: int = 0

    def __post_init__(self):
        self._images: List[np.ndarray] = []
        self._meta_rows: List[Dict] = []
        self._index_rows: List[Tuple[str, Optional[str], int, int]] = []
        self.shard_idx = int(self.start_shard_idx)
        self.written_images = 0

    def _encode_storage_dtype(self, arr: np.ndarray) -> np.ndarray:
        if self.storage_dtype == "float16":
            return arr.astype(np.float16, copy=False)
        if self.storage_dtype == "float32":
            return arr.astype(np.float32, copy=False)
        if self.storage_dtype == "uint16":
            return np.clip(arr * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        raise ValueError(f"Unsupported storage dtype: {self.storage_dtype}")

    def add(self, image_2d_float01: np.ndarray, meta: Dict) -> None:
        self._images.append(image_2d_float01)
        self._meta_rows.append(meta)
        if len(self._images) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._images:
            return

        images = np.stack(self._images, axis=0)
        images = self._encode_storage_dtype(images)

        meta_rows = []
        for row_idx, row in enumerate(self._meta_rows):
            r = dict(row)
            r["shard"] = int(self.shard_idx)
            r["row"] = int(row_idx)
            meta_rows.append(r)
            self._index_rows.append(
                (
                    str(r["sop_uid"]),
                    str(r.get("accession")) if r.get("accession") is not None else None,
                    int(self.shard_idx),
                    int(row_idx),
                )
            )

        img_path = self.out_dir / f"images_{self.shard_idx:05d}.npy"
        meta_path = self.out_dir / f"meta_{self.shard_idx:05d}.parquet"
        np.save(img_path, images, allow_pickle=False)
        pd.DataFrame(meta_rows).to_parquet(meta_path, index=False)

        self.written_images += len(meta_rows)
        self.shard_idx += 1
        self._images.clear()
        self._meta_rows.clear()

    @property
    def index_rows(self) -> List[Tuple[str, Optional[str], int, int]]:
        return self._index_rows


def detect_index_format(path: str) -> str:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return "parquet"
    if p.endswith(".csv"):
        return "csv"
    raise ValueError(f"Unsupported master index format for {path}. Use .csv or .parquet")


def read_master_index(path: str) -> pd.DataFrame:
    fmt = detect_index_format(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description="Build sharded image cache for CXR training.")
    ap.add_argument("--master_index", required=True, help="CSV/Parquet master index.")
    ap.add_argument("--out_dir", required=True, help="Output directory for sharded image cache.")
    ap.add_argument("--image_column", default="image_path")
    ap.add_argument("--accession_column", default="accession")
    ap.add_argument("--sop_uid_column", default="SOPInstanceUID")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--shard_size", type=int, default=4096)
    ap.add_argument("--storage_dtype", choices=["float16", "float32", "uint16"], default="float16")
    ap.add_argument("--keep_aspect_ratio", action="store_true", help="Preserve aspect ratio by padding to square.")
    ap.add_argument("--pad_value", type=float, default=0.0)
    ap.add_argument("--resample", choices=list(RESAMPLE_MAP.keys()), default="bilinear")
    ap.add_argument(
        "--contrast_mode",
        choices=["none", "hist_eq", "percentile"],
        default="none",
        help="Optional per-image contrast preprocessing before resize.",
    )
    ap.add_argument("--contrast_low_pct", type=float, default=1.0)
    ap.add_argument("--contrast_high_pct", type=float, default=99.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Append to existing shards in out_dir.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_master_index(args.master_index)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    for c in [args.image_column, args.sop_uid_column]:
        if c not in df.columns:
            raise KeyError(f"master index missing required column: {c}")
    if args.accession_column not in df.columns:
        print(f"[WARN] accession column '{args.accession_column}' missing. accession will be empty in meta/index.")

    existing_sops = set()
    existing_index_path = out_dir / "image_sharded_index.parquet"
    existing_index_df = None
    start_shard_idx = 0

    existing_shards = detect_existing_shards(out_dir)
    if existing_shards:
        if not args.resume:
            raise FileExistsError(
                f"Found existing shard files in {out_dir}. Use --resume or a new output directory."
            )
        start_shard_idx = max(existing_shards) + 1
        if existing_index_path.exists():
            existing_index_df = pd.read_parquet(existing_index_path)
            if "sop_uid" in existing_index_df.columns:
                existing_sops = set(existing_index_df["sop_uid"].astype(str).tolist())
        print(f"[INFO] Resume enabled. Existing shards={len(existing_shards)}, starting at shard {start_shard_idx}.")

    writer = ShardWriter(
        out_dir=out_dir,
        shard_size=max(1, int(args.shard_size)),
        storage_dtype=args.storage_dtype,
        start_shard_idx=start_shard_idx,
    )

    skipped_existing = 0
    skipped_errors = 0
    processed = 0

    iterator = df.itertuples(index=False)
    for row in tqdm(iterator, total=len(df), ncols=90):
        rowd = row._asdict()
        sop = str(rowd.get(args.sop_uid_column, ""))
        if not sop:
            skipped_errors += 1
            continue

        if sop in existing_sops:
            skipped_existing += 1
            continue

        image_path = str(rowd.get(args.image_column, ""))
        accession = rowd.get(args.accession_column, None)
        accession = str(accession) if accession is not None else None

        if not image_path or not os.path.exists(image_path):
            skipped_errors += 1
            continue

        try:
            img = load_png_to_float01_1ch(image_path)
            src_h, src_w = int(img.shape[0]), int(img.shape[1])
            img = apply_contrast(
                img,
                mode=args.contrast_mode,
                low_pct=float(args.contrast_low_pct),
                high_pct=float(args.contrast_high_pct),
            )
            img = resize_to_square(
                img,
                out_size=int(args.image_size),
                keep_aspect_ratio=bool(args.keep_aspect_ratio),
                pad_value=float(args.pad_value),
                resample_name=args.resample,
            )
            img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
        except Exception as e:
            skipped_errors += 1
            print(f"[WARN] skip {sop}: {type(e).__name__}: {e}")
            continue

        meta = {
            "sop_uid": sop,
            "SOPInstanceUID": sop,
            "accession": accession,
            "image_path": image_path,
            "source_height": src_h,
            "source_width": src_w,
            "processed_height": int(args.image_size),
            "processed_width": int(args.image_size),
        }
        writer.add(img, meta)
        processed += 1

    writer.flush()

    new_index_df = pd.DataFrame(
        writer.index_rows,
        columns=["sop_uid", "accession", "shard", "row"],
    )

    if existing_index_df is not None and not existing_index_df.empty:
        # Normalize potential legacy column names
        if "SOPInstanceUID" in existing_index_df.columns and "sop_uid" not in existing_index_df.columns:
            existing_index_df = existing_index_df.rename(columns={"SOPInstanceUID": "sop_uid"})
        keep_cols = ["sop_uid", "accession", "shard", "row"]
        for c in keep_cols:
            if c not in existing_index_df.columns:
                existing_index_df[c] = None
        existing_index_df = existing_index_df[keep_cols]
        final_index_df = pd.concat([existing_index_df, new_index_df], ignore_index=True)
        final_index_df = final_index_df.drop_duplicates(subset=["sop_uid"], keep="last")
    else:
        final_index_df = new_index_df

    final_index_df.to_parquet(existing_index_path, index=False)

    summary = {
        "format": "sharded_image_cache_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_master_index": args.master_index,
        "image_column": args.image_column,
        "accession_column": args.accession_column,
        "sop_uid_column": args.sop_uid_column,
        "image_size": int(args.image_size),
        "shard_size": int(args.shard_size),
        "storage_dtype": args.storage_dtype,
        "keep_aspect_ratio": bool(args.keep_aspect_ratio),
        "pad_value": float(args.pad_value),
        "resample": args.resample,
        "contrast_mode": args.contrast_mode,
        "contrast_low_pct": float(args.contrast_low_pct),
        "contrast_high_pct": float(args.contrast_high_pct),
        "processed_new_images": int(processed),
        "skipped_existing": int(skipped_existing),
        "skipped_errors": int(skipped_errors),
        "num_images_total": int(len(final_index_df)),
        "num_shards_total": int(len(detect_existing_shards(out_dir))),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] Sharded image cache created.")
    print(f"  out_dir: {out_dir}")
    print(f"  processed_new_images: {processed}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  skipped_errors: {skipped_errors}")
    print(f"  num_images_total: {len(final_index_df)}")
    print(f"  num_shards_total: {summary['num_shards_total']}")


if __name__ == "__main__":
    main()
