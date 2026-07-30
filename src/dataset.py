# dataset.py
# ============================================================================
# Dataset and split utilities for the VisionConceptModel pretraining pipeline.
#
# Provides:
#   - Master-index builders that join image, report-embedding and label sources.
#   - Patient/accession/random group splitting helpers.
#   - CXR image transforms (normalization, augmentation, corner cutout).
#   - Two Dataset classes:
#       * CXRMultimodalDataset  - reads PNG images + embeddings/labels on the fly.
#       * PTCachedCXRDataset     - reads pre-cached (sharded or per-SOP) tensors,
#                                  with optional text/label "override" stores.
#   - make_dataloader / compute_concept_sample_weights for concept-aware
#     weighted sampling (see the training script).
# ============================================================================

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# define as plain tuples (not tensors) at module level
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

RAD_DINO_MAIRA_2_MEAN = (0.5307, 0.5307, 0.5307)
RAD_DINO_MAIRA_2_STD  = (0.2583, 0.2583, 0.2583)
RAD_DINO_MAIRA_2_RESCALE = 1.0 / 255.0 # no rescaling needed if using ToTensor()


# -------------------------
# Path + build config
# -------------------------

@dataclass
class DatasetPaths:
    image_index_path: str   # parquet index of image files (SOPInstanceUID -> image_path)
    report_index_path: str  # parquet index of reports (accession -> report json path)
    embedding_dir: str      # directory of pre-computed report text embeddings
    label_dir: Optional[str] = None  # directory of structured report labels (optional)


@dataclass
class DatasetBuildConfig:
    """
    Controls which examples are kept in the master index.
    """
    require_image: bool = True
    require_embedding: bool = True
    require_label: bool = False
    max_rows: int | None = None      # new: limit for quick tests
    sample_frac: float | None = None # optional random fraction for quick tests
    seed: int = 42


# -------------------------
# Master index
# -------------------------

def build_master_index_fast(
    paths: DatasetPaths,
    cfg: DatasetBuildConfig,
) -> pd.DataFrame:
    # Load base tables
    img_df = pd.read_parquet(paths.image_index_path)
    rep_map = (
        pd.read_parquet(paths.report_index_path)[
            ["AccessionNumber_json", "AccessionPseudonym"]
        ]
        .rename(
            columns={
                "AccessionNumber_json": "accession",
                "AccessionPseudonym": "accession_pseudo",
            }
        )
    )

    img_df = img_df.rename(columns={"AccessionNumber": "accession_pseudo"})

    # Optional downsampling before merge to speed up tests
    if cfg.sample_frac is not None:
        img_df = img_df.sample(frac=cfg.sample_frac, random_state=cfg.seed)
    if cfg.max_rows is not None and len(img_df) > cfg.max_rows:
        img_df = img_df.sample(n=cfg.max_rows, random_state=cfg.seed)

    df = img_df.merge(rep_map, on="accession_pseudo", how="left")
    df = df[~df["accession"].isna()].copy()
    df["accession"] = df["accession"].astype(str)

    if "png_path" not in df.columns:
        raise KeyError("dataset_index.parquet must contain 'png_path'.")
    df["image_path"] = df["png_path"].astype(str)

    # Choose patient_id
    if "patient_number_norm" in df.columns:
        df["patient_id"] = df["patient_number_norm"].astype(str)
    elif "PatientID" in df.columns:
        df["patient_id"] = df["PatientID"].astype(str)
    else:
        df["patient_id"] = "unknown"

    # Pre-index embedding availability by filename stem
    if cfg.require_embedding or True:
        if not os.path.isdir(paths.embedding_dir):
            raise NotADirectoryError(paths.embedding_dir)
        emb_files = os.listdir(paths.embedding_dir)
        emb_accs = {os.path.splitext(f)[0] for f in emb_files if f.endswith(".parquet")}
    else:
        emb_accs = set()

    df["embedding_path"] = df["accession"].apply(
        lambda acc: os.path.join(paths.embedding_dir, f"{acc}.parquet")
    )
    df["has_embedding"] = df["accession"].isin(emb_accs)

    # Pre-index label availability
    if paths.label_dir is not None:
        if not os.path.isdir(paths.label_dir):
            raise NotADirectoryError(paths.label_dir)
        label_files = os.listdir(paths.label_dir)
        # filenames like "<acc>_chest_standard.json"
        label_accs = {
            f.split("_chest_standard.json")[0]
            for f in label_files
            if f.endswith("_chest_standard.json")
        }
        df["label_path"] = df["accession"].apply(
            lambda acc: os.path.join(paths.label_dir, f"{acc}_chest_standard.json")
        )
        df["has_label"] = df["accession"].isin(label_accs)
    else:
        df["label_path"] = ""
        df["has_label"] = False

    # If image existence is a concern and all png_paths are valid, you can skip this check for speed
    if cfg.require_image:
        df = df[df["image_path"].apply(os.path.exists)]

    if cfg.require_embedding:
        df = df[df["has_embedding"]]

    if cfg.require_label:
        df = df[df["has_label"]]

    df = df.reset_index(drop=True)
    return df

def build_master_index(
    paths: DatasetPaths,
    cfg: DatasetBuildConfig,
) -> pd.DataFrame:
    """
    Build a per-image master index.

    Output columns:
      - image_path
      - accession          (real accession, used for embeddings + labels)
      - accession_pseudo   (from image index)
      - patient_id         (best pseudonym)
      - embedding_path, has_embedding
      - label_path, has_label
    """
    img_df = pd.read_parquet(paths.image_index_path)

    rep_map = (
        pd.read_parquet(paths.report_index_path)[
            ["AccessionNumber_json", "AccessionPseudonym"]
        ]
        .rename(
            columns={
                "AccessionNumber_json": "accession",
                "AccessionPseudonym": "accession_pseudo",
            }
        )
    )

    if "AccessionNumber" not in img_df.columns:
        raise KeyError("dataset_index.parquet must contain 'AccessionNumber' (pseudonym).")

    img_df = img_df.rename(columns={"AccessionNumber": "accession_pseudo"})

    df = img_df.merge(rep_map, on="accession_pseudo", how="left")

    # Drop rows without mapping (cannot link to text/labels)
    df = df[~df["accession"].isna()].copy()
    df["accession"] = df["accession"].astype(str)

    if "png_path" not in df.columns:
        raise KeyError("dataset_index.parquet must contain 'png_path'.")
    df["image_path"] = df["png_path"].astype(str)

    # Patient id
    if "patient_number_norm" in df.columns:
        df["patient_id"] = df["patient_number_norm"].astype(str)
    elif "PatientID" in df.columns:
        df["patient_id"] = df["PatientID"].astype(str)
    else:
        df["patient_id"] = "unknown"

    # Embedding paths
    def _emb_path(acc: str) -> str:
        return os.path.join(paths.embedding_dir, f"{acc}.parquet")

    df["embedding_path"] = df["accession"].apply(_emb_path)
    df["has_embedding"] = df["embedding_path"].apply(os.path.exists)

    # Label paths
    if paths.label_dir is not None:
        def _label_path(acc: str) -> str:
            return os.path.join(paths.label_dir, f"{acc}_chest_standard.json")
        df["label_path"] = df["accession"].apply(_label_path)
        df["has_label"] = df["label_path"].apply(os.path.exists)
    else:
        df["label_path"] = ""
        df["has_label"] = False

    # Filter according to config
    mask = np.ones(len(df), dtype=bool)
    if cfg.require_image:
        mask &= df["image_path"].apply(os.path.exists).values
    if cfg.require_embedding:
        mask &= df["has_embedding"].values
    if cfg.require_label:
        mask &= df["has_label"].values

    df = df[mask].reset_index(drop=True)
    return df


# -------------------------
# Split utilities
# -------------------------

def _group_id(df: pd.DataFrame, split_on: str) -> pd.Series:
    """
    split_on: 'accession' | 'patient' | 'image'
    """
    if split_on == "accession":
        return df["accession"].astype(str)
    if split_on == "patient":
        # Prefer original PatientID for patient-level split; fallback keeps compatibility
        # with older prebuilt master indices that only contain patient_id.
        if "PatientID" in df.columns:
            return df["PatientID"].astype(str)
        if "patient_id" in df.columns:
            return df["patient_id"].astype(str)
        raise KeyError("split_on='patient' requires 'PatientID' or 'patient_id' column.")
    if split_on == "image":
        return pd.Series([f"img_{i}" for i in range(len(df))], index=df.index)
    raise ValueError(f"Unsupported split_on={split_on}")


def create_group_splits(
    df: pd.DataFrame,
    out_path: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_on: str = "accession",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create split assignment by groups to avoid leakage between views.

    Recommended:
      - split_on='accession': prevents different views of same report across splits.
      - split_on='patient': stricter, prevents any exams of same patient across splits.

    Writes Parquet with columns: group_id, split.
    """
    if not (0 <= val_frac < 1 and 0 <= test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("Require 0 <= val_frac,test_frac and val_frac+test_frac < 1.")

    df = df.copy()
    df["group_id"] = _group_id(df, split_on)

    groups = df["group_id"].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    train_groups = set(groups[n_test + n_val:])

    rows = []
    for g in groups:
        if g in test_groups:
            s = "test"
        elif g in val_groups:
            s = "val"
        else:
            s = "train"
        rows.append({"group_id": str(g), "split": s})

    split_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    split_df.to_parquet(out_path, index=False)
    return split_df


def apply_group_splits(
    df: pd.DataFrame,
    split_path: str,
    split_on: str,
    split: str,
) -> pd.DataFrame:
    """
    Filter master index to one split according to saved group assignments.
    """
    df = df.copy()
    df["group_id"] = _group_id(df, split_on).astype(str)

    split_df = pd.read_parquet(split_path)
    split_df["group_id"] = split_df["group_id"].astype(str)
    m = dict(zip(split_df["group_id"], split_df["split"]))

    mask = df["group_id"].map(m) == split
    return df[mask].reset_index(drop=True)


# -------------------------
# Image transforms
# -------------------------


class To3Channels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [H,W] or [C,H,W] in [0,1]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x


class AddGaussianNoise:
    def __init__(self, std: float = 0.01, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            noise = torch.randn_like(x) * self.std
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)
        return x


class NormalizeCXR:
    """
    mode:
    - 'none'
    - 'imagenet'
    - 'rad_dino_maira_2'
    - 'cxr_global'
    - 'cxr_per_image'
    - 'cxr_percentile'
    """

    def __init__(self, mode="imagenet", cxr_mean=None, cxr_std=None,
                 p_low=1.0, p_high=99.0, eps=1e-6,
                 rad_auto_rescale: bool = True):
        self.mode = mode
        self.cxr_mean = cxr_mean
        self.cxr_std = cxr_std
        self.p_low = p_low
        self.p_high = p_high
        self.eps = eps
        self.rad_auto_rescale = rad_auto_rescale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            x = x.float()
        else:
            x = x.to(torch.float32)

        if self.mode == "none":
            return x

        if self.mode == "imagenet":
            mean = torch.as_tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(-1, 1, 1)
            std  = torch.as_tensor(IMAGENET_STD,  dtype=x.dtype, device=x.device).view(-1, 1, 1)
            return (x - mean) / std

        if self.mode == "rad_dino_maira_2":
            # RAD-DINO-MAIRA-2 processor: do_rescale=1/255 then normalize with mean/std below. :contentReference[oaicite:1]{index=1}
            if self.rad_auto_rescale:
                # If your pipeline feeds 0..255 floats/uints, bring to 0..1.
                # (If you already use ToTensor(), values are already 0..1 and this does nothing.)
                if x.max().item() > 1.5:
                    x = x * RAD_DINO_MAIRA_2_RESCALE

            mean = torch.as_tensor(RAD_DINO_MAIRA_2_MEAN, dtype=x.dtype, device=x.device).view(-1, 1, 1)
            std  = torch.as_tensor(RAD_DINO_MAIRA_2_STD,  dtype=x.dtype, device=x.device).view(-1, 1, 1)

            # expects 3-channel input; you said you already provide that
            return (x - mean) / std

        if self.mode == "cxr_global":
            if self.cxr_mean is None or self.cxr_std is None or self.cxr_std < self.eps:
                return x
            return (x - self.cxr_mean) / max(self.cxr_std, self.eps)

        if self.mode == "cxr_per_image":
            m = x.mean()
            s = x.std()
            if s < self.eps:
                return x - m
            return (x - m) / s

        if self.mode == "cxr_percentile":
            lo = torch.quantile(x, self.p_low / 100.0)
            hi = torch.quantile(x, self.p_high / 100.0)
            if hi <= lo + self.eps:
                return x
            x = (x - lo) / (hi - lo)
            x = torch.clamp(x, 0.0, 1.0)
            return x

        return x


import torch
import torch.nn as nn
import random

class CornerCutout(nn.Module):
    """
    Fast, fine-grained Corner Cutout for PyTorch Tensors.
    Allows separate probabilities for each corner and asymmetric cutout sizes.
    """
    def __init__(
        self, 
        cutout_size=(0.15, 0.15), 
        corner_probs={'tl': 0.7, 'tr': 0.7, 'bl': 0.0, 'br': 0.0},
        fill_value=0
    ):
        """
        Args:
            cutout_size (tuple): (height_pct, width_pct) of the cutout box.
                                 e.g., (0.15, 0.10) cuts 15% vertical, 10% horizontal.
            corner_probs (dict): Dictionary defining probability of erasing each corner.
                                 Keys: 'tl' (top-left), 'tr' (top-right), 
                                       'bl' (bottom-left), 'br' (bottom-right).
                                 Values: Float 0.0 to 1.0.
            fill_value (float): Value to fill erased regions with (usually 0).
        """
        super().__init__()
        self.h_pct, self.w_pct = cutout_size
        self.probs = corner_probs
        self.fill = fill_value

    def forward(self, img):
        """
        Args:
            img (Tensor): Image tensor of shape [C, H, W]
        """
        # Efficiently get dimensions without moving data
        _, h, w = img.shape
        
        # Calculate cutout limits in pixels
        cut_h = int(h * self.h_pct)
        cut_w = int(w * self.w_pct)

        # We use a single random batch call for speed if needed, 
        # but simple python random is fast enough for 4 float comparisons.
        
        # Top-Left
        if self.probs.get('tl', 0) > 0 and random.random() < self.probs['tl']:
            img[:, 0:cut_h, 0:cut_w] = self.fill

        # Top-Right
        if self.probs.get('tr', 0) > 0 and random.random() < self.probs['tr']:
            img[:, 0:cut_h, w-cut_w:w] = self.fill

        # Bottom-Left
        if self.probs.get('bl', 0) > 0 and random.random() < self.probs['bl']:
            img[:, h-cut_h:h, 0:cut_w] = self.fill

        # Bottom-Right
        if self.probs.get('br', 0) > 0 and random.random() < self.probs['br']:
            img[:, h-cut_h:h, w-cut_w:w] = self.fill
            
        return img
    

from torchvision.transforms import InterpolationMode as IM

def build_transform(
    image_size: int,
    augment: bool,
    norm_mode: str = "imagenet",
    cxr_mean: float | None = None,
    cxr_std: float | None = None,
    affine_degrees: float | tuple[float, float] = 5,
    affine_translate: tuple[float, float] = (0.02, 0.02),
    cornercutout_size: tuple[float, float] = (0.2, 0.25),
    cornercutout_probs: dict | None = None,
    cornercutout_fill: float = 0.0,
) -> T.Compose:
    """
    16-bit PNG -> float32 [3,H,W] with selectable normalization.

    norm_mode:
      - 'imagenet'     : use with ImageNet-pretrained backbones.
      - 'none'         : only [0,1] scaling.
      - 'cxr_global'   : global mean/std you computed offline.
      - 'cxr_per_image': per-image standardization.
      - 'cxr_percentile': percentile-based clipping in [0,1].
    """
    ops = []
        # T.ToTensor(),   # 16-bit -> [0,1] float, linear
        # To3Channels(),
    

    if augment:
        if cornercutout_probs is None:
            cornercutout_probs = {
                "tl": 0.7,
                "tr": 0.85,
                "bl": 0.0,
                "br": 0.0,
            }
        # conservative CXR-safe augmentations
        # ops += [
        #     T.RandomRotation(degrees=5, interpolation=IM.NEAREST),
        #     T.RandomAffine(
        #         degrees=0,
        #         translate=(0.02, 0.02),
        #         scale=(0.98, 1.02),
        #     ),
        #     #AddGaussianNoise(std=0.01, p=0.3),
        # ]
        # -----------------------------------------------------------
        # TRAIN PIPELINE: Random Marker Removal
        # -----------------------------------------------------------
        # 1. RandomResizedCrop: The main fix for markers.
        # scale=(0.8, 1.0): We take a crop that is 80% to 100% of the original area.
        # This acts like a "Zoom". The edges (markers) are frequently cut off.
        # ops.append(
        #     T.RandomResizedCrop(
        #         size=(image_size, image_size),
        #         scale=(0.6, 1.0),     # Zoom range (1.0 = no zoom, 0.8 = 20% zoom in)
        #         ratio=(0.9, 1.1),     # Keep aspect ratio mostly square to avoid distorting ribs
        #         interpolation=IM.NEAREST
        #     )
        # )
        
        # 2. Rotations and other Affine
        # We removed 'scale' from RandomAffine because RandomResizedCrop handles it better.
        ops.append(
            T.RandomAffine(
                degrees=affine_degrees,            # +/- degrees
                translate=affine_translate,
                fill=0,               # Fill empty space with black
                interpolation=IM.NEAREST
            )
        )

        ops.append(T.Resize((image_size, image_size), interpolation=IM.NEAREST))

        # 3. APPLY CUSTOM CORNER CUTOUT
        # Scenario: Eliminate markers which are usually Top-Right or Top-Left.
        # We target Top corners aggressively (80% chance), Bottom corners rarely (0%).
        ops.append(
            CornerCutout(
                cutout_size=cornercutout_size, # 15% of H and W
                corner_probs=cornercutout_probs,
                fill_value=cornercutout_fill
            )
        )

    else:
        ops.append(T.Resize((image_size, image_size), interpolation=IM.NEAREST))

    ops.append(NormalizeCXR(
        mode=norm_mode,
        cxr_mean=cxr_mean,
        cxr_std=cxr_std,
    ))

    return T.Compose(ops)


# -------------------------
# Embeddings + labels helpers
# -------------------------



def load_accession_embeddings(path: str) -> Dict[str, np.ndarray]:
    """
    Load one accession's embeddings into dict[field] = np.ndarray(dim,).
    Missing file -> {}.
    """
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    if "field" not in df.columns or "embedding" not in df.columns:
        raise KeyError(f"Embedding parquet {path} missing 'field'/'embedding'.")
    out: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        field = str(row["field"])
        emb = np.asarray(row["embedding"], dtype="float32")
        out[field] = emb
    return out


def extract_binary_label(report: Optional[dict], concept_name: str) -> Tuple[float, bool]:
    """
    Map a structured-report JSON concept to a binary label + validity mask.

    Expects the structured JSON produced by the report_struct_label pipeline:
    a top-level ``data`` object whose concept leaves are ``{"text": ..., "label": ...}``.

    concept_name examples:
      'support_devices.airway'
      'support_devices.chest_drain'
      'thoracic_organs.heart'
      'pathologies.lung.pneumonia'

    Label rules:
      - support_devices.*:
          'present'     -> 1
          'not present' -> 0
      - thoracic_organs.*, pathologies.* (0-3 severity score):
          0   -> unknown (mask=False, ignored)
          1   -> 0 (normal)
          2,3 -> 1 (pathologic)
      - anything missing / unexpected -> (0.0, False)

    Returns:
        (label, valid) where ``valid`` is False when the entry should be masked out.
    """
    if report is None:
        return 0.0, False

    # The report JSON wraps the concept tree in a "data" object.
    data = report.get("data", report.get("Data", report))
    node = data
    for k in concept_name.split("."):
        if not isinstance(node, dict) or k not in node:
            return 0.0, False
        node = node[k]

    # Concept leaves are {"text": ..., "label": ...}; fall back to the raw value.
    val = node.get("label") if isinstance(node, dict) else node

    # support_devices style (presence string)
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "present":
            return 1.0, True
        if v == "not present":
            return 0.0, True
        return 0.0, False

    # thoracic_organs / pathologies style (0-3 severity score)
    if isinstance(val, (int, float)):
        if val == 0:
            return 0.0, False  # unknown -> mask out
        if val == 1:
            return 0.0, True
        if val in (2, 3):
            return 1.0, True
        return 0.0, False

    return 0.0, False


import os
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, Tuple, Optional, List, Set, Any
import hashlib


def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _detect_accession_column(meta_path: Path) -> str:
    """
    Return the per-row key column of a shard's metadata parquet.

    The report_struct_label pipeline keys rows by 'report_id', while internally
    prepared caches key them by 'accession'. This lets the sharded stores consume
    either: 'accession' is used when present, otherwise 'report_id' (treated as
    the accession identifier).
    """
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(str(meta_path)).schema_arrow.names)
    if "accession" in names:
        return "accession"
    if "report_id" in names:
        return "report_id"
    raise KeyError(f"{meta_path} has neither an 'accession' nor a 'report_id' column.")


class ShardedClsStore:
    """
    Random access to CLS labels/masks stored as:
      - labels_00000.npy (N,K) float
      - masks_00000.npy  (N,K) bool
      - meta_00000.parquet (N rows: accession, ...)

    Builds index: accession -> (shard_idx, row_idx)
    """

    def __init__(
        self,
        root_dir: str,
        *,
        index_cache_name: str = "cls_sharded_index.parquet",
        shard_memmap_cache_size: int = 4,
        build_index_if_missing: bool = True,
        expected_concepts_sha1: Optional[str] = None,
        expected_K: Optional[int] = None,
    ):
        self.root = Path(root_dir)
        self.index_path = self.root / index_cache_name
        self.shard_memmap_cache_size = shard_memmap_cache_size

        self._index: Dict[str, Tuple[int, int]] = {}  # accession -> (shard,row)
        self._shard_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()  # shard -> (labels, masks)
        self._K: Optional[int] = None

        # Optional safety checks (highly recommended)
        self._expected_concepts_sha1 = expected_concepts_sha1
        self._expected_K = expected_K
        self._validate_summary_json_if_present()

        if self.index_path.exists():
            self._load_index_parquet(self.index_path)
        elif build_index_if_missing:
            self._build_index_from_meta()
            self._save_index_parquet(self.index_path)
        else:
            raise FileNotFoundError(f"Index not found: {self.index_path}")

    def _validate_summary_json_if_present(self) -> None:
        """
        If your sharding script wrote summary.json (recommended), validate K and concept order/hash.
        """
        summary_path = self.root / "summary.json"
        if not summary_path.exists():
            return

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return

        K = summary.get("K")
        csha = summary.get("concepts_sha1")

        if self._expected_K is not None and K is not None and int(K) != int(self._expected_K):
            raise ValueError(f"CLS override K mismatch: shard K={K} vs expected K={self._expected_K}")

        if self._expected_concepts_sha1 and csha and csha != self._expected_concepts_sha1:
            raise ValueError(
                "CLS override concept hash mismatch (likely different concept order/list). "
                f"shard={csha} expected={self._expected_concepts_sha1}"
            )

    @property
    def K(self) -> int:
        if self._K is None:
            shards = self._list_shards()
            if not shards:
                raise FileNotFoundError(f"No meta_*.parquet + labels_*.npy + masks_*.npy in {self.root}")
            lab = np.load(self.root / f"labels_{shards[0]:05d}.npy", mmap_mode="r")
            self._K = int(lab.shape[1])
        return self._K

    def _list_shards(self) -> List[int]:
        meta = {int(p.stem.split("_")[1]) for p in self.root.glob("meta_*.parquet")}
        lab  = {int(p.stem.split("_")[1]) for p in self.root.glob("labels_*.npy")}
        msk  = {int(p.stem.split("_")[1]) for p in self.root.glob("masks_*.npy")}
        return sorted(meta & lab & msk)

    def _build_index_from_meta(self) -> None:
        shards = self._list_shards()
        if not shards:
            raise FileNotFoundError(f"No shard triplets found in {self.root}")

        tmp: Dict[str, Tuple[int, int]] = {}

        for sidx in shards:
            meta_path = self.root / f"meta_{sidx:05d}.parquet"
            key_col = _detect_accession_column(meta_path)
            df = pd.read_parquet(meta_path, columns=[key_col])

            # IMPORTANT: row_idx must correspond to row in labels/masks arrays.
            # Using enumerate over the column is safest.
            for row_idx, acc in enumerate(df[key_col].astype(str).tolist()):
                if acc not in tmp:
                    tmp[acc] = (sidx, int(row_idx))

        self._index = tmp

    def _save_index_parquet(self, path: Path) -> None:
        rows = [(acc, s, r) for acc, (s, r) in self._index.items()]
        df = pd.DataFrame(rows, columns=["accession", "shard", "row"])
        df.to_parquet(path, index=False)

    def _load_index_parquet(self, path: Path) -> None:
        df = pd.read_parquet(path)
        self._index = {
            str(acc): (int(shard), int(row))
            for acc, shard, row in zip(df["accession"].astype(str), df["shard"].astype(int), df["row"].astype(int))
        }

    def _get_memmaps(self, shard_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        lab_path = self.root / f"labels_{shard_idx:05d}.npy"
        msk_path = self.root / f"masks_{shard_idx:05d}.npy"
        lab = np.load(lab_path, mmap_mode="r")
        msk = np.load(msk_path, mmap_mode="r")

        self._shard_cache[shard_idx] = (lab, msk)
        self._shard_cache.move_to_end(shard_idx)

        while self.shard_memmap_cache_size and len(self._shard_cache) > self.shard_memmap_cache_size:
            self._shard_cache.popitem(last=False)

        if self._K is None:
            self._K = int(lab.shape[1])

        return lab, msk

    def get(self, accession: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pos = self._index.get(str(accession))
        if pos is None:
            return None
        shard_idx, row_idx = pos
        lab, msk = self._get_memmaps(shard_idx)

        # Force writable, contiguous arrays
        labels = np.array(lab[row_idx], dtype=np.float32, copy=True)  # (K,)
        masks  = np.array(msk[row_idx], dtype=bool,    copy=True)     # (K,)
        return labels, masks



class ShardedEmbeddingStore:
    """
    Provides random access to embeddings stored as:
      - embeddings_00000.npy (N,D)
      - meta_00000.parquet (N rows: accession, field, ...)
    by building an in-memory index: (accession, field) -> (shard_idx, row_idx).
    """
    def __init__(
        self,
        root_dir: str,
        needed_fields: Set[str],
        *,
        index_cache_name: str = "sharded_index.parquet",
        shard_memmap_cache_size: int = 4,
        build_index_if_missing: bool = True,
    ):
        self.root = Path(root_dir)
        self.needed_fields = set(needed_fields)
        self.index_path = self.root / index_cache_name
        self.shard_memmap_cache_size = shard_memmap_cache_size

        self._index: Dict[str, Dict[str, Tuple[int, int]]] = {}  # accession -> field -> (shard,row)
        self._emb_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()  # shard -> memmap
        self._dim: Optional[int] = None

        if self.index_path.exists():
            self._load_index_parquet(self.index_path)
        elif build_index_if_missing:
            self._build_index_from_meta()
            # persist for next runs
            self._save_index_parquet(self.index_path)
        else:
            raise FileNotFoundError(f"Index not found: {self.index_path}")

    @property
    def dim(self) -> int:
        if self._dim is None:
            # infer from first embeddings shard
            emb_files = sorted(self.root.glob("embeddings_*.npy"))
            if not emb_files:
                raise FileNotFoundError(f"No embeddings_*.npy in {self.root}")
            arr = np.load(emb_files[0], mmap_mode="r")
            self._dim = int(arr.shape[1])
        return self._dim

    def _list_shards(self) -> List[int]:
        meta = {int(p.stem.split("_")[1]) for p in self.root.glob("meta_*.parquet")}
        emb = {int(p.stem.split("_")[1]) for p in self.root.glob("embeddings_*.npy")}
        return sorted(meta & emb)

    def _build_index_from_meta(self) -> None:
        shards = self._list_shards()
        if not shards:
            raise FileNotFoundError(f"No meta_*.parquet + embeddings_*.npy pairs found in {self.root}")

        tmp: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)

        for sidx in shards:
            meta_path = self.root / f"meta_{sidx:05d}.parquet"
            # Only read columns we need
            key_col = _detect_accession_column(meta_path)
            df = pd.read_parquet(meta_path, columns=[key_col, "field"])
            # Filter to relevant fields only (huge speed/memory win)
            df = df[df["field"].isin(self.needed_fields)]
            if df.empty:
                continue

            # df index corresponds to row_idx in embeddings file
            # Keep first occurrence per (accession, field)
            for row_idx, acc, field in zip(df.index.to_numpy(),
                                           df[key_col].astype(str).tolist(),
                                           df["field"].astype(str).tolist()):
                if field not in tmp[acc]:
                    tmp[acc][field] = (sidx, int(row_idx))

        self._index = dict(tmp)

    def _save_index_parquet(self, path: Path) -> None:
        rows = []
        for acc, fmap in self._index.items():
            for field, (sidx, ridx) in fmap.items():
                rows.append((acc, field, sidx, ridx))
        df = pd.DataFrame(rows, columns=["accession", "field", "shard", "row"])
        df.to_parquet(path, index=False)

    def _load_index_parquet(self, path: Path) -> None:
        df = pd.read_parquet(path)
        tmp: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
        for acc, field, shard, row in zip(df["accession"].astype(str),
                                        df["field"].astype(str),
                                        df["shard"].astype(int),
                                        df["row"].astype(int)):
            tmp[acc][field] = (int(shard), int(row))
        self._index = dict(tmp)

    def _get_memmap(self, shard_idx: int) -> np.ndarray:
        if shard_idx in self._emb_cache:
            self._emb_cache.move_to_end(shard_idx)
            return self._emb_cache[shard_idx]

        p = self.root / f"embeddings_{shard_idx:05d}.npy"
        arr = np.load(p, mmap_mode="r")  # memmap
        self._emb_cache[shard_idx] = arr
        self._emb_cache.move_to_end(shard_idx)

        while self.shard_memmap_cache_size and len(self._emb_cache) > self.shard_memmap_cache_size:
            self._emb_cache.popitem(last=False)

        if self._dim is None:
            self._dim = int(arr.shape[1])
        return arr

    def get_vec(self, accession: str, field: str) -> Optional[np.ndarray]:
        fmap = self._index.get(str(accession))
        if not fmap:
            return None
        pos = fmap.get(field)
        if pos is None:
            return None
        shard_idx, row_idx = pos
        arr = self._get_memmap(shard_idx)
        # return as float32 for downstream torch
        return np.asarray(arr[row_idx], dtype=np.float32)


class ShardedImageStore:
    """
    Random access to image shards stored as:
      - images_00000.npy (N,H,W) or (N,1,H,W)
      - meta_00000.parquet (N rows with at least sop_uid/SOPInstanceUID)

    Builds an index:
      sop_uid -> (shard_idx, row_idx, accession)
    """

    def __init__(
        self,
        root_dir: str,
        *,
        index_cache_name: str = "image_sharded_index.parquet",
        shard_memmap_cache_size: int = 4,
        build_index_if_missing: bool = True,
    ):
        self.root = Path(root_dir)
        self.index_path = self.root / index_cache_name
        self.shard_memmap_cache_size = shard_memmap_cache_size

        self._index: Dict[str, Tuple[int, int, Optional[str]]] = {}
        self._img_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        if self.index_path.exists():
            self._load_index_parquet(self.index_path)
        elif build_index_if_missing:
            self._build_index_from_meta()
            self._save_index_parquet(self.index_path)
        else:
            raise FileNotFoundError(f"Index not found: {self.index_path}")

    def _list_shards(self) -> List[int]:
        meta = {int(p.stem.split("_")[1]) for p in self.root.glob("meta_*.parquet")}
        imgs = {int(p.stem.split("_")[1]) for p in self.root.glob("images_*.npy")}
        return sorted(meta & imgs)

    def _sop_col(self, df: pd.DataFrame) -> str:
        if "sop_uid" in df.columns:
            return "sop_uid"
        if "SOPInstanceUID" in df.columns:
            return "SOPInstanceUID"
        raise KeyError("meta shard missing sop column ('sop_uid' or 'SOPInstanceUID').")

    def _build_index_from_meta(self) -> None:
        shards = self._list_shards()
        if not shards:
            raise FileNotFoundError(f"No meta_*.parquet + images_*.npy pairs found in {self.root}")

        out: Dict[str, Tuple[int, int, Optional[str]]] = {}
        for sidx in shards:
            meta_path = self.root / f"meta_{sidx:05d}.parquet"
            df = pd.read_parquet(meta_path)
            sop_col = self._sop_col(df)
            has_row = "row" in df.columns
            has_accession = "accession" in df.columns

            for default_row_idx, row in enumerate(df.itertuples(index=False)):
                rowd = row._asdict()
                sop = str(rowd.get(sop_col, ""))
                if not sop:
                    continue
                row_idx = int(rowd["row"]) if has_row and rowd.get("row") is not None else int(default_row_idx)
                accession = str(rowd["accession"]) if has_accession and rowd.get("accession") is not None else None
                if sop not in out:
                    out[sop] = (int(sidx), int(row_idx), accession)

        self._index = out

    def _save_index_parquet(self, path: Path) -> None:
        rows = [(sop, s, r, acc) for sop, (s, r, acc) in self._index.items()]
        df = pd.DataFrame(rows, columns=["sop_uid", "shard", "row", "accession"])
        df.to_parquet(path, index=False)

    def _load_index_parquet(self, path: Path) -> None:
        df = pd.read_parquet(path)
        sop_col = "sop_uid" if "sop_uid" in df.columns else "SOPInstanceUID"
        acc_col = "accession" if "accession" in df.columns else None
        self._index = {}
        for _, row in df.iterrows():
            sop = str(row[sop_col])
            sidx = int(row["shard"])
            ridx = int(row["row"])
            acc = str(row[acc_col]) if (acc_col is not None and pd.notna(row[acc_col])) else None
            self._index[sop] = (sidx, ridx, acc)

    def _get_memmap(self, shard_idx: int) -> np.ndarray:
        if shard_idx in self._img_cache:
            self._img_cache.move_to_end(shard_idx)
            return self._img_cache[shard_idx]

        p = self.root / f"images_{shard_idx:05d}.npy"
        arr = np.load(p, mmap_mode="r")
        self._img_cache[shard_idx] = arr
        self._img_cache.move_to_end(shard_idx)

        while self.shard_memmap_cache_size and len(self._img_cache) > self.shard_memmap_cache_size:
            self._img_cache.popitem(last=False)

        return arr

    def get(self, sop_uid: str) -> Tuple[np.ndarray, Optional[str]]:
        pos = self._index.get(str(sop_uid))
        if pos is None:
            raise KeyError(f"SOP not found in sharded image cache: {sop_uid}")

        shard_idx, row_idx, accession = pos
        arr = self._get_memmap(shard_idx)
        img = arr[row_idx]
        return img, accession




# -------------------------
# Dataset
# -------------------------

class CXRMultimodalDataset(Dataset):
    """
    Per-image multimodal dataset for VisionConceptModel.

    Loads chest X-ray images from PNG files together with the pre-computed
    per-concept text embeddings (and optional classification labels) referenced
    by the master index.

    __getitem__ returns:
      - images:           [3,H,W] float32
      - text_emb:         [K,text_dim] float32
      - text_mask:        [K] bool
      - cls_labels:       [K] float32 (if include_cls_labels)
      - cls_mask:         [K] bool   (if include_cls_labels)
      - accession:        str

    Assumptions:
      - concept_names match the report JSON hierarchy, e.g.
        'pathologies.pleura.pleural_effusion'.
      - text embedding fields are 'structured.<concept_name>'.
      - Missing embeddings -> mask=False and zero vector.
      - Missing labels or 0-coded (no info) -> cls_mask=False.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        concept_names: List[str],
        embedding_dir: str,
        label_dir: Optional[str] = None,
        include_cls_labels: bool = True,
        image_size: int = 512,
        augment: bool = False,
        max_samples: Optional[int] = None,
        norm_mode: str = "imagenet",
        cxr_mean: float | None = None,
        cxr_std: float | None = None,
        affine_degrees: float | tuple[float, float] = 5,
        affine_translate: tuple[float, float] = (0.02, 0.02),
        cornercutout_size: tuple[float, float] = (0.2, 0.25),
        cornercutout_probs: dict | None = None,
        cornercutout_fill: float = 0.0,
        text_concept_names: Optional[List[str]] = None,
        cls_concept_names: Optional[List[str]] = None,
    ):
        super().__init__()

        if max_samples is not None and len(index_df) > max_samples:
            index_df = index_df.sample(n=max_samples, random_state=0).reset_index(drop=True)

        for c in ["image_path", "accession"]:
            if c not in index_df.columns:
                raise KeyError(f"index_df missing '{c}'.")

        self.df = index_df.reset_index(drop=True)
        base_names = list(concept_names)
        self.text_concept_names = list(text_concept_names) if text_concept_names is not None else base_names
        self.cls_concept_names = list(cls_concept_names) if cls_concept_names is not None else base_names
        self.concept_names = self.text_concept_names
        self.embedding_dir = embedding_dir
        self.label_dir = label_dir
        self.include_cls_labels = include_cls_labels

        self.transform = build_transform(
            image_size=image_size,
            augment=augment,
            norm_mode=norm_mode,
            cxr_mean=cxr_mean,
            cxr_std=cxr_std,
            affine_degrees=affine_degrees,
            affine_translate=affine_translate,
            cornercutout_size=cornercutout_size,
            cornercutout_probs=cornercutout_probs,
            cornercutout_fill=cornercutout_fill,
        )

        self._emb_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._label_cache: Dict[str, Optional[dict]] = {}
        self._text_dim: Optional[int] = None

    # --- internal helpers ---

    def _load_image(self, path: str) -> torch.Tensor:
        # Read PNG with PIL
        with Image.open(path) as img:
            # Keep grayscale. If not a recognized grayscale mode, convert to 16-bit grayscale.
            if img.mode not in ("I;16", "I", "L"):
                img = img.convert("I;16")
            arr = np.array(img)

        # Robust integer → [0,1] scaling
        if np.issubdtype(arr.dtype, np.integer):
            # Typical case: uint16 from 'I;16'
            if arr.dtype == np.uint16:
                x = arr.astype(np.float32) / 65535.0
            # Signed ints coming from PIL 'I' but actually holding 0..65535
            elif arr.dtype in (np.int16, np.int32) and arr.min() >= 0 and arr.max() <= 65535:
                x = arr.astype(np.float32) / 65535.0
            # 8-bit fallback
            elif arr.dtype == np.uint8:
                x = arr.astype(np.float32) / 255.0
            else:
                # Last-resort min-max (rare)
                a, b = float(arr.min()), float(arr.max())
                x = (arr.astype(np.float32) - a) / max(b - a, 1.0)
        else:
            # Float input: clamp to [0,1]
            x = np.clip(arr.astype(np.float32), 0.0, 1.0)

        # To torch [1,H,W] float32 then make 3 channels
        t = torch.from_numpy(x).unsqueeze(0)           # [1,H,W]
        t = t.repeat(3, 1, 1)                          # [3,H,W]

        # Now apply tensor-based transforms (geom + normalization)
        return self.transform(t)

    def _get_embeddings(self, accession: str) -> Dict[str, np.ndarray]:
        if accession in self._emb_cache:
            emb = self._emb_cache[accession]
        else:
            path = os.path.join(self.embedding_dir, f"{accession}.parquet")
            emb = load_accession_embeddings(path)
            self._emb_cache[accession] = emb

        if self._text_dim is None and emb:
            self._text_dim = len(next(iter(emb.values())))
        return emb

    def _get_report(self, accession: str) -> Optional[dict]:
        if self.label_dir is None:
            return None
        if accession in self._label_cache:
            return self._label_cache[accession]
        path = os.path.join(self.label_dir, f"{accession}_chest_standard.json")
        if not os.path.exists(path):
            rep = None
        else:
            with open(path, "r", encoding="utf-8") as f:
                rep = json.load(f)
        self._label_cache[accession] = rep
        return rep

    def _build_text_tensors(
        self,
        accession: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_map = self._get_embeddings(accession)
        D = self._text_dim or 768
        K = len(self.text_concept_names)

        text_emb = np.zeros((K, D), dtype="float32")
        text_mask = np.zeros((K,), dtype=bool)

        for i, cname in enumerate(self.text_concept_names):
            field = f"structured.{cname}"
            vec = emb_map.get(field)
            if vec is not None and len(vec) == D:
                text_emb[i] = vec
                text_mask[i] = True

        return (
            torch.from_numpy(text_emb),
            torch.from_numpy(text_mask),
        )

    def _build_cls_tensors(
        self,
        accession: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rep = self._get_report(accession)
        K = len(self.cls_concept_names)
        labels = np.zeros((K,), dtype="float32")
        mask = np.zeros((K,), dtype=bool)

        if rep is None:
            return torch.from_numpy(labels), torch.from_numpy(mask)

        for i, cname in enumerate(self.cls_concept_names):
            y, valid = extract_binary_label(rep, cname)
            if valid:
                labels[i] = float(y)
                mask[i] = True

        return torch.from_numpy(labels), torch.from_numpy(mask)

    # --- Dataset API ---

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        accession = str(row["accession"])

        sample: Dict[str, torch.Tensor] = {}

        sample["images"] = self._load_image(img_path)

        text_emb, text_mask = self._build_text_tensors(accession)
        sample["text_emb"] = text_emb
        sample["text_mask"] = text_mask

        if self.include_cls_labels:
            cls_labels, cls_mask = self._build_cls_tensors(accession)
            sample["cls_labels"] = cls_labels
            sample["cls_mask"] = cls_mask

        # metadata (not used by model directly)
        sample["accession"] = accession
        #sample["image_path"] = img_path

        return sample

# chached dataset version and helpers:
class PTCachedCXRDataset(Dataset):
    """
    Loads cached images from either:
      - legacy per-SOP .pt files (prepare_pt_cache.py), or
      - sharded image cache (images_*.npy + meta_*.parquet).

    Text/CLS can come from cache (legacy .pt) and/or override sources.
    """
    def __init__(self,
                 df,
                 concept_names: List[str],
                 preproc_dir: str | None = None,
                 image_size: int = 512,
                 augment: bool = False,
                 norm_mode: str = "imagenet",
                 include_cls_labels: bool = True,
                 accession_column: str = "accession",
                 sop_uid_column: str = "SOPInstanceUID",
                 cxr_mean: float | None = None,
                 cxr_std: float | None = None,
                 p_low: float = 1.0,
                 p_high: float = 99.0,
                 affine_degrees: float | tuple[float, float] = 5,
                 affine_translate: tuple[float, float] = (0.02, 0.02),
                 cornercutout_size: tuple[float, float] = (0.2, 0.25),
                 cornercutout_probs: dict | None = None,
                 cornercutout_fill: float = 0.0,
                 # --- NEW: image cache backend ---
                 cached_dataset_path: str | None = None,
                 image_cache_mode: str = "auto",  # "auto" | "pt" | "sharded"
                 image_sharded_index_cache_name: str = "image_sharded_index.parquet",
                 image_sharded_memmap_cache_size: int = 4,
                 # --- NEW ---
                 text_override_dir: str | None = None,
                 use_text_override: bool = False,
                 text_override_strict: bool = False,
                 text_override_cache_size: int = 256,
                 text_override_mode: str = "auto",  # NEW
                 text_override_index_cache_name: str = "sharded_index.parquet",  # NEW
                 text_override_shard_memmap_cache_size: int = 4,  # NEW
                 # --- NEW: CLS override ---
                 cls_override_dir: str | None = None,
                 use_cls_override: bool = False,
                 cls_override_strict: bool = False,
                 cls_override_cache_size: int = 256,
                 cls_override_mode: str = "auto",  # "auto" | "pt" | "sharded"
                 cls_override_index_cache_name: str = "cls_sharded_index.parquet",
                 cls_override_shard_memmap_cache_size: int = 4,
                 cls_override_concept_names: Optional[List[str]] = None,
                 text_concept_names: Optional[List[str]] = None,
                 cls_concept_names: Optional[List[str]] = None,

                 ):
        if cached_dataset_path is not None:
            preproc_dir = cached_dataset_path
        if preproc_dir is None:
            raise ValueError("Either preproc_dir or cached_dataset_path must be provided.")

        self.df = df.reset_index(drop=True)
        base_names = list(concept_names)
        self.text_concept_names = list(text_concept_names) if text_concept_names is not None else base_names
        self.cls_concept_names = list(cls_concept_names) if cls_concept_names is not None else base_names
        self.concept_names = self.text_concept_names
        self.preproc_dir = preproc_dir
        self.cached_dataset_path = preproc_dir
        self.image_size = image_size
        self.include_cls_labels = include_cls_labels
        self.acc_col = accession_column
        self.sop_col = sop_uid_column
        self.image_cache_mode = image_cache_mode
        self.image_sharded_index_cache_name = image_sharded_index_cache_name
        self.image_sharded_memmap_cache_size = image_sharded_memmap_cache_size
        self._image_sharded_store: Optional[ShardedImageStore] = None
        self.text_override_dir = text_override_dir
        self.use_text_override = use_text_override
        self.text_override_strict = text_override_strict
        self.text_override_cache_size = text_override_cache_size

        # per-worker LRU cache: accession -> dict of tensors

        self.text_override_mode = text_override_mode
        self.text_override_index_cache_name = text_override_index_cache_name
        self.text_override_shard_memmap_cache_size = text_override_shard_memmap_cache_size

        self._text_override_cache = OrderedDict()
        self._sharded_store = None
        self.cls_override_dir = cls_override_dir
        self.use_cls_override = use_cls_override
        self.cls_override_strict = cls_override_strict
        self.cls_override_cache_size = cls_override_cache_size
        self.cls_override_mode = cls_override_mode
        self.cls_override_index_cache_name = cls_override_index_cache_name
        self.cls_override_shard_memmap_cache_size = cls_override_shard_memmap_cache_size
        self.cls_override_concept_names = (
            list(cls_override_concept_names)
            if cls_override_concept_names is not None
            else list(self.cls_concept_names)
        )
        override_idx = {name: i for i, name in enumerate(self.cls_override_concept_names)}
        missing_cls = [name for name in self.cls_concept_names if name not in override_idx]
        if missing_cls:
            raise ValueError(
                "Requested CLS concept(s) not found in cls_override_concept_names: "
                f"{missing_cls}"
            )
        self._cls_override_select_idx = [
            override_idx[name] for name in self.cls_concept_names
        ]
        if self.cls_override_concept_names == self.cls_concept_names:
            self._cls_override_select_idx = None

        self._cls_override_cache = OrderedDict()
        self._cls_sharded_store = None

        self._image_cache_mode_effective = self._detect_image_cache_mode(self.preproc_dir, self.image_cache_mode)
        if self._image_cache_mode_effective == "sharded":
            self._image_sharded_store = ShardedImageStore(
                self.preproc_dir,
                index_cache_name=self.image_sharded_index_cache_name,
                shard_memmap_cache_size=self.image_sharded_memmap_cache_size,
                build_index_if_missing=True,
            )
        print(f"Image cache mode: {self._image_cache_mode_effective}")

        if self.use_text_override and self.text_override_dir:
            mode = self._detect_override_mode(self.text_override_dir, self.text_override_mode)
            self._override_mode_effective = mode

            if mode == "sharded":
                needed_fields = {f"structured.{c}" for c in self.text_concept_names}
                self._sharded_store = ShardedEmbeddingStore(
                    self.text_override_dir,
                    needed_fields=needed_fields,
                    index_cache_name=self.text_override_index_cache_name,
                    shard_memmap_cache_size=self.text_override_shard_memmap_cache_size,
                    build_index_if_missing=True,
                )
            else:
                self._override_mode_effective = "pt"

            print(f"Text override mode: {self._override_mode_effective}")

        if self.use_cls_override and self.cls_override_dir:
            mode = self._detect_cls_override_mode(self.cls_override_dir, self.cls_override_mode)
            self._cls_override_mode_effective = mode

            if mode == "sharded":
                expected_sha = sha1_str("|".join(self.cls_override_concept_names))
                expected_K = len(self.cls_override_concept_names)

                self._cls_sharded_store = ShardedClsStore(
                    self.cls_override_dir,
                    index_cache_name=self.cls_override_index_cache_name,
                    shard_memmap_cache_size=self.cls_override_shard_memmap_cache_size,
                    build_index_if_missing=True,
                    expected_concepts_sha1=expected_sha,
                    expected_K=expected_K,
                )
            else:
                self._cls_override_mode_effective = "pt"

            print(f"CLS override mode: {self._cls_override_mode_effective}")




        if len(self.df) == 0:
            raise ValueError("Empty dataframe")
        first_sop = str(self.df.iloc[0][self.sop_col])
        if self._image_cache_mode_effective == "pt":
            sample_path = os.path.join(self.preproc_dir, f"{first_sop}.pt")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Cache file not found: {sample_path}")
        else:
            if self._image_sharded_store is None:
                raise RuntimeError("Sharded image cache mode selected but image store is not initialized.")
            try:
                self._image_sharded_store.get(first_sop)
            except KeyError as e:
                raise FileNotFoundError(
                    f"First SOP ({first_sop}) not found in sharded image cache at {self.preproc_dir}"
                ) from e

        # full transform pipeline on tensors
        self.transform = build_transform(
            image_size=image_size,
            augment=augment,
            norm_mode=norm_mode,
            cxr_mean=cxr_mean,
            cxr_std=cxr_std,
            affine_degrees=affine_degrees,
            affine_translate=affine_translate,
            cornercutout_size=cornercutout_size,
            cornercutout_probs=cornercutout_probs,
            cornercutout_fill=cornercutout_fill,
            # p_low=p_low,
            # p_high=p_high,
        )

    def __len__(self): return len(self.df)

    def _detect_image_cache_mode(self, d: str, mode: str) -> str:
        if mode in ("pt", "sharded"):
            return mode
        dd = Path(d)
        if any(dd.glob("meta_*.parquet")) and any(dd.glob("images_*.npy")):
            return "sharded"
        if any(dd.glob("*.pt")):
            return "pt"
        raise FileNotFoundError(
            f"Could not detect image cache mode in {d}. "
            "Expected either *.pt files or meta_*.parquet + images_*.npy shards."
        )

    def _detect_override_mode(self, d: str, mode: str) -> str:
        if mode in ("pt", "sharded"):
            return mode
        # auto-detect
        dd = Path(d)
        if any(dd.glob("meta_*.parquet")) and any(dd.glob("embeddings_*.npy")):
            return "sharded"
        # fallback old style
        return "pt"
    
    def _detect_cls_override_mode(self, d: str, mode: str) -> str:
        if mode in ("pt", "sharded"):
            return mode
        dd = Path(d)
        if any(dd.glob("meta_*.parquet")) and any(dd.glob("labels_*.npy")) and any(dd.glob("masks_*.npy")):
            return "sharded"
        return "pt"

    def _load_image_1ch_from_sharded(self, sop_uid: str) -> Tuple[torch.Tensor, Optional[str]]:
        if self._image_sharded_store is None:
            raise RuntimeError("Sharded image store not initialized.")

        img_np, accession = self._image_sharded_store.get(sop_uid)
        arr = np.asarray(img_np)

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(f"Unexpected sharded image shape for SOP {sop_uid}: {arr.shape}")
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected sharded image ndim for SOP {sop_uid}: {arr.ndim}")

        if np.issubdtype(arr.dtype, np.floating):
            img = torch.from_numpy(arr.astype(np.float32, copy=False)).clamp(0.0, 1.0)
        elif arr.dtype == np.uint16:
            img = torch.from_numpy(arr.astype(np.float32)) / 65535.0
        elif arr.dtype == np.uint8:
            img = torch.from_numpy(arr.astype(np.float32)) / 255.0
        else:
            a = float(arr.min())
            b = float(arr.max())
            img = (torch.from_numpy(arr.astype(np.float32)) - a) / max((b - a), 1e-6)

        return img.unsqueeze(0), accession


    def _get_text_override(self, accession: str) -> dict | None:
        if not self.text_override_dir:
            return None

        accession = str(accession)

        # LRU hit
        if accession in self._text_override_cache:
            self._text_override_cache.move_to_end(accession)
            return self._text_override_cache[accession]

        mode = getattr(self, "_override_mode_effective", "pt")

        if mode == "pt":
            path = os.path.join(self.text_override_dir, f"{accession}.pt")
            if not os.path.exists(path):
                return None
            data = torch.load(path, map_location="cpu", weights_only=True)
            pack = {
                "text_emb": data["text_emb"],
                "text_mask": data["text_mask"],
            }

        else:
            # sharded: assemble tensors on the fly from (accession, field) -> embedding rows
            if self._sharded_store is None:
                return None

            D = self._sharded_store.dim
            K = len(self.text_concept_names)

            text_emb = torch.zeros((K, D), dtype=torch.float32)
            text_mask = torch.zeros((K,), dtype=torch.bool)

            for i, cname in enumerate(self.text_concept_names):
                field = f"structured.{cname}"
                v = self._sharded_store.get_vec(accession, field)
                if v is not None and v.shape[0] == D:
                    # Sharded arrays may be read-only memory maps. Copy before
                    # wrapping to avoid undefined behavior if a tensor is mutated.
                    text_emb[i] = torch.from_numpy(np.array(v, copy=True))
                    text_mask[i] = True

            pack = {
                "text_emb": text_emb,
                "text_mask": text_mask,
            }

        self._text_override_cache[accession] = pack
        self._text_override_cache.move_to_end(accession)

        if self.text_override_cache_size and len(self._text_override_cache) > self.text_override_cache_size:
            self._text_override_cache.popitem(last=False)

        return pack
    
    def _get_cls_override(self, accession: str) -> dict | None:
        if not self.cls_override_dir:
            return None

        accession = str(accession)

        if accession in self._cls_override_cache:
            self._cls_override_cache.move_to_end(accession)
            return self._cls_override_cache[accession]

        mode = getattr(self, "_cls_override_mode_effective", "pt")

        if mode == "pt":
            path = os.path.join(self.cls_override_dir, f"{accession}.pt")
            if not os.path.exists(path):
                return None
            data = torch.load(path, map_location="cpu", weights_only=True)
            cls_labels = data["cls_labels"]
            cls_mask = data["cls_mask"]

        else:
            if self._cls_sharded_store is None:
                return None

            got = self._cls_sharded_store.get(accession)
            if got is None:
                return None

            labels_np, mask_np = got
            cls_labels = torch.from_numpy(np.array(labels_np, copy=True))  # (K_override,)
            cls_mask = torch.from_numpy(np.array(mask_np, copy=True))      # (K_override,)

        if self._cls_override_select_idx is not None:
            idx = torch.as_tensor(self._cls_override_select_idx, dtype=torch.long)
            cls_labels = cls_labels.index_select(0, idx)
            cls_mask = cls_mask.index_select(0, idx)

        pack = {
            "cls_labels": cls_labels,
            "cls_mask": cls_mask,
        }

        self._cls_override_cache[accession] = pack
        self._cls_override_cache.move_to_end(accession)

        if self.cls_override_cache_size and len(self._cls_override_cache) > self.cls_override_cache_size:
            self._cls_override_cache.popitem(last=False)

        return pack



    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sop = str(self.df.iloc[idx][self.sop_col])
        out: Dict[str, torch.Tensor]

        if self._image_cache_mode_effective == "pt":
            path = os.path.join(self.preproc_dir, f"{sop}.pt")
            samp = torch.load(path, map_location="cpu", weights_only=True)

            img1 = samp["images_1ch"]  # float16
            img01 = img1.float().repeat(3, 1, 1)
            img = self.transform(img01)
            out = {"images": img}

            if self.acc_col in self.df.columns:
                accession = str(samp.get("accession", self.df.iloc[idx][self.acc_col]))
            else:
                accession = str(samp.get("accession", ""))

            # base (legacy .pt cache may include text/labels)
            if "text_emb" in samp and "text_mask" in samp:
                if samp["text_emb"].shape[0] == len(self.text_concept_names):
                    out["text_emb"] = samp["text_emb"]
                    out["text_mask"] = samp["text_mask"]
                elif not (self.use_text_override and self.text_override_dir):
                    raise ValueError(
                        "Cached text_emb K mismatch. Enable text_override or regenerate cache."
                    )

            if self.include_cls_labels and "cls_labels" in samp and "cls_mask" in samp:
                if samp["cls_labels"].shape[0] == len(self.cls_concept_names):
                    out["cls_labels"] = samp["cls_labels"]
                    out["cls_mask"] = samp["cls_mask"]
                elif not (self.use_cls_override and self.cls_override_dir):
                    raise ValueError(
                        "Cached cls_labels K mismatch. Enable cls_override or regenerate cache."
                    )
        else:
            img1, accession_from_store = self._load_image_1ch_from_sharded(sop)
            img01 = img1.float().repeat(3, 1, 1)
            img = self.transform(img01)
            out = {"images": img}

            if self.acc_col in self.df.columns:
                accession = str(self.df.iloc[idx][self.acc_col])
            elif accession_from_store is not None:
                accession = str(accession_from_store)
            else:
                accession = ""

        # optional text override by accession
        if self.use_text_override and self.text_override_dir:
            ov = self._get_text_override(accession)
            if ov is None and self.text_override_strict:
                raise FileNotFoundError(f"Missing text override cache for accession={accession}")
            if ov is not None:
                out["text_emb"] = ov["text_emb"]
                out["text_mask"] = ov["text_mask"]

        if self._image_cache_mode_effective == "sharded":
            # Image shards are image-only. Require text from override sources.
            if "text_emb" not in out or "text_mask" not in out:
                raise ValueError(
                    "Sharded image cache does not include text embeddings. "
                    "Enable text_override_dir/use_text_override."
                )

        # CLS from override (or legacy cache if already present in out)
        if self.include_cls_labels:
            if self.use_cls_override and self.cls_override_dir:
                ov = self._get_cls_override(accession)
                if ov is None and self.cls_override_strict:
                    raise FileNotFoundError(f"Missing CLS override for accession={accession}")
                if ov is not None:
                    out["cls_labels"] = ov["cls_labels"]
                    out["cls_mask"] = ov["cls_mask"]

            if self._image_cache_mode_effective == "sharded" and ("cls_labels" not in out or "cls_mask" not in out):
                raise ValueError(
                    "Sharded image cache does not include CLS labels. "
                    "Enable cls_override_dir/use_cls_override or set include_cls_labels=False."
                )

        return out

    



# -------------------------
# Dataloader helper
# -------------------------

def make_dataloader(dataset, batch_size, shuffle, num_workers=10,
                    prefetch_factor=6, persistent_workers=False,
                    pin_memory=True, drop_last=False,
                    sample_weights=None):  # NEW: optional per-sample weights for WeightedRandomSampler
    """
    Build a DataLoader. When sample_weights is provided (array-like of length len(dataset)),
    a WeightedRandomSampler is used instead of shuffle so that samples are drawn proportionally
    to their weight. This is the mechanism for Strategy B (concept-aware oversampling).
    """
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.float64),
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,          # sampler is mutually exclusive with shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=drop_last,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        #pin_memory_device=pin_memory_device,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )


def compute_concept_sample_weights(
    df: pd.DataFrame,
    concept_names: List[str],
    text_override_dir: str,
    acc_col: str = "accession",
    coverage_csv_path: Optional[str] = None,
    rare_threshold: float = 0.15,
    target_coverage: float = 0.15,
    weights_cache_path: Optional[str] = None,
    split_filter: str = "train",
    index_cache_name: str = "sharded_index.parquet",
) -> np.ndarray:
    """
    Compute per-sample importance weights for WeightedRandomSampler (Strategy B).

    For each concept c whose text coverage < rare_threshold, samples that carry text for
    concept c receive an additive bonus weight:
        rarity_mult_c = max(0, target_coverage / coverage_c - 1)
    The final weight for sample i is:
        weight_i = 1 + sum_c [ has_text_{i,c} * rarity_mult_c ]

    Frequent concepts (coverage >= rare_threshold) contribute zero bonus, so their
    samples are only drawn at the baseline rate.  Rare-concept samples are oversampled
    proportionally to how far their coverage is below target_coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Per-sample master index for the training split (len = N_samples).
    concept_names : list[str]
        Ordered list of alignment concept names (without "structured." prefix).
    text_override_dir : str
        Root directory of the sharded embedding store.  The function reads
        ``{text_override_dir}/{index_cache_name}`` to discover which accessions
        have embeddings for which concept fields.
    acc_col : str
        Column in df that holds the accession ID (links samples to embedding entries).
    coverage_csv_path : str | None
        Path to a CSV file with columns [split, concept_name, pct_samples_with_text].
        When provided AND the matching split rows exist, the per-concept coverage
        fractions are read from this file (fast; no dataset scan required).
        Example: the text_stats_align_concepts.csv generated during preprocessing.
        If None or missing, coverage is estimated from the sharded index directly.
    rare_threshold : float
        Concepts with text coverage fraction < rare_threshold are considered rare and
        their samples receive upsampling bonus.  Default: 0.15 (15 %).
    target_coverage : float
        The coverage fraction used to compute the upsampling strength.
        rarity_mult_c = target_coverage / coverage_c - 1.
        At target_coverage = 0.15 and coverage_c = 0.012 (a rare concept), this
        gives rarity_mult_c ≈ 11.5, so samples with that concept's text are
        ~12.5× more likely to be drawn.
    weights_cache_path : str | None
        If provided, the computed weight array is saved here as a .npy file.
        On subsequent calls with the same path the cached array is returned directly,
        skipping all computation.  Useful to avoid re-scanning the sharded index on
        every training run.
    split_filter : str
        Row filter applied to the coverage CSV's 'split' column.  Default: 'train'.
    index_cache_name : str
        Filename of the sharded embedding index inside text_override_dir.
        Default: 'sharded_index.parquet' (ShardedEmbeddingStore default).

    Returns
    -------
    weights : np.ndarray, shape [N], dtype float32
        Per-sample weights (all >= 1.0).
    """
    # ── 1. Cache hit ────────────────────────────────────────────────────────────
    if weights_cache_path and os.path.exists(weights_cache_path):
        print(f"[WeightedSampler] Loading cached sample weights from {weights_cache_path}")
        arr = np.load(weights_cache_path)
        if len(arr) == len(df):
            return arr.astype(np.float32)
        print(f"[WeightedSampler] Cache length mismatch ({len(arr)} vs {len(df)}). Recomputing.")

    concept_set = set(concept_names)
    K = len(concept_names)
    concept_idx = {c: i for i, c in enumerate(concept_names)}

    # ── 2. Per-concept rarity multipliers ────────────────────────────────────────
    rarity_mult = np.zeros(K, dtype=np.float64)
    coverage_from_csv: Optional[dict] = None  # concept -> pct fraction [0,1]

    if coverage_csv_path and os.path.exists(coverage_csv_path):
        cov_df = pd.read_csv(coverage_csv_path)
        # Filter to the requested split; if that split has no rows (e.g. the
        # report pipeline emits a single "all" split when no split file is given),
        # fall back to using all rows.
        if "split" in cov_df.columns:
            sel = cov_df[cov_df["split"] == split_filter]
            cov_df = sel if len(sel) > 0 else cov_df
        cols = set(cov_df.columns)
        coverage_from_csv = {}
        for _, row in cov_df.iterrows():
            cname = str(row.get("concept_name", ""))
            if cname not in concept_idx:
                continue
            # Accept either a direct percentage column, or the report pipeline's
            # count columns (n_samples_with_text / n_samples_total).
            if "pct_samples_with_text" in cols:
                pct = float(row["pct_samples_with_text"]) / 100.0
            elif "pct_accessions_with_text" in cols:
                pct = float(row["pct_accessions_with_text"]) / 100.0
            elif {"n_samples_with_text", "n_samples_total"} <= cols:
                total = float(row["n_samples_total"])
                pct = float(row["n_samples_with_text"]) / total if total > 0 else 0.0
            else:
                raise ValueError(
                    "Coverage CSV must contain 'pct_samples_with_text', "
                    "'pct_accessions_with_text', or 'n_samples_with_text'+'n_samples_total'. "
                    f"Got columns: {sorted(cols)}"
                )
            coverage_from_csv[cname] = pct
        for cname, pct in coverage_from_csv.items():
            if pct < rare_threshold:
                i = concept_idx[cname]
                rarity_mult[i] = max(0.0, target_coverage / max(pct, 1e-8) - 1.0)
        n_below_gate = sum(1 for pct in coverage_from_csv.values() if pct < rare_threshold)
        n_upsampled = int((rarity_mult > 0).sum())
        print(f"[WeightedSampler] Coverage loaded from CSV. "
              f"{n_below_gate}/{K} concepts below rare_threshold={rare_threshold:.2f}; "
              f"{n_upsampled} actually upsampled (pct < target_coverage={target_coverage:.2f}).")
    else:
        # Rarity computed later from the sharded index counts
        coverage_from_csv = None
        print(f"[WeightedSampler] No coverage CSV found. Will estimate coverage from sharded index.")

    # ── 3. Read sharded embedding index ──────────────────────────────────────────
    index_path = os.path.join(text_override_dir, index_cache_name)
    if not os.path.exists(index_path):
        print(f"[WeightedSampler] WARNING: sharded index not found at {index_path}. "
              "Returning uniform weights.")
        return np.ones(len(df), dtype=np.float32)

    print(f"[WeightedSampler] Reading sharded index: {index_path}")
    idx_df = pd.read_parquet(index_path, columns=["accession", "field"])

    # Filter to concept fields only
    concept_fields = {f"structured.{c}" for c in concept_names}
    idx_df = idx_df[idx_df["field"].isin(concept_fields)].copy()
    idx_df["_cname"] = idx_df["field"].str.slice(start=len("structured."))

    # Build: accession (str) -> set of covered concept names
    acc_to_concepts: dict = (
        idx_df.groupby("accession")["_cname"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )
    acc_to_concepts = {str(k): v for k, v in acc_to_concepts.items()}

    # ── 4. If no CSV, estimate rarity from sharded index counts ──────────────────
    if coverage_from_csv is None:
        unique_accs_in_df = set(df[acc_col].astype(str).unique())
        n_accs = max(len(unique_accs_in_df), 1)
        for c in concept_names:
            count = sum(1 for acc, cset in acc_to_concepts.items()
                        if acc in unique_accs_in_df and c in cset)
            pct = count / n_accs
            if pct < rare_threshold:
                i = concept_idx[c]
                rarity_mult[i] = max(0.0, target_coverage / max(pct, 1e-8) - 1.0)
        n_upsampled = int((rarity_mult > 0).sum())
        print(f"[WeightedSampler] {n_upsampled}/{K} concepts will be upsampled "
              f"(pct < target_coverage={target_coverage:.2f}; rare_threshold gate={rare_threshold:.2f}).")

    # Print per-concept multipliers for the rare ones
    for i, c in enumerate(concept_names):
        if rarity_mult[i] > 0:
            print(f"  [{c}]  rarity_mult = {rarity_mult[i]:.2f}")

    # ── 5. Compute per-sample weights ────────────────────────────────────────────
    accessions = df[acc_col].astype(str).values
    weights = np.ones(len(df), dtype=np.float64)

    for i, acc in enumerate(accessions):
        covered = acc_to_concepts.get(acc, set())
        extra = 0.0
        for c in covered:
            ci = concept_idx.get(c)
            if ci is not None:
                extra += rarity_mult[ci]
        weights[i] = 1.0 + extra

    weights = weights.astype(np.float32)

    # ── 6. Save cache ─────────────────────────────────────────────────────────────
    if weights_cache_path:
        cache_dir = os.path.dirname(weights_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(weights_cache_path, weights)
        print(f"[WeightedSampler] Saved sample weights to {weights_cache_path}")

    nonuniform = int((weights > 1.0).sum())
    print(f"[WeightedSampler] {nonuniform}/{len(weights)} samples have weight > 1.0. "
          f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    return weights
