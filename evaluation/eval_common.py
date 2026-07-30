# -*- coding: utf-8 -*-
"""
Shared helpers for the RadPRISM evaluation scripts.

Provides configuration resolution, model construction, classification metrics,
text embedding backends and concept-retrieval utilities used by both
``evaluate_internal.py`` and ``evaluate_chexlocalize.py``.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# Shared source modules live at the repository root (RadPRISM/src).
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ModelConfig, VisionConceptModel  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config loading + run-config resolution
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts; values from ``override`` win."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def resolve_effective_run_config(
    run_dir: str,
    use_parent_pretrain_config: bool = True,
    parent_pretrain_config_path: Optional[str] = None,
    auto_parent_from_finetune_config: bool = True,
) -> Dict[str, Any]:
    """
    Load ``run_dir/config.json`` and, for fine-tune runs, merge the parent
    pretraining config underneath it (so the full 'model' block is available).
    """
    run_cfg = load_json(str(Path(run_dir) / "config.json"))

    parent_path = parent_pretrain_config_path
    if parent_path is None:
        parent_path = (run_cfg.get("pretrain") or {}).get("config_path")

    should_try_parent = use_parent_pretrain_config or (
        auto_parent_from_finetune_config and ("model" not in run_cfg)
    )
    if not should_try_parent or not parent_path:
        if "model" not in run_cfg and not parent_path:
            raise KeyError(
                "Run config has no 'model' block and no parent config path. "
                "Set model.parent_pretrain_config_path in the eval config."
            )
        return run_cfg

    if not Path(parent_path).exists():
        # Parent config missing but run config is self-contained -> use it as-is.
        if "model" in run_cfg:
            return run_cfg
        raise FileNotFoundError(f"Parent pretrain config not found: {parent_path}")

    return deep_merge_dicts(load_json(parent_path), run_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Concept lists + cls-head map
# ─────────────────────────────────────────────────────────────────────────────

def resolve_concept_lists(cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    align = cfg.get("align_concept_names") or cfg.get("concept_names")
    cls = cfg.get("cls_concept_names") or cfg.get("concept_names") or align
    if align is None or cls is None:
        raise KeyError("config.json is missing concept lists.")
    return list(align), list(cls)


def infer_cls_head_map(cls_names: List[str], align_names: List[str]) -> Optional[List[int]]:
    """None when cls == align (identity); otherwise map each cls concept to its align index."""
    if len(cls_names) == len(align_names) and cls_names == align_names:
        return None
    align_idx = {name: i for i, name in enumerate(align_names)}
    missing = [c for c in cls_names if c not in align_idx]
    if missing:
        raise ValueError(f"cls concepts not found among align concepts: {missing}")
    return [int(align_idx[c]) for c in cls_names]


def normalize_or_infer_cls_head_map(
    cfg: Dict[str, Any], cls_names: List[str], align_names: List[str]
) -> Optional[List[int]]:
    explicit = cfg.get("cls_head_map")
    if explicit is not None:
        out = [int(x) for x in explicit]
        if len(out) != len(cls_names):
            raise ValueError("cls_head_map length mismatch with cls_concept_names.")
        if any(x < 0 or x >= len(align_names) for x in out):
            raise ValueError("cls_head_map contains out-of-range indices.")
        return out
    return infer_cls_head_map(cls_names, align_names)


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model_from_config(
    cfg: Dict[str, Any], device: str, record_attn_maps: bool = False
) -> Tuple[VisionConceptModel, List[str], List[str], Optional[List[int]]]:
    """Rebuild a VisionConceptModel from a saved run config (no weights loaded)."""
    model_cfg = cfg["model"]
    align_names, cls_names = resolve_concept_lists(cfg)

    use_cls_heads = bool(model_cfg.get("use_cls_heads", True))
    if bool((cfg.get("losses") or {}).get("use_cls_loss", False)):
        use_cls_heads = True
    cls_head_map = normalize_or_infer_cls_head_map(cfg, cls_names, align_names) if use_cls_heads else None

    mcfg = ModelConfig(
        concept_names=align_names,
        cls_concept_names=cls_names if use_cls_heads else None,
        cls_head_map=cls_head_map,
        use_cls_heads=use_cls_heads,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg.get("n_heads", 12),
        dropout=model_cfg.get("dropout", 0.1),
        vision_backbone=model_cfg.get("vision_backbone", "rad_dino_maira_2"),
        pretrained_vision=model_cfg.get("pretrained_vision", False),
        vision_weights_path=(model_cfg.get("vision_weights_path") or None),
        text_in_dim=model_cfg.get("text_in_dim", 768),
        project_text=model_cfg.get("project_text", True),
        rad_dino_model_dir=model_cfg.get("rad_dino_model_dir", None),
        dinov2_repo_path=model_cfg.get("dinov2_repo_path", None),
        dinov2_weights_path=model_cfg.get("dinov2_weights_path", None),
        record_attn_maps=record_attn_maps,
    )
    model = VisionConceptModel(mcfg).to(device)
    model.eval()
    return model, align_names, cls_names, cls_head_map


def load_model_weights(model: VisionConceptModel, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] {len(missing)} missing keys when loading checkpoint.")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys when loading checkpoint.")


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics + thresholds
# ─────────────────────────────────────────────────────────────────────────────

def compute_best_thresholds(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Best F1 and Youden-J thresholds on a 1-D score/label vector."""
    labels = labels.astype(int)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    result = {"n_pos": n_pos, "n_neg": n_neg,
              "best_f1": float("nan"), "thr_f1": float("nan"),
              "best_j": float("nan"), "thr_j": float("nan")}
    if n_pos == 0 or n_neg == 0:
        return result

    best_f1, best_j, thr_f1, thr_j = -1.0, -1.0, 0.5, 0.5
    for thr in np.linspace(0.0, 1.0, 501):
        pred = (scores >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        tn = int(np.sum((pred == 0) & (labels == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        j = recall + specificity - 1.0
        if f1 > best_f1:
            best_f1, thr_f1 = f1, thr
        if j > best_j:
            best_j, thr_j = j, thr
    result.update(best_f1=float(best_f1), thr_f1=float(thr_f1),
                  best_j=float(best_j), thr_j=float(thr_j))
    return result


def per_concept_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """AUROC/AUPRC + best-F1 (threshold-free) for one concept's valid entries."""
    out = {"auroc": float("nan"), "auprc": float("nan"), "best_f1": 0.0, "thr_f1": 0.5}
    y_true = y_true.astype(int)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return out
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        pass
    out["auprc"] = float(average_precision_score(y_true, y_score))
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precisions * recalls / (precisions + recalls)
    f1 = np.nan_to_num(f1)
    bi = int(np.argmax(f1))
    out["best_f1"] = float(f1[bi])
    out["thr_f1"] = float(thresholds[bi]) if bi < len(thresholds) else 0.5
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Text embedding backends
# ─────────────────────────────────────────────────────────────────────────────

class TextEmbedder:
    """
    Encode text into the model's ``text_in_dim`` space for zero-shot prompting
    and concept retrieval.

    Backends:
      - "sentence_transformer": a local SentenceTransformer model (e.g. Qwen3-Embedding).
      - "auto_model":          a local HuggingFace AutoModel with mean/cls pooling.
      - "hash":                deterministic pseudo-random vectors (no model). For the
                               dummy dataset / smoke tests only — NOT meaningful.
    """

    def __init__(self, backend: str, dim: int, model_dir: Optional[str] = None,
                 device: str = "cpu", precision: str = "float32", normalize: bool = True,
                 pooling: str = "mean", max_length: int = 512, batch_size: int = 32,
                 truncate_dim: Optional[int] = None):
        self.backend = backend
        self.dim = int(dim)
        self.device = device
        self.normalize = normalize
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim or dim
        self._model = None
        self._tokenizer = None
        if backend in ("sentence_transformer", "auto_model"):
            if not model_dir:
                raise ValueError(f"backend='{backend}' requires a model_dir.")
            self.model_dir = model_dir
            self._dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                           "float32": torch.float32}.get(precision, torch.float32)
            self._load()
        elif backend != "hash":
            raise ValueError(f"Unknown embedding backend: {backend}")

    def _load(self):
        if self.backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_dir, device=self.device)
        else:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "hash":
            return self._encode_hash(texts)
        if self.backend == "sentence_transformer":
            vecs = self._model.encode(list(texts), batch_size=self.batch_size,
                                      convert_to_numpy=True, normalize_embeddings=False)
            vecs = np.asarray(vecs, dtype=np.float32)[:, : self.truncate_dim]
        else:
            vecs = self._encode_auto_model(texts)
        if self.normalize:
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return vecs.astype(np.float32)

    def _encode_hash(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(t.encode("utf-8"), "little", signed=False) % (2 ** 32) if t else 0
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out

    @torch.no_grad()
    def _encode_auto_model(self, texts: Sequence[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            enc = self._tokenizer(batch, padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._model(**enc)
            hidden = out.last_hidden_state
            if self.pooling == "cls":
                vec = hidden[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            chunks.append(vec.float().cpu().numpy()[:, : self.truncate_dim])
        return np.concatenate(chunks, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Concept retrieval
# ─────────────────────────────────────────────────────────────────────────────

def project_text_to_query_space(raw: np.ndarray, model: VisionConceptModel) -> torch.Tensor:
    """Project raw text embeddings [N, text_in_dim] into the model's concept space."""
    x = torch.from_numpy(np.asarray(raw, dtype=np.float32))
    text_proj = getattr(model, "text_proj", None)
    if text_proj is not None:
        dev = next(model.parameters()).device
        with torch.no_grad():
            x = text_proj(x.to(dev)).cpu()
    return F.normalize(x, dim=1)


def build_retrieval_db_from_custom(
    custom_texts_by_concept: Dict[str, List[str]],
    align_names: List[str],
    embedder: TextEmbedder,
) -> Dict[int, Dict[str, Any]]:
    """
    Build a per-concept retrieval DB from hand-authored sentences.

    Returns {align_idx: {"raw": np.ndarray[N, text_in_dim], "meta": [{"text": ...}]}}.
    """
    name_to_idx = {n: i for i, n in enumerate(align_names)}
    db: Dict[int, Dict[str, Any]] = {}
    for cname, texts in custom_texts_by_concept.items():
        if cname not in name_to_idx:
            print(f"[retrieval] WARNING: unknown concept '{cname}' in custom text DB, skipped.")
            continue
        texts = [str(t).strip() for t in texts if str(t).strip()]
        if not texts:
            continue
        raw = embedder.encode(texts)
        db[name_to_idx[cname]] = {"raw": raw, "meta": [{"text": t, "concept": cname} for t in texts]}
    return db


def query_retrieval_db(
    v_concepts: torch.Tensor,          # [K_align, d_model] (single image)
    db: Dict[int, Dict[str, Any]],
    model: VisionConceptModel,
    align_names: List[str],
    top_k: int = 3,
    concept_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Return the top-k unique text matches per requested concept for one image."""
    proj_cache: Dict[int, torch.Tensor] = {}
    results: List[Dict[str, Any]] = []
    indices = list(concept_indices) if concept_indices is not None else sorted(db.keys())
    for aidx in indices:
        entry = db.get(int(aidx))
        if entry is None:
            continue
        if aidx not in proj_cache:
            proj_cache[aidx] = project_text_to_query_space(entry["raw"], model)
        embs = proj_cache[aidx]                       # [N, d_model], normalized
        vq = F.normalize(v_concepts[aidx], dim=0)
        sims = torch.mv(embs, vq)
        order = torch.argsort(sims, descending=True)
        seen, kept = set(), 0
        for j in order.tolist():
            txt = str(entry["meta"][j].get("text", "")).strip()
            if txt in seen:
                continue
            seen.add(txt)
            results.append({
                "concept": align_names[aidx],
                "align_idx": int(aidx),
                "rank": kept + 1,
                "sim": float(sims[j]),
                "text": txt,
            })
            kept += 1
            if kept >= top_k:
                break
    return results
