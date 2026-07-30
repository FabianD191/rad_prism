# losses.py
# ============================================================================
# Loss functions for the vision-language concept-alignment model.
#
# Two loss families are supported:
#   1. Concept alignment (InfoNCE) - the core pretraining objective. For every
#      clinical concept the model learns to match its image-derived concept
#      embedding with the text embedding of the corresponding report snippet,
#      using an in-batch contrastive (InfoNCE) loss.
#   2. Binary classification (BCE) - an optional auxiliary head that predicts
#      concept presence/severity from the image alone. Disabled during pure
#      alignment pretraining, but kept here because the downstream fine-tuning
#      stage reuses it.
# ============================================================================

from dataclasses import dataclass
from typing import Dict, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last dimension of a tensor (numerically safe)."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@dataclass
class LossConfig:
    """
    Configuration for :func:`compute_losses`.

    The alignment loss is the primary pretraining objective; the classification
    loss is an optional auxiliary term that requires the model's classification
    heads to be enabled.
    """
    # ── Which losses are active ──────────────────────────────────────────────
    use_cls_loss: bool = False
    use_align_loss: bool = True

    # ── Classification (BCE) parameters ──────────────────────────────────────
    w_cls: float = 1.0
    pos_weight: Optional[torch.Tensor] = None            # per-concept BCE pos-weight [K]
    cls_concept_balanced_reduction: bool = False         # equal mean across active concepts

    # ── Concept alignment (InfoNCE) parameters ───────────────────────────────
    w_align: float = 1.0
    concept_averaged_align: bool = True                  # macro-average over concepts
    tau_align: float = 0.07                              # temperature (fixed fallback)
    learnable_concept_taus: bool = False                 # learn one temperature per concept
    align_log_tau_per_concept: Optional[torch.Tensor] = None  # log-temperatures [K]
    symmetric_align: bool = True                         # image->text and text->image


# ----------------------------------------------------------------------------
# 1. Classification losses (BCE)
# ----------------------------------------------------------------------------

def build_tempered_pos_weight_from_counts(
    pos_counts: torch.Tensor,
    neg_counts: torch.Tensor,
    gamma: float = 1.0,
    prior_smoothing_alpha: float = 0.0,
    global_pos_prior: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build concept-wise tempered BCE pos-weights from positive/negative counts.

    Raw ratio:      r_c = (1 - p_c) / p_c
    Tempered ratio: w_c = r_c ** gamma

    Optional Bayesian-style prior smoothing:
      p_c = (pos_c + alpha * p_global) / (n_c + alpha), alpha >= 0
    with p_global inferred from aggregate counts if not provided.
    """
    if pos_counts.shape != neg_counts.shape:
        raise ValueError(
            f"pos_counts and neg_counts must have same shape, got {tuple(pos_counts.shape)} vs {tuple(neg_counts.shape)}"
        )
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if prior_smoothing_alpha < 0:
        raise ValueError(f"prior_smoothing_alpha must be >= 0, got {prior_smoothing_alpha}")

    pos = pos_counts.float()
    neg = neg_counts.float()
    total = (pos + neg).clamp(min=0.0)

    if prior_smoothing_alpha > 0.0:
        if global_pos_prior is None:
            global_pos_prior_t = (pos.sum() + eps) / (total.sum() + 2.0 * eps)
        else:
            global_pos_prior_t = torch.tensor(float(global_pos_prior), device=pos.device, dtype=pos.dtype)
        global_pos_prior_t = global_pos_prior_t.clamp(min=eps, max=1.0 - eps)
        p = (pos + prior_smoothing_alpha * global_pos_prior_t) / (total + prior_smoothing_alpha).clamp(min=eps)
    else:
        # +1 smoothing mirrors common stable ratio estimate from counts.
        p = (pos + 1.0) / (total + 2.0)

    p = p.clamp(min=eps, max=1.0 - eps)
    ratio = (1.0 - p) / p
    return ratio.pow(float(gamma))


def masked_bce_with_logits(
    logits: torch.Tensor,     # [B,K]
    targets: torch.Tensor,    # [B,K]
    mask: torch.Tensor,       # [B,K] bool
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    concept_balanced_reduction: bool = False,
) -> torch.Tensor:
    """Standard BCE on valid entries only (entries where ``mask`` is True)."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    if pos_weight is not None:
        # keep [K] and broadcast across batch
        if pos_weight.dim() != 1 or pos_weight.numel() != logits.shape[1]:
            raise ValueError(f"pos_weight must be [K], got {tuple(pos_weight.shape)}")
        pos_weight = pos_weight.to(logits.device)

    # optional label smoothing toward 0.5
    if label_smoothing > 0.0:
        if not (0.0 <= label_smoothing < 0.5):
            raise ValueError(f"label_smoothing should be in [0, 0.5), got {label_smoothing}")
        with torch.no_grad():
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # elementwise, no flattening
    loss_el = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='none'
    )

    m = mask.to(loss_el.dtype)
    weighted = loss_el * m

    if not concept_balanced_reduction:
        return weighted.sum() / m.sum().clamp(min=1.0)

    # Concept-balanced reduction:
    # 1) mean loss per concept over its valid samples
    # 2) equal average over concepts that have at least one valid sample
    denom_per_concept = m.sum(dim=0)              # [K]
    valid_concepts = denom_per_concept > 0
    if valid_concepts.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    mean_per_concept = weighted.sum(dim=0)[valid_concepts] / denom_per_concept[valid_concepts].clamp(min=1.0)
    return mean_per_concept.mean()


# ----------------------------------------------------------------------------
# 2. Concept alignment losses (InfoNCE)
# ----------------------------------------------------------------------------

def concept_alignment_infonce(
    v_concepts: torch.Tensor, u_text: torch.Tensor, text_mask: torch.Tensor,
    tau: float = 0.07, symmetric: bool = True
) -> torch.Tensor:
    """
    Pair-weighted concept InfoNCE.

    Computes a separate in-batch InfoNCE loss per concept and averages the
    concept losses weighted by the number of valid pairs, so frequent concepts
    dominate the total. See :func:`concept_averaged_infonce` for the macro
    (equal-weight) variant used by default.
    """
    v = l2n(v_concepts)
    u = l2n(u_text)
    B, K, _ = v.shape

    total_weighted_loss = 0.0
    total_pairs = 0

    for c in range(K):
        mask_c = text_mask[:, c]
        idx = torch.nonzero(mask_c, as_tuple=False).squeeze(-1)
        Bc = idx.numel()
        if Bc <= 1:
            continue

        v_c = v[idx, c, :]
        u_c = u[idx, c, :]
        logits = v_c @ u_c.t() / tau
        targets = torch.arange(Bc, device=v.device)
        loss = F.cross_entropy(logits, targets)
        if symmetric:
            loss = 0.5 * (loss + F.cross_entropy(logits.t(), targets))

        total_weighted_loss += Bc * loss
        total_pairs += Bc

    if total_pairs == 0:
        return torch.tensor(0.0, device=v.device, requires_grad=True)
    return total_weighted_loss / total_pairs


def concept_averaged_infonce(
    v_concepts: torch.Tensor, u_text: torch.Tensor, text_mask: torch.Tensor,
    tau: Union[float, torch.Tensor] = 0.07, symmetric: bool = True,
) -> torch.Tensor:
    """
    Concept-averaged (macro) InfoNCE.

    Computes a separate in-batch InfoNCE for each concept that has at least two
    valid pairs in the batch, then averages the per-concept losses equally. This
    prevents frequent concepts from dominating the objective, which matters for
    the highly skewed text-coverage distribution across clinical concepts.

    Parameters
    ----------
    v_concepts : [B, K, D]
        Image-derived concept embeddings.
    u_text : [B, K, D]
        Projected text embeddings for each concept.
    text_mask : [B, K] bool
        True where a report snippet (and thus a positive pair) exists.
    tau : float | torch.Tensor
        Temperature. A scalar applies to all concepts; a [K] tensor provides a
        learnable per-concept temperature.
    symmetric : bool
        If True, average the image->text and text->image directions.
    """
    v = l2n(v_concepts)
    u = l2n(u_text)
    B, K, _ = v.shape

    # Resolve the temperature: either a shared scalar or one value per concept.
    tau_tensor = tau if torch.is_tensor(tau) else None
    tau_scalar: Optional[float] = None
    if tau_tensor is not None:
        if tau_tensor.ndim == 0:
            pass
        elif tau_tensor.ndim == 1:
            if tau_tensor.numel() != K:
                raise ValueError(
                    f"Per-concept tau length mismatch: expected {K}, got {tau_tensor.numel()}."
                )
        else:
            raise ValueError(f"tau tensor must be scalar or [K], got shape {tuple(tau_tensor.shape)}")
    else:
        tau_scalar = float(tau)
        if tau_scalar <= 0:
            raise ValueError(f"tau must be > 0, got {tau_scalar}")

    losses: List[torch.Tensor] = []

    for c in range(K):
        mask_c = text_mask[:, c]
        idx = torch.nonzero(mask_c, as_tuple=False).squeeze(-1)
        Bc = idx.numel()

        # Need at least two pairs to form a contrastive problem.
        if Bc <= 1:
            continue

        v_c = v[idx, c, :]  # [Bc, D]
        u_c = u[idx, c, :]  # [Bc, D]

        # Resolve per-concept temperature.
        if tau_tensor is not None:
            tau_c = tau_tensor if tau_tensor.ndim == 0 else tau_tensor[c]
            tau_c = tau_c.clamp(min=1e-6)
        else:
            tau_c = max(tau_scalar, 1e-6)

        logits = v_c @ u_c.t() / tau_c
        targets = torch.arange(Bc, device=v.device)
        loss = F.cross_entropy(logits, targets)
        if symmetric:
            loss = 0.5 * (loss + F.cross_entropy(logits.t(), targets))

        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=v.device, requires_grad=True)

    return torch.stack(losses).mean()


# ----------------------------------------------------------------------------
# 3. Loss orchestration
# ----------------------------------------------------------------------------

def compute_losses(
    batch: Dict,
    outputs: Dict,
    model: nn.Module,
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    Combine the active losses into a single total according to ``cfg``.

    Returns a dict with the scalar ``total`` plus the individual ``cls`` and
    ``align`` components (each 0.0 when its loss is disabled or has no valid
    entries in the batch).
    """
    loss_total = torch.tensor(0.0, device=outputs["v_concepts"].device)
    losses = {"total": 0.0, "cls": 0.0, "align": 0.0}

    # 1. Classification (BCE)
    if cfg.use_cls_loss and (batch.get("cls_labels") is not None):
        if outputs.get("concept_logits") is None:
            raise ValueError("concept_logits is None. Enable classification heads in the model.")
        l_cls = masked_bce_with_logits(
            outputs["concept_logits"],
            batch["cls_labels"],
            batch["cls_mask"],
            pos_weight=cfg.pos_weight,
            concept_balanced_reduction=cfg.cls_concept_balanced_reduction,
        )
        loss_total += cfg.w_cls * l_cls
        losses["cls"] = l_cls

    # 2. Concept alignment (InfoNCE)
    if cfg.use_align_loss and (batch.get("text_emb") is not None):
        # Per-concept learnable temperature (if enabled), else the fixed scalar.
        align_tau = cfg.tau_align
        if (
            cfg.concept_averaged_align
            and cfg.learnable_concept_taus
            and (cfg.align_log_tau_per_concept is not None)
        ):
            align_tau = torch.exp(cfg.align_log_tau_per_concept)

        if cfg.concept_averaged_align:
            l_align = concept_averaged_infonce(
                outputs["v_concepts"],
                outputs["u_text"],
                batch["text_mask"],
                tau=align_tau,
                symmetric=cfg.symmetric_align,
            )
        else:
            l_align = concept_alignment_infonce(
                outputs["v_concepts"],
                outputs["u_text"],
                batch["text_mask"],
                tau=cfg.tau_align,
                symmetric=cfg.symmetric_align,
            )

        loss_total += cfg.w_align * l_align
        losses["align"] = l_align

    losses["total"] = loss_total
    return losses
