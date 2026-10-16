"""Skeleton-recall auxiliary loss (P5, docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md).

[Kirchhoff et al., Skeleton Recall Loss (2024)](https://arxiv.org/abs/2404.03010)
and [Does the Skeleton-Recall Loss Really Work? (2025)](https://arxiv.org/abs/2508.11374)
provide conflicting empirical evidence on thin-structure segmentation. Neither
settles GEMS performance, so this module implements a small matched
auxiliary-loss ablation: the control soft-Tversky (alpha=0.2 / beta=0.8, the
incumbent training loss) plus a fixed-weight skeleton-recall term that rewards
connectivity along fault traces. Connectivity is a diagnostic, not a
replacement for distance-weighted scoring -- the gate still decides on the
pooled distance-weighted Tversky with fault/trace non-regression safeguards.

Construction (an adaptation for GEMS, dependency-free):
- Soft skeleton: the white top-hat ``x - open(x)`` plus Lantuejoul opening
  residues ``erode^k(x) - open(erode^k(x))`` over ``iterations`` erosions, all
  with 3x3 min/max pooling and zero padding (no wraparound). On binary input
  this is exactly the morphological skeleton; on soft predictions it is
  differentiable (subgradients through min/max/relu/clamp only).
- The top-hat term is load-bearing, not decorative: pure Lantuejoul erosion
  annihilates sub-structuring-element thin structures -- fault traces are
  1-3 px wide -- leaving exactly zero gradient on the pixels the recall term
  samples. The top-hat preserves the thin-structure gradient path, so a
  predicted 1 px line skeletonizes to itself.
- Target skeleton: the same operator applied to ``truth & valid``,
  thresholded at 0.5 (exactly {0, 1} on binary input; the threshold is a
  formality).
- Recall term: ``1 - sum(S_pred * T * valid) / (sum(T) + eps)`` per patch,
  defined as 0.0 when a patch holds no truth skeleton. The auxiliary term
  must not invent signal on background-only patches; the primary loss
  handles those.

Honesty rules (same discipline as P1/P3):
1. The target skeleton is built from training-side truth only
   (``truth & valid``); held-out fault/trace pixels never define it.
2. The base term is the control soft-Tversky restricted to supervised
   pixels, so with ``skeleton_weight = 0.0`` the loss reduces exactly to
   the incumbent's training objective on the same batch.
3. No wraparound: min/max pooling uses zero padding. Border pixels see a
   biased operator within ``iterations`` px of the patch edge; training
   windows come from the buffer protocol, so this is documented, not hidden.
4. A patchwise ratio is a surrogate for a global ratio; patch sampling and
   the mean reduction are documented rather than described as exact.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def _validate_weight(skeleton_weight: float) -> None:
    if not 0.0 <= skeleton_weight <= 1.0:
        raise ValueError("skeleton_weight must lie in [0, 1]")


def _validate_iterations(skeleton_iterations: int) -> None:
    if (
        not isinstance(skeleton_iterations, (int, np.integer))
        or isinstance(skeleton_iterations, bool)
        or not 1 <= int(skeleton_iterations) <= 12
    ):
        raise ValueError("skeleton_iterations must be an integer in [1, 12]")


def _validate_params(skeleton_weight: float, skeleton_iterations: int) -> None:
    _validate_weight(skeleton_weight)
    _validate_iterations(skeleton_iterations)


def _min_pool3(x: np.ndarray) -> np.ndarray:
    """3x3 erosion (stride 1, zero padding -- no wraparound)."""
    return minimum_filter(x, size=3, mode="constant", cval=0.0)


def _max_pool3(x: np.ndarray) -> np.ndarray:
    """3x3 dilation (stride 1, zero padding -- no wraparound)."""
    return maximum_filter(x, size=3, mode="constant", cval=0.0)


def soft_skeletonize(prob: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Differentiable soft skeleton of a 2D probability map.

    ``top-hat(x) + sum_k relu(erode^k(x) - open(erode^k(x)))``, clipped to
    [0, 1]. On binary input the terms are disjoint, so the output is exactly
    {0, 1}: the morphological skeleton, with 1 px structures preserved via
    the top-hat term. The torch twin in :func:`build_skeleton_recall_loss`
    implements the identical operator with max-pooling primitives.
    """
    _validate_iterations(iterations)
    x = np.asarray(prob, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("prob must be a 2D array")
    skel = np.maximum(x - _max_pool3(_min_pool3(x)), 0.0)
    for _ in range(int(iterations)):
        x = _min_pool3(x)
        skel = skel + np.maximum(x - _max_pool3(_min_pool3(x)), 0.0)
    return np.clip(skel, 0.0, 1.0)


def target_skeleton(
    truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    iterations: int = 5,
) -> np.ndarray:
    """Hard skeleton of the training-side truth (the recall target).

    Built from ``truth & valid`` only, so held-out fault/trace pixels never
    define the target. Returns a {0, 1} float array.
    """
    _validate_iterations(iterations)
    gt = np.asarray(truth).astype(bool)
    if gt.ndim != 2:
        raise ValueError("truth must be a 2D array")
    valid = np.ones_like(gt, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match truth shape")
    return (soft_skeletonize((gt & valid).astype(np.float64), iterations) > 0.5).astype(np.float64)


def skeleton_recall_loss_value(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    iterations: int = 5,
    eps: float = 1e-12,
) -> float:
    """Scalar ``1 - recall`` of the truth skeleton on one patch.

    ``prediction`` holds probabilities in [0, 1]. Returns 0.0 when the patch
    contains no truth-skeleton pixels -- the auxiliary term contributes no
    signal there by construction.
    """
    _validate_iterations(iterations)
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth).astype(bool)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")
    valid = np.ones_like(gt, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match prediction shape")

    target = target_skeleton(gt, valid, iterations)
    denom = float(target.sum())
    if denom == 0.0:
        return 0.0
    prob = np.where(valid, pred, 0.0)
    skel_pred = soft_skeletonize(prob, iterations)
    recall = float((skel_pred * target).sum() / (denom + eps))
    return float(np.clip(1.0 - recall, 0.0, 1.0))


def masked_soft_tversky_loss_value(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    eps: float = 1e-12,
) -> float:
    """NumPy reference for the control soft-Tversky on supervised pixels.

    The torch module's base term must agree with this; with
    ``skeleton_weight = 0.0`` the full loss reduces exactly to it.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth).astype(bool)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")
    valid = np.ones_like(gt, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match prediction shape")

    prob = np.where(valid, pred, 0.0)
    y = (gt & valid).astype(np.float64)
    tp = float((prob * y).sum())
    fp = float((prob * (1.0 - y)).sum())
    fn = float(((1.0 - prob) * y).sum())
    return 1.0 - tp / (tp + alpha * fp + beta * fn + eps)


def build_skeleton_recall_loss(
    *,
    alpha: float = 0.2,
    beta: float = 0.8,
    skeleton_weight: float = 0.25,
    skeleton_iterations: int = 5,
    eps: float = 1e-12,
):
    """Build the skeleton-recall auxiliary torch loss.

    Returns ``base_soft_tversky + skeleton_weight * skeleton_recall`` with
    ``needs_spatial=True``. torch is imported here (not at module top) so
    this module stays importable on machines without torch. The returned
    module accepts ``(logits, targets, valid=None)`` with (B, H, W) tensors
    and keeps the full 2D structure so the skeleton operator stays spatially
    meaningful. The target skeleton is built per patch on CPU from
    training-side truth only (``truth & valid``), mirroring the P3 pattern.

    Note: the base term is the masked soft-Tversky implemented inline (not
    the smp module) so the supervision mask is honored in *both* terms --
    held-out fault/trace pixels contribute to neither, keeping the ablation
    matched to the incumbent ``loss: tversky`` objective.
    """
    import torch
    from torch import nn

    _validate_params(skeleton_weight, skeleton_iterations)
    skeleton_iterations = int(skeleton_iterations)

    def _tminpool3(x):
        return -nn.functional.max_pool2d(-x, 3, 1, 1)

    def _tmaxpool3(x):
        return nn.functional.max_pool2d(x, 3, 1, 1)

    def _soft_skeletonize_torch(p):
        skel = torch.relu(p - _tmaxpool3(_tminpool3(p)))
        x = p
        for _ in range(skeleton_iterations):
            x = _tminpool3(x)
            skel = skel + torch.relu(x - _tmaxpool3(_tminpool3(x)))
        return skel.clamp(0.0, 1.0)

    class SkeletonRecallLoss(nn.Module):
        needs_spatial = True  # train_full_map.py must not flatten the batch

        def __init__(self) -> None:
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.skeleton_weight = skeleton_weight
            self.skeleton_iterations = skeleton_iterations
            self.eps = eps

        def forward(self, logits, targets, valid=None):
            prob = torch.sigmoid(logits)
            if valid is None:
                valid = torch.ones_like(targets, dtype=torch.bool)
            else:
                valid = valid.bool()
            v = valid.to(prob.dtype)
            prob_m = prob * v
            truth = ((targets > 0.5) & valid).to(prob.dtype)
            # Base: control soft-Tversky on supervised pixels only.
            tp = (prob_m * truth).sum(dim=(1, 2))
            fp = (prob_m * (1.0 - truth)).sum(dim=(1, 2))
            fn = ((1.0 - prob_m) * truth).sum(dim=(1, 2))
            base = 1.0 - tp / (tp + self.alpha * fp + self.beta * fn + self.eps)
            # Auxiliary: skeleton recall against the training-side target.
            truth_np = ((targets > 0.5) & valid).detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            tgt = np.stack(
                [target_skeleton(truth_np[b], valid_np[b], self.skeleton_iterations) for b in range(truth.shape[0])]
            )
            target = torch.as_tensor(tgt, dtype=prob.dtype, device=prob.device)
            denom = target.sum(dim=(1, 2))
            skel_pred = _soft_skeletonize_torch(prob_m)
            recall = (skel_pred * target).sum(dim=(1, 2)) / (denom + self.eps)
            aux = torch.where(denom > 0, 1.0 - recall, torch.zeros_like(recall)).clamp(0.0, 1.0)
            return (base + self.skeleton_weight * aux).mean()

    return SkeletonRecallLoss()
