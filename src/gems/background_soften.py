"""Background-loss softening for incomplete labels (P3, docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md).

Known fault labels are incomplete: geologist selection is not uniform and
unmapped faults exist in the background. Plain Tversky treats every
non-labeled pixel as hard background, so predicting a plausible but
unmapped fault near a known trace pays the full false-positive cost.
The competition's own distance-weighted metric already forgives
near-truth predictions (300 m tolerance, about 3 px); this loss extends
a milder version of that tolerance to training by downweighting the
false-positive penalty for background pixels in an *ambiguity band*
around training-side truth.

Honesty rules (same discipline as P1):
1. Ambiguity is fit from training-side evidence only: truth pixels
   outside the valid supervision mask never define the band.
2. No pseudolabeling: band pixels are NOT promoted to positives — the
   TP and FN terms are unchanged, nothing is rewarded for predicting in
   the band.
3. No flooding reward: background outside the band keeps the full FP
   weight, so flooded predictions still pay full cost.
4. No wraparound: the band is built with a Euclidean distance transform,
   which never wraps at array borders.

With ``soften_factor = 1.0`` the loss reduces exactly to the standard
soft Tversky; with ``soften_factor < 1.0`` the weighted FP term can only
shrink, so the softened loss is always <= the control loss on the same
prediction. The reduction is fixed in config before evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class SoftenedTverskyComponents:
    """Soft-Tversky evidence with a weighted FP term on one patch."""

    tp: float
    fp_w: float
    fn: float

    def score(self, alpha: float = 0.2, beta: float = 0.8, eps: float = 1e-12) -> float:
        return self.tp / (self.tp + alpha * self.fp_w + beta * self.fn + eps)

    def loss(self, alpha: float = 0.2, beta: float = 0.8, eps: float = 1e-12) -> float:
        return 1.0 - self.score(alpha, beta, eps)


def _validate_params(radius_pixels: float, soften_factor: float) -> None:
    if not radius_pixels > 0:
        raise ValueError("ambiguity radius_pixels must be positive")
    if not 0.0 < soften_factor <= 1.0:
        raise ValueError("background_soften_factor must lie in (0, 1]")


def ambiguity_weights(
    truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    radius_pixels: float = 3.0,
    soften_factor: float = 0.5,
) -> np.ndarray:
    """Per-pixel FP weight map for the softened Tversky loss.

    Returns 1.0 everywhere valid, ``soften_factor`` on background pixels
    within ``radius_pixels`` (strictly less, matching P1's convention) of
    a truth pixel, and 0.0 on invalid pixels. Truth pixels themselves are
    not background and keep weight 1.0 (their weight is unused since the
    FP term multiplies by ``1 - y``).
    """
    _validate_params(radius_pixels, soften_factor)
    gt = np.asarray(truth).astype(bool)
    if gt.ndim != 2:
        raise ValueError("truth must be a 2D array")
    valid = np.ones_like(gt, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match truth shape")

    gt_eff = gt & valid
    weights = np.where(valid, 1.0, 0.0)
    if gt_eff.any():
        distance_to_truth = distance_transform_edt(~gt_eff)
        band = (distance_to_truth < radius_pixels) & ~gt_eff & valid
        weights = np.where(band, soften_factor, weights)
    return weights.astype(np.float64)


def softened_tversky_components(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    radius_pixels: float = 3.0,
    soften_factor: float = 0.5,
    alpha: float = 0.2,
    beta: float = 0.8,
) -> SoftenedTverskyComponents:
    """NumPy reference for the background-softened Tversky evidence.

    ``prediction`` holds probabilities in [0, 1]. Only the FP term is
    weighted (by :func:`ambiguity_weights`); TP and FN are the standard
    soft-Tversky terms, so band predictions are never rewarded.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth).astype(bool)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")
    valid = np.ones_like(gt, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match prediction shape")

    weights = ambiguity_weights(gt, valid, radius_pixels=radius_pixels, soften_factor=soften_factor)
    prob = np.where(valid, pred, 0.0)
    y = gt & valid
    tp = float((prob * y).sum())
    fp_w = float((weights * prob * (1.0 - y)).sum())
    fn = float(((1.0 - prob) * y).sum())
    return SoftenedTverskyComponents(tp=tp, fp_w=fp_w, fn=fn)


def softened_tversky_loss_value(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    radius_pixels: float = 3.0,
    soften_factor: float = 0.5,
    alpha: float = 0.2,
    beta: float = 0.8,
    eps: float = 1e-12,
) -> float:
    """Scalar loss ``1 - TP/(TP + alpha FP_w + beta FN)`` on one patch."""
    components = softened_tversky_components(
        prediction,
        truth,
        valid_mask=valid_mask,
        radius_pixels=radius_pixels,
        soften_factor=soften_factor,
        alpha=alpha,
        beta=beta,
    )
    return components.loss(alpha=alpha, beta=beta, eps=eps)


def build_background_softened_loss(
    *,
    alpha: float = 0.2,
    beta: float = 0.8,
    ambiguity_radius_pixels: float = 3.0,
    background_soften_factor: float = 0.5,
    eps: float = 1e-12,
):
    """Build the background-softened Tversky torch loss.

    torch is imported here (not at module top) so this module stays
    importable on machines without torch. The returned module accepts
    ``(logits, targets, valid=None)`` with (B, H, W) tensors and keeps
    the full 2D structure so the ambiguity band stays spatially
    meaningful. The ambiguity band is built from training-side truth
    only (``truth & valid``), per the honesty rules above.
    """
    import torch
    from torch import nn

    _validate_params(ambiguity_radius_pixels, background_soften_factor)

    class BackgroundSoftenedTverskyLoss(nn.Module):
        needs_spatial = True  # train_full_map.py must not flatten the batch

        def __init__(self) -> None:
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.ambiguity_radius_pixels = ambiguity_radius_pixels
            self.background_soften_factor = background_soften_factor
            self.eps = eps

        def forward(self, logits, targets, valid=None):
            prob = torch.sigmoid(logits)
            if valid is None:
                valid = torch.ones_like(targets, dtype=torch.bool)
            else:
                valid = valid.bool()
            truth = (targets > 0.5) & valid
            prob = prob * valid.to(prob.dtype)
            truth_np = truth.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            weights = np.stack(
                [
                    ambiguity_weights(
                        truth_np[b],
                        valid_np[b],
                        radius_pixels=self.ambiguity_radius_pixels,
                        soften_factor=self.background_soften_factor,
                    )
                    for b in range(truth.shape[0])
                ]
            )
            weight = torch.as_tensor(weights, dtype=prob.dtype, device=prob.device)
            tp = (prob * truth.to(prob.dtype)).sum(dim=(1, 2))
            fp_w = (weight * prob * (~truth).to(prob.dtype)).sum(dim=(1, 2))
            fn = ((1.0 - prob) * truth.to(prob.dtype)).sum(dim=(1, 2))
            score = tp / (tp + self.alpha * fp_w + self.beta * fn + self.eps)
            return (1.0 - score).mean()

    return BackgroundSoftenedTverskyLoss()
