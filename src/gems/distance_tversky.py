"""Distance-aware differentiable Tversky loss (P1, docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md).

Adapts the competition's distance-weighted Tversky equations directly as a
training loss. For evaluated label set G on a prediction grid p (probabilities
in [0, 1]), with a triangular distance kernel of radius ``radius_pixels``::

    m_g(p) = max_{x: |x-g| <= r} p_x (1 - |x-g| / r)
    T  = sum_{g in G} m_g(p)          (matched truth mass)
    FN = |G| - T
    F  = sum_x p_x [1 - max_g (1 - |x-g| / r)_+]   (unmatched prediction mass)

    L_distance = 1 - T / (T + alpha F + beta FN + eps)

The finite local max admits subgradients almost everywhere. Two rules keep the
loss honest:

1. Three-pixel context halos: contributions are counted on a ``core`` region
   only, but the prediction and truth context extends ``halo_pixels`` beyond
   the core. No wraparound at array borders.
2. Validity and supervision masks: predictions outside the valid region are
   zeroed, and truth pixels outside the valid region never enter the truth
   distance kernel (no withheld-label leakage into the FP field).

A patchwise ratio is a surrogate for the global full-map ratio: border context
is zero-filled and the ratio is reduced per patch. Patch sampling and reduction
are documented here rather than presented as exact global training.

Because an all-background patch gives only a weak gradient through this ratio
alone, the training recipe blends it with the conventional Tversky loss
(``loss_distance_weight`` in [0, 1], fixed in config before evaluation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class DistanceTverskyComponents:
    """Distance-weighted evidence measured on one patch's core region."""

    tp_w: float
    fp_w: float
    fn_w: float

    def score(self, alpha: float = 0.2, beta: float = 0.8, eps: float = 1e-12) -> float:
        return self.tp_w / (self.tp_w + alpha * self.fp_w + beta * self.fn_w + eps)

    def loss(self, alpha: float = 0.2, beta: float = 0.8, eps: float = 1e-12) -> float:
        return 1.0 - self.score(alpha, beta, eps)


def kernel_offsets(radius_pixels: float = 3.0) -> list[tuple[int, int, float]]:
    """Integer (dy, dx) offsets within the radius with triangular weights.

    Euclidean distance, matching ``gems.metric``. Offsets with zero weight are
    dropped; they cannot change the max-projection.
    """
    if radius_pixels <= 0:
        raise ValueError("radius_pixels must be positive")
    radius_int = math.ceil(radius_pixels)
    offsets: list[tuple[int, int, float]] = []
    for dy in range(-radius_int, radius_int + 1):
        for dx in range(-radius_int, radius_int + 1):
            distance = math.hypot(dy, dx)
            if distance >= radius_pixels:
                continue
            weight = 1.0 - distance / radius_pixels
            if weight > 0.0:
                offsets.append((dy, dx, weight))
    if not offsets:
        raise ValueError("radius_pixels too small to cover any offset")
    return offsets


def _shift_no_wrap(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2D array with zero fill (never wraps around edges)."""
    out = np.zeros_like(array)
    h, w = array.shape
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def distance_tversky_components(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    core_mask: np.ndarray | None = None,
    radius_pixels: float = 3.0,
) -> DistanceTverskyComponents:
    """NumPy reference for the patch distance-Tversky evidence.

    ``prediction`` holds probabilities in [0, 1] over the full patch
    (core + halo). ``core_mask`` selects the core region whose contributions
    are counted; when ``None`` the whole patch is the core. ``valid_mask``
    marks supervised pixels: predictions outside it are zeroed and truth
    outside it never enters the distance kernel.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth).astype(bool)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")
    valid = (
        np.ones_like(gt, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    )
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match prediction shape")
    core = np.ones_like(gt, dtype=bool) if core_mask is None else np.asarray(core_mask).astype(bool)
    if core.shape != gt.shape:
        raise ValueError("core_mask must match prediction shape")

    pred = np.where(valid, pred, 0.0)
    gt_eff = gt & valid

    best_near_truth = np.zeros_like(pred)
    for dy, dx, weight in kernel_offsets(radius_pixels):
        shifted = _shift_no_wrap(pred, dy, dx)
        np.maximum(best_near_truth, shifted * weight, out=best_near_truth)

    core_truth = core & gt_eff
    tp_w = float(best_near_truth[core_truth].sum())
    fn_w = float(core_truth.sum()) - tp_w

    if gt_eff.any():
        distance_to_truth = distance_transform_edt(~gt_eff)
        truth_kernel = np.maximum(1.0 - distance_to_truth / radius_pixels, 0.0)
    else:
        truth_kernel = np.zeros_like(pred)
    fp_w = float((pred * (1.0 - truth_kernel) * core).sum())
    return DistanceTverskyComponents(tp_w=tp_w, fp_w=fp_w, fn_w=fn_w)


def distance_tversky_loss_value(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    core_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
    eps: float = 1e-12,
) -> float:
    """Scalar loss ``1 - T/(T + alpha F + beta FN)`` on one patch."""
    components = distance_tversky_components(
        prediction,
        truth,
        valid_mask=valid_mask,
        core_mask=core_mask,
        radius_pixels=radius_pixels,
    )
    return components.loss(alpha=alpha, beta=beta, eps=eps)


def truth_kernel_numpy(
    truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    radius_pixels: float = 3.0,
) -> np.ndarray:
    """Triangular truth-proximity kernel ``max(1 - d/radius, 0)``.

    Used by the torch loss to build the FP field from training-side truth
    only: truth outside the valid mask never enters the kernel.
    """
    gt = np.asarray(truth).astype(bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape != gt.shape:
            raise ValueError("valid_mask must match truth shape")
        gt = gt & valid
    if not gt.any():
        return np.zeros(gt.shape, dtype=np.float64)
    distance_to_truth = distance_transform_edt(~gt)
    return np.maximum(1.0 - distance_to_truth / radius_pixels, 0.0)


def build_distance_tversky_loss(
    *,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
    blend_weight: float = 0.5,
    halo_pixels: int = 0,
    eps: float = 1e-12,
):
    """Build the blended torch loss ``(1-w)*Tversky + w*L_distance``.

    torch is imported here (not at module top) so this module stays
    importable on machines without torch. The returned module accepts
    ``(logits, targets, valid=None)`` with (B, H, W) tensors and keeps the
    full 2D structure so the distance kernel stays spatially meaningful.
    """
    import torch
    from segmentation_models_pytorch.losses import TverskyLoss
    from torch import nn

    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must lie in [0, 1]")
    if halo_pixels < 0:
        raise ValueError("halo_pixels must be non-negative")

    offsets = kernel_offsets(radius_pixels)
    radius_int = math.ceil(radius_pixels)

    class DistanceTverskyLoss(nn.Module):
        needs_spatial = True  # train_full_map.py must not flatten the batch

        def __init__(self) -> None:
            super().__init__()
            self.base = TverskyLoss(alpha=alpha, beta=beta, mode="binary")
            self.alpha = alpha
            self.beta = beta
            self.radius_pixels = radius_pixels
            self.blend_weight = blend_weight
            self.halo_pixels = halo_pixels
            self.eps = eps

        def _distance_term(self, prob, truth, valid, core):
            batch = prob.shape[0]
            # Max-projection of the triangular kernel over the prediction.
            padded = nn.functional.pad(prob, (radius_int,) * 4)
            best = torch.zeros_like(prob)
            h, w = prob.shape[1], prob.shape[2]
            for dy, dx, weight in offsets:
                shifted = padded[
                    :,
                    radius_int + dy : radius_int + dy + h,
                    radius_int + dx : radius_int + dx + w,
                ]
                torch.maximum(best, shifted * weight, out=best)
            t = (best * core * truth).sum(dim=(1, 2))
            fn = (core * truth).sum(dim=(1, 2)) - t
            # FP field: truth kernel built from training-side truth only.
            kernels = []
            truth_np = (truth & valid).detach().cpu().numpy().astype(bool)
            valid_np = valid.detach().cpu().numpy().astype(bool)
            for b in range(batch):
                kernels.append(
                    truth_kernel_numpy(truth_np[b], valid_np[b], radius_pixels=self.radius_pixels)
                )
            kernel = torch.as_tensor(np.stack(kernels), dtype=prob.dtype, device=prob.device)
            f = (prob * (1.0 - kernel) * core).sum(dim=(1, 2))
            return 1.0 - t / (t + self.alpha * f + self.beta * fn + self.eps)

        def forward(self, logits, targets, valid=None):
            prob = torch.sigmoid(logits)
            if valid is None:
                valid = torch.ones_like(targets, dtype=torch.bool)
            else:
                valid = valid.bool()
            truth = (targets > 0.5) & valid
            prob = prob * valid.to(prob.dtype)
            if self.halo_pixels:
                core = torch.zeros_like(valid)
                hp = self.halo_pixels
                core[:, hp : prob.shape[1] - hp, hp : prob.shape[2] - hp] = True
            else:
                core = torch.ones_like(valid)
            distance_term = self._distance_term(prob, truth, valid, core).mean()
            base_term = self.base(logits.unsqueeze(1), targets.unsqueeze(1).to(logits.dtype))
            return (1.0 - self.blend_weight) * base_term + self.blend_weight * distance_term

    return DistanceTverskyLoss()


def build_training_loss(config: dict):
    """Return ``(criterion, needs_spatial)`` from a training config.

    ``loss: tversky`` keeps the existing segmentation-models-pytorch Tversky
    loss. ``loss: distance_tversky`` returns the blended distance-aware loss,
    which requires the spatial batch structure (``needs_spatial=True``).
    ``loss: background_softened`` returns the P3 background-softened Tversky
    (FP penalty downweighted in the ambiguity band around training-side
    truth), which also requires the spatial batch structure.
    """
    training = config.get("training", {})
    name = str(training.get("loss", "tversky")).lower()
    alpha = float(training.get("alpha", 0.2))
    beta = float(training.get("beta", 0.8))
    if name == "tversky":
        from segmentation_models_pytorch.losses import TverskyLoss

        return TverskyLoss(alpha=alpha, beta=beta, mode="binary"), False
    if name == "distance_tversky":
        return (
            build_distance_tversky_loss(
                alpha=alpha,
                beta=beta,
                radius_pixels=float(training.get("loss_radius_pixels", 3.0)),
                blend_weight=float(training.get("loss_distance_weight", 0.5)),
                halo_pixels=int(training.get("loss_halo_pixels", 0)),
            ),
            True,
        )
    if name == "background_softened":
        from gems.background_soften import build_background_softened_loss

        return (
            build_background_softened_loss(
                alpha=alpha,
                beta=beta,
                ambiguity_radius_pixels=float(training.get("loss_ambiguity_radius_pixels", 3.0)),
                background_soften_factor=float(training.get("loss_background_soften_factor", 0.5)),
            ),
            True,
        )
    raise ValueError(f"unknown training loss: {name!r}")
