from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def _shift_with_fill(array: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    """Shift a 2D array without wraparound."""
    out = np.full_like(array, fill)
    h, w = array.shape

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy) if dy < 0 else h

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx) if dx < 0 else w

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def distance_weighted_tversky(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
    valid_mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Compute the GEMS-style distance-weighted Tversky index.

    The competition uses a triangular distance kernel with 300 m support on a
    100 m raster, i.e. a radius of 3 pixels. This implementation follows the
    published formulas.

    Non-finite prediction values are permitted only outside the valid mask.
    This matches submission rasters that use NaN outside the valid study area.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth).astype(bool)

    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")
    if alpha < 0 or beta < 0 or radius_pixels <= 0:
        raise ValueError("alpha/beta must be non-negative and radius_pixels must be positive")

    if valid_mask is None:
        valid = np.ones_like(gt, dtype=bool)
    else:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape != gt.shape:
            raise ValueError("valid_mask must match prediction shape")

    valid_predictions = pred[valid]
    if not np.all(np.isfinite(valid_predictions)):
        raise ValueError("prediction contains non-finite values inside the valid region")
    if valid_predictions.size and (
        valid_predictions.min() < 0.0 or valid_predictions.max() > 1.0
    ):
        raise ValueError("prediction values must lie in [0, 1] inside the valid region")

    pred = np.where(valid, pred, 0.0)
    gt = gt & valid

    best_near_truth = np.zeros_like(pred)
    radius_int = math.ceil(radius_pixels)
    for dy in range(-radius_int, radius_int + 1):
        for dx in range(-radius_int, radius_int + 1):
            distance = math.hypot(dy, dx)
            if distance > radius_pixels:
                continue
            weight = max(1.0 - distance / radius_pixels, 0.0)
            shifted = _shift_with_fill(pred, dy, dx, fill=0.0)
            np.maximum(best_near_truth, shifted * weight, out=best_near_truth)

    matched_truth = best_near_truth[gt]
    tp_w = float(matched_truth.sum())
    fn_w = float((1.0 - matched_truth).sum())

    if gt.any():
        distance_to_truth = distance_transform_edt(~gt)
        truth_kernel = np.maximum(1.0 - distance_to_truth / radius_pixels, 0.0)
    else:
        truth_kernel = np.zeros_like(pred)

    fp_w = float((pred * (1.0 - truth_kernel) * valid).sum())
    denominator = tp_w + alpha * fp_w + beta * fn_w + eps
    return tp_w / denominator
