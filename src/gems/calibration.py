"""Exact binary-threshold search for the published distance-weighted metric."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class ThresholdCurve:
    thresholds: np.ndarray
    scores: np.ndarray
    true_positive_weight: np.ndarray
    false_positive_weight: np.ndarray
    truth_pixels: int

    @property
    def best_threshold(self) -> float:
        # Descending thresholds break exact score ties toward fewer predictions.
        return float(self.thresholds[int(np.argmax(self.scores))])


def binary_threshold(prediction: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the >= convention; NaN remains NaN outside an evaluation region."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    pred = np.asarray(prediction, dtype=np.float64)
    return np.where(np.isfinite(pred), (pred >= threshold).astype(float), np.nan)


def exact_threshold_curve(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
    eps: float = 1e-12,
) -> ThresholdCurve:
    """Score every distinct binary prediction attainable by a global threshold.

    Each truth pixel contributes an event when an activated nearby prediction
    improves its best triangular-kernel match. FP contributions are additive.
    Sorting and cumulatively summing these events avoids rescoring a full raster
    for every threshold. Memory is O(N + K*G), where N is valid pixels, G is truth
    pixels and K is the number of offsets with positive kernel weight.

    These are *fit* scores, never independent evidence of generalization. Select
    a threshold on calibration predictions and score it unchanged on a separate
    holdout whose labels did not inform model training or threshold selection.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth, dtype=bool)
    if pred.ndim != 2 or gt.shape != pred.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")
    valid = np.ones(pred.shape, dtype=bool) if valid_mask is None else np.asarray(
        valid_mask, dtype=bool
    )
    if valid.shape != pred.shape or not valid.any():
        raise ValueError("valid_mask must match raster shape and contain valid pixels")
    if not all(math.isfinite(x) for x in (alpha, beta, radius_pixels, eps)):
        raise ValueError("metric parameters must be finite")
    if alpha < 0 or beta < 0 or radius_pixels <= 0 or eps <= 0:
        raise ValueError("invalid metric parameters")
    values = pred[valid]
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("valid predictions must be finite and in [0, 1]")
    gt = gt & valid
    rows, cols = np.nonzero(gt)
    count = len(rows)
    # Include an explicit no-prediction state, even when the maximum equals 1.
    thresholds = np.concatenate(([np.nextafter(values.max(), np.inf)], np.unique(values)[::-1]))
    ascending = thresholds[::-1]
    fp_events = np.zeros(len(thresholds), dtype=np.float64)
    if count:
        kernel = np.maximum(1 - distance_transform_edt(~gt) / radius_pixels, 0)
    else:
        kernel = np.zeros_like(pred)
    indices = len(thresholds) - 1 - np.searchsorted(ascending, values)
    np.add.at(fp_events, indices, 1 - kernel[valid])
    tp_events = np.zeros_like(fp_events)
    if count:
        offsets = []
        radius = math.ceil(radius_pixels)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                weight = 1 - math.hypot(dy, dx) / radius_pixels
                if weight > 0:
                    offsets.append((dy, dx, weight))
        nearby = np.full((count, len(offsets)), -np.inf)
        weights = np.zeros_like(nearby)
        height, width = pred.shape
        for column, (dy, dx, weight) in enumerate(offsets):
            yy, xx = rows + dy, cols + dx
            inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            selected = np.flatnonzero(inside)
            selected = selected[valid[yy[selected], xx[selected]]]
            nearby[selected, column] = pred[yy[selected], xx[selected]]
            weights[selected, column] = weight
        order = np.argsort(-nearby, axis=1, kind="stable")
        activation = np.take_along_axis(nearby, order, axis=1)
        ordered_weights = np.take_along_axis(weights, order, axis=1)
        best = np.maximum.accumulate(ordered_weights, axis=1)
        increments = np.diff(best, axis=1, prepend=0.0)
        events = (increments > 0) & np.isfinite(activation)
        indices = len(thresholds) - 1 - np.searchsorted(ascending, activation[events])
        np.add.at(tp_events, indices, increments[events])
    tp = np.cumsum(tp_events)
    fp = np.cumsum(fp_events)
    fn = np.maximum(count - tp, 0)
    scores = tp / (tp + alpha * fp + beta * fn + eps)
    return ThresholdCurve(thresholds, scores, tp, fp, count)
