"""Threshold and prediction optimization for the Tversky metric.

The competition uses a distance-weighted Tversky index with alpha=0.2, beta=0.8,
and 300m tolerance. This heavily weights recall over precision, so the optimal
probability threshold is typically much lower than 0.5.

This module provides tools to find optimal thresholds and potentially refine
predictions to maximize the competition metric.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gems.metric import distance_weighted_tversky


@dataclass(frozen=True)
class ThresholdResult:
    """Result from threshold optimization."""

    optimal_threshold: float
    optimal_score: float
    scores_by_threshold: dict[float, float]
    method: str


def optimize_threshold_grid(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
    thresholds: np.ndarray | None = None,
    progress_callback: Callable[[float, float], None] | None = None,
) -> ThresholdResult:
    """Find optimal threshold via grid search.

    This exhaustively evaluates the Tversky metric at each threshold and
    returns the threshold that maximizes the score.

    The default search grid emphasizes the lower probability range where
    the optimal threshold likely lies given beta=0.8 > alpha=0.2.

    Args:
        prediction: Continuous probability array in [0, 1]
        truth: Binary ground truth
        valid_mask: Optional mask for valid evaluation pixels
        alpha: Tversky alpha (false positive penalty weight)
        beta: Tversky beta (false negative penalty weight)
        radius_pixels: Distance tolerance in pixels
        thresholds: Custom threshold grid (default: emphasis on [0, 0.5])
        progress_callback: Optional callback(threshold, score) for monitoring

    Returns:
        ThresholdResult with optimal threshold and scores
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth).astype(bool)

    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")

    if thresholds is None:
        # Dense sampling in [0, 0.5] where optimal likely lies,
        # sparser in [0.5, 1.0]
        thresholds = np.concatenate(
            [
                np.linspace(0.0, 0.5, 51),  # every 0.01 up to 0.5
                np.linspace(0.52, 1.0, 25),  # every 0.02 above 0.5
            ]
        )

    scores: dict[float, float] = {}
    best_threshold = 0.0
    best_score = -np.inf

    for threshold in thresholds:
        # Threshold predictions remain as soft values,
        # just evaluating at this threshold level
        binary_pred = (pred >= threshold).astype(np.float32)

        score = distance_weighted_tversky(
            binary_pred,
            gt,
            alpha=alpha,
            beta=beta,
            radius_pixels=radius_pixels,
            valid_mask=valid_mask,
        )

        scores[float(threshold)] = score

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

        if progress_callback is not None:
            progress_callback(float(threshold), score)

    return ThresholdResult(
        optimal_threshold=best_threshold,
        optimal_score=best_score,
        scores_by_threshold=scores,
        method="grid_search",
    )


def optimize_threshold_golden(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
    bracket: tuple[float, float] = (0.0, 1.0),
    tolerance: float = 1e-3,
    max_iterations: int = 50,
    progress_callback: Callable[[float, float], None] | None = None,
) -> ThresholdResult:
    """Find optimal threshold via golden section search.

    This is more efficient than grid search for smooth objective functions,
    but may miss local optima. Use grid search first to understand the
    landscape, then golden search for refinement if needed.

    Args:
        prediction: Continuous probability array in [0, 1]
        truth: Binary ground truth
        valid_mask: Optional mask for valid evaluation pixels
        alpha: Tversky alpha (false positive penalty weight)
        beta: Tversky beta (false negative penalty weight)
        radius_pixels: Distance tolerance in pixels
        bracket: Search interval (default [0, 1])
        tolerance: Convergence tolerance
        max_iterations: Maximum search iterations
        progress_callback: Optional callback(threshold, score) for monitoring

    Returns:
        ThresholdResult with optimal threshold and scores
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth).astype(bool)

    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and truth must be same-shape 2D arrays")

    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    resphi = 2 - phi

    def evaluate(threshold: float) -> float:
        binary_pred = (pred >= threshold).astype(np.float32)
        score = distance_weighted_tversky(
            binary_pred,
            gt,
            alpha=alpha,
            beta=beta,
            radius_pixels=radius_pixels,
            valid_mask=valid_mask,
        )
        if progress_callback is not None:
            progress_callback(threshold, score)
        return score

    a, b = bracket
    scores: dict[float, float] = {}

    # Initial points
    c = b - resphi * (b - a)
    d = a + resphi * (b - a)
    score_c = evaluate(c)
    score_d = evaluate(d)
    scores[c] = score_c
    scores[d] = score_d

    for _ in range(max_iterations):
        if b - a < tolerance:
            break

        # Golden section maximization (note: we want maximum, not minimum)
        if score_c > score_d:
            b = d
            d = c
            score_d = score_c
            c = b - resphi * (b - a)
            score_c = evaluate(c)
            scores[c] = score_c
        else:
            a = c
            c = d
            score_c = score_d
            d = a + resphi * (b - a)
            score_d = evaluate(d)
            scores[d] = score_d

    optimal_threshold = (a + b) / 2
    optimal_score = evaluate(optimal_threshold)
    scores[optimal_threshold] = optimal_score

    return ThresholdResult(
        optimal_threshold=optimal_threshold,
        optimal_score=optimal_score,
        scores_by_threshold=scores,
        method="golden_section",
    )


def calibrate_predictions(
    prediction: np.ndarray,
    *,
    optimal_threshold: float,
    scale: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """Recalibrate predictions given an optimal threshold.

    Simple linear scaling that maps optimal_threshold to 0.5. This can help
    when downstream tools expect 0.5 as the decision boundary.

    For more sophisticated calibration (isotonic regression, Platt scaling),
    use scikit-learn with held-out calibration data.

    Args:
        prediction: Original probability predictions
        optimal_threshold: Threshold that maximizes Tversky metric
        scale: Additional multiplicative scaling factor
        clip: Whether to clip output to [0, 1]

    Returns:
        Recalibrated predictions
    """
    pred = np.asarray(prediction, dtype=np.float64)

    if optimal_threshold <= 0 or optimal_threshold >= 1:
        raise ValueError("optimal_threshold must be in (0, 1)")

    # Linear scaling: map optimal_threshold -> 0.5
    scaling_factor = 0.5 / optimal_threshold
    calibrated = pred * scaling_factor * scale

    if clip:
        calibrated = np.clip(calibrated, 0.0, 1.0)

    return calibrated.astype(pred.dtype)
