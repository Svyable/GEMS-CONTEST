"""Deprecated threshold optimization (superseded by calibration.py).

DEPRECATION NOTICE: This module is deprecated. Use gems.calibration instead.

The exact_threshold_curve() function in gems.calibration evaluates every distinct
threshold state efficiently via event-based cumulative sums, making it both faster
and more accurate than the grid-search approach here.

Migration guide:
- optimize_threshold_grid() -> use exact_threshold_curve() from gems.calibration
- optimize_threshold_golden() -> use exact_threshold_curve() (finds exact global optimum)
- calibrate_predictions() -> unchanged, kept for compatibility

This module now wraps gems.calibration for backward compatibility.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gems.calibration import exact_threshold_curve


@dataclass(frozen=True)
class ThresholdResult:
    """Result from threshold optimization (deprecated)."""

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
    """Find optimal threshold (DEPRECATED: use gems.calibration.exact_threshold_curve).

    This now wraps exact_threshold_curve() for backward compatibility.
    The exact method evaluates every distinct threshold state efficiently
    and is both faster and more accurate than grid search.

    Args:
        prediction: Continuous probability array in [0, 1]
        truth: Binary ground truth
        valid_mask: Optional mask for valid evaluation pixels
        alpha: Tversky alpha (false positive penalty weight)
        beta: Tversky beta (false negative penalty weight)
        radius_pixels: Distance tolerance in pixels
        thresholds: Ignored (exact method evaluates all thresholds)
        progress_callback: Ignored (exact method is fast)

    Returns:
        ThresholdResult with optimal threshold and scores
    """
    warnings.warn(
        "optimize_threshold_grid() is deprecated. Use gems.calibration.exact_threshold_curve() "
        "for exact, efficient threshold optimization.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use exact method from calibration.py
    curve = exact_threshold_curve(
        prediction,
        truth,
        valid_mask=valid_mask,
        alpha=alpha,
        beta=beta,
        radius_pixels=radius_pixels,
    )

    scores_dict = {
        float(t): float(s) for t, s in zip(curve.thresholds, curve.scores, strict=False)
    }

    return ThresholdResult(
        optimal_threshold=curve.best_threshold,
        optimal_score=float(curve.scores.max()),
        scores_by_threshold=scores_dict,
        method="exact",
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
    """Find optimal threshold (DEPRECATED: use gems.calibration.exact_threshold_curve).

    This now wraps exact_threshold_curve() for backward compatibility.
    The exact method finds the global optimum directly without search,
    making golden section search unnecessary.

    Args:
        prediction: Continuous probability array in [0, 1]
        truth: Binary ground truth
        valid_mask: Optional mask for valid evaluation pixels
        alpha: Tversky alpha (false positive penalty weight)
        beta: Tversky beta (false negative penalty weight)
        radius_pixels: Distance tolerance in pixels
        bracket: Ignored (exact method evaluates all thresholds)
        tolerance: Ignored
        max_iterations: Ignored
        progress_callback: Ignored

    Returns:
        ThresholdResult with optimal threshold and scores
    """
    warnings.warn(
        "optimize_threshold_golden() is deprecated. Use gems.calibration.exact_threshold_curve() "
        "which finds the exact global optimum without search.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use exact method from calibration.py
    curve = exact_threshold_curve(
        prediction,
        truth,
        valid_mask=valid_mask,
        alpha=alpha,
        beta=beta,
        radius_pixels=radius_pixels,
    )

    scores_dict = {
        float(t): float(s) for t, s in zip(curve.thresholds, curve.scores, strict=False)
    }

    return ThresholdResult(
        optimal_threshold=curve.best_threshold,
        optimal_score=float(curve.scores.max()),
        scores_by_threshold=scores_dict,
        method="exact",
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
