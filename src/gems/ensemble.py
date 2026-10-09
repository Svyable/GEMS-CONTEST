"""Ensemble and calibration framework for GEMS competition (roadmap #6).

Combines fold-wise OOF probability rasters from multiple candidate models using
various fusion strategies (mean, metric-fit weighted mean, rank-average). Applies
leave-one-fold-out (LOFO) weight fitting to prevent leakage: weights for fold k
are fit on all other folds, so fold k's score does not use weights fit on fold k's
labels. Output GeoTIFFs pass validate_submission.py and are scored through
verify_candidate.py and logged to docs/candidate-trials.jsonl.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

from gems.calibration import exact_threshold_curve
from gems.metric import distance_weighted_tversky


@dataclass(frozen=True)
class EnsembleMember:
    """One candidate model contributing to an ensemble."""

    name: str
    fold_predictions: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class EnsembleWeights:
    """Leave-one-fold-out (LOFO) weights for ensemble members.

    weights[fold_id] gives the weight vector for that fold, fitted on all
    other folds. Length of each weight vector equals the number of members.
    """

    weights: Mapping[int, np.ndarray]
    method: str


EnsembleMethod = Literal["mean", "weighted_mean", "rank_average"]


def _check_aligned(members: Sequence[EnsembleMember], fold_ids: Sequence[int]) -> None:
    """Verify all members have predictions for all folds with matching shapes."""
    if not members:
        raise ValueError("at least one ensemble member is required")
    if not fold_ids:
        raise ValueError("at least one fold is required")

    reference_shape = None
    for member in members:
        if set(member.fold_predictions.keys()) != set(fold_ids):
            raise ValueError(
                f"member {member.name} has folds {sorted(member.fold_predictions.keys())}, "
                f"expected {sorted(fold_ids)}"
            )
        for fold in fold_ids:
            pred = np.asarray(member.fold_predictions[fold], dtype=np.float64)
            if pred.ndim != 2:
                raise ValueError(f"member {member.name} fold {fold} must be 2D")
            if reference_shape is None:
                reference_shape = pred.shape
            elif pred.shape != reference_shape:
                raise ValueError(
                    f"member {member.name} fold {fold} shape {pred.shape} "
                    f"does not match reference {reference_shape}"
                )


def ensemble_mean(members: Sequence[EnsembleMember], fold_ids: Sequence[int]) -> dict[int, np.ndarray]:
    """Simple arithmetic mean of all member predictions per fold."""
    _check_aligned(members, fold_ids)
    combined = {}
    for fold in fold_ids:
        predictions = [np.asarray(m.fold_predictions[fold], dtype=np.float64) for m in members]
        combined[fold] = np.mean(predictions, axis=0).astype(np.float32)
    return combined


def ensemble_rank_average(
    members: Sequence[EnsembleMember], fold_ids: Sequence[int]
) -> dict[int, np.ndarray]:
    """Rank-average ensemble: average rank of each pixel across members.

    Converts each member's predictions to ranks (0-based, normalized to [0,1]),
    then averages the ranks. This is robust to scale differences between models.
    """
    _check_aligned(members, fold_ids)
    combined = {}
    for fold in fold_ids:
        predictions = [np.asarray(m.fold_predictions[fold], dtype=np.float64) for m in members]
        # Convert each prediction to normalized ranks
        ranks = []
        for pred in predictions:
            finite_mask = np.isfinite(pred)
            rank_pred = np.zeros_like(pred, dtype=np.float64)
            if finite_mask.any():
                finite_values = pred[finite_mask]
                # argsort gives indices that would sort the array
                # argsort(argsort(x)) gives ranks
                sorted_indices = np.argsort(finite_values)
                rank_values = np.empty_like(finite_values)
                rank_values[sorted_indices] = np.arange(len(finite_values), dtype=np.float64)
                # Normalize to [0, 1]
                if len(finite_values) > 1:
                    rank_values = rank_values / (len(finite_values) - 1)
                else:
                    rank_values = np.array([0.5])
                rank_pred[finite_mask] = rank_values
            rank_pred[~finite_mask] = np.nan
            ranks.append(rank_pred)
        combined[fold] = np.nanmean(ranks, axis=0).astype(np.float32)
    return combined


def fit_lofo_weights(
    members: Sequence[EnsembleMember],
    fold_ids: Sequence[int],
    truth: np.ndarray,
    fold_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
) -> EnsembleWeights:
    """Fit leave-one-fold-out (LOFO) ensemble weights using grid search.

    For each fold k, fits weights on all other folds (j != k) to maximize the
    mean score across those folds. Weights are non-negative and sum to 1.
    This ensures fold k's score does not use weights fit on fold k's labels.

    Grid search over: equal weights, best-single-model, and all 0.1-step
    weight combinations that sum to 1 (for 2-3 members).
    """
    _check_aligned(members, fold_ids)
    gt = np.asarray(truth, dtype=bool)
    folds = np.asarray(fold_map, dtype=np.int16)
    if gt.ndim != 2 or folds.shape != gt.shape:
        raise ValueError("truth and fold_map must be same-shape 2D arrays")

    valid = np.ones(gt.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match raster shape")

    n_members = len(members)
    if n_members == 1:
        # Single member: trivial weights
        return EnsembleWeights(
            weights={fold: np.array([1.0]) for fold in fold_ids},
            method="weighted_mean",
        )

    # Generate weight candidates
    weight_candidates: list[np.ndarray] = []
    # Equal weights
    weight_candidates.append(np.ones(n_members, dtype=np.float64) / n_members)
    # Best single model (one-hot for each member)
    for i in range(n_members):
        w = np.zeros(n_members, dtype=np.float64)
        w[i] = 1.0
        weight_candidates.append(w)

    # For 2-3 members, enumerate all 0.1-step combinations
    if n_members == 2:
        for a in range(11):
            w = np.array([a / 10.0, 1.0 - a / 10.0], dtype=np.float64)
            weight_candidates.append(w)
    elif n_members == 3:
        for a in range(11):
            for b in range(11 - a):
                c = 10 - a - b
                w = np.array([a / 10.0, b / 10.0, c / 10.0], dtype=np.float64)
                weight_candidates.append(w)

    # Deduplicate
    unique_weights = []
    seen = set()
    for w in weight_candidates:
        key = tuple(np.round(w, decimals=3))
        if key not in seen:
            seen.add(key)
            unique_weights.append(w)

    # Fit weights for each fold using leave-one-fold-out
    lofo_weights = {}
    for target_fold in fold_ids:
        # Fit on all other folds
        other_folds = [f for f in fold_ids if f != target_fold]
        if not other_folds:
            # Only one fold total: use equal weights
            lofo_weights[target_fold] = np.ones(n_members, dtype=np.float64) / n_members
            continue

        best_weights = None
        best_mean_score = -np.inf

        for weights in unique_weights:
            # Compute weighted average on each other fold and score it
            scores = []
            for fold in other_folds:
                mask = valid & (folds == fold)
                if not mask.any():
                    continue
                # Weighted combination
                predictions = [
                    np.asarray(m.fold_predictions[fold], dtype=np.float64) for m in members
                ]
                weighted = np.zeros_like(predictions[0], dtype=np.float64)
                for i, pred in enumerate(predictions):
                    weighted += weights[i] * pred
                score = distance_weighted_tversky(
                    weighted,
                    gt,
                    alpha=alpha,
                    beta=beta,
                    radius_pixels=radius_pixels,
                    valid_mask=mask,
                )
                scores.append(score)

            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    best_weights = weights.copy()

        if best_weights is None:
            # Fallback to equal weights
            best_weights = np.ones(n_members, dtype=np.float64) / n_members

        lofo_weights[target_fold] = best_weights

    return EnsembleWeights(weights=lofo_weights, method="weighted_mean")


def apply_ensemble(
    members: Sequence[EnsembleMember],
    fold_ids: Sequence[int],
    method: EnsembleMethod = "mean",
    weights: EnsembleWeights | None = None,
) -> dict[int, np.ndarray]:
    """Combine member predictions using the specified method.

    For method='weighted_mean', weights must be provided (typically from
    fit_lofo_weights). Each fold uses its corresponding LOFO weight vector.
    """
    _check_aligned(members, fold_ids)

    if method == "mean":
        return ensemble_mean(members, fold_ids)
    elif method == "rank_average":
        return ensemble_rank_average(members, fold_ids)
    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("weights required for method='weighted_mean'")
        if set(weights.weights.keys()) != set(fold_ids):
            raise ValueError(
                f"weights folds {sorted(weights.weights.keys())} "
                f"do not match requested {sorted(fold_ids)}"
            )
        combined = {}
        for fold in fold_ids:
            w = weights.weights[fold]
            if len(w) != len(members):
                raise ValueError(
                    f"fold {fold} weights length {len(w)} does not match "
                    f"member count {len(members)}"
                )
            predictions = [np.asarray(m.fold_predictions[fold], dtype=np.float64) for m in members]
            weighted = np.zeros_like(predictions[0], dtype=np.float64)
            for i, pred in enumerate(predictions):
                weighted += w[i] * pred
            combined[fold] = weighted.astype(np.float32)
        return combined
    else:
        raise ValueError(f"unknown ensemble method: {method}")


def load_fold_predictions(
    pattern: str | Path,
    fold_ids: Sequence[int],
    *,
    reference_path: str | Path | None = None,
) -> dict[int, np.ndarray]:
    """Load fold-specific prediction rasters using a path pattern.

    pattern must contain '{fold}', e.g. 'runs/model-a/fold-{fold}.tif'.
    If reference_path is provided, checks that all predictions match its
    georeferencing.
    """
    pattern = str(pattern)
    if "{fold}" not in pattern:
        raise ValueError("pattern must contain '{fold}' placeholder")

    reference = None
    if reference_path is not None:
        reference = rasterio.open(reference_path)

    predictions = {}
    try:
        for fold in fold_ids:
            path = pattern.format(fold=fold)
            with rasterio.open(path) as src:
                if reference is not None:
                    if (src.height, src.width) != (reference.height, reference.width):
                        raise ValueError(f"fold {fold}: shape mismatch with reference")
                    if src.crs != reference.crs or src.transform != reference.transform:
                        raise ValueError(f"fold {fold}: georeferencing mismatch with reference")
                predictions[fold] = src.read(1).astype(np.float32)
    finally:
        if reference is not None:
            reference.close()

    return predictions


def calibrate_ensemble_threshold(
    predictions: Mapping[int, np.ndarray],
    truth: np.ndarray,
    fold_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
) -> dict[int, float]:
    """Find optimal binary threshold for each fold using exact threshold search.

    Returns dict mapping fold_id -> optimal threshold for that fold.
    Uses the canonical exact_threshold_curve from calibration.py.
    """
    gt = np.asarray(truth, dtype=bool)
    folds = np.asarray(fold_map, dtype=np.int16)
    if gt.ndim != 2 or folds.shape != gt.shape:
        raise ValueError("truth and fold_map must be same-shape 2D arrays")

    valid = np.ones(gt.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != gt.shape:
        raise ValueError("valid_mask must match raster shape")

    thresholds = {}
    for fold, pred in predictions.items():
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != gt.shape:
            raise ValueError(f"fold {fold} prediction has wrong shape")
        mask = valid & (folds == fold)
        if not mask.any():
            thresholds[fold] = 0.5  # Default if no valid pixels
            continue
        curve = exact_threshold_curve(
            pred,
            gt,
            valid_mask=mask,
            alpha=alpha,
            beta=beta,
            radius_pixels=radius_pixels,
        )
        thresholds[fold] = curve.best_threshold

    return thresholds
