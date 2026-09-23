from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean, pstdev

import numpy as np

from gems.cv import fault_discovery_fold
from gems.metric import distance_weighted_tversky


@dataclass(frozen=True)
class FoldMetric:
    fold: int
    score: float
    valid_pixels: int
    truth_pixels: int


def _valid_or_all(shape: tuple[int, int], valid_mask: np.ndarray | None) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError("valid_mask must match raster shape")
    return valid


def evaluate_spatial_predictions(
    predictions: Mapping[int, np.ndarray],
    truth: np.ndarray,
    fold_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
) -> dict:
    """Score one fold-specific prediction raster per spatial fold.

    Predictions are stitched only on their validation regions to form one OOF
    raster, which is then scored globally in addition to per-fold scores.
    """
    gt = np.asarray(truth).astype(bool)
    folds = np.asarray(fold_map)
    if gt.ndim != 2 or folds.shape != gt.shape:
        raise ValueError("truth and fold_map must be same-shape 2D arrays")

    valid = _valid_or_all(gt.shape, valid_mask)
    fold_ids = sorted(int(v) for v in np.unique(folds[valid]) if v >= 0)
    if not fold_ids:
        raise ValueError("fold_map contains no non-negative folds in the valid region")
    if set(predictions) != set(fold_ids):
        raise ValueError(
            f"prediction folds {sorted(predictions)} do not match fold map {fold_ids}"
        )

    oof = np.full(gt.shape, np.nan, dtype=np.float32)
    per_fold: list[FoldMetric] = []
    for fold in fold_ids:
        prediction = np.asarray(predictions[fold], dtype=np.float64)
        if prediction.shape != gt.shape:
            raise ValueError(f"prediction for fold {fold} has wrong shape")
        mask = valid & (folds == fold)
        score = distance_weighted_tversky(
            prediction,
            gt,
            alpha=alpha,
            beta=beta,
            radius_pixels=radius_pixels,
            valid_mask=mask,
        )
        oof[mask] = prediction[mask]
        per_fold.append(
            FoldMetric(
                fold=fold,
                score=score,
                valid_pixels=int(mask.sum()),
                truth_pixels=int((gt & mask).sum()),
            )
        )

    oof_valid = valid & (folds >= 0)
    aggregate = distance_weighted_tversky(
        oof,
        gt,
        alpha=alpha,
        beta=beta,
        radius_pixels=radius_pixels,
        valid_mask=oof_valid,
    )
    return {
        "scheme": "spatial",
        "aggregate_score": aggregate,
        "macro_mean": mean(item.score for item in per_fold),
        "macro_std": pstdev(item.score for item in per_fold),
        "folds": [item.__dict__ for item in per_fold],
    }


def evaluate_fault_discovery_predictions(
    predictions: Mapping[int, np.ndarray],
    truth: np.ndarray,
    component_folds: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    known_fault_exclusion_pixels: int = 0,
    alpha: float = 0.2,
    beta: float = 0.8,
    radius_pixels: float = 3.0,
) -> dict:
    """Score fold-specific models on complete held-out fault components.

    Evaluation masks overlap on non-fault background, so there is intentionally
    no synthetic global score. Report macro fold statistics instead.
    """
    gt = np.asarray(truth).astype(bool)
    folds = np.asarray(component_folds)
    if gt.ndim != 2 or folds.shape != gt.shape:
        raise ValueError("truth and component_folds must be same-shape 2D arrays")

    valid = _valid_or_all(gt.shape, valid_mask)
    fold_ids = sorted(int(v) for v in np.unique(folds[gt & valid]) if v >= 0)
    if not fold_ids:
        raise ValueError("component_folds contains no held-out fault folds")
    if set(predictions) != set(fold_ids):
        raise ValueError(
            f"prediction folds {sorted(predictions)} do not match fold map {fold_ids}"
        )

    per_fold: list[FoldMetric] = []
    for fold in fold_ids:
        prediction = np.asarray(predictions[fold], dtype=np.float64)
        if prediction.shape != gt.shape:
            raise ValueError(f"prediction for fold {fold} has wrong shape")
        spec = fault_discovery_fold(
            gt,
            folds,
            fold=fold,
            valid_mask=valid,
            known_fault_exclusion_pixels=known_fault_exclusion_pixels,
        )
        score = distance_weighted_tversky(
            prediction,
            spec.validation_truth,
            alpha=alpha,
            beta=beta,
            radius_pixels=radius_pixels,
            valid_mask=spec.evaluation_mask,
        )
        per_fold.append(
            FoldMetric(
                fold=fold,
                score=score,
                valid_pixels=int(spec.evaluation_mask.sum()),
                truth_pixels=int(spec.validation_truth.sum()),
            )
        )

    return {
        "scheme": "fault",
        "macro_mean": mean(item.score for item in per_fold),
        "macro_std": pstdev(item.score for item in per_fold),
        "folds": [item.__dict__ for item in per_fold],
    }
