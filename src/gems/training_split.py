"""Candidate supervision contracts for spatial, fault and endpoint validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label, maximum, minimum

from gems.cv import fault_discovery_fold, trace_completion_fold


@dataclass(frozen=True)
class TrainingSplit:
    scheme: str
    train_mask: np.ndarray
    train_truth: np.ndarray
    evaluation_mask: np.ndarray
    evaluation_truth: np.ndarray

    def summary(self) -> dict:
        return {
            "scheme": self.scheme,
            "train_pixels": int(self.train_mask.sum()),
            "train_fault_pixels": int((self.train_truth & self.train_mask).sum()),
            "evaluation_pixels": int(self.evaluation_mask.sum()),
            "evaluation_fault_pixels": int((self.evaluation_truth & self.evaluation_mask).sum()),
        }


def candidate_training_split(
    truth: np.ndarray,
    folds: np.ndarray,
    *,
    scheme: str,
    fold: int,
    valid_mask: np.ndarray,
    buffer_pixels: int,
    known_fault_exclusion_pixels: int = 0,
) -> TrainingSplit:
    gt = np.asarray(truth, dtype=bool)
    values = np.asarray(folds)
    valid = np.asarray(valid_mask, dtype=bool)
    if gt.ndim != 2 or values.shape != gt.shape or valid.shape != gt.shape:
        raise ValueError("truth, fold map and valid mask must be same-shape 2D arrays")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values[valid] < -1):
        raise ValueError("fold map must contain integer fold IDs or -1")
    if fold < 0 or buffer_pixels < 0 or known_fault_exclusion_pixels < 0:
        raise ValueError("fold and buffer sizes must be non-negative")
    gt = gt & valid
    if scheme == "spatial":
        if np.any(valid & (values < 0)):
            raise ValueError("spatial fold map must assign every valid pixel")
        if known_fault_exclusion_pixels:
            raise ValueError("known-fault exclusion is supported only for fault/trace CV")
        held = valid & (values == fold)
        if not held.any():
            raise ValueError("requested fold has no valid pixels")
        y, x = np.ogrid[-buffer_pixels:buffer_pixels + 1, -buffer_pixels:buffer_pixels + 1]
        excluded = binary_dilation(held, structure=x * x + y * y <= buffer_pixels**2)
        train = valid & ~excluded
        result = TrainingSplit(scheme, train, gt & train, held, gt & held)
    elif scheme in ("fault", "trace"):
        if np.any(valid & ~gt & (values >= 0)):
            raise ValueError("fault/trace fold IDs may only label positive truth pixels")
        withheld = gt & (values == fold)
        if not withheld.any():
            raise ValueError("requested fold has no valid fault pixels")
        components, count = label(gt, structure=np.ones((3, 3)))
        ids = np.arange(1, count + 1)
        if scheme == "fault":
            if np.any(gt & (values < 0)):
                raise ValueError("complete-fault CV must assign every valid fault pixel")
            if np.any(maximum(values, components, ids) != minimum(values, components, ids)):
                raise ValueError("complete fault components must not be split across folds")
            factory = fault_discovery_fold
        else:
            # A trace endpoint must leave some of that same component as known truth.
            if np.any(minimum(withheld.astype(np.uint8), components, ids) == 1):
                raise ValueError("trace CV must retain a body for every withheld component")
            factory = trace_completion_fold
        spec = factory(gt, values, fold=fold, buffer_pixels=buffer_pixels,
                       valid_mask=valid,
                       known_fault_exclusion_pixels=known_fault_exclusion_pixels)
        result = TrainingSplit(scheme, spec.train_valid_mask,
                               spec.train_truth & spec.train_valid_mask,
                               spec.evaluation_mask, spec.validation_truth)
    else:
        raise ValueError("scheme must be spatial, fault or trace")
    if not result.train_mask.any() or not result.train_truth.any():
        raise ValueError("split leaves no supervised training faults")
    if not (result.evaluation_truth & result.evaluation_mask).any():
        raise ValueError("split leaves no evaluable held-out faults")
    return result


def masked_training_arrays(
    features: np.ndarray, split: TrainingSplit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copies used for training; withheld features/labels never enter the model.

    Keep the original normalized features for full-region inference. Zero-filled
    holes are a deliberate discovery-training ablation, recorded in run metadata.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 3 or x.shape[:2] != split.train_mask.shape:
        raise ValueError("features must be HWC and match the split")
    inputs = np.where(split.train_mask[..., None], x, 0.0)
    targets = np.where(split.train_mask, split.train_truth, 0).astype(np.float32)
    return inputs, targets, split.train_mask.copy()
