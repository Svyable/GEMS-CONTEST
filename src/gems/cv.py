from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label as connected_components


@dataclass(frozen=True)
class SpatialFold:
    train_mask: np.ndarray
    validation_mask: np.ndarray
    excluded_from_training: np.ndarray


@dataclass(frozen=True)
class FaultDiscoveryFold:
    train_valid_mask: np.ndarray
    train_truth: np.ndarray
    validation_truth: np.ndarray
    evaluation_mask: np.ndarray
    component_folds: np.ndarray


def _disk(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def assign_spatial_blocks(
    shape: tuple[int, int],
    *,
    block_size: int,
    n_folds: int,
    seed: int = 0,
) -> np.ndarray:
    """Assign rectangular image blocks to balanced, reproducible folds."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    n_rows = math.ceil(height / block_size)
    n_cols = math.ceil(width / block_size)
    n_blocks = n_rows * n_cols
    if n_folds > n_blocks:
        raise ValueError("n_folds cannot exceed the number of spatial blocks")

    order = np.arange(n_blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    flat = np.empty(n_blocks, dtype=np.int16)
    flat[order] = np.arange(n_blocks, dtype=np.int64) % n_folds
    return flat.reshape(n_rows, n_cols)


def spatial_fold_masks(
    assignments: np.ndarray,
    shape: tuple[int, int],
    *,
    fold: int,
    block_size: int,
    buffer_pixels: int = 0,
    valid_mask: np.ndarray | None = None,
) -> SpatialFold:
    """Expand block assignments to pixel masks, buffering validation from training."""
    if assignments.ndim != 2:
        raise ValueError("assignments must be a 2D block grid")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if buffer_pixels < 0:
        raise ValueError("buffer_pixels must be non-negative")
    if fold < 0 or fold > int(assignments.max()):
        raise ValueError("fold is outside assignment range")

    height, width = shape
    expected = (math.ceil(height / block_size), math.ceil(width / block_size))
    if assignments.shape != expected:
        raise ValueError(f"assignment grid must have shape {expected}, got {assignments.shape}")

    rows = np.arange(height) // block_size
    cols = np.arange(width) // block_size
    validation = assignments[rows[:, None], cols[None, :]] == fold

    valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if valid.shape != shape:
        raise ValueError("valid_mask must match shape")
    validation &= valid

    excluded = validation
    if buffer_pixels:
        excluded = binary_dilation(validation, structure=_disk(buffer_pixels))
    excluded &= valid
    train = valid & ~excluded
    return SpatialFold(train, validation, excluded)


def assign_fault_components(
    labels: np.ndarray,
    *,
    n_folds: int,
    seed: int = 0,
) -> np.ndarray:
    """Assign complete 8-connected fault components to size-balanced folds."""
    truth = np.asarray(labels) > 0
    if truth.ndim != 2:
        raise ValueError("labels must be a 2D raster")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    component_ids, n_components = connected_components(truth, structure=np.ones((3, 3)))
    if n_components < n_folds:
        raise ValueError(
            f"need at least {n_folds} connected fault components, found {n_components}"
        )

    sizes = np.bincount(component_ids.ravel(), minlength=n_components + 1)[1:]
    rng = np.random.default_rng(seed)
    tie_break = rng.random(n_components)
    order = np.lexsort((tie_break, -sizes))

    loads = np.zeros(n_folds, dtype=np.int64)
    component_to_fold = np.full(n_components + 1, -1, dtype=np.int16)
    for zero_based_component in order:
        candidate_folds = np.flatnonzero(loads == loads.min())
        chosen = int(rng.choice(candidate_folds))
        component_id = int(zero_based_component) + 1
        component_to_fold[component_id] = chosen
        loads[chosen] += int(sizes[zero_based_component])

    pixel_folds = component_to_fold[component_ids]
    pixel_folds[~truth] = -1
    return pixel_folds


def fault_discovery_fold(
    labels: np.ndarray,
    component_folds: np.ndarray,
    *,
    fold: int,
    buffer_pixels: int = 0,
    valid_mask: np.ndarray | None = None,
    known_fault_exclusion_pixels: int = 0,
) -> FaultDiscoveryFold:
    """Build masks for a held-out-fault rediscovery experiment."""
    truth = np.asarray(labels) > 0
    folds = np.asarray(component_folds)
    if truth.ndim != 2 or folds.shape != truth.shape:
        raise ValueError("labels and component_folds must be same-shape 2D arrays")
    if buffer_pixels < 0 or known_fault_exclusion_pixels < 0:
        raise ValueError("buffer sizes must be non-negative")
    if not np.any(folds == fold):
        raise ValueError(f"fold {fold} contains no fault pixels")

    valid = np.ones_like(truth, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if valid.shape != truth.shape:
        raise ValueError("valid_mask must match labels")

    validation_truth = truth & (folds == fold) & valid
    train_truth = truth & (folds >= 0) & (folds != fold) & valid

    withheld_buffer = binary_dilation(validation_truth, structure=_disk(buffer_pixels))
    train_valid = valid & ~withheld_buffer

    known_exclusion = train_truth
    if known_fault_exclusion_pixels:
        known_exclusion = binary_dilation(
            train_truth, structure=_disk(known_fault_exclusion_pixels)
        )
    evaluation_mask = valid & ~known_exclusion

    return FaultDiscoveryFold(
        train_valid_mask=train_valid,
        train_truth=train_truth,
        validation_truth=validation_truth,
        evaluation_mask=evaluation_mask,
        component_folds=folds,
    )
