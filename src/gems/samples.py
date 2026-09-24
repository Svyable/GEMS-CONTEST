from __future__ import annotations

import numpy as np

from gems.tiling import tile_starts


def training_origins(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    patch_size: int,
    step: int,
    negative_ratio: float,
    seed: int,
    allowed_mask: np.ndarray | None = None,
) -> tuple[tuple[int, int], ...]:
    """Choose deterministic training windows, including background patches.

    Every fully covered window that contains a fault pixel and overlaps the
    valid study area is kept. Background windows are sampled, without
    replacement, up to ``negative_ratio`` times the positive count. When
    ``allowed_mask`` is set, a window is kept only if it lies entirely inside
    that mask, which is how a buffered spatial fold stays out of training.
    """
    truth = np.asarray(labels) > 0
    valid = np.asarray(valid_mask, dtype=bool)
    if truth.ndim != 2 or valid.shape != truth.shape:
        raise ValueError("labels and valid_mask must be the same 2D shape")
    allowed = None if allowed_mask is None else np.asarray(allowed_mask, dtype=bool)
    if allowed is not None and allowed.shape != truth.shape:
        raise ValueError("allowed_mask must match labels")
    if patch_size <= 1:
        raise ValueError("patch_size must be greater than 1")
    if step <= 0 or step > patch_size:
        raise ValueError("step must satisfy 0 < step <= patch_size")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")

    height, width = truth.shape
    overlap = patch_size - step
    row_starts = tile_starts(height, patch_size, overlap)
    col_starts = tile_starts(width, patch_size, overlap)

    positive: list[tuple[int, int]] = []
    negative: list[tuple[int, int]] = []
    for row in row_starts:
        row_end = row + patch_size
        for col in col_starts:
            col_end = col + patch_size
            if not bool(valid[row:row_end, col:col_end].any()):
                continue
            if allowed is not None and not bool(allowed[row:row_end, col:col_end].all()):
                continue
            origin = (int(row), int(col))
            if bool(truth[row:row_end, col:col_end].any()):
                positive.append(origin)
            else:
                negative.append(origin)

    rng = np.random.default_rng(seed)
    n_negative = min(len(negative), int(negative_ratio * len(positive)))
    if n_negative:
        chosen = rng.choice(len(negative), n_negative, replace=False)
        chosen.sort()
        sampled = [negative[int(index)] for index in chosen]
    else:
        sampled = []
    return tuple(positive + sampled)
