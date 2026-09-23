from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Window:
    row: int
    col: int
    height: int
    width: int


def tile_starts(length: int, patch_size: int, overlap: int) -> tuple[int, ...]:
    """Return deterministic tile starts that guarantee full axis coverage."""
    if length <= 0:
        raise ValueError("length must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")
    if length <= patch_size:
        return (0,)

    stride = patch_size - overlap
    starts = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def generate_windows(
    shape: tuple[int, int],
    *,
    patch_size: int,
    overlap: int,
) -> tuple[Window, ...]:
    """Generate row-major windows covering the full raster."""
    height, width = shape
    rows = tile_starts(height, patch_size, overlap)
    cols = tile_starts(width, patch_size, overlap)
    return tuple(
        Window(
            row=row,
            col=col,
            height=min(patch_size, height - row),
            width=min(patch_size, width - col),
        )
        for row in rows
        for col in cols
    )


def extract_patch(
    image: np.ndarray,
    window: Window,
    *,
    patch_size: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Extract one tile and pad smaller edge rasters to the requested patch size."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be 2D or HWC")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if window.row < 0 or window.col < 0:
        raise ValueError("window origin must be non-negative")
    if window.height <= 0 or window.width <= 0:
        raise ValueError("window dimensions must be positive")
    if window.row + window.height > array.shape[0]:
        raise ValueError("window extends beyond image height")
    if window.col + window.width > array.shape[1]:
        raise ValueError("window extends beyond image width")
    if window.height > patch_size or window.width > patch_size:
        raise ValueError("window cannot exceed patch_size")

    dtype = np.result_type(array.dtype, np.float32)
    if array.ndim == 2:
        patch = np.full((patch_size, patch_size), fill_value, dtype=dtype)
        patch[: window.height, : window.width] = array[
            window.row : window.row + window.height,
            window.col : window.col + window.width,
        ]
    else:
        patch = np.full(
            (patch_size, patch_size, array.shape[2]),
            fill_value,
            dtype=dtype,
        )
        patch[: window.height, : window.width] = array[
            window.row : window.row + window.height,
            window.col : window.col + window.width,
            :,
        ]
    return patch


def cosine_blend_weights(patch_size: int, floor: float = 0.05) -> np.ndarray:
    """Create nonzero center-weighted 2D blending weights."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0 < floor <= 1:
        raise ValueError("floor must lie in (0, 1]")
    if patch_size <= 2:
        return np.ones((patch_size, patch_size), dtype=np.float32)

    axis = np.hanning(patch_size).astype(np.float32)
    axis = np.maximum(axis, floor)
    weights = np.outer(axis, axis)
    weights /= weights.max()
    return weights.astype(np.float32)


def blend_predictions(
    shape: tuple[int, int],
    windows: Sequence[Window],
    predictions: Iterable[np.ndarray],
    *,
    patch_size: int,
    valid_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Blend overlapping prediction tiles into one full-resolution raster."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape must be positive")
    if weights is None:
        weights = cosine_blend_weights(patch_size)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (patch_size, patch_size):
        raise ValueError("weights must match patch_size")

    predictions = list(predictions)
    if len(predictions) != len(windows):
        raise ValueError("predictions and windows must have equal length")

    total = np.zeros(shape, dtype=np.float64)
    weight_sum = np.zeros(shape, dtype=np.float64)

    for window, prediction in zip(windows, predictions, strict=True):
        patch = np.asarray(prediction, dtype=np.float64)
        if patch.ndim != 2 or patch.shape[0] < window.height or patch.shape[1] < window.width:
            raise ValueError("each prediction must be a 2D patch covering its window")

        used = patch[: window.height, : window.width]
        if not np.all(np.isfinite(used)):
            raise ValueError("prediction patches must be finite inside their windows")
        tile_weights = weights[: window.height, : window.width]

        row_slice = slice(window.row, window.row + window.height)
        col_slice = slice(window.col, window.col + window.width)
        total[row_slice, col_slice] += used * tile_weights
        weight_sum[row_slice, col_slice] += tile_weights

    output = np.full(shape, np.nan, dtype=np.float32)
    covered = weight_sum > 0
    output[covered] = (total[covered] / weight_sum[covered]).astype(np.float32)

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != shape:
            raise ValueError("valid_mask must match shape")
        output[~valid] = np.nan
    return output
