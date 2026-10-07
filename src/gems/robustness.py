"""Robustness perturbations for spatially registered raster feature stacks."""

import numpy as np


NEIGHBOR_SHIFTS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def shift_feature_channels(
    features: np.ndarray,
    channels,
    *,
    row_offset: int,
    col_offset: int,
    valid_mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Shift selected HWC channels without wraparound.

    Positive row offsets move source values downward; positive column offsets move
    them rightward. Destination pixels whose source is outside the raster or outside
    the valid study region receive fill_value. Unselected channels are unchanged.
    """
    x = np.asarray(features, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if x.ndim != 3 or valid.shape != x.shape[:2]:
        raise ValueError("features must be HWC and valid_mask must match HW")
    indices = tuple(int(index) for index in channels)
    if not indices:
        raise ValueError("at least one channel is required")
    if len(set(indices)) != len(indices):
        raise ValueError("channels must be unique")
    if min(indices) < 0 or max(indices) >= x.shape[-1]:
        raise ValueError("channel index outside feature stack")
    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite")

    output = x.copy()
    height, width = valid.shape

    src_r0 = max(0, -row_offset)
    src_r1 = min(height, height - row_offset)
    src_c0 = max(0, -col_offset)
    src_c1 = min(width, width - col_offset)
    dst_r0 = max(0, row_offset)
    dst_r1 = min(height, height + row_offset)
    dst_c0 = max(0, col_offset)
    dst_c1 = min(width, width + col_offset)

    output[..., indices] = np.float32(fill_value)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return output

    source_valid = valid[src_r0:src_r1, src_c0:src_c1]
    destination_valid = valid[dst_r0:dst_r1, dst_c0:dst_c1]
    transfer = source_valid & destination_valid
    for index in indices:
        source = x[src_r0:src_r1, src_c0:src_c1, index]
        destination = output[dst_r0:dst_r1, dst_c0:dst_c1, index]
        destination[transfer] = source[transfer]
    return output


def windows_intersecting_mask(
    windows,
    mask: np.ndarray,
) -> tuple:
    """Keep only windows touching at least one selected evaluation pixel."""
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2:
        raise ValueError("mask must be 2D")
    result = []
    for window in windows:
        if (
            window.row < 0
            or window.col < 0
            or window.row + window.height > selected.shape[0]
            or window.col + window.width > selected.shape[1]
        ):
            raise ValueError("window lies outside mask")
        tile = selected[
            window.row : window.row + window.height,
            window.col : window.col + window.width,
        ]
        if tile.any():
            result.append(window)
    return tuple(result)
