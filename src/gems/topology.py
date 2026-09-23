from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import find_objects
from scipy.ndimage import label as connected_components


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = np.quantile(values.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    names = ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    return {name: float(value) for name, value in zip(names, qs, strict=True)}


def fault_component_report(
    labels: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    connectivity: int = 8,
    pixel_size_x_m: float = 100.0,
    pixel_size_y_m: float = 100.0,
    top_n: int = 20,
) -> dict:
    """Describe connected fault topology without deciding whether it is suitable for CV."""
    truth = np.asarray(labels) > 0
    if truth.ndim != 2:
        raise ValueError("labels must be a 2D raster")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("pixel sizes must be positive")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")

    if valid_mask is None:
        valid = np.ones_like(truth, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != truth.shape:
            raise ValueError("valid_mask must match labels")
    truth &= valid

    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.uint8,
    )
    component_ids, n_components = connected_components(truth, structure=structure)
    sizes = np.bincount(component_ids.ravel(), minlength=n_components + 1)[1:]
    slices = find_objects(component_ids)

    total_fault = int(truth.sum())
    valid_pixels = int(valid.sum())
    order = np.argsort(-sizes) if sizes.size else np.array([], dtype=int)
    components = []
    for zero_based_id in order[:top_n]:
        component_id = int(zero_based_id) + 1
        bbox = slices[zero_based_id]
        if bbox is None:
            continue
        row_slice, col_slice = bbox
        rows = int(row_slice.stop - row_slice.start)
        cols = int(col_slice.stop - col_slice.start)
        pixels = int(sizes[zero_based_id])
        components.append(
            {
                "component_id": component_id,
                "pixels": pixels,
                "share_of_fault_pixels": pixels / total_fault if total_fault else 0.0,
                "bbox": {
                    "row_start": int(row_slice.start),
                    "row_stop": int(row_slice.stop),
                    "col_start": int(col_slice.start),
                    "col_stop": int(col_slice.stop),
                    "height_pixels": rows,
                    "width_pixels": cols,
                    "height_m": rows * float(pixel_size_y_m),
                    "width_m": cols * float(pixel_size_x_m),
                },
            }
        )

    largest = int(sizes.max()) if sizes.size else 0
    return {
        "connectivity": connectivity,
        "valid_pixels": valid_pixels,
        "fault_pixels": total_fault,
        "fault_prevalence": total_fault / valid_pixels if valid_pixels else 0.0,
        "component_count": int(n_components),
        "component_size_pixels": _quantiles(sizes),
        "largest_component_pixels": largest,
        "largest_component_share": largest / total_fault if total_fault else 0.0,
        "top_components": components,
    }


def spatial_block_report(
    labels: np.ndarray,
    *,
    block_size: int,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """Describe how sparse fault supervision is across candidate spatial blocks."""
    truth = np.asarray(labels) > 0
    if truth.ndim != 2:
        raise ValueError("labels must be a 2D raster")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if valid_mask is None:
        valid = np.ones_like(truth, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != truth.shape:
            raise ValueError("valid_mask must match labels")
    truth &= valid

    height, width = truth.shape
    block_rows = math.ceil(height / block_size)
    block_cols = math.ceil(width / block_size)
    counts = []
    valid_counts = []
    for row in range(block_rows):
        r0 = row * block_size
        r1 = min(height, r0 + block_size)
        for col in range(block_cols):
            c0 = col * block_size
            c1 = min(width, c0 + block_size)
            counts.append(int(truth[r0:r1, c0:c1].sum()))
            valid_counts.append(int(valid[r0:r1, c0:c1].sum()))

    counts_array = np.asarray(counts, dtype=np.int64)
    valid_array = np.asarray(valid_counts, dtype=np.int64)
    usable = valid_array > 0
    positive = (counts_array > 0) & usable
    positive_counts = counts_array[positive]
    return {
        "block_size_pixels": block_size,
        "block_grid": [block_rows, block_cols],
        "total_blocks": int(counts_array.size),
        "valid_blocks": int(usable.sum()),
        "fault_positive_blocks": int(positive.sum()),
        "fault_positive_block_share": (
            float(positive.sum() / usable.sum()) if usable.any() else 0.0
        ),
        "fault_pixels_per_positive_block": _quantiles(positive_counts),
    }
