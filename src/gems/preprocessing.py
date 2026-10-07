"""Candidate preprocessing; organizer reproduction intentionally uses its own recipe."""

from __future__ import annotations

import numpy as np


def normalize_training_features(
    features: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Fit channel ranges on training pixels only, then transform the full raster.

    Held-out values may fall outside [0, 1]; clipping would erase extrapolation.
    Missing values become zero. Training-constant channels become zero everywhere
    because no scale can be estimated without consulting validation data.
    """
    x = np.asarray(features, dtype=np.float32)
    mask = np.asarray(fit_mask, dtype=bool)
    if x.ndim != 3 or x.shape[-1] == 0 or mask.shape != x.shape[:2]:
        raise ValueError("features must be HWC and fit_mask must match HW")
    if not mask.any():
        raise ValueError("normalization requires training pixels")
    output = np.zeros_like(x)
    minimum, maximum, counts, constants = [], [], [], []
    for band in range(x.shape[-1]):
        values = x[..., band]
        finite = np.isfinite(values)
        selected = values[mask & finite]
        if not selected.size:
            raise ValueError(f"channel {band} has no finite training values")
        low, high = float(selected.min()), float(selected.max())
        if high != low:
            scaled = (values[finite].astype(np.float64) - low) / (high - low)
            if np.any(np.abs(scaled) > np.finfo(np.float32).max):
                raise ValueError(f"channel {band} overflows float32 after normalization")
            output[..., band][finite] = scaled.astype(np.float32)
        else:
            constants.append(band)
        minimum.append(low)
        maximum.append(high)
        counts.append(int(selected.size))
    return output, {
        "method": "training_region_channel_minmax_v1",
        "fit_pixels": int(mask.sum()),
        "finite_fit_pixels_per_channel": counts,
        "minimum": minimum,
        "maximum": maximum,
        "constant_channels": constants,
        "nonfinite_fill": 0.0,
        "clip": False,
    }
