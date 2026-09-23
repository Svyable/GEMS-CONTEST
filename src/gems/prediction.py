from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def write_prediction_like_template(
    prediction: np.ndarray,
    *,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write a float32 prediction GeoTIFF aligned exactly to a template raster."""
    values = np.asarray(prediction, dtype=np.float32)
    with rasterio.open(template_path) as template:
        expected_shape = (template.height, template.width)
        if values.shape != expected_shape:
            raise ValueError(
                f"prediction shape {values.shape} does not match template {expected_shape}"
            )
        valid = template.dataset_mask() > 0
        profile = template.profile.copy()

    inside = values[valid]
    if inside.size == 0:
        raise ValueError("template contains no valid pixels")
    if not np.all(np.isfinite(inside)):
        raise ValueError("prediction contains non-finite values inside the valid region")
    if float(inside.min()) < 0.0 or float(inside.max()) > 1.0:
        raise ValueError("prediction values inside the valid region must lie in [0, 1]")

    output = values.copy()
    output[~valid] = np.nan
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)
    return path
