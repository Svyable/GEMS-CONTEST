from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


@dataclass(frozen=True)
class SubmissionReport:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _same_nodata(left: object, right: object) -> bool:
    """Compare GeoTIFF nodata values, treating every NaN as the same sentinel."""
    if left is None or right is None:
        return left is None and right is None
    left_value = np.asarray(left)
    right_value = np.asarray(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(left_value == right_value)


def validate_submission(submission_path: str | Path, template_path: str | Path) -> SubmissionReport:
    """Validate a prediction GeoTIFF against the organizer's sample submission."""
    errors: list[str] = []
    warnings: list[str] = []

    with rasterio.open(template_path) as ref, rasterio.open(submission_path) as sub:
        if sub.count != 1:
            errors.append(f"expected 1 band, found {sub.count}")
        if sub.dtypes[0] != "float32":
            errors.append(f"expected float32, found {sub.dtypes[0]}")
        if sub.crs != ref.crs:
            errors.append(f"CRS mismatch: submission={sub.crs}, template={ref.crs}")
        if sub.crs is None or sub.crs.to_epsg() != 32611:
            errors.append(f"expected EPSG:32611, found {sub.crs}")
        if sub.transform != ref.transform:
            errors.append("geotransform/resolution/origin differs from template")
        if sub.width != ref.width or sub.height != ref.height:
            errors.append(
                f"shape mismatch: submission={(sub.height, sub.width)}, "
                f"template={(ref.height, ref.width)}"
            )
        if sub.bounds != ref.bounds:
            errors.append(f"bounds mismatch: submission={sub.bounds}, template={ref.bounds}")

        data = sub.read(1, masked=True)
        finite = np.asarray(data.compressed())
        if finite.size == 0:
            errors.append("submission contains no finite/unmasked predictions")
        else:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            if lo < 0.0 or hi > 1.0:
                errors.append(f"prediction range must be [0, 1], found [{lo}, {hi}]")

        if not _same_nodata(sub.nodata, ref.nodata):
            warnings.append(
                f"nodata metadata differs: submission={sub.nodata}, template={ref.nodata}"
            )

        ref_mask = ref.dataset_mask() > 0
        sub_mask = sub.dataset_mask() > 0
        if not np.array_equal(ref_mask, sub_mask):
            warnings.append("dataset valid-data mask differs from template")

    return SubmissionReport(not errors, tuple(errors), tuple(warnings))
