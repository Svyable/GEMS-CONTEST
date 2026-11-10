"""Submission validator tests (steps 1, 36).

The validator checks a prediction GeoTIFF against the organizer's
template: one float32 band, EPSG:32611, same shape/transform/bounds,
and — since step 36 — the valid-region contract directionally: finite
predictions outside the template's valid region are an error, missing
predictions inside it are an error, and a constant map earns a warning
because it would burn a rolling submission slot for no signal.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from gems.submission import validate_submission

WIDTH, HEIGHT = 5, 4


def _ramp() -> np.ndarray:
    """Non-constant values in [0, 1] over the valid region."""
    return np.linspace(0.05, 0.95, HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)


def _write(path: Path, data: np.ndarray, *, dtype="float32", nodata=np.nan):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=1,
        dtype=dtype,
        crs="EPSG:32611",
        transform=from_origin(300000, 4400000, 100, 100),
        nodata=nodata if dtype == "float32" else None,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def _template(path: Path) -> np.ndarray:
    """Template with row 0 invalid (nodata) and the rest valid."""
    data = _ramp()
    data[0, :] = np.nan
    _write(path, data)
    return data


def _submission(path: Path, data: np.ndarray, **kwargs):
    _write(path, data, **kwargs)


def test_valid_submission(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    template = _template(ref)
    _submission(sub, template)
    report = validate_submission(sub, ref)
    assert report.ok, report.errors
    assert report.warnings == ()


def test_rejects_wrong_dtype(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    _template(ref)
    data = np.arange(HEIGHT * WIDTH, dtype=np.uint8).reshape(HEIGHT, WIDTH)
    _submission(sub, data, dtype="uint8")
    report = validate_submission(sub, ref)
    assert not report.ok
    assert any("float32" in e for e in report.errors)


def test_rejects_predictions_outside_valid_region(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    template = _template(ref)
    data = template.copy()
    data[0, :] = 0.5  # finite where the template is nodata
    _submission(sub, data)
    report = validate_submission(sub, ref)
    assert not report.ok
    assert any("outside" in e for e in report.errors)


def test_rejects_nan_inside_valid_region(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    template = _template(ref)
    data = template.copy()
    data[2, 2] = np.nan  # a hole inside the valid region
    _submission(sub, data)
    report = validate_submission(sub, ref)
    assert not report.ok
    assert any("inside" in e for e in report.errors)


def test_rejects_out_of_range(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    template = _template(ref)
    data = template.copy()
    data[2, 2] = 1.5
    _submission(sub, data)
    report = validate_submission(sub, ref)
    assert not report.ok
    assert any("[0, 1]" in e for e in report.errors)


def test_rejects_no_finite_predictions(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    _template(ref)
    _submission(sub, np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32))
    report = validate_submission(sub, ref)
    assert not report.ok
    assert any("no finite predictions" in e for e in report.errors)


def test_warns_on_constant_predictions(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    template = _template(ref)
    data = template.copy()
    data[1:, :] = 0.4  # constant inside the valid region
    _submission(sub, data)
    report = validate_submission(sub, ref)
    assert report.ok, report.errors
    assert any("constant" in w for w in report.warnings)


def test_warns_on_nodata_metadata_difference(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    template = _template(ref)
    # Same values, but the submission declares no nodata: its NaN row
    # is then read as data, so the outside-valid check must still fire
    # as an error rather than hiding behind the metadata warning.
    data = template.copy()
    data[0, :] = 0.5
    _submission(sub, data, nodata=None)
    report = validate_submission(sub, ref)
    assert not report.ok
    assert any("outside" in e for e in report.errors)
