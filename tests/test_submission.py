from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from gems.submission import validate_submission


def _write(path: Path, *, dtype="float32", count=1):
    data = np.zeros((count, 4, 5), dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=5,
        height=4,
        count=count,
        dtype=dtype,
        crs="EPSG:32611",
        transform=from_origin(300000, 4400000, 100, 100),
        nodata=np.nan if dtype == "float32" else None,
    ) as dst:
        dst.write(data)


def test_valid_submission(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    _write(ref)
    _write(sub)
    report = validate_submission(sub, ref)
    assert report.ok, report.errors
    assert report.warnings == ()


def test_rejects_wrong_dtype(tmp_path):
    ref = tmp_path / "ref.tif"
    sub = tmp_path / "sub.tif"
    _write(ref)
    _write(sub, dtype="uint8")
    report = validate_submission(sub, ref)
    assert not report.ok
    assert any("float32" in e for e in report.errors)
