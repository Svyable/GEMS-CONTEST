import hashlib

import numpy as np
import rasterio
from rasterio.transform import from_origin

from gems.data import fingerprint_file, fingerprint_files, raster_alignment_errors


def _write_raster(path, *, width=5, height=4, x0=300000):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(x0, 4400000, 100, 100),
        nodata=np.nan,
    ) as dst:
        dst.write(np.zeros((1, height, width), dtype="float32"))
        dst.update_tags(1, description="demo", data_category="test")


def test_fingerprint_plain_file(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"gems")
    fp = fingerprint_file(path, root=tmp_path)
    assert fp["path"] == "x.txt"
    assert fp["sha256"] == hashlib.sha256(b"gems").hexdigest()
    assert fp["bytes"] == 4


def test_raster_fingerprint_has_stable_geometry(tmp_path):
    path = tmp_path / "x.tif"
    _write_raster(path)
    fp = fingerprint_file(path, root=tmp_path)
    assert fp["raster"]["crs"] == "EPSG:32611"
    assert fp["raster"]["resolution"] == [100.0, 100.0]
    assert fp["raster"]["band_tags"][0]["description"] == "demo"


def test_manifest_hash_independent_of_input_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    left = fingerprint_files([a, b], root=tmp_path)
    right = fingerprint_files([b, a], root=tmp_path)
    assert left["manifest_sha256"] == right["manifest_sha256"]


def test_alignment_reports_mismatch(tmp_path):
    features, labels = tmp_path / "features.tif", tmp_path / "labels.tif"
    _write_raster(features)
    _write_raster(labels, x0=300100)
    assert "geotransform mismatch" in raster_alignment_errors(features, labels)
