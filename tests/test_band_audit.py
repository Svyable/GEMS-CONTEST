"""Synthetic, offline gates; no restricted competition pixels or weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gems.band_audit import (
    BAND_NAMES,
    FEATURE_FILE,
    FILES,
    LABEL_FILE,
    TEMPLATE_FILE,
    audit_manifest,
    build_report,
)
from gems.data import fingerprint_files


def _write_tif(path: Path, data: np.ndarray, nodata: float, *, origin: int = 200000):
    with rasterio.open(
        path, "w", driver="GTiff", count=data.shape[0], width=data.shape[2],
        height=data.shape[1], dtype=data.dtype, nodata=nodata,
        crs="EPSG:32611", transform=from_origin(origin, 4400000, 100, 100),
    ) as dst:
        dst.write(data)
        if path.name == FEATURE_FILE:
            for index, name in enumerate(BAND_NAMES, 1):
                dst.update_tags(index, band_name=name)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    shape = (8, 8)
    features = np.ones((len(BAND_NAMES), *shape), dtype="float32")
    features[0, 0, 0] = -np.finfo("float32").max
    labels = np.zeros((1, *shape), dtype="int8")
    labels[0, 4, 4] = 1
    template = np.zeros((1, *shape), dtype="float32")
    paths = {FILE_NAME: tmp_path / FILE_NAME for FILE_NAME in FILES}
    _write_tif(paths[FEATURE_FILE], features, float(-np.finfo("float32").max))
    _write_tif(paths[LABEL_FILE], labels, -1)
    _write_tif(paths[TEMPLATE_FILE], template, np.nan)
    manifest = fingerprint_files(paths.values(), root=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, paths


def _scan(manifest_path: Path, paths: dict[str, Path], **kwargs):
    return build_report(
        manifest_path, features=paths[FEATURE_FILE], labels=paths[LABEL_FILE],
        template=paths[TEMPLATE_FILE], **kwargs,
    )


def test_live_gated_manifest_passes_but_physics_stays_blocked(tmp_path):
    manifest, paths = _fixture(tmp_path)
    report = _scan(manifest, paths)
    assert report["status"] == "PASS"
    assert report["live"]["sha256_verified"]
    assert report["live"]["study_masks"]["feature_valid"] == 64
    assert report["live"]["band_statistics"][0]["missing_pixels"] == 1
    assert report["live"]["band_statistics"][0]["valid_pixels"] == 63
    assert report["live"]["valid_label_value_counts"] == {"0": 63, "1": 1}
    assert report["physical_transform_gate"].startswith("BLOCKED")
    assert report["declared_units"] == 0


def test_metadata_only_makes_no_live_integrity_claim(tmp_path):
    manifest, _ = _fixture(tmp_path)
    report = build_report(manifest)
    assert report["status"] == "METADATA_ONLY"
    assert "live" not in report
    assert report["grids_match_in_manifest"]


def test_sha_detects_different_pixels_with_identical_geometry(tmp_path):
    manifest, paths = _fixture(tmp_path)
    with rasterio.open(paths[FEATURE_FILE], "r+") as dst:
        dst.write(np.full((8, 8), 3.0, dtype="float32"), 3)
    report = _scan(manifest, paths)
    assert report["status"] == "FAIL"
    assert any("SHA256" in e for e in report["live"]["errors"])


def test_live_tags_detect_reordered_bands_even_with_matching_crs(tmp_path):
    manifest, paths = _fixture(tmp_path)
    with rasterio.open(paths[FEATURE_FILE], "r+") as dst:
        dst.update_tags(1, band_name="tmi")
    report = _scan(manifest, paths)
    assert report["status"] == "FAIL"
    assert any("metadata differs" in e for e in report["live"]["errors"])


def test_shifted_template_grid_fails_closed(tmp_path):
    manifest, paths = _fixture(tmp_path)
    with rasterio.open(paths[TEMPLATE_FILE]) as ds:
        arr = ds.read()
    _write_tif(paths[TEMPLATE_FILE], arr, np.nan, origin=200100)
    report = _scan(manifest, paths)
    assert report["status"] == "FAIL"
    assert any("grid differs" in e for e in report["live"]["errors"])


def test_unexpected_valid_label_values_fail_closed(tmp_path):
    manifest, paths = _fixture(tmp_path)
    with rasterio.open(paths[LABEL_FILE], "r+") as dst:
        arr = dst.read(1)
        arr[1, 2] = 2
        dst.write(arr, 1)
    report = _scan(manifest, paths)
    assert report["status"] == "FAIL"
    assert report["live"]["unexpected_valid_label_pixels"] == 1


def test_mask_disagreement_is_reported_not_hidden(tmp_path):
    manifest, paths = _fixture(tmp_path)
    with rasterio.open(paths[LABEL_FILE], "r+") as dst:
        arr = dst.read(1)
        arr[0, 4] = -1
        dst.write(arr, 1)
    report = _scan(manifest, paths)
    assert report["live"]["study_masks"]["feature_label_disagreement"] == 1
    assert any("valid-data masks differ" in warning for warning in report["live"]["warnings"])


def test_skipping_hashes_never_claims_full_verification(tmp_path):
    manifest, paths = _fixture(tmp_path)
    report = _scan(manifest, paths, verify_hashes=False)
    assert report["status"] == "UNVERIFIED_HASHES"
    assert not report["live"]["sha256_verified"]


def test_incomplete_triplet_rejected(tmp_path):
    manifest, paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="all three"):
        build_report(manifest, features=paths[FEATURE_FILE])


def test_manifest_wrong_band_order_rejected(tmp_path):
    manifest, _ = _fixture(tmp_path)
    document = json.loads(manifest.read_text())
    for file_entry in document["files"]:
        if file_entry["path"] == FEATURE_FILE:
            file_entry["raster"]["band_tags"][0]["band_name"] = "other"
    assert audit_manifest(document)["status"] == "MANIFEST_FAILURE"


def test_mutated_manifest_fails_its_own_checksum(tmp_path):
    manifest, _ = _fixture(tmp_path)
    document = json.loads(manifest.read_text())
    for entry in document["files"]:
        if entry["path"] == FEATURE_FILE:
            entry["raster"]["band_tags"][0]["description"] = "untrusted metadata"
    assert "manifest SHA256 self-check failed" in audit_manifest(document)["errors"]


def test_manual_snapshot_is_never_validated_against_live_files(tmp_path):
    manifest, paths = _fixture(tmp_path)
    document = json.loads(manifest.read_text())
    document["snapshot_description"] = "this is a manually transcribed subset"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="manual snapshot"):
        _scan(manifest, paths)


def test_mismatched_dimensions_skip_unsafe_pixel_scan(tmp_path):
    manifest, paths = _fixture(tmp_path)
    smaller = np.zeros((1, 6, 8), dtype="float32")
    _write_tif(paths[TEMPLATE_FILE], smaller, np.nan)
    result = _scan(manifest, paths)
    assert result["status"] == "FAIL"
    assert result["live"]["scan_skipped"] == "incompatible grids"


def test_cli_metadata_only_is_explicit(tmp_path):
    import subprocess
    import sys

    manifest, _ = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "scripts/audit_band_integrity.py", "--manifest", str(manifest),
         "--output", str(output)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["status"] == "METADATA_ONLY"
    assert "potential-field=BLOCKED" in run.stdout


def test_cli_bad_manifest_exits_nonzero(tmp_path):
    import subprocess
    import sys

    manifest, _ = _fixture(tmp_path)
    document = json.loads(manifest.read_text())
    document["files"][0]["raster"]["count"] = 99
    manifest.write_text(json.dumps(document))
    output = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "scripts/audit_band_integrity.py", "--manifest", str(manifest),
         "--output", str(output)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 2, run.stderr
    assert json.loads(output.read_text())["status"] == "MANIFEST_FAILURE"
