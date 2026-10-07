"""End-to-end scorer report protocol test with synthetic GeoTIFFs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from gems.data import sha256_file


def test_score_cv_emits_auditable_protocol(tmp_path: Path):
    height = width = 24
    truth = np.zeros((height, width), dtype=np.uint8)
    truth[3, 3] = 1
    truth[15, 18] = 1
    folds = np.zeros((height, width), dtype=np.int16)
    folds[:, width // 2:] = 1
    common = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": "EPSG:32611",
        "transform": from_origin(500000, 4000000, 100, 100),
    }
    paths = {}
    for name, array in (("truth", truth), ("folds", folds)):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", dtype=array.dtype, **common) as dst:
            dst.write(array, 1)
        paths[name] = path
    for fold in (0, 1):
        with rasterio.open(
            tmp_path / f"prediction-{fold}.tif", "w", dtype="float32", **common
        ) as dst:
            dst.write(truth.astype(np.float32), 1)
    result_path = tmp_path / "scores.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/score_cv.py",
            "--truth", str(paths["truth"]),
            "--fold-map", str(paths["folds"]),
            "--scheme", "spatial",
            "--prediction-pattern", str(tmp_path / "prediction-{fold}.tif"),
            "--output-json", str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(result_path.read_text())
    assert payload == json.loads(proc.stdout)
    protocol = payload["evaluation_protocol"]
    assert protocol["schema_version"] == 1
    assert protocol["metric"] == "distance_weighted_tversky"
    assert (protocol["alpha"], protocol["beta"], protocol["radius_pixels"]) == (0.2, 0.8, 3.0)
    assert protocol["known_fault_exclusion_pixels"] == 0
    assert protocol["truth_sha256"] == sha256_file(paths["truth"])
    assert protocol["fold_map_sha256"] == sha256_file(paths["folds"])
    assert payload["macro_mean"] > 0.99999999
