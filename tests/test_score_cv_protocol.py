"""End-to-end scorer report protocol test with synthetic GeoTIFFs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gems.data import sha256_file


@pytest.mark.parametrize("scheme", ["spatial", "fault", "trace"])
def test_score_cv_emits_auditable_protocol(tmp_path: Path, scheme):
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
            "--scheme", scheme,
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
    assert protocol["schema_version"] == 2
    if scheme == "spatial":
        assert protocol["aggregation"] == "stitched_spatial_oof"
        assert payload["aggregate_components"]["truth_pixels"] == 2
        assert payload["aggregate_components"]["valid_pixels"] == height * width
    else:
        assert protocol["aggregation"] == "overlapping_background_macro"
        assert "aggregate_score" not in payload
        assert "aggregate_components" not in payload
    assert protocol["metric"] == "distance_weighted_tversky"
    assert (protocol["alpha"], protocol["beta"], protocol["radius_pixels"]) == (0.2, 0.8, 3.0)
    assert protocol["known_fault_exclusion_pixels"] == 0
    assert protocol["truth_sha256"] == sha256_file(paths["truth"])
    assert protocol["fold_map_sha256"] == sha256_file(paths["folds"])
    assert payload["macro_mean"] > 0.99999999


@pytest.mark.parametrize("metadata_case", ["valid", "mixed", "missing", "invalid_json"])
def test_score_cv_metrics_pattern_carries_training_protocol(tmp_path: Path, metadata_case):
    """--metrics-pattern must surface the training protocol the gate compares."""
    height = width = 24
    truth = np.zeros((height, width), dtype=np.uint8)
    truth[3, 3] = 1
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
        metrics = {
            "buffer_pixels": 16,
            "seed": 20260922,
            "epochs": [{"optimizer_steps": 3}],
            "train_step": 32,
            "config_sha256": "ab" * 32,
        }
        (tmp_path / f"metrics-{fold}.json").write_text(json.dumps(metrics))
    if metadata_case == "mixed":
        metrics["seed"] += 1
        (tmp_path / "metrics-1.json").write_text(json.dumps(metrics))
    elif metadata_case == "missing":
        (tmp_path / "metrics-1.json").unlink()
    elif metadata_case == "invalid_json":
        (tmp_path / "metrics-1.json").write_text("[")
    result_path = tmp_path / "scores.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/score_cv.py",
            "--truth", str(paths["truth"]),
            "--fold-map", str(paths["folds"]),
            "--scheme", "spatial",
            "--prediction-pattern", str(tmp_path / "prediction-{fold}.tif"),
            "--metrics-pattern", str(tmp_path / "metrics-{fold}.json"),
            "--output-json", str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    if metadata_case != "valid":
        assert proc.returncode != 0
        assert "training protocol" in proc.stderr
        assert not result_path.exists()
        return
    assert proc.returncode == 0, proc.stderr
    protocol = json.loads(result_path.read_text())["evaluation_protocol"]
    assert protocol["buffer_pixels"] == 16
    assert protocol["seed"] == 20260922
    assert protocol["epochs"] == 1
    assert protocol["train_step"] == 32
    assert protocol["config_sha256"] == "ab" * 32


def test_score_cv_without_metrics_pattern_omits_training_protocol(tmp_path: Path):
    """Without --metrics-pattern the gate must see no training fields to compare."""
    height = width = 24
    truth = np.zeros((height, width), dtype=np.uint8)
    truth[3, 3] = 1
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
    protocol = json.loads(result_path.read_text())["evaluation_protocol"]
    for field in ("buffer_pixels", "seed", "epochs", "train_step", "config_sha256"):
        assert field not in protocol
