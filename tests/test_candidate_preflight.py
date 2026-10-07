"""Exercise the actual candidate CLI before optional ML imports or training."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.mark.parametrize(
    "problem, message",
    [
        ("shift", "geotransform mismatch"),
        ("sparse", "spatial fold map must assign"),
        ("empty", "no valid pixels"),
        ("float", "integer raster"),
    ],
)
def test_candidate_rejects_invalid_fold_artifact(tmp_path, monkeypatch, problem, message):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("candidate", root / "scripts/train_full_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {}
    for name in ["features", "labels", "template", "fold"]:
        path = tmp_path / f"{name}.tif"
        paths[name] = str(path)
        dtype = "int16" if name == "fold" and problem != "float" else "float32"
        values = np.ones((4, 4), dtype=dtype)
        if name == "fold":
            values[:2] = 0
            if problem == "sparse":
                values[3] = -1
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype=dtype,
            crs="EPSG:32611",
            transform=from_origin(
                300100 if name == "fold" and problem == "shift" else 300000, 4400000, 100, 100
            ),
        ) as dst:
            dst.write(values, 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_full_map",
            "--features",
            paths["features"],
            "--labels",
            paths["labels"],
            "--template",
            paths["template"],
            "--fold-map",
            paths["fold"],
            "--fold",
            "99" if problem == "empty" else "0",
            "--output",
            str(tmp_path / "out.tif"),
            "--config",
            str(root / "configs/resnet18_fold0.yaml"),
        ],
    )
    with pytest.raises(SystemExit, match=message):
        module.main()
    assert not (tmp_path / "out.tif").exists()
