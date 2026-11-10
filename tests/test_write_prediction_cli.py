"""End-to-end tests for scripts/write_prediction.py (step 36).

The CLI is the last link before an upload: .npy probability raster in,
template-aligned GeoTIFF out, validated before it prints OK. These tests
exercise it on synthetic data — a clean run exits 0, and an out-of-range
raster exits 1 with an ERROR line instead of a traceback.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from write_prediction import main

WIDTH, HEIGHT = 5, 4


def _template(path: Path) -> None:
    data = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    data[:, 0, :] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(300000, 4400000, 100, 100),
        nodata=np.nan,
    ) as dst:
        dst.write(data)


def _run_cli(tmp_path: Path, values: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> int:
    template = tmp_path / "template.tif"
    _template(template)
    npy = tmp_path / "prediction.npy"
    np.save(npy, values)
    output = tmp_path / "out.tif"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_prediction.py",
            "--prediction-npy",
            str(npy),
            "--template",
            str(template),
            "--output",
            str(output),
        ],
    )
    return main()


def test_cli_writes_validated_submission(tmp_path, monkeypatch, capsys):
    values = np.linspace(0.05, 0.95, HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    assert _run_cli(tmp_path, values, monkeypatch) == 0
    assert "OK: wrote validated submission-format raster" in capsys.readouterr().out
    with rasterio.open(tmp_path / "out.tif") as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert np.isnan(src.read(1)[0]).all()


def test_cli_rejects_out_of_range_without_traceback(tmp_path, monkeypatch, capsys):
    values = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    assert _run_cli(tmp_path, values, monkeypatch) == 1
    assert "ERROR:" in capsys.readouterr().out


def test_cli_rejects_shape_mismatch(tmp_path, monkeypatch, capsys):
    values = np.zeros((2, 2), dtype=np.float32)
    assert _run_cli(tmp_path, values, monkeypatch) == 1
    assert "ERROR:" in capsys.readouterr().out
