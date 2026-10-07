import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.mark.parametrize("problem", [None, "overlap", "shift", "nonbinary", "nodata"])
def test_calibration_cli_report_and_fail_closed_inputs(tmp_path, monkeypatch, problem):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "calibration_cli", root / "scripts/calibrate_threshold.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth = np.zeros((40, 40), dtype="float32")
    truth[4, 4] = truth[34, 34] = 1
    prediction = truth * 0.7 + (1 - truth) * 0.1
    fit = np.zeros_like(truth)
    fit[:10, :10] = 1
    evaluation = np.zeros_like(truth)
    evaluation[30:, 30:] = 1
    if problem == "overlap":
        evaluation[4, 4] = 1
    if problem == "nonbinary":
        fit[4, 4] = 2
    if problem == "nodata":
        prediction[34, 34] = np.nan
    paths = {}
    for name, values in [("truth", truth), ("prediction", prediction),
                         ("calibration-mask", fit), ("evaluation-mask", evaluation)]:
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=40, height=40, count=1,
            dtype="float32", crs="EPSG:32611", nodata=np.nan,
            transform=from_origin(300100 if problem == "shift" and name == "prediction"
                                  else 300000, 4400000, 100, 100),
        ) as dst:
            dst.write(values, 1)
        paths[name] = str(path)
    output = tmp_path / "report.json"
    argv = ["calibrate_threshold"]
    for name, path in paths.items():
        argv.extend([f"--{name}", path])
    argv.extend(["--output-json", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    if problem:
        with pytest.raises(SystemExit):
            module.main()
        assert not output.exists()
    else:
        assert module.main() == 0
        report = json.loads(output.read_text())
        assert report["threshold"] == pytest.approx(0.7)
        assert report["evaluation"]["binary_score"] == pytest.approx(1)
        assert len(report["input_sha256"]) == 4
