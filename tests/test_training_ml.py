"""Optional ML tests, required by the CPU training-smoke CI job."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
from segmentation_models_pytorch.losses import TverskyLoss

from gems.cv import assign_fault_components, assign_trace_endpoints
from gems.evaluation import (
    evaluate_fault_discovery_predictions,
    evaluate_trace_completion_predictions,
)


def _trainer():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("trainer", root / "scripts/train_full_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_masked_loss_has_no_gradient_on_withheld_logits():
    module = _trainer()
    logits = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[[True, True], [False, False]]])
    criterion = TverskyLoss(alpha=0.2, beta=0.8, mode="binary")
    loss = module.masked_binary_loss(criterion, logits, target, mask)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.count_nonzero(logits.grad[mask]) > 0
    mutated_logits, mutated_target = logits.detach().clone(), target.clone()
    mutated_logits[~mask], mutated_target[~mask] = 999, 999
    other = module.masked_binary_loss(criterion, mutated_logits, mutated_target, mask)
    torch.testing.assert_close(loss.detach(), other)
    assert module.masked_binary_loss(criterion, logits, target, torch.zeros_like(mask)) is None


@pytest.mark.parametrize("scheme", ["fault", "trace"])
def test_actual_unet_train_infer_validate_and_discovery_score(tmp_path, monkeypatch, scheme):
    module = _trainer()
    truth = np.zeros((96, 128), dtype="uint8")
    truth[16, 5:100] = truth[80, 5:100] = 1
    folds = (assign_fault_components(truth, n_folds=2) if scheme == "fault" else
             assign_trace_endpoints(truth, n_folds=2, min_pixels=24))
    features = np.random.default_rng(3).random((3, *truth.shape), dtype=np.float32)
    profile = {"driver": "GTiff", "width": 128, "height": 96, "crs": "EPSG:32611",
               "transform": from_origin(300000, 4400000, 100, 100)}
    paths = {}
    for name, data, nodata in [("features", features, None),
                               ("labels", truth[None], None),
                               ("template", np.zeros((1, 96, 128), dtype="float32"), np.nan),
                               ("fold-map", folds[None], -1)]:
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", **profile, dtype=data.dtype,
                           count=data.shape[0], nodata=nodata) as dst:
            dst.write(data)
        paths[name] = str(path)
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/resnet18_fold0.yaml").read_text())
    config["model"]["encoder_weights"] = None  # No network/checkpoint download.
    config["training"].update(epochs=1, batch_size=2)
    config["patches"].update(patch_size=64, train_step=64)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output, metrics = tmp_path / "prediction.tif", tmp_path / "metrics.json"
    argv = ["train_full_map"]
    for name, path in paths.items():
        argv.extend([f"--{name}", path])
    argv.extend(["--cv-scheme", scheme, "--buffer-pixels", "3", "--fold", "0",
                 "--overlap", "16", "--config", str(config_path), "--output", str(output),
                 "--metrics-json", str(metrics)])
    monkeypatch.setattr(sys, "argv", argv)
    # Explicitly exercise the reproducible CPU path even on a GPU test host.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert module.main() == 0
    finally:
        torch.set_num_threads(previous_threads)
    report = json.loads(metrics.read_text())
    assert report["validation"]["scheme"] == scheme
    assert report["training_loss_mask"] == "supervised_pixels_only"
    assert report["epochs"][0]["optimizer_steps"] > 0
    assert report["train_step"] == 64  # protocol-equality gate reads this field
    with rasterio.open(output) as src:
        predicted = src.read(1)
    evaluator = (evaluate_fault_discovery_predictions if scheme == "fault" else
                 evaluate_trace_completion_predictions)
    result = evaluator({0: predicted, 1: predicted}, truth, folds)
    assert report["holdout_distance_weighted_tversky"] == pytest.approx(
        result["folds"][0]["score"], abs=1e-10
    )

def test_registration_sensitivity_runs_on_spatial_holdout(tmp_path, monkeypatch):
    module = _trainer()
    truth = np.zeros((64, 96), dtype="uint8")
    truth[16, 5:90] = 1
    truth[48, 5:90] = 1
    folds = np.zeros_like(truth, dtype="int16")
    folds[:, 48:] = 1
    features = np.random.default_rng(7).random((3, *truth.shape), dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "width": 96,
        "height": 64,
        "crs": "EPSG:32611",
        "transform": from_origin(300000, 4400000, 100, 100),
    }
    paths = {}
    for name, data, nodata in [
        ("features", features, None),
        ("labels", truth[None], None),
        ("template", np.zeros((1, 64, 96), dtype="float32"), np.nan),
        ("fold-map", folds[None], -1),
    ]:
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            **profile,
            dtype=data.dtype,
            count=data.shape[0],
            nodata=nodata,
        ) as dst:
            dst.write(data)
            if name == "features":
                dst.update_tags(1, data_category="magnetic_data")
                dst.update_tags(2, data_category="gravity_data")
                dst.update_tags(3, data_category="topographic")
        paths[name] = str(path)

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/resnet18_fold0.yaml").read_text())
    config["model"]["encoder_weights"] = None
    config["training"].update(epochs=1, batch_size=2)
    config["patches"].update(patch_size=32, train_step=32)
    config["robustness"] = {
        "registration_sensitivity": {"categories": ["magnetic_data"]}
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "prediction.tif"
    metrics = tmp_path / "metrics.json"
    sensitivity = tmp_path / "registration.json"
    argv = ["train_full_map"]
    for name, path in paths.items():
        argv.extend([f"--{name}", path])
    argv.extend(
        [
            "--cv-scheme",
            "spatial",
            "--buffer-pixels",
            "2",
            "--fold",
            "0",
            "--overlap",
            "8",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--metrics-json",
            str(metrics),
            "--registration-sensitivity-json",
            str(sensitivity),
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert module.main() == 0
    finally:
        torch.set_num_threads(previous_threads)

    report = json.loads(sensitivity.read_text())
    assert report["method"] == "one_pixel_feature_family_translation_v1"
    assert report["pixel_resolution"] == [100.0, 100.0]
    assert set(report["categories"]) == {"magnetic_data"}
    category = report["categories"]["magnetic_data"]
    assert category["channels_zero_based"] == [0]
    assert len(category["shifts"]) == 8
    assert all("delta_from_baseline" in item for item in category["shifts"])

