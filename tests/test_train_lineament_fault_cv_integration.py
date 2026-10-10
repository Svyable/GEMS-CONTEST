"""Integration test: train_full_map.py accepts lineament features under fault/trace CV."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from gems.cv import assign_fault_components


def _trainer():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("trainer", root / "scripts/train_full_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_test_raster(path, data, categories=None):
    """Write a synthetic GeoTIFF with optional band category tags."""
    if data.ndim == 2:
        data = data[..., np.newaxis]
    H, W, C = data.shape
    
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=W,
        height=H,
        count=C,
        dtype=data.dtype,
        crs=CRS.from_epsg(32611),
        transform=from_origin(300000, 4400000, 100, 100),
    ) as dst:
        for band in range(C):
            dst.write(data[:, :, band], band + 1)
            if categories and band < len(categories):
                dst.update_tags(band + 1, data_category=categories[band])


def test_train_full_map_accepts_lineament_features_under_fault_cv(tmp_path):
    """Verify train_full_map.py runs with lineament features under --cv-scheme fault."""
    # Create synthetic features with category tags
    H, W = 40, 50
    rng = np.random.default_rng(123)
    features = rng.normal(0, 1, (H, W, 4)).astype(np.float32)
    features_path = tmp_path / "features.tif"
    _write_test_raster(
        features_path,
        features,
        categories=["magnetic_data", "gravity_data", "magnetic_data", "topographic"],
    )
    
    # Create synthetic labels with two separate fault components
    labels = np.zeros((H, W), dtype=np.uint8)
    labels[10:15, 10:30] = 1  # horizontal fault
    labels[25:35, 20:25] = 1  # vertical fault
    labels_path = tmp_path / "labels.tif"
    _write_test_raster(labels_path, labels)
    
    # Create fold map
    component_folds = assign_fault_components(labels, n_folds=2, seed=0)
    fold_map_path = tmp_path / "folds.tif"
    _write_test_raster(fold_map_path, component_folds.astype(np.int16))
    
    # Create template (same shape as labels)
    template_path = tmp_path / "template.tif"
    _write_test_raster(template_path, labels)
    
    # Create minimal config with lineament features
    config_yaml = """
patches:
  patch_size: 16
  train_step: 8
model:
  encoder: resnet18
  encoder_weights: imagenet
  classes: 1
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.0001
  alpha: 0.2
  beta: 0.8
augmentation:
  random_resized_crop:
    scale: [0.8, 1.0]
    ratio: [0.9, 1.1]
  horizontal_flip_probability: 0.5
  vertical_flip_probability: 0.5
  random_rotation_degrees: 15
derived_features:
  lineament:
    enabled: true
    categories:
      - magnetic_data
      - gravity_data
    kind: gradient_energy
    sigma_pixels: 1.0
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    
    output_path = tmp_path / "output.tif"
    metrics_path = tmp_path / "metrics.json"
    
    # Import and run train_full_map main
    import sys
    import json
    sys.argv = [
        "train_full_map.py",
        "--features", str(features_path),
        "--labels", str(labels_path),
        "--template", str(template_path),
        "--output", str(output_path),
        "--config", str(config_path),
        "--fold-map", str(fold_map_path),
        "--cv-scheme", "fault",
        "--fold", "0",
        "--buffer-pixels", "6",
        "--seed", "42",
        "--metrics-json", str(metrics_path),
    ]
    
    # This should not raise the "fault/trace CV needs a masked-input transform protocol" error
    # The old code would have raised SystemExit before computing lineament features.
    # Now it should proceed to lineament computation and only fail at torch import (if torch unavailable)
    try:
        trainer = _trainer()
        returncode = trainer.main()
        
        # If torch is available, verify full success
        assert returncode == 0, "train_full_map should succeed with lineament + fault CV"
        assert output_path.exists(), "output prediction should be written"
        assert metrics_path.exists(), "metrics JSON should be written"
        
        # Verify metrics include lineament metadata
        metrics = json.loads(metrics_path.read_text())
        assert metrics["derived_features"] is not None
        assert metrics["derived_features"]["lineament"]["kind"] == "gradient_energy"
        assert metrics["derived_features"]["lineament"]["derived_channels"] == 2  # 2 groups
        assert metrics["validation"]["scheme"] == "fault"
        
    except SystemExit as exc:
        # If torch is unavailable, verify we got past the lineament feature computation
        # The old code would have raised "fault/trace CV needs a masked-input transform protocol"
        # before ever trying to import torch
        if "ML dependencies are missing" in str(exc):
            pytest.skip("ML dependencies not available (expected, test verified protocol works)")
        else:
            # Re-raise any other SystemExit (like the old protocol refusal)
            raise


@pytest.mark.parametrize("kind", ["structure_tensor_coherence", "mumford_shah_log"])
def test_train_full_map_accepts_advanced_lineament_kinds_under_trace_cv(tmp_path, kind):
    """Verify advanced lineament kinds work under trace CV."""
    H, W = 50, 60
    rng = np.random.default_rng(456)
    features = rng.normal(0, 1, (H, W, 2)).astype(np.float32)
    features_path = tmp_path / "features.tif"
    _write_test_raster(features_path, features, categories=["magnetic_data", "gravity_data"])
    
    # Create labels with long traces
    labels = np.zeros((H, W), dtype=np.uint8)
    labels[15, 10:50] = 1  # long horizontal trace
    labels[35, 10:50] = 1  # another long trace
    labels_path = tmp_path / "labels.tif"
    _write_test_raster(labels_path, labels)
    
    # Create trace endpoint fold map
    from gems.cv import assign_trace_endpoints
    endpoint_folds = assign_trace_endpoints(
        labels, n_folds=2, seed=1, min_pixels=20, endpoint_fraction=0.25
    )
    fold_map_path = tmp_path / "folds.tif"
    _write_test_raster(fold_map_path, endpoint_folds.astype(np.int16))
    
    template_path = tmp_path / "template.tif"
    _write_test_raster(template_path, labels)
    
    # Config with advanced lineament kind
    extra_params = ""
    if kind == "structure_tensor_coherence":
        extra_params = "    structure_tensor_window: 2.0\n"
    
    config_yaml = f"""
patches:
  patch_size: 16
  train_step: 8
model:
  encoder: resnet18
  encoder_weights: imagenet
  classes: 1
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.0001
  alpha: 0.2
  beta: 0.8
augmentation:
  random_resized_crop:
    scale: [0.8, 1.0]
    ratio: [0.9, 1.1]
  horizontal_flip_probability: 0.5
  vertical_flip_probability: 0.5
  random_rotation_degrees: 10
derived_features:
  lineament:
    enabled: true
    categories:
      - magnetic_data
    kind: {kind}
    sigma_pixels: 1.0
{extra_params}"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)
    
    output_path = tmp_path / "output.tif"
    
    import sys
    sys.argv = [
        "train_full_map.py",
        "--features", str(features_path),
        "--labels", str(labels_path),
        "--template", str(template_path),
        "--output", str(output_path),
        "--config", str(config_path),
        "--fold-map", str(fold_map_path),
        "--cv-scheme", "trace",
        "--fold", "0",
        "--buffer-pixels", "16",  # large enough for all lineament kinds
        "--seed", "99",
    ]
    
    try:
        trainer = _trainer()
        returncode = trainer.main()
        
        # If torch is available, verify success
        assert returncode == 0, f"{kind} + trace CV should succeed"
        assert output_path.exists()
        
    except SystemExit as exc:
        # Accept torch import error (proves we got past the old protocol refusal)
        if "ML dependencies are missing" in str(exc):
            pytest.skip("ML dependencies not available (expected, test verified protocol works)")
        else:
            raise
