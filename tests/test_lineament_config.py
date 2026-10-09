"""Regression checks for runnable lineament configs and held-out data isolation."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from gems.lineament import (
    grouped_lineament_features,
    lineament_kwargs,
    required_lineament_buffer,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "kind", "buffer"),
    [
        ("resnet18_structure_tensor.yaml", "structure_tensor_coherence", 17),
        ("resnet18_ridge_valley.yaml", "ridge_valley_response", 30),
        ("resnet18_steerable_filter.yaml", "steerable_filter", 7),
    ],
)
def test_ablation_configs_are_runnable_and_options_reach_features(filename, kind, buffer):
    config = yaml.safe_load((ROOT / "configs" / filename).read_text())
    assert config["patches"]["patch_size"] == 128
    assert config["model"]["encoder"] == "resnet18"
    assert config["training"]["epochs"] == 20
    spec = config["derived_features"]["lineament"]
    assert spec["enabled"]
    assert "geodetic_strain" in spec["categories"]
    options = lineament_kwargs(spec)
    assert options["kind"] == kind
    assert required_lineament_buffer(options) == buffer

    _, xx = np.indices((48, 48))
    features = np.exp(-((xx - 24) / 3.0) ** 2).astype(np.float32)[..., None]
    derived, metadata = grouped_lineament_features(
        features,
        {"magnetic_data": (0,)},
        valid_mask=np.ones((48, 48), dtype=bool),
        **options,
    )
    assert derived.shape == (48, 48, 1)
    assert np.isfinite(derived).all()
    assert metadata["kind"] == kind
    if kind == "ridge_valley_response":
        assert metadata["multiscale_sigmas"] == [1.0, 2.0, 4.0]
    elif kind == "steerable_filter":
        assert metadata["steerable_wavelength"] == 8.0
        assert metadata["steerable_orientations"] == 8
    else:
        assert metadata["structure_tensor_window"] == 3.0


@pytest.mark.parametrize(
    "kind",
    ["mumford_shah_log", "ridge_valley_response", "steerable_filter"],
)
def test_held_out_outlier_cannot_change_training_region_feature_scaling(kind):
    rng = np.random.default_rng(17)
    features = rng.normal(size=(160, 160, 1)).astype(np.float32)
    features[:, 110, 0] += 5
    modified = features.copy()
    modified[:45, :45, 0] = 10000

    valid = np.ones((160, 160), dtype=bool)
    training = np.zeros_like(valid)
    training[90:140, 90:140] = True
    kwargs = {
        "kind": kind,
        "sigma_pixels": 1.0,
        "structure_tensor_window": 3.0,
        "ridge_valley_scales": (1.0, 2.0, 4.0),
    }
    baseline, meta = grouped_lineament_features(
        features, {"signal": (0,)}, valid_mask=valid,
        normalization_mask=training, **kwargs,
    )
    poisoned, poisoned_meta = grouped_lineament_features(
        modified, {"signal": (0,)}, valid_mask=valid,
        normalization_mask=training, **kwargs,
    )
    np.testing.assert_allclose(
        baseline[105:125, 105:125], poisoned[105:125, 105:125],
        atol=1e-6, rtol=1e-6,
    )
    assert meta["normalization_pixels"] == 2500
    assert poisoned_meta["normalization_pixels"] == 2500


def test_lineament_rejects_invalid_normalization_mask():
    features = np.ones((12, 12, 1), dtype=np.float32)
    valid = np.ones((12, 12), dtype=bool)
    valid[0, 0] = False

    for fit_mask in (
        np.zeros((12, 12), dtype=bool),
        np.ones((12, 12), dtype=bool),
        np.ones((2, 2), dtype=bool),
    ):
        with pytest.raises(ValueError, match="normalization_mask"):
            grouped_lineament_features(
                features, {"signal": (0,)},
                valid_mask=valid, normalization_mask=fit_mask,
            )
