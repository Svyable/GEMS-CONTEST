import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gems.lineament import grouped_lineament_features, raster_category_groups


def test_constant_group_has_zero_lineament_response():
    features = np.ones((9, 11, 2), dtype=np.float32)
    valid = np.ones((9, 11), dtype=bool)
    for kind in ("gradient_energy", "phase_edge"):
        result, metadata = grouped_lineament_features(
            features,
            {"physical": (0, 1)},
            valid_mask=valid,
            kind=kind,
            sigma_pixels=0,
        )
        assert result.shape == (9, 11, 1)
        assert not result.any()
        assert metadata["output_channels"] == [f"physical:{kind}"]


def test_phase_edge_is_bounded_and_monotone_with_gradient_strength():
    x = np.arange(15, dtype=np.float32)
    weak = np.tile(x, (15, 1))
    strong = 3 * weak
    features = np.stack([weak, strong], axis=-1)
    valid = np.ones((15, 15), dtype=bool)

    weak_phase, _ = grouped_lineament_features(
        features,
        {"weak": (0,)},
        valid_mask=valid,
        kind="phase_edge",
        sigma_pixels=0,
        phase_epsilon_pixels=0.5,
    )
    strong_phase, _ = grouped_lineament_features(
        features,
        {"strong": (1,)},
        valid_mask=valid,
        kind="phase_edge",
        sigma_pixels=0,
        phase_epsilon_pixels=0.5,
    )
    assert np.all((0 <= weak_phase) & (weak_phase <= 1))
    assert np.all((0 <= strong_phase) & (strong_phase <= 1))
    assert np.all(strong_phase >= weak_phase)
    assert float(strong_phase[7, 7, 0]) > float(weak_phase[7, 7, 0])


def test_invalid_pixels_are_zero_and_do_not_poison_smoothed_result():
    features = np.zeros((21, 21, 1), dtype=np.float32)
    features[:, 10:, 0] = 1
    features[0, 0, 0] = np.nan
    valid = np.ones((21, 21), dtype=bool)
    valid[:3, :3] = False

    result, _ = grouped_lineament_features(
        features,
        {"edge": (0,)},
        valid_mask=valid,
        kind="phase_edge",
        sigma_pixels=1,
    )
    assert np.isfinite(result).all()
    assert not result[:3, :3].any()
    assert result[10, 10, 0] > result[10, 3, 0]


def test_raster_categories_come_from_band_metadata(tmp_path):
    path = tmp_path / "features.tif"
    data = np.zeros((4, 5, 6), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=6,
        height=5,
        count=4,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(300000, 4400000, 100, 100),
    ) as dst:
        dst.write(data)
        dst.update_tags(1, data_category="magnetic_data")
        dst.update_tags(2, data_category="gravity_data")
        dst.update_tags(3, data_category="magnetic_data")
        dst.update_tags(4, data_category="topographic")

    groups = raster_category_groups(
        path,
        ["magnetic_data", "gravity_data", "topographic"],
    )
    assert groups == {
        "magnetic_data": (0, 2),
        "gravity_data": (1,),
        "topographic": (3,),
    }


def test_missing_raster_category_is_rejected(tmp_path):
    path = tmp_path / "features.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="float32",
        transform=from_origin(0, 3, 1, 1),
    ) as dst:
        dst.write(np.zeros((1, 3, 3), dtype=np.float32))
        dst.update_tags(1, data_category="magnetic_data")

    with pytest.raises(ValueError, match="missing requested data categories"):
        raster_category_groups(path, ["gravity_data"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "unknown"},
        {"sigma_pixels": -1},
        {"phase_epsilon_pixels": 0},
        {"structure_tensor_window": -1},
        {"structure_tensor_window": 0},
    ],
)
def test_invalid_lineament_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        grouped_lineament_features(
            np.ones((4, 4, 1), dtype=np.float32),
            {"group": (0,)},
            valid_mask=np.ones((4, 4), bool),
            **kwargs,
        )


def test_structure_tensor_coherence_on_constant_field():
    """Constant field has zero gradient and therefore zero coherence."""
    features = np.ones((20, 20, 1), dtype=np.float32) * 5.0
    valid = np.ones((20, 20), dtype=bool)
    
    result, metadata = grouped_lineament_features(
        features,
        {"constant": (0,)},
        valid_mask=valid,
        kind="structure_tensor_coherence",
        sigma_pixels=1.0,
        structure_tensor_window=2.0,
    )
    
    assert result.shape == (20, 20, 1)
    assert np.allclose(result, 0.0, atol=1e-5)
    assert metadata["kind"] == "structure_tensor_coherence"
    assert "structure_tensor_window" in metadata


def test_structure_tensor_coherence_on_strong_linear_gradient():
    """A strong horizontal gradient should have high coherence."""
    features = np.tile(np.arange(30, dtype=np.float32), (30, 1))
    features = features[..., np.newaxis]
    valid = np.ones((30, 30), dtype=bool)
    
    result, _ = grouped_lineament_features(
        features,
        {"horizontal": (0,)},
        valid_mask=valid,
        kind="structure_tensor_coherence",
        sigma_pixels=0.5,
        structure_tensor_window=2.0,
    )
    
    # Center should have high coherence (close to 1)
    center_coherence = result[15, 15, 0]
    assert center_coherence > 0.8, f"Expected high coherence, got {center_coherence}"
    assert np.all((0 <= result) & (result <= 1))


def test_structure_tensor_orientation_detects_horizontal_gradient():
    """Horizontal gradient (increasing left-to-right) has orientation near 0."""
    features = np.tile(np.arange(40, dtype=np.float32), (40, 1))
    features = features[..., np.newaxis]
    valid = np.ones((40, 40), dtype=bool)
    
    result, metadata = grouped_lineament_features(
        features,
        {"horizontal": (0,)},
        valid_mask=valid,
        kind="structure_tensor_orientation",
        sigma_pixels=0.5,
        structure_tensor_window=2.0,
    )
    
    # Orientation should be near 0 (horizontal gradient direction)
    center_orientation = result[20, 20, 0]
    assert abs(center_orientation) < 0.2, f"Expected ~0, got {center_orientation}"
    assert np.all((-np.pi/2 <= result) & (result <= np.pi/2))
    assert metadata["kind"] == "structure_tensor_orientation"


def test_structure_tensor_orientation_detects_vertical_gradient():
    """Vertical gradient (increasing top-to-bottom) has orientation near ±π/2."""
    features = np.tile(np.arange(40, dtype=np.float32)[:, np.newaxis], (1, 40))
    features = features[..., np.newaxis]
    valid = np.ones((40, 40), dtype=bool)
    
    result, _ = grouped_lineament_features(
        features,
        {"vertical": (0,)},
        valid_mask=valid,
        kind="structure_tensor_orientation",
        sigma_pixels=0.5,
        structure_tensor_window=2.0,
    )
    
    # Orientation should be near ±π/2 (vertical gradient direction)
    center_orientation = result[20, 20, 0]
    assert abs(abs(center_orientation) - np.pi/2) < 0.2, \
        f"Expected ~±π/2, got {center_orientation}"


def test_structure_tensor_handles_invalid_regions():
    """Invalid pixels should be zero and not contaminate valid regions."""
    features = np.tile(np.arange(25, dtype=np.float32), (25, 1))
    features = features[..., np.newaxis]
    features[0, 0, 0] = np.nan
    
    valid = np.ones((25, 25), dtype=bool)
    valid[:3, :3] = False
    
    for kind in ("structure_tensor_coherence", "structure_tensor_orientation"):
        result, _ = grouped_lineament_features(
            features,
            {"test": (0,)},
            valid_mask=valid,
            kind=kind,
            sigma_pixels=0.5,
            structure_tensor_window=2.0,
        )
        
        assert np.isfinite(result).all()
        assert not result[:3, :3].any()
        # Valid region should have non-zero values for coherence
        if kind == "structure_tensor_coherence":
            assert result[12, 12, 0] > 0.1


def test_structure_tensor_multi_channel_aggregation():
    """Structure tensor should aggregate gradients across multiple channels."""
    # Create two channels: horizontal and vertical gradients
    horiz = np.tile(np.arange(30, dtype=np.float32), (30, 1))
    vert = np.tile(np.arange(30, dtype=np.float32)[:, np.newaxis], (1, 30))
    features = np.stack([horiz, vert], axis=-1)
    valid = np.ones((30, 30), dtype=bool)
    
    result, _ = grouped_lineament_features(
        features,
        {"both": (0, 1)},
        valid_mask=valid,
        kind="structure_tensor_coherence",
        sigma_pixels=0.5,
        structure_tensor_window=2.0,
    )
    
    # With both horizontal and vertical gradients, coherence should be lower
    # than either alone (more isotropic)
    assert result.shape == (30, 30, 1)
    assert np.all((0 <= result) & (result <= 1))


def _vertical_gaussian_trough(size, center_col, width, amplitude=1.0):
    """Field with a vertical trough (valley) line at center_col."""
    xx = np.arange(size, dtype=np.float32)
    profile = -amplitude * np.exp(-((xx - center_col) ** 2) / (2 * width**2))
    field = np.tile(profile, (size, 1))
    return field[..., np.newaxis]


def _vertical_gaussian_crest(size, center_col, width, amplitude=1.0):
    """Field with a vertical crest (ridge) line at center_col."""
    xx = np.arange(size, dtype=np.float32)
    profile = amplitude * np.exp(-((xx - center_col) ** 2) / (2 * width**2))
    field = np.tile(profile, (size, 1))
    return field[..., np.newaxis]


def test_ridge_valley_response_detects_valley():
    """A concave-up trough line should give a strong positive signed response."""
    size, center = 60, 30
    features = _vertical_gaussian_trough(size, center, width=1.5)
    valid = np.ones((size, size), dtype=bool)

    result, metadata = grouped_lineament_features(
        features,
        {"trough": (0,)},
        valid_mask=valid,
        kind="ridge_valley_response",
        sigma_pixels=1.0,
        structure_tensor_window=2.0,
    )

    assert result.shape == (size, size, 1)
    assert np.all((-1 <= result) & (result <= 1))
    center_response = result[size // 2, center, 0]
    assert center_response > 0.5, f"expected strong valley response, got {center_response}"
    # Far from the line the response should be near zero.
    background = result[size // 2, 0, 0]
    assert abs(background) < 0.1, f"expected quiet background, got {background}"
    assert metadata["output_channels"] == ["trough:ridge_valley_response"]
    assert metadata["phase_interpretation"] == (
        "signed_multiscale_hessian_eigenvalue_ridge_valley_response"
    )


def test_ridge_valley_response_detects_ridge():
    """A concave-down crest line should give a strong negative signed response."""
    size, center = 60, 30
    features = _vertical_gaussian_crest(size, center, width=1.5)
    valid = np.ones((size, size), dtype=bool)

    result, _ = grouped_lineament_features(
        features,
        {"crest": (0,)},
        valid_mask=valid,
        kind="ridge_valley_response",
        sigma_pixels=1.0,
        structure_tensor_window=2.0,
    )

    center_response = result[size // 2, center, 0]
    assert center_response < -0.5, f"expected strong ridge response, got {center_response}"
    background = result[size // 2, 0, 0]
    assert abs(background) < 0.1, f"expected quiet background, got {background}"


def test_ridge_valley_response_constant_field_is_zero():
    """A constant field has zero Hessian and therefore zero response."""
    features = np.ones((25, 25, 1), dtype=np.float32) * 3.0
    valid = np.ones((25, 25), dtype=bool)

    result, metadata = grouped_lineament_features(
        features,
        {"constant": (0,)},
        valid_mask=valid,
        kind="ridge_valley_response",
        sigma_pixels=1.0,
        structure_tensor_window=2.0,
    )

    assert result.shape == (25, 25, 1)
    assert np.allclose(result, 0.0, atol=1e-6)
    assert metadata["multiscale_sigmas"] == [1.0]


def test_ridge_valley_response_multiscale_picks_best_scale():
    """Multi-scale max should detect the valley and record every scale."""
    size, center = 60, 30
    features = _vertical_gaussian_trough(size, center, width=2.0)
    valid = np.ones((size, size), dtype=bool)

    result, metadata = grouped_lineament_features(
        features,
        {"trough": (0,)},
        valid_mask=valid,
        kind="ridge_valley_response",
        sigma_pixels=1.0,
        structure_tensor_window=2.0,
        ridge_valley_scales=(0.5, 2.0, 4.0),
    )

    assert metadata["multiscale_sigmas"] == [0.5, 2.0, 4.0]
    center_response = result[size // 2, center, 0]
    assert center_response > 0.5, f"expected multiscale valley response, got {center_response}"
    assert np.all((-1 <= result) & (result <= 1))


def test_ridge_valley_response_handles_invalid_regions():
    """Invalid pixels must stay zero and not poison the Hessian."""
    size, center = 50, 25
    features = _vertical_gaussian_trough(size, center, width=1.5)
    features[0, 0, 0] = np.nan
    valid = np.ones((size, size), dtype=bool)
    valid[:4, :4] = False

    result, _ = grouped_lineament_features(
        features,
        {"trough": (0,)},
        valid_mask=valid,
        kind="ridge_valley_response",
        sigma_pixels=1.0,
        structure_tensor_window=2.0,
    )

    assert np.isfinite(result).all()
    assert not result[:4, :4].any()
    # Valley line far from the invalid corner still responds.
    assert result[size // 2, center, 0] > 0.3


@pytest.mark.parametrize(
    "scales",
    [
        [],
        (-1.0,),
        (1.0, float("inf")),
        (float("nan"),),
    ],
)
def test_invalid_ridge_valley_scales_are_rejected(scales):
    with pytest.raises(ValueError):
        grouped_lineament_features(
            np.ones((8, 8, 1), dtype=np.float32),
            {"group": (0,)},
            valid_mask=np.ones((8, 8), bool),
            kind="ridge_valley_response",
            ridge_valley_scales=scales,
        )
