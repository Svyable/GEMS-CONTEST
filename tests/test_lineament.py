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
