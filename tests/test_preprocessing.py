import numpy as np
import pytest

from gems.preprocessing import normalize_training_features


def test_heldout_and_buffer_extremes_cannot_change_training_scaling():
    x = np.array([0, 2, 4, 100, 200, 300], dtype=np.float32).reshape(2, 3, 1)
    fit = np.array([[True, True, True], [False, False, False]])
    baseline, stats = normalize_training_features(x, fit)
    x[~fit] = -10000
    changed, changed_stats = normalize_training_features(x, fit)
    np.testing.assert_array_equal(baseline[fit], changed[fit])
    np.testing.assert_array_equal(baseline[fit, 0], [0, 0.5, 1])
    assert stats == changed_stats
    assert baseline[1, 0, 0] == 25  # no clipping/refitting on holdout


def test_missing_and_constant_channels_remain_finite():
    x = np.array([[[2, 1], [2, np.nan]], [[99, 3], [np.inf, 5]]], dtype=np.float32)
    fit = np.array([[True, True], [False, False]])
    result, stats = normalize_training_features(x, fit)
    assert np.isfinite(result).all()
    assert not result.any()
    assert stats["constant_channels"] == [0, 1]
    assert stats["finite_fit_pixels_per_channel"] == [2, 1]


def test_per_channel_ranges_use_only_finite_training_values():
    x = np.array([[[1, 10], [3, 20]], [[np.nan, np.inf], [5, 30]]])
    result, stats = normalize_training_features(x, np.ones((2, 2), bool))
    np.testing.assert_array_equal(result[0, 1], [0.5, 0.5])
    np.testing.assert_array_equal(result[1, 0], [0, 0])
    assert stats["minimum"] == [1, 10]
    assert stats["maximum"] == [5, 30]


@pytest.mark.parametrize("mask", [np.zeros((2, 2), bool), np.ones((3, 2), bool)])
def test_invalid_fit_mask_rejected(mask):
    with pytest.raises(ValueError):
        normalize_training_features(np.ones((2, 2, 1)), mask)


def test_heldout_finite_values_cannot_rescue_missing_training_channel():
    x = np.array([[[np.nan], [np.inf]], [[1], [2]]])
    with pytest.raises(ValueError, match="no finite training"):
        normalize_training_features(x, np.array([[True, True], [False, False]]))
