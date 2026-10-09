import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gems.ensemble import (
    EnsembleMember,
    EnsembleWeights,
    apply_ensemble,
    calibrate_ensemble_threshold,
    ensemble_mean,
    ensemble_rank_average,
    fit_lofo_weights,
    load_fold_predictions,
)


def test_ensemble_mean_averages_predictions():
    """Simple mean ensemble returns arithmetic average."""
    fold_predictions_a = {0: np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)}
    fold_predictions_b = {0: np.array([[0.4, 0.6], [0.8, 1.0]], dtype=np.float32)}
    members = [
        EnsembleMember("model_a", fold_predictions_a),
        EnsembleMember("model_b", fold_predictions_b),
    ]

    result = ensemble_mean(members, [0])

    expected = np.array([[0.3, 0.5], [0.7, 0.9]], dtype=np.float32)
    assert result[0].shape == (2, 2)
    assert np.allclose(result[0], expected, atol=1e-6)


def test_ensemble_mean_with_multiple_folds():
    """Mean ensemble handles multiple folds."""
    members = [
        EnsembleMember(
            "model_a",
            {
                0: np.array([[0.2]], dtype=np.float32),
                1: np.array([[0.4]], dtype=np.float32),
            },
        ),
        EnsembleMember(
            "model_b",
            {
                0: np.array([[0.4]], dtype=np.float32),
                1: np.array([[0.6]], dtype=np.float32),
            },
        ),
    ]

    result = ensemble_mean(members, [0, 1])

    assert np.allclose(result[0], [[0.3]], atol=1e-6)
    assert np.allclose(result[1], [[0.5]], atol=1e-6)


def test_rank_average_normalizes_to_ranks():
    """Rank-average converts predictions to normalized ranks before averaging."""
    pred_a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    pred_b = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    members = [
        EnsembleMember("a", {0: pred_a}),
        EnsembleMember("b", {0: pred_b}),
    ]

    result = ensemble_rank_average(members, [0])

    assert result[0].shape == (2, 2)
    flat = result[0].flatten()
    assert all(flat[i] <= flat[i + 1] for i in range(len(flat) - 1))


def test_rank_average_handles_different_scales():
    """Rank-average is robust to scale differences."""
    pred_a = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    pred_b = np.array([[0.0, 50.0, 100.0]], dtype=np.float32)
    members = [
        EnsembleMember("a", {0: pred_a}),
        EnsembleMember("b", {0: pred_b}),
    ]

    result = ensemble_rank_average(members, [0])

    assert result[0].shape == (1, 3)
    assert np.allclose(result[0], [[0.0, 0.5, 1.0]], atol=1e-6)


def test_lofo_weights_for_fold_k_do_not_use_fold_k_labels():
    """LOFO weights: weights for fold k are fitted on other folds only.

    This test proves fold k's weight vector does not depend on fold k's labels.
    We fit weights with original fold-0 labels, then change fold-0 labels to
    all zeros and refit. The weight vector for fold 0 should be identical
    because it was fit on fold 1's score only.
    """
    members = [
        EnsembleMember(
            "model_a",
            {
                0: np.array([[0.8, 0.2], [0.9, 0.1]], dtype=np.float32),
                1: np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            },
        ),
        EnsembleMember(
            "model_b",
            {
                0: np.array([[0.3, 0.4], [0.2, 0.3]], dtype=np.float32),
                1: np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
            },
        ),
    ]

    truth_original = np.array([[True, False], [True, False]], dtype=bool)
    fold_map = np.array([[0, 0], [1, 1]], dtype=np.int16)

    weights_original = fit_lofo_weights(members, [0, 1], truth_original, fold_map)

    truth_modified = np.array([[False, False], [True, False]], dtype=bool)

    weights_modified = fit_lofo_weights(members, [0, 1], truth_modified, fold_map)

    assert np.allclose(weights_original.weights[0], weights_modified.weights[0], atol=1e-9), (
        "LOFO violation: fold 0 weights changed when fold 0 labels changed, "
        "but they should only depend on fold 1"
    )


def test_lofo_weights_single_fold_uses_equal_weights():
    """With only one fold, LOFO falls back to equal weights."""
    members = [
        EnsembleMember("a", {0: np.array([[0.5]], dtype=np.float32)}),
        EnsembleMember("b", {0: np.array([[0.6]], dtype=np.float32)}),
    ]
    truth = np.array([[True]], dtype=bool)
    fold_map = np.array([[0]], dtype=np.int16)

    weights = fit_lofo_weights(members, [0], truth, fold_map)

    assert np.allclose(weights.weights[0], [0.5, 0.5], atol=1e-6)


def test_lofo_weights_single_member_is_one():
    """With a single member, LOFO returns weight of 1.0."""
    members = [EnsembleMember("only", {0: np.array([[0.7]], dtype=np.float32)})]
    truth = np.array([[True]], dtype=bool)
    fold_map = np.array([[0]], dtype=np.int16)

    weights = fit_lofo_weights(members, [0], truth, fold_map)

    assert np.allclose(weights.weights[0], [1.0], atol=1e-9)


def test_weighted_ensemble_applies_lofo_weights():
    """Weighted ensemble uses fold-specific LOFO weights."""
    members = [
        EnsembleMember(
            "a",
            {
                0: np.array([[1.0, 0.0]], dtype=np.float32),
                1: np.array([[0.0, 1.0]], dtype=np.float32),
            },
        ),
        EnsembleMember(
            "b",
            {
                0: np.array([[0.0, 1.0]], dtype=np.float32),
                1: np.array([[1.0, 0.0]], dtype=np.float32),
            },
        ),
    ]
    weights = EnsembleWeights(
        weights={
            0: np.array([0.7, 0.3]),
            1: np.array([0.2, 0.8]),
        },
        method="weighted_mean",
    )

    result = apply_ensemble(members, [0, 1], method="weighted_mean", weights=weights)

    assert np.allclose(result[0], [[0.7, 0.3]], atol=1e-6)
    assert np.allclose(result[1], [[0.8, 0.2]], atol=1e-6)


def test_apply_ensemble_mean_method():
    """apply_ensemble with method='mean' returns arithmetic mean."""
    members = [
        EnsembleMember("a", {0: np.array([[0.2, 0.8]], dtype=np.float32)}),
        EnsembleMember("b", {0: np.array([[0.4, 0.6]], dtype=np.float32)}),
    ]

    result = apply_ensemble(members, [0], method="mean")

    assert np.allclose(result[0], [[0.3, 0.7]], atol=1e-6)


def test_apply_ensemble_rank_average_method():
    """apply_ensemble with method='rank_average' returns rank-averaged predictions."""
    members = [
        EnsembleMember("a", {0: np.array([[0.1, 0.9]], dtype=np.float32)}),
        EnsembleMember("b", {0: np.array([[10.0, 90.0]], dtype=np.float32)}),
    ]

    result = apply_ensemble(members, [0], method="rank_average")

    assert result[0].shape == (1, 2)
    assert result[0][0, 0] < result[0][0, 1]


def test_apply_ensemble_requires_weights_for_weighted_mean():
    """apply_ensemble method='weighted_mean' requires weights argument."""
    members = [EnsembleMember("a", {0: np.array([[0.5]], dtype=np.float32)})]

    with pytest.raises(ValueError, match="weights required"):
        apply_ensemble(members, [0], method="weighted_mean", weights=None)


def test_calibrate_ensemble_threshold_finds_optimal_per_fold():
    """Threshold calibration finds best threshold for each fold."""
    predictions = {
        0: np.array([[0.3, 0.7]], dtype=np.float32),
        1: np.array([[0.2, 0.9]], dtype=np.float32),
    }
    truth = np.array([[False, True]], dtype=bool)
    fold_map = np.array([[0, 1]], dtype=np.int16)

    thresholds = calibrate_ensemble_threshold(predictions, truth, fold_map, radius_pixels=1.0)

    assert 0 in thresholds
    assert 1 in thresholds
    assert 0.3 < thresholds[0] < 0.7
    assert 0.2 < thresholds[1] < 0.9


def test_calibrate_threshold_handles_no_valid_pixels():
    """Threshold calibration returns default 0.5 when fold has no valid pixels."""
    predictions = {0: np.array([[0.5]], dtype=np.float32)}
    truth = np.array([[True]], dtype=bool)
    fold_map = np.array([[0]], dtype=np.int16)
    valid_mask = np.array([[False]], dtype=bool)

    thresholds = calibrate_ensemble_threshold(
        predictions, truth, fold_map, valid_mask=valid_mask
    )

    assert thresholds[0] == 0.5


def test_load_fold_predictions_reads_pattern(tmp_path):
    """load_fold_predictions reads GeoTIFFs using {fold} pattern."""
    for fold in [0, 1]:
        path = tmp_path / f"fold-{fold}.tif"
        data = np.full((5, 5), fold * 0.5, dtype=np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=5,
            height=5,
            count=1,
            dtype="float32",
            crs="EPSG:32611",
            transform=from_origin(0, 5, 1, 1),
        ) as dst:
            dst.write(data, 1)

    pattern = tmp_path / "fold-{fold}.tif"
    predictions = load_fold_predictions(pattern, [0, 1])

    assert 0 in predictions
    assert 1 in predictions
    assert np.allclose(predictions[0], 0.0, atol=1e-6)
    assert np.allclose(predictions[1], 0.5, atol=1e-6)


def test_load_fold_predictions_checks_georeferencing(tmp_path):
    """load_fold_predictions rejects misaligned rasters when reference is given."""
    ref_path = tmp_path / "reference.tif"
    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(0, 10, 1, 1),
    ) as dst:
        dst.write(np.zeros((10, 10), dtype=np.float32), 1)

    fold_path = tmp_path / "fold-0.tif"
    with rasterio.open(
        fold_path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(0, 5, 1, 1),
    ) as dst:
        dst.write(np.zeros((5, 5), dtype=np.float32), 1)

    pattern = tmp_path / "fold-{fold}.tif"
    with pytest.raises(ValueError, match="shape mismatch"):
        load_fold_predictions(pattern, [0], reference_path=ref_path)


def test_ensemble_rejects_mismatched_fold_ids():
    """Ensemble functions reject members with missing folds."""
    members = [
        EnsembleMember("a", {0: np.array([[0.5]], dtype=np.float32)}),
        EnsembleMember("b", {1: np.array([[0.6]], dtype=np.float32)}),
    ]

    with pytest.raises(ValueError, match="has folds.*expected"):
        ensemble_mean(members, [0, 1])


def test_ensemble_rejects_mismatched_shapes():
    """Ensemble functions reject members with inconsistent shapes."""
    members = [
        EnsembleMember("a", {0: np.array([[0.5, 0.6]], dtype=np.float32)}),
        EnsembleMember("b", {0: np.array([[0.7]], dtype=np.float32)}),
    ]

    with pytest.raises(ValueError, match="does not match reference"):
        ensemble_mean(members, [0])


def test_load_fold_predictions_requires_fold_placeholder():
    """load_fold_predictions rejects patterns without {fold}."""
    with pytest.raises(ValueError, match="must contain.*fold"):
        load_fold_predictions("no-placeholder.tif", [0])
