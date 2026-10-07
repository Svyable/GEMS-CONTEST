"""Tests for threshold optimization."""

import numpy as np
import pytest

from gems.optimization import (
    calibrate_predictions,
    optimize_threshold_golden,
    optimize_threshold_grid,
)


def _make_sparse_linear_pattern(shape: tuple[int, int], density: float = 0.01) -> np.ndarray:
    """Create a sparse linear pattern resembling fault traces."""
    truth = np.zeros(shape, dtype=bool)
    h, w = shape

    # Diagonal lines
    for i in range(0, h, int(1 / density)):
        if i < h:
            truth[i, min(i, w - 1)] = True
            if i + 1 < h and min(i + 1, w - 1) < w:
                truth[i + 1, min(i + 1, w - 1)] = True

    # Horizontal line
    mid = h // 2
    truth[mid, :] = True

    return truth


def test_optimize_threshold_grid_synthetic():
    """Test grid search on synthetic data."""
    np.random.seed(42)
    shape = (128, 128)
    truth = _make_sparse_linear_pattern(shape, density=0.02)

    # Simulate imperfect predictions with noise
    prediction = truth.astype(np.float32) * 0.8 + np.random.uniform(0, 0.2, shape)
    prediction = np.clip(prediction, 0, 1)

    result = optimize_threshold_grid(
        prediction,
        truth,
        alpha=0.2,
        beta=0.8,
        radius_pixels=3.0,
    )

    assert 0.0 <= result.optimal_threshold <= 1.0
    assert result.optimal_score >= 0.0
    assert result.method == "grid_search"
    assert len(result.scores_by_threshold) > 0

    # With beta=0.8 heavily weighting recall, optimal threshold should be low
    assert result.optimal_threshold < 0.5, "Expected low threshold for high-recall metric"

    # Check that score is better than always predicting zero
    zero_score = result.scores_by_threshold.get(0.0)
    if zero_score is not None:
        assert result.optimal_score >= zero_score


def test_optimize_threshold_grid_custom_thresholds():
    """Test grid search with custom threshold array."""
    np.random.seed(43)
    shape = (64, 64)
    truth = _make_sparse_linear_pattern(shape, density=0.03)
    prediction = truth.astype(np.float32) * 0.6 + np.random.uniform(0, 0.3, shape)
    prediction = np.clip(prediction, 0, 1)

    custom_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    result = optimize_threshold_grid(
        prediction, truth, thresholds=custom_thresholds, radius_pixels=3.0
    )

    assert result.optimal_threshold in custom_thresholds
    assert len(result.scores_by_threshold) == len(custom_thresholds)


def test_optimize_threshold_grid_with_mask():
    """Test grid search with valid mask."""
    np.random.seed(44)
    shape = (100, 100)
    truth = _make_sparse_linear_pattern(shape, density=0.02)
    prediction = truth.astype(np.float32) * 0.7 + np.random.uniform(0, 0.25, shape)
    prediction = np.clip(prediction, 0, 1)

    # Mask out edges
    valid_mask = np.ones(shape, dtype=bool)
    valid_mask[:10, :] = False
    valid_mask[-10:, :] = False
    valid_mask[:, :10] = False
    valid_mask[:, -10:] = False

    result = optimize_threshold_grid(
        prediction, truth, valid_mask=valid_mask, radius_pixels=3.0
    )

    assert 0.0 <= result.optimal_threshold <= 1.0
    assert result.optimal_score >= 0.0


def test_optimize_threshold_grid_progress_callback():
    """Test that progress callback is invoked."""
    np.random.seed(45)
    shape = (64, 64)
    truth = _make_sparse_linear_pattern(shape, density=0.02)
    prediction = truth.astype(np.float32) * 0.5 + 0.2

    callback_invocations = []

    def callback(threshold: float, score: float):
        callback_invocations.append((threshold, score))

    optimize_threshold_grid(
        prediction,
        truth,
        thresholds=np.linspace(0.1, 0.9, 5),
        progress_callback=callback,
        radius_pixels=3.0,
    )

    assert len(callback_invocations) == 5
    assert all(isinstance(t, float) and isinstance(s, float) for t, s in callback_invocations)


def test_optimize_threshold_golden_synthetic():
    """Test golden section search on synthetic data."""
    np.random.seed(46)
    shape = (96, 96)
    truth = _make_sparse_linear_pattern(shape, density=0.025)
    prediction = truth.astype(np.float32) * 0.75 + np.random.uniform(0, 0.2, shape)
    prediction = np.clip(prediction, 0, 1)

    result = optimize_threshold_golden(
        prediction,
        truth,
        alpha=0.2,
        beta=0.8,
        radius_pixels=3.0,
        tolerance=0.01,
    )

    assert 0.0 <= result.optimal_threshold <= 1.0
    assert result.optimal_score >= 0.0
    assert result.method == "golden_section"
    assert len(result.scores_by_threshold) > 0

    # Should converge to a reasonable threshold
    assert result.optimal_threshold < 1.0


def test_optimize_threshold_golden_custom_bracket():
    """Test golden section search with custom bracket."""
    np.random.seed(47)
    shape = (80, 80)
    truth = _make_sparse_linear_pattern(shape, density=0.02)
    prediction = truth.astype(np.float32) * 0.4 + 0.15

    result = optimize_threshold_golden(
        prediction,
        truth,
        bracket=(0.1, 0.6),
        tolerance=0.01,
        radius_pixels=3.0,
    )

    assert 0.1 <= result.optimal_threshold <= 0.6


def test_optimize_threshold_golden_max_iterations():
    """Test that golden search respects max_iterations."""
    np.random.seed(48)
    shape = (64, 64)
    truth = _make_sparse_linear_pattern(shape, density=0.02)
    prediction = truth.astype(np.float32) * 0.5 + 0.2

    result = optimize_threshold_golden(
        prediction,
        truth,
        max_iterations=5,
        tolerance=1e-6,  # tight tolerance that won't be reached
        radius_pixels=3.0,
    )

    # Should terminate due to max_iterations, not tolerance
    assert len(result.scores_by_threshold) <= 12  # 2 + 2*5 evaluations


def test_optimize_threshold_grid_shape_mismatch():
    """Test that shape mismatches are caught."""
    prediction = np.random.rand(64, 64)
    truth = np.random.rand(32, 32) > 0.5

    with pytest.raises(ValueError, match="same-shape"):
        optimize_threshold_grid(prediction, truth)


def test_optimize_threshold_golden_shape_mismatch():
    """Test that shape mismatches are caught."""
    prediction = np.random.rand(64, 64)
    truth = np.random.rand(64, 32) > 0.5

    with pytest.raises(ValueError, match="same-shape"):
        optimize_threshold_golden(prediction, truth)


def test_calibrate_predictions_basic():
    """Test basic prediction calibration."""
    np.random.seed(49)
    prediction = np.random.uniform(0, 1, (50, 50))
    optimal_threshold = 0.3

    calibrated = calibrate_predictions(prediction, optimal_threshold=optimal_threshold)

    assert calibrated.shape == prediction.shape
    assert calibrated.dtype == prediction.dtype
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)

    # Pixels at optimal_threshold should map to ~0.5
    test_val = np.array([[optimal_threshold]])
    result = calibrate_predictions(test_val, optimal_threshold=optimal_threshold)
    assert np.isclose(result[0, 0], 0.5, atol=1e-6)


def test_calibrate_predictions_no_clip():
    """Test calibration without clipping."""
    prediction = np.array([[0.8]])
    optimal_threshold = 0.2  # scaling factor = 0.5/0.2 = 2.5

    calibrated = calibrate_predictions(prediction, optimal_threshold=optimal_threshold, clip=False)

    # 0.8 * 2.5 = 2.0, which exceeds 1.0
    assert calibrated[0, 0] == 2.0


def test_calibrate_predictions_with_scale():
    """Test calibration with additional scaling."""
    prediction = np.array([[0.4]])
    optimal_threshold = 0.4
    scale = 0.8

    calibrated = calibrate_predictions(
        prediction, optimal_threshold=optimal_threshold, scale=scale
    )

    # 0.4 * (0.5/0.4) * 0.8 = 0.4
    assert np.isclose(calibrated[0, 0], 0.4, atol=1e-6)


def test_calibrate_predictions_invalid_threshold():
    """Test that invalid thresholds are rejected."""
    prediction = np.random.rand(10, 10)

    with pytest.raises(ValueError, match="must be in"):
        calibrate_predictions(prediction, optimal_threshold=0.0)

    with pytest.raises(ValueError, match="must be in"):
        calibrate_predictions(prediction, optimal_threshold=1.0)

    with pytest.raises(ValueError, match="must be in"):
        calibrate_predictions(prediction, optimal_threshold=-0.1)

    with pytest.raises(ValueError, match="must be in"):
        calibrate_predictions(prediction, optimal_threshold=1.5)


def test_optimization_workflow_integration():
    """Test a complete optimization workflow."""
    np.random.seed(50)
    shape = (100, 100)

    # Create ground truth with multiple linear structures
    truth = np.zeros(shape, dtype=bool)
    truth[25, :] = True  # horizontal
    truth[:, 40] = True  # vertical
    truth[range(60, 90), range(20, 50)] = True  # diagonal

    # Simulate noisy predictions
    prediction = truth.astype(np.float32) * 0.7
    noise = np.random.uniform(0, 0.25, shape)
    prediction = np.clip(prediction + noise, 0, 1)

    # Optimize with grid search
    grid_result = optimize_threshold_grid(
        prediction, truth, alpha=0.2, beta=0.8, radius_pixels=3.0
    )

    # Optimize with golden section
    golden_result = optimize_threshold_golden(
        prediction, truth, alpha=0.2, beta=0.8, radius_pixels=3.0, tolerance=0.005
    )

    # Both methods should find thresholds in valid range
    assert 0.0 <= grid_result.optimal_threshold <= 1.0
    assert 0.0 <= golden_result.optimal_threshold <= 1.0

    # Both should achieve high scores on this easy synthetic problem
    assert grid_result.optimal_score > 0.5
    assert golden_result.optimal_score > 0.5

    # Calibrate predictions
    calibrated = calibrate_predictions(
        prediction, optimal_threshold=grid_result.optimal_threshold
    )

    assert calibrated.shape == prediction.shape
    assert np.all(np.isfinite(calibrated))
    assert np.all((calibrated >= 0) & (calibrated <= 1))
