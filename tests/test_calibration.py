"""Check the optimized event sweep against an independent coordinate oracle."""

import math

import numpy as np
import pytest

from gems.calibration import binary_threshold, exact_threshold_curve
from gems.metric import distance_weighted_tversky


def _oracle(pred, truth, valid, threshold, radius, alpha, beta):
    predicted = np.argwhere(valid & (pred >= threshold))
    positive = np.argwhere(valid & truth)
    tp = sum(
        max((max(1 - np.linalg.norm(p - g) / radius, 0) for p in predicted), default=0)
        for g in positive
    )
    fp = sum(
        1 - max((max(1 - np.linalg.norm(p - g) / radius, 0) for g in positive), default=0)
        for p in predicted
    )
    return tp / (tp + alpha * fp + beta * (len(positive) - tp) + 1e-12)


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.7, 3.0])
@pytest.mark.parametrize("seed", range(4))
def test_every_threshold_matches_independent_oracle_and_existing_metric(radius, seed):
    rng = np.random.default_rng(seed)
    pred = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0], size=(5, 6))
    truth = rng.random(pred.shape) < 0.2
    valid = rng.random(pred.shape) > 0.2
    pred[~valid] = np.nan
    truth[0, 0] = True  # Border cases are intentional.
    curve = exact_threshold_curve(pred, truth, valid_mask=valid, radius_pixels=radius)
    for threshold, score in zip(curve.thresholds, curve.scores, strict=True):
        expected = _oracle(pred, truth, valid, threshold, radius, 0.2, 0.8)
        assert score == pytest.approx(expected, abs=1e-12)
        assert score == pytest.approx(distance_weighted_tversky(
            binary_threshold(pred, threshold), truth, valid_mask=valid, radius_pixels=radius
        ), abs=1e-12)


def test_ties_activate_together_and_higher_threshold_wins_score_ties():
    pred = np.array([[0.8, 0.8], [0.0, 0.0]])
    curve = exact_threshold_curve(pred, np.zeros_like(pred, dtype=bool))
    assert len(curve.thresholds) == 3
    assert curve.best_threshold > pred.max()
    assert not binary_threshold(pred, curve.best_threshold).any()


def test_single_fault_offset_and_duplicate_match_are_not_double_counted():
    pred = np.zeros((7, 7))
    truth = np.zeros_like(pred, dtype=bool)
    truth[3, 3] = True
    pred[3, 4] = pred[4, 3] = 0.9
    curve = exact_threshold_curve(pred, truth)
    assert curve.true_positive_weight[1] == pytest.approx(2 / 3)
    assert curve.false_positive_weight[1] == pytest.approx(2 / 3)


def test_transpose_invariance():
    rng = np.random.default_rng(4)
    pred = rng.random((8, 9))
    truth = rng.random(pred.shape) < 0.1
    first = exact_threshold_curve(pred, truth)
    second = exact_threshold_curve(pred.T, truth.T)
    np.testing.assert_allclose(first.scores, second.scores, atol=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_probabilities_fail_closed(value):
    with pytest.raises(ValueError, match="predictions"):
        exact_threshold_curve(np.array([[value]]), np.array([[True]]))


@pytest.mark.parametrize("parameter", ["alpha", "beta", "radius_pixels", "eps"])
def test_nonfinite_metric_parameters_rejected(parameter):
    with pytest.raises(ValueError, match="finite"):
        exact_threshold_curve(np.zeros((2, 2)), np.zeros((2, 2)), **{parameter: math.nan})


def test_empty_mask_rejected():
    with pytest.raises(ValueError, match="valid pixels"):
        exact_threshold_curve(np.zeros((2, 2)), np.zeros((2, 2)),
                              valid_mask=np.zeros((2, 2), dtype=bool))
