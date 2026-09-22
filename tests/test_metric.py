import numpy as np
import pytest

from gems.metric import distance_weighted_tversky


def test_perfect_prediction_scores_one():
    truth = np.zeros((9, 9), dtype=bool)
    truth[4, 4] = True
    pred = truth.astype(float)
    assert distance_weighted_tversky(pred, truth) == pytest.approx(1.0, abs=1e-10)


def test_three_pixel_offset_gets_no_credit_at_support_boundary():
    truth = np.zeros((9, 9), dtype=bool)
    truth[4, 4] = True
    pred = np.zeros((9, 9), dtype=float)
    pred[4, 7] = 1.0
    assert distance_weighted_tversky(pred, truth) == pytest.approx(0.0, abs=1e-12)


def test_one_pixel_offset_gets_partial_credit():
    truth = np.zeros((9, 9), dtype=bool)
    truth[4, 4] = True
    pred = np.zeros((9, 9), dtype=float)
    pred[4, 5] = 1.0
    score = distance_weighted_tversky(pred, truth)
    assert 0.0 < score < 1.0


def test_mask_excludes_predictions():
    truth = np.zeros((7, 7), dtype=bool)
    pred = np.zeros((7, 7), dtype=float)
    pred[0, 0] = 1.0
    valid = np.ones((7, 7), dtype=bool)
    valid[0, 0] = False
    assert distance_weighted_tversky(pred, truth, valid_mask=valid) == pytest.approx(0.0)


def test_rejects_out_of_range_probabilities():
    pred = np.zeros((3, 3), dtype=float)
    pred[1, 1] = 1.1
    truth = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        distance_weighted_tversky(pred, truth)
