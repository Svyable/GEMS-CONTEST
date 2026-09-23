import numpy as np
import pytest

from gems.evaluation import (
    evaluate_fault_discovery_predictions,
    evaluate_spatial_predictions,
)


def test_spatial_perfect_oof_scores_one():
    truth = np.zeros((12, 12), dtype=bool)
    truth[2, 2] = True
    truth[8, 8] = True
    fold_map = np.zeros((12, 12), dtype=np.int16)
    fold_map[:, 6:] = 1
    predictions = {0: truth.astype(float), 1: truth.astype(float)}
    result = evaluate_spatial_predictions(predictions, truth, fold_map)
    assert result["aggregate_score"] == pytest.approx(1.0, abs=1e-10)
    assert result["macro_mean"] == pytest.approx(1.0, abs=1e-10)
    assert len(result["folds"]) == 2


def test_spatial_oof_allows_nan_outside_each_fold():
    truth = np.zeros((8, 8), dtype=bool)
    truth[2, 2] = True
    truth[2, 6] = True
    fold_map = np.zeros((8, 8), dtype=np.int16)
    fold_map[:, 4:] = 1
    left = np.full((8, 8), np.nan)
    right = np.full((8, 8), np.nan)
    left[:, :4] = truth[:, :4]
    right[:, 4:] = truth[:, 4:]
    result = evaluate_spatial_predictions({0: left, 1: right}, truth, fold_map)
    assert result["aggregate_score"] == pytest.approx(1.0, abs=1e-10)


def _fault_case():
    truth = np.zeros((12, 12), dtype=bool)
    truth[2:4, 2] = True
    truth[8:10, 9] = True
    folds = np.full((12, 12), -1, dtype=np.int16)
    folds[2:4, 2] = 0
    folds[8:10, 9] = 1
    return truth, folds


def test_fault_discovery_perfect_fold_models_score_one():
    truth, folds = _fault_case()
    prediction0 = np.zeros((12, 12), dtype=float)
    prediction1 = np.zeros((12, 12), dtype=float)
    prediction0[2:4, 2] = 1.0
    prediction1[8:10, 9] = 1.0
    result = evaluate_fault_discovery_predictions(
        {0: prediction0, 1: prediction1},
        truth,
        folds,
    )
    assert result["macro_mean"] == pytest.approx(1.0, abs=1e-10)
    assert "aggregate_score" not in result


def test_fault_discovery_requires_all_prediction_folds():
    truth, folds = _fault_case()
    with pytest.raises(ValueError, match="prediction folds"):
        evaluate_fault_discovery_predictions(
            {0: np.zeros((12, 12))},
            truth,
            folds,
        )
