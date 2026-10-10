import numpy as np
import pytest

from gems.evaluation import (
    evaluate_fault_discovery_predictions,
    evaluate_spatial_predictions,
    evaluate_trace_completion_predictions,
)
from gems.metric import distance_weighted_tversky, distance_weighted_tversky_components


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


def test_spatial_pool_retains_distance_matches_across_fold_seam():
    truth = np.zeros((10, 12), dtype=bool)
    truth[4, 5] = True
    folds = np.zeros(truth.shape, dtype=np.int16)
    folds[:, 6:] = 1
    predictions = {f: np.zeros(truth.shape) for f in (0, 1)}
    predictions[1][4, 6] = 1.0
    result = evaluate_spatial_predictions(predictions, truth, folds)
    # The right-fold prediction matches left-fold truth after stitching. Neither
    # independently masked fold score sees this match; their TP sum is zero.
    assert result["macro_mean"] == 0
    assert result["aggregate_score"] == pytest.approx(2 / 3)
    assert result["aggregate_components"]["tp_w"] == pytest.approx(2 / 3)
    assert result["aggregate_components"]["fp_w"] == pytest.approx(1 / 3)
    assert result["aggregate_components"]["fn_w"] == pytest.approx(1 / 3)


def test_real_rasters_reverse_macro_and_pooled_candidate_ranking():
    truth = np.zeros((30, 40), dtype=bool)
    truth[5, 5] = True
    truth[10:19, 30] = True
    folds = np.zeros(truth.shape, dtype=np.int16)
    folds[:, 20:] = 1
    incumbent = evaluate_spatial_predictions(
        {0: truth.astype(float) * 0, 1: truth.astype(float) * 0.7}, truth, folds
    )
    candidate = evaluate_spatial_predictions(
        {0: truth.astype(float) * 0.9, 1: truth.astype(float) * 0.5}, truth, folds
    )
    assert candidate["macro_mean"] > incumbent["macro_mean"]
    assert candidate["aggregate_score"] < incumbent["aggregate_score"]
    from gems.verification import compare_view

    protocol = {
        "schema_version": 2, "aggregation": "stitched_spatial_oof",
        "metric": "distance_weighted_tversky", "alpha": 0.2, "beta": 0.8,
        "radius_pixels": 3.0, "truth_sha256": "a" * 64,
        "fold_map_sha256": "b" * 64, "known_fault_exclusion_pixels": 0,
    }
    incumbent["evaluation_protocol"] = candidate["evaluation_protocol"] = protocol
    comp = compare_view("spatial", incumbent, candidate, trials_before=0)
    assert comp.comparable
    assert comp.delta < 0
    assert not comp.won


def test_aggregate_components_match_scalar_metric_with_invalid_region():
    rng = np.random.default_rng(2026)
    truth = rng.random((12, 14)) > 0.95
    valid = np.ones(truth.shape, dtype=bool)
    valid[:3, :3] = False
    pred = rng.random(truth.shape)
    pred[~valid] = np.nan
    evidence = distance_weighted_tversky_components(pred, truth, valid_mask=valid)
    assert evidence.tp_w + evidence.fn_w == pytest.approx((truth & valid).sum())
    for alpha, beta in ((0.2, 0.8), (0.5, 0.5), (1.0, 0.0)):
        assert evidence.score(alpha, beta) == pytest.approx(
            distance_weighted_tversky(pred, truth, alpha=alpha, beta=beta, valid_mask=valid)
        )


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


def test_trace_scores_endpoints_and_excludes_retained_bodies():
    truth = np.zeros((24, 24), dtype=bool)
    truth[4, 2:12] = truth[18, 2:12] = True
    folds = np.full(truth.shape, -1, dtype=np.int16)
    folds[4, 10:12], folds[18, 10:12] = 0, 1
    predictions = {fold: truth.astype(float) for fold in [0, 1]}
    # Models may predict retained known bodies; they must not count as TP or FP.
    result = evaluate_trace_completion_predictions(predictions, truth, folds)
    assert result["scheme"] == "trace"
    assert result["macro_mean"] == pytest.approx(1)
    assert "aggregate_score" not in result
    assert [item["truth_pixels"] for item in result["folds"]] == [2, 2]
