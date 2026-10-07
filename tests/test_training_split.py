import numpy as np
import pytest

from gems.cv import assign_fault_components, assign_trace_endpoints
from gems.preprocessing import normalize_training_features
from gems.samples import training_origins
from gems.training_split import candidate_training_split, masked_training_arrays


def _case():
    truth = np.zeros((64, 64), dtype=bool)
    truth[10, 5:45] = True
    truth[35, 5:45] = True
    valid = np.ones_like(truth)
    return truth, valid


@pytest.mark.parametrize("scheme", ["fault", "trace"])
def test_discovery_inputs_targets_and_sampling_do_not_observe_withheld_pixels(scheme):
    truth, valid = _case()
    folds = (assign_fault_components(truth, n_folds=2) if scheme == "fault" else
             assign_trace_endpoints(truth, n_folds=2, min_pixels=24))
    split = candidate_training_split(truth, folds, scheme=scheme, fold=0,
                                     valid_mask=valid, buffer_pixels=3)
    assert not (split.train_mask & split.evaluation_truth).any()
    assert not (split.train_truth & split.evaluation_truth).any()
    assert not (split.evaluation_mask & (truth & ~split.evaluation_truth)).any()
    features = np.random.default_rng(4).random((*truth.shape, 3)).astype(np.float32)
    poisoned = features.copy()
    poisoned[~split.train_mask] = 1e9
    first, stats = normalize_training_features(features, split.train_mask)
    second, other_stats = normalize_training_features(poisoned, split.train_mask)
    assert stats == other_stats
    x, y, mask = masked_training_arrays(first, split)
    other_x, other_y, other_mask = masked_training_arrays(second, split)
    np.testing.assert_array_equal(x, other_x)
    np.testing.assert_array_equal(y, other_y)
    np.testing.assert_array_equal(mask, other_mask)
    assert (x[~mask] == 0).all() and (y[~mask] == 0).all()
    origins = training_origins(y, valid, patch_size=32, step=16, negative_ratio=1,
                               seed=0, allowed_mask=mask, min_allowed_fraction=0.5)
    assert origins
    assert any(not mask[r:r + 32, c:c + 32].all() for r, c in origins)
    if scheme == "trace":
        # Some of the withheld endpoint's own component body remains supervised.
        row = np.nonzero(split.evaluation_truth)[0][0]
        assert split.train_truth[row].any()


@pytest.mark.parametrize("problem", ["background", "unassigned", "split", "invalid_fold"])
def test_complete_fault_maps_fail_closed(problem):
    truth, valid = _case()
    folds = assign_fault_components(truth, n_folds=2)
    fold = 0
    if problem == "background":
        folds[0, 0] = 0
    elif problem == "unassigned":
        folds[10, 5] = -1
    elif problem == "split":
        folds[10, 5] = 1 - folds[10, 5]
    else:
        fold = 9
    with pytest.raises(ValueError):
        candidate_training_split(truth, folds, scheme="fault", fold=fold,
                                 valid_mask=valid, buffer_pixels=3)


def test_trace_cannot_disguise_whole_component_holdout_as_endpoint_validation():
    truth, valid = _case()
    folds = assign_fault_components(truth, n_folds=2)
    with pytest.raises(ValueError, match="retain a body"):
        candidate_training_split(truth, folds, scheme="trace", fold=0,
                                 valid_mask=valid, buffer_pixels=3)


def test_partial_window_classification_ignores_withheld_truth():
    truth = np.zeros((8, 16), dtype=bool)
    truth[0, 0] = truth[0, 8] = True
    allowed = np.ones_like(truth)
    allowed[0, 8] = False
    origins = training_origins(truth, np.ones_like(truth), patch_size=8, step=8,
                               negative_ratio=0, seed=0, allowed_mask=allowed,
                               min_allowed_fraction=0.5)
    assert origins == ((0, 0),)
