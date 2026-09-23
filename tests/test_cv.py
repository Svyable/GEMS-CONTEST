import numpy as np
from scipy.ndimage import binary_dilation

from gems.cv import (
    assign_fault_components,
    assign_spatial_blocks,
    fault_discovery_fold,
    spatial_fold_masks,
)


def test_spatial_block_assignment_is_balanced_and_deterministic():
    a = assign_spatial_blocks((100, 90), block_size=20, n_folds=5, seed=7)
    b = assign_spatial_blocks((100, 90), block_size=20, n_folds=5, seed=7)
    assert np.array_equal(a, b)
    counts = np.bincount(a.ravel(), minlength=5)
    assert counts.max() - counts.min() <= 1


def test_spatial_buffer_prevents_train_validation_contact():
    assignments = assign_spatial_blocks((20, 20), block_size=10, n_folds=2, seed=2)
    fold = spatial_fold_masks(
        assignments,
        (20, 20),
        fold=0,
        block_size=10,
        buffer_pixels=2,
    )
    dilated_validation = binary_dilation(fold.validation_mask, iterations=2)
    assert not np.any(fold.train_mask & dilated_validation)
    assert not np.any(fold.train_mask & fold.validation_mask)


def test_spatial_valid_mask_is_honored():
    valid = np.ones((12, 12), dtype=bool)
    valid[:2] = False
    assignments = assign_spatial_blocks((12, 12), block_size=6, n_folds=2, seed=0)
    fold = spatial_fold_masks(
        assignments,
        (12, 12),
        fold=0,
        block_size=6,
        valid_mask=valid,
    )
    assert not fold.train_mask[:2].any()
    assert not fold.validation_mask[:2].any()


def _component_labels():
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[2:5, 2] = 1
    labels[2:6, 8] = 1
    labels[8, 2:7] = 1
    labels[10:16, 10] = 1
    labels[14, 2:9] = 1
    labels[3:7, 15] = 1
    return labels


def test_fault_components_are_never_split_between_folds():
    labels = _component_labels()
    folds = assign_fault_components(labels, n_folds=3, seed=11)
    slices = [
        folds[2:5, 2],
        folds[2:6, 8],
        folds[8, 2:7],
        folds[10:16, 10],
        folds[14, 2:9],
        folds[3:7, 15],
    ]
    assert all(np.unique(values).size == 1 for values in slices)


def test_fault_holdout_excludes_withheld_neighborhood_from_training():
    labels = _component_labels()
    folds = assign_fault_components(labels, n_folds=3, seed=4)
    fold_id = int(folds[labels.astype(bool)][0])
    held = fault_discovery_fold(
        labels,
        folds,
        fold=fold_id,
        buffer_pixels=2,
    )
    assert held.validation_truth.any()
    assert not np.any(held.train_truth & held.validation_truth)
    assert not np.any(held.train_valid_mask & binary_dilation(held.validation_truth, iterations=2))
    assert not np.any(held.evaluation_mask & held.train_truth)
