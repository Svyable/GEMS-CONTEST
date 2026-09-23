import numpy as np

from gems.visual_qa import fold_preview_rgb, preview_stride


def test_preview_stride_respects_max_dimension():
    assert preview_stride((100, 80), 100) == 1
    assert preview_stride((401, 200), 200) == 3


def test_spatial_preview_preserves_thin_fault_presence():
    folds = np.zeros((8, 8), dtype=np.int16)
    folds[:, 4:] = 1
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[1, 1] = 1
    rgb = fold_preview_rgb(
        folds,
        scheme="spatial",
        labels=labels,
        max_dimension=4,
    )
    assert rgb.shape == (4, 4, 3)
    assert np.any(np.all(rgb == 255, axis=-1))


def test_fault_preview_colors_assigned_components_and_flags_unassigned_truth():
    folds = np.full((6, 6), -1, dtype=np.int16)
    folds[1, 1] = 0
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1, 1] = 1
    labels[4, 4] = 1
    rgb = fold_preview_rgb(
        folds,
        scheme="fault",
        labels=labels,
        max_dimension=6,
    )
    assert not np.array_equal(rgb[1, 1], np.array([36, 36, 36], dtype=np.uint8))
    assert np.array_equal(rgb[4, 4], np.array([255, 255, 255], dtype=np.uint8))


def test_invalid_region_is_black():
    folds = np.zeros((4, 4), dtype=np.int16)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    rgb = fold_preview_rgb(
        folds,
        scheme="spatial",
        valid_mask=valid,
        max_dimension=4,
    )
    assert np.array_equal(rgb[0, 0], np.zeros(3, dtype=np.uint8))
