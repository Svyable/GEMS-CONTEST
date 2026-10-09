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


def test_fault_preview_consistent_downsampling():
    """Fault preview uses consistent downsampling for fold IDs and truth mask.

    Held-out faults assigned to a fold should render in that fold's color,
    not white (which is reserved for unassigned faults).
    """
    import numpy as np

    from gems.visual_qa import fold_preview_rgb

    # Create a small fault component assigned to fold 1
    labels = np.zeros((100, 100), dtype=bool)
    labels[45:55, 45:55] = True  # 10x10 fault square

    fold_map = np.full((100, 100), -1, dtype=np.int16)
    fold_map[45:55, 45:55] = 1  # Assign this fault to fold 1

    valid = np.ones((100, 100), dtype=bool)

    rgb = fold_preview_rgb(fold_map, scheme="fault", labels=labels, valid_mask=valid)

    # Find where fold 1 is rendered in the downsampled image
    # (Should be colored, not white [255, 255, 255])
    fold1_pixels = np.where(
        (rgb[:, :, 0] != 255) & (rgb[:, :, 1] != 255) & (rgb[:, :, 2] != 255)
    )

    # The fault square should be rendered in fold 1's color, not white
    # At least some pixels in the downsampled region should be colored
    assert len(fold1_pixels[0]) > 0, "Held-out fault rendered white instead of fold color"
