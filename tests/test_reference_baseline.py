import numpy as np

from gems.reference_baseline import (
    assemble_nonoverlap_patches,
    make_reference_split,
    normalize_reference_features,
    patchify,
    reference_padding,
)


def test_reference_padding_preserves_notebook_full_patch_quirk():
    assert reference_padding((256, 384), 128) == (128, 128)
    assert reference_padding((257, 385), 128) == (127, 127)


def test_patchify_and_assemble_nonoverlap_round_trip():
    image = np.arange(64).reshape(8, 8)
    patches = patchify(image, (4, 4), 4)
    assert patches.shape == (2, 2, 4, 4)
    reconstructed = assemble_nonoverlap_patches(patches)
    assert np.array_equal(reconstructed, image)


def test_normalization_is_per_channel_and_nan_aware():
    x = np.array(
        [
            [[0.0, 10.0], [1.0, np.nan]],
            [[2.0, 20.0], [4.0, 30.0]],
        ],
        dtype=np.float32,
    )
    normalized = normalize_reference_features(x)
    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 1, 0] == 1.0
    assert normalized[0, 0, 1] == 0.0
    assert normalized[1, 1, 1] == 1.0
    assert np.isnan(normalized[0, 1, 1])


def test_reference_split_is_deterministic_and_holds_out_whole_tiles():
    features = np.ones((16, 16, 5), dtype=np.float32)
    labels = np.zeros((16, 16), dtype=np.float32)
    labels[1, 1] = 1
    labels[1, 9] = 1
    labels[9, 1] = 1
    labels[9, 9] = 1

    first = make_reference_split(
        features,
        labels,
        patch_size=8,
        test_proportion=0.5,
        seed=10,
        train_step=4,
    )
    second = make_reference_split(
        features,
        labels,
        patch_size=8,
        test_proportion=0.5,
        seed=10,
        train_step=4,
    )

    assert first.test_indices == second.test_indices
    assert first.x_test.shape[0] == 2
    assert first.x_test.shape[1:] == (5, 8, 8)
    assert first.y_test.sum() == 2
    assert first.y_train.sum() > 0
