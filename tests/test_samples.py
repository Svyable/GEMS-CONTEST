import numpy as np

from gems.samples import training_origins


def test_training_origins_keep_every_positive_window_and_cap_negatives():
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[4:8, 4:20] = 1
    valid = np.ones_like(labels, dtype=bool)

    origins = training_origins(
        labels,
        valid,
        patch_size=16,
        step=8,
        negative_ratio=1.0,
        seed=7,
    )
    positive = [origin for origin in origins if labels[origin[0] : origin[0] + 16, origin[1] : origin[1] + 16].any()]
    negative = [origin for origin in origins if origin not in positive]

    assert positive
    assert len(negative) == len(positive)
    assert all(
        not labels[row : row + 16, col : col + 16].any() for row, col in negative
    )
    again = training_origins(
        labels,
        valid,
        patch_size=16,
        step=8,
        negative_ratio=1.0,
        seed=7,
    )
    assert origins == again


def test_training_origins_skip_windows_outside_the_study_area():
    labels = np.zeros((24, 24), dtype=np.uint8)
    labels[0, 0] = 1
    valid = np.zeros_like(labels, dtype=bool)
    valid[:, 12:] = True

    origins = training_origins(
        labels,
        valid,
        patch_size=8,
        step=8,
        negative_ratio=0.0,
        seed=0,
    )
    assert origins == ()
