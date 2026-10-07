import numpy as np
import pytest

from gems.robustness import NEIGHBOR_SHIFTS, shift_feature_channels, windows_intersecting_mask
from gems.tiling import Window


def test_neighbor_shifts_are_all_one_pixel_neighbors():
    assert len(NEIGHBOR_SHIFTS) == 8
    assert len(set(NEIGHBOR_SHIFTS)) == 8
    assert (0, 0) not in NEIGHBOR_SHIFTS
    assert all(max(abs(row), abs(col)) == 1 for row, col in NEIGHBOR_SHIFTS)


def test_shift_selected_channel_without_wraparound():
    features = np.zeros((3, 4, 2), dtype=np.float32)
    features[..., 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
    features[..., 1] = 99
    valid = np.ones((3, 4), bool)

    shifted = shift_feature_channels(
        features,
        [0],
        row_offset=1,
        col_offset=-1,
        valid_mask=valid,
        fill_value=-5,
    )

    np.testing.assert_array_equal(shifted[..., 1], features[..., 1])
    assert shifted[1, 0, 0] == features[0, 1, 0]
    assert shifted[2, 2, 0] == features[1, 3, 0]
    assert np.all(shifted[0, :, 0] == -5)
    assert np.all(shifted[:, 3, 0] == -5)


def test_invalid_source_pixels_become_fill_value():
    features = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    valid = np.ones((3, 3), bool)
    valid[1, 1] = False
    shifted = shift_feature_channels(
        features,
        [0],
        row_offset=0,
        col_offset=1,
        valid_mask=valid,
        fill_value=0,
    )
    assert shifted[1, 2, 0] == 0
    assert shifted[1, 1, 0] == features[1, 0, 0]


def test_windows_intersecting_mask_filters_to_evaluation_area():
    windows = (
        Window(0, 0, 4, 4),
        Window(0, 4, 4, 4),
        Window(4, 0, 4, 4),
        Window(4, 4, 4, 4),
    )
    mask = np.zeros((8, 8), bool)
    mask[5, 6] = True
    assert windows_intersecting_mask(windows, mask) == (windows[3],)


@pytest.mark.parametrize(
    "channels,row_offset,col_offset",
    [
        ([], 0, 1),
        ([2], 0, 1),
        ([0, 0], 1, 0),
    ],
)
def test_invalid_channel_selection_is_rejected(channels, row_offset, col_offset):
    with pytest.raises(ValueError):
        shift_feature_channels(
            np.zeros((3, 3, 2), dtype=np.float32),
            channels,
            row_offset=row_offset,
            col_offset=col_offset,
            valid_mask=np.ones((3, 3), bool),
        )
