import numpy as np

from gems.tiling import (
    blend_predictions,
    extract_patch,
    generate_windows,
    tile_starts,
)


def test_tile_starts_cover_axis_and_end_exactly():
    starts = tile_starts(41, patch_size=16, overlap=4)
    assert starts[0] == 0
    assert starts[-1] == 25
    coverage = np.zeros(41, dtype=bool)
    for start in starts:
        coverage[start : start + 16] = True
    assert coverage.all()


def test_overlap_blending_reconstructs_source_values():
    image = np.arange(37 * 41, dtype=np.float32).reshape(37, 41)
    windows = generate_windows(image.shape, patch_size=16, overlap=6)
    patches = [
        extract_patch(image, window, patch_size=16, fill_value=0.0)
        for window in windows
    ]
    reconstructed = blend_predictions(
        image.shape,
        windows,
        patches,
        patch_size=16,
    )
    assert np.allclose(reconstructed, image)


def test_small_raster_is_padded_but_blends_back_to_original_shape():
    image = np.full((7, 9), 0.25, dtype=np.float32)
    windows = generate_windows(image.shape, patch_size=16, overlap=4)
    assert len(windows) == 1
    patch = extract_patch(image, windows[0], patch_size=16, fill_value=0.0)
    assert patch.shape == (16, 16)
    output = blend_predictions(
        image.shape,
        windows,
        [patch],
        patch_size=16,
    )
    assert output.shape == image.shape
    assert np.allclose(output, image)


def test_valid_mask_sets_outside_pixels_to_nan():
    image = np.full((10, 10), 0.7, dtype=np.float32)
    windows = generate_windows(image.shape, patch_size=8, overlap=2)
    patches = [extract_patch(image, window, patch_size=8) for window in windows]
    valid = np.ones(image.shape, dtype=bool)
    valid[:2, :3] = False
    output = blend_predictions(
        image.shape,
        windows,
        patches,
        patch_size=8,
        valid_mask=valid,
    )
    assert np.isnan(output[:2, :3]).all()
    assert np.allclose(output[valid], 0.7)
