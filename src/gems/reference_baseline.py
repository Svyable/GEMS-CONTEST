from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio


@dataclass(frozen=True)
class ReferenceSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_indices: tuple[tuple[int, int], ...]
    padded_shape: tuple[int, int]


def load_reference_arrays(
    feature_path: str,
    label_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load arrays using the organizer notebook's preprocessing conventions."""
    with rasterio.open(feature_path) as src:
        feature_profile = src.profile.copy()
        features = src.read().astype(np.float32)

    features[features < -1e38] = np.nan
    features = np.moveaxis(features, 0, -1)

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype(np.float32)

    labels[labels < 1] = 0
    return features, labels, feature_profile


def normalize_reference_features(features: np.ndarray) -> np.ndarray:
    """Match the organizer notebook's full-raster per-channel min/max normalization."""
    x = np.asarray(features, dtype=np.float32)
    mins = np.nanmin(x, axis=(0, 1))
    maxs = np.nanmax(x, axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - mins) / (maxs - mins)


def reference_padding(shape: tuple[int, int], patch_size: int) -> tuple[int, int]:
    """Match the notebook exactly, including a full patch pad when divisible."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    dims = np.asarray(shape, dtype=np.int64)
    if dims.size != 2 or np.any(dims <= 0):
        raise ValueError("shape must contain two positive dimensions")
    pad = patch_size - dims % patch_size
    return int(pad[0]), int(pad[1])


def patchify(image: np.ndarray, patch_shape: tuple[int, ...], step: int) -> np.ndarray:
    """NumPy equivalent of skimage.util.view_as_windows used by the notebook."""
    if step <= 0:
        raise ValueError("step must be positive")
    if image.ndim != len(patch_shape):
        raise ValueError("patch_shape dimensionality must match image")
    if any(p <= 0 for p in patch_shape):
        raise ValueError("patch dimensions must be positive")
    windows = np.lib.stride_tricks.sliding_window_view(image, patch_shape)
    leading = tuple(slice(None, None, step) for _ in patch_shape)
    return windows[leading]


def _unpatchify_nonoverlap_2d(patches: np.ndarray) -> np.ndarray:
    rows, cols, patch_h, patch_w = patches.shape
    return patches.transpose(0, 2, 1, 3).reshape(rows * patch_h, cols * patch_w)


def _unpatchify_nonoverlap_3d(patches: np.ndarray) -> np.ndarray:
    if patches.shape[2] != 1:
        raise ValueError("feature patches must span the full channel dimension")
    squeezed = patches[:, :, 0]
    rows, cols, patch_h, patch_w, channels = squeezed.shape
    return squeezed.transpose(0, 2, 1, 3, 4).reshape(
        rows * patch_h,
        cols * patch_w,
        channels,
    )


def assemble_nonoverlap_patches(patches: np.ndarray) -> np.ndarray:
    """Reassemble a non-overlapping 2D patch grid."""
    if patches.ndim != 4:
        raise ValueError("expected patch grid with shape rows, cols, height, width")
    return _unpatchify_nonoverlap_2d(patches)


def make_reference_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    patch_size: int = 128,
    test_proportion: float = 0.5,
    seed: int = 0,
    train_step: int = 32,
    elevation_channel: int = 4,
) -> ReferenceSplit:
    """Reproduce the organizer notebook's positive-patch Monte Carlo split in NumPy."""
    x = np.asarray(features)
    y = np.asarray(labels)
    if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
        raise ValueError("features must be HWC and labels must match HxW")
    if not 0 <= test_proportion <= 1:
        raise ValueError("test_proportion must lie in [0, 1]")
    if not 0 <= elevation_channel < x.shape[-1]:
        raise ValueError("elevation_channel is outside the feature stack")

    pad_y, pad_x = reference_padding(y.shape, patch_size)
    x_pad = np.pad(
        x,
        ((0, pad_y), (0, pad_x), (0, 0)),
        mode="constant",
        constant_values=np.nan,
    )
    y_pad = np.pad(y, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)

    n_features = x.shape[-1]
    x_tiles = patchify(
        x_pad,
        (patch_size, patch_size, n_features),
        patch_size,
    ).copy()
    y_tiles = patchify(y_pad, (patch_size, patch_size), patch_size).copy()

    good_tiles: list[tuple[int, int]] = []
    for row in range(y_tiles.shape[0]):
        for col in range(y_tiles.shape[1]):
            has_fault = bool(np.any(y_tiles[row, col]))
            elevation = x_tiles[row, col, 0, :, :, elevation_channel]
            if has_fault and not np.all(np.isnan(elevation)):
                good_tiles.append((row, col))

    n_test = int(test_proportion * len(good_tiles))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(good_tiles), n_test, replace=False)
    test_indices = tuple(good_tiles[int(i)] for i in chosen)

    x_test = np.zeros((n_test, patch_size, patch_size, n_features), dtype=np.float32)
    y_test = np.zeros((n_test, patch_size, patch_size), dtype=np.float32)
    for index, (row, col) in enumerate(test_indices):
        x_test[index] = x_tiles[row, col]
        y_test[index] = y_tiles[row, col]
        x_tiles[row, col] = 0
        y_tiles[row, col] = 0

    x_recovered = _unpatchify_nonoverlap_3d(x_tiles)
    y_recovered = _unpatchify_nonoverlap_2d(y_tiles)

    x_windows = patchify(
        x_recovered,
        (patch_size, patch_size, n_features),
        train_step,
    )
    y_windows = patchify(y_recovered, (patch_size, patch_size), train_step)

    train_indices: list[tuple[int, int]] = []
    for row in range(y_windows.shape[0]):
        for col in range(y_windows.shape[1]):
            has_fault = bool(np.any(y_windows[row, col]))
            elevation = x_windows[row, col, 0, :, :, elevation_channel]
            if has_fault and not np.all(np.isnan(elevation)):
                train_indices.append((row, col))

    x_train = np.zeros(
        (len(train_indices), patch_size, patch_size, n_features),
        dtype=np.float32,
    )
    y_train = np.zeros((len(train_indices), patch_size, patch_size), dtype=np.float32)
    for index, (row, col) in enumerate(train_indices):
        x_train[index] = x_windows[row, col, 0]
        y_train[index] = y_windows[row, col]

    x_train = np.nan_to_num(x_train).transpose(0, 3, 1, 2)
    x_test = np.nan_to_num(x_test).transpose(0, 3, 1, 2)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)

    return ReferenceSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        test_indices=test_indices,
        padded_shape=y_pad.shape,
    )
