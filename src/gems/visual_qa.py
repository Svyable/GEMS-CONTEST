import numpy as np


_PALETTE = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [188, 189, 34],
        [23, 190, 207],
        [127, 127, 127],
    ],
    dtype=np.uint8,
)


def _block_any(mask: np.ndarray, stride: int) -> np.ndarray:
    height, width = mask.shape
    out_h = (height + stride - 1) // stride
    out_w = (width + stride - 1) // stride
    padded = np.zeros((out_h * stride, out_w * stride), dtype=bool)
    padded[:height, :width] = mask
    return padded.reshape(out_h, stride, out_w, stride).any(axis=(1, 3))


def _sample_centers(array: np.ndarray, stride: int) -> np.ndarray:
    height, width = array.shape
    row0 = min(stride // 2, height - 1)
    col0 = min(stride // 2, width - 1)
    sampled = array[row0::stride, col0::stride]
    out_h = (height + stride - 1) // stride
    out_w = (width + stride - 1) // stride
    if sampled.shape == (out_h, out_w):
        return sampled

    result = np.empty((out_h, out_w), dtype=array.dtype)
    for row in range(out_h):
        src_row = min(row * stride + stride // 2, height - 1)
        for col in range(out_w):
            src_col = min(col * stride + stride // 2, width - 1)
            result[row, col] = array[src_row, src_col]
    return result


def preview_stride(shape: tuple[int, int], max_dimension: int) -> int:
    """Choose an integer downsampling stride bounded by max_dimension."""
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape must be positive")
    return max(1, (max(height, width) + max_dimension - 1) // max_dimension)


def fold_preview_rgb(
    fold_map: np.ndarray,
    *,
    scheme: str,
    labels: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    max_dimension: int = 1600,
) -> np.ndarray:
    """Render a compact RGB QA preview for spatial or fault-component folds."""
    folds = np.asarray(fold_map)
    if folds.ndim != 2:
        raise ValueError("fold_map must be 2D")
    if scheme not in {"spatial", "fault"}:
        raise ValueError("scheme must be 'spatial' or 'fault'")

    shape = folds.shape
    if labels is None:
        truth = np.zeros(shape, dtype=bool)
    else:
        truth = np.asarray(labels) > 0
        if truth.shape != shape:
            raise ValueError("labels must match fold_map")

    if valid_mask is None:
        valid = np.ones(shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != shape:
            raise ValueError("valid_mask must match fold_map")

    stride = preview_stride(shape, max_dimension)
    sampled_folds = _sample_centers(folds, stride)
    valid_small = _block_any(valid, stride)
    truth_small = _block_any(truth & valid, stride)

    rgb = np.zeros((*sampled_folds.shape, 3), dtype=np.uint8)
    rgb[valid_small] = np.array([36, 36, 36], dtype=np.uint8)

    nonnegative = sampled_folds >= 0
    for fold in np.unique(sampled_folds[nonnegative]):
        fold_int = int(fold)
        rgb[(sampled_folds == fold) & valid_small] = _PALETTE[fold_int % len(_PALETTE)]

    if scheme == "spatial":
        rgb[truth_small] = np.array([255, 255, 255], dtype=np.uint8)
    else:
        unassigned_truth = truth_small & (sampled_folds < 0)
        rgb[unassigned_truth] = np.array([255, 255, 255], dtype=np.uint8)

    rgb[~valid_small] = 0
    return rgb
