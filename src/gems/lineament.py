"""Derived lineament features for physically grouped GEMS raster bands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

from gems.mumford_shah import compute_edge_strength


def raster_category_groups(
    feature_path: str | Path,
    categories: Sequence[str],
    *,
    tag: str = "data_category",
) -> dict[str, tuple[int, ...]]:
    """Return zero-based band indices grouped by a raster metadata tag.

    The official GEMS feature raster carries data_category tags such as
    magnetic_data and gravity_data. Reading those tags avoids brittle
    hard-coded band numbers when constructing derived physical-family features.
    """
    requested = tuple(str(category) for category in categories)
    if not requested:
        raise ValueError("at least one feature category is required")
    if len(set(requested)) != len(requested):
        raise ValueError("feature categories must be unique")

    grouped: dict[str, list[int]] = {category: [] for category in requested}
    with rasterio.open(feature_path) as src:
        for index in range(src.count):
            value = src.tags(index + 1).get(tag)
            if value in grouped:
                grouped[value].append(index)

    missing = [category for category, indices in grouped.items() if not indices]
    if missing:
        raise ValueError(
            "feature raster is missing requested data categories: " + ", ".join(missing)
        )
    return {category: tuple(indices) for category, indices in grouped.items()}


def _masked_gaussian(
    values: np.ndarray,
    valid_mask: np.ndarray,
    sigma_pixels: float,
) -> np.ndarray:
    finite = valid_mask & np.isfinite(values)
    if sigma_pixels == 0:
        return np.where(finite, values, 0.0).astype(np.float32, copy=False)

    weights = gaussian_filter(
        finite.astype(np.float32),
        sigma=sigma_pixels,
        mode="nearest",
    )
    numerator = gaussian_filter(
        np.where(finite, values, 0.0).astype(np.float32, copy=False),
        sigma=sigma_pixels,
        mode="nearest",
    )
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, weights, out=result, where=weights > 1e-6)
    return result


def grouped_lineament_features(
    features: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    valid_mask: np.ndarray,
    kind: str = "phase_edge",
    sigma_pixels: float = 1.0,
    phase_epsilon_pixels: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Create one derived lineament channel per physical feature family.

    gradient_energy is a transparent control: the root-mean-square spatial
    gradient magnitude across all channels in a group.

    phase_edge maps the same group gradient energy through the local
    Ambrosio--Tortorelli fixed-image phase relation

        edge = (4 * epsilon * energy) / (1 + 4 * epsilon * energy)

    after optional Gaussian denoising. This is intentionally not claimed to
    solve the full alternating Mumford--Shah/Ambrosio--Tortorelli variational
    problem; the phase-field diffusion term is omitted to keep the transform
    deterministic, scalable to the full GEMS raster, and suitable for a clean
    ablation against the gradient control.

    mumford_shah_log uses Laplacian of Gaussian (LoG) edge detection on the
    normalized stack (per-band or vector norm across groups). This properly
    distinguishes sharp discontinuities (high second derivative) from smooth
    gradients (zero second derivative), as described in roadmap priority #3.
    """
    x = np.asarray(features, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if x.ndim != 3 or valid.shape != x.shape[:2]:
        raise ValueError("features must be HWC and valid_mask must match HW")
    if kind not in {"gradient_energy", "phase_edge", "mumford_shah_log"}:
        raise ValueError("kind must be 'gradient_energy', 'phase_edge', or 'mumford_shah_log'")
    if sigma_pixels < 0 or not np.isfinite(sigma_pixels):
        raise ValueError("sigma_pixels must be finite and nonnegative")
    if phase_epsilon_pixels <= 0 or not np.isfinite(phase_epsilon_pixels):
        raise ValueError("phase_epsilon_pixels must be finite and positive")
    if not groups:
        raise ValueError("at least one feature group is required")

    outputs: list[np.ndarray] = []
    output_names: list[str] = []
    normalized_groups: dict[str, list[int]] = {}

    for name, raw_indices in groups.items():
        indices = tuple(int(index) for index in raw_indices)
        if not indices:
            raise ValueError(f"feature group {name!r} is empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"feature group {name!r} contains duplicate channels")
        if min(indices) < 0 or max(indices) >= x.shape[-1]:
            raise ValueError(f"feature group {name!r} references a channel outside the stack")

        if kind == "mumford_shah_log":
            # Use Laplacian of Gaussian edge detection on normalized bands
            # First normalize each band to [0, 1] using training region stats
            normalized_stack = []
            for index in indices:
                band = x[..., index]
                valid_values = band[valid & np.isfinite(band)]
                if len(valid_values) > 0:
                    p_low = np.percentile(valid_values, 2.0)
                    p_high = np.percentile(valid_values, 98.0)
                    if p_high > p_low:
                        normalized = (band - p_low) / (p_high - p_low)
                        normalized = np.clip(normalized, 0, 1)
                    else:
                        normalized = np.full_like(band, 0.5)
                else:
                    normalized = np.full_like(band, 0.5)
                # Apply smoothing
                smoothed = _masked_gaussian(normalized, valid, sigma_pixels)
                normalized_stack.append(smoothed)
            
            # Compute vector norm of gradients across all bands
            energy = np.zeros(x.shape[:2], dtype=np.float32)
            for smoothed in normalized_stack:
                grad_y, grad_x = np.gradient(smoothed)
                energy += (
                    grad_x.astype(np.float32, copy=False) ** 2
                    + grad_y.astype(np.float32, copy=False) ** 2
                )
            grad_norm = np.sqrt(energy / len(indices))
            
            # Normalize gradient norm to [0, 1]
            if grad_norm.max() > 0:
                grad_norm = grad_norm / grad_norm.max()
            
            # Apply LoG edge detection
            derived = compute_edge_strength(
                grad_norm,
                epsilon=phase_epsilon_pixels,
                lambda_smooth=0.1,
            )
        else:
            # Original gradient_energy and phase_edge implementations
            energy = np.zeros(x.shape[:2], dtype=np.float32)
            for index in indices:
                smoothed = _masked_gaussian(x[..., index], valid, sigma_pixels)
                grad_y, grad_x = np.gradient(smoothed)
                energy += (
                    grad_x.astype(np.float32, copy=False) ** 2
                    + grad_y.astype(np.float32, copy=False) ** 2
                )
            energy /= np.float32(len(indices))

            if kind == "gradient_energy":
                derived = np.sqrt(energy).astype(np.float32, copy=False)
            else:  # phase_edge
                scaled = np.float32(4.0 * phase_epsilon_pixels) * energy
                derived = scaled / (np.float32(1.0) + scaled)

        derived[~valid] = 0.0
        outputs.append(derived.astype(np.float32, copy=False))
        output_names.append(f"{name}:{kind}")
        normalized_groups[str(name)] = list(indices)

    stacked = np.stack(outputs, axis=-1).astype(np.float32, copy=False)
    
    # Set interpretation metadata
    if kind == "phase_edge":
        interpretation = "local_fixed_image_ambrosio_tortorelli_relation_without_phase_diffusion"
    elif kind == "mumford_shah_log":
        interpretation = "laplacian_of_gaussian_edge_detection_with_vector_norm_across_bands"
    else:
        interpretation = None
    
    return stacked, {
        "method": "grouped_lineament_v1",
        "kind": kind,
        "sigma_pixels": float(sigma_pixels),
        "phase_epsilon_pixels": float(phase_epsilon_pixels),
        "groups_zero_based": normalized_groups,
        "output_channels": output_names,
        "phase_interpretation": interpretation,
    }
