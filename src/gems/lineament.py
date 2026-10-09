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


def _symmetric_eigenvalues(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending order) of the 2x2 symmetric matrix [[a, c], [c, b]].

    Computed elementwise: trace = a + b, det = a*b - c**2,
    lambda = (trace +/- sqrt(trace**2 - 4*det)) / 2.
    """
    trace = a + b
    det = a * b - c * c
    discriminant = np.maximum(trace * trace - 4.0 * det, 0.0)
    sqrt_disc = np.sqrt(discriminant)
    lambda_max = (trace + sqrt_disc) / 2.0
    lambda_min = (trace - sqrt_disc) / 2.0
    return lambda_max, lambda_min


def grouped_lineament_features(
    features: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    valid_mask: np.ndarray,
    kind: str = "phase_edge",
    sigma_pixels: float = 1.0,
    phase_epsilon_pixels: float = 1.0,
    structure_tensor_window: float = 3.0,
    ridge_valley_scales: Sequence[float] | None = None,
    steerable_wavelength: float = 8.0,
    steerable_orientations: int = 8,
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

    structure_tensor_coherence computes the coherence (λ1 - λ2) / (λ1 + λ2 + ε)
    from the smoothed structure tensor, where λ1 ≥ λ2 are eigenvalues. High
    coherence indicates a strong directional structure (lineaments), while low
    coherence indicates isotropic or noisy regions. Window size is controlled
    by structure_tensor_window (sigma for gradient-product smoothing).

    structure_tensor_orientation computes the dominant orientation angle
    (in radians, [-π/2, π/2]) from the structure tensor eigenvector. This
    indicates the direction of maximum change perpendicular to the lineament.

    ridge_valley_response computes the Hessian of each smoothed band, takes
    the eigenvalues λmax >= λmin of the cross-channel-averaged Hessian, and
    forms a signed valley-following channel:

        valley = relu(λmax - λmin) * relu(λmax)    (concave-up trough)
        ridge  = relu(λmax - λmin) * relu(-λmin)   (concave-down crest)
        signed = valley - ridge

    The eigenvalue-gap factor suppresses isotropic blobs; the sign term
    distinguishes valleys (fault-parallel topographic troughs, conductivity
    lows) from ridges. With ridge_valley_scales set to a tuple of pixel
    scales, the response is computed at each scale and the valley/ridge terms
    are maxed independently across the pyramid (Frangi-style multi-scale
    detection). The signed channel is normalized by the 99th percentile of
    its magnitude over the valid region (fold-pure: statistics from valid
    only) and clipped to [-1, 1]; 0 means no directional second-order
    structure.

    steerable_filter applies Gabor-style directional filters at multiple
    orientations to detect linear structures at any angle. At each of
    steerable_orientations equally-spaced angles from 0 to π, a complex Gabor
    kernel (Gaussian envelope × sinusoidal carrier at steerable_wavelength)
    convolves with smoothed bands. The maximum magnitude across orientations
    gives the lineament strength independent of fault strike direction,
    normalized to [0,1] by the 99th percentile over the valid region
    (fold-pure). This completes roadmap priority #5 (multi-scale gradient
    and structure-tensor features).
    """
    x = np.asarray(features, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if x.ndim != 3 or valid.shape != x.shape[:2]:
        raise ValueError("features must be HWC and valid_mask must match HW")
    allowed_kinds = {
        "gradient_energy", "phase_edge", "mumford_shah_log",
        "structure_tensor_coherence", "structure_tensor_orientation",
        "ridge_valley_response", "steerable_filter",
    }
    if kind not in allowed_kinds:
        raise ValueError(f"kind must be one of {allowed_kinds}")
    if sigma_pixels < 0 or not np.isfinite(sigma_pixels):
        raise ValueError("sigma_pixels must be finite and nonnegative")
    if phase_epsilon_pixels <= 0 or not np.isfinite(phase_epsilon_pixels):
        raise ValueError("phase_epsilon_pixels must be finite and positive")
    if structure_tensor_window <= 0 or not np.isfinite(structure_tensor_window):
        raise ValueError("structure_tensor_window must be finite and positive")
    if ridge_valley_scales is not None:
        ridge_valley_scales = tuple(float(s) for s in ridge_valley_scales)
        if not ridge_valley_scales:
            raise ValueError("ridge_valley_scales must be non-empty when given")
        if any(s < 0 or not np.isfinite(s) for s in ridge_valley_scales):
            raise ValueError("ridge_valley_scales must be finite and nonnegative")
    if steerable_wavelength <= 0 or not np.isfinite(steerable_wavelength):
        raise ValueError("steerable_wavelength must be finite and positive")
    if steerable_orientations < 2 or steerable_orientations > 32:
        raise ValueError("steerable_orientations must be between 2 and 32")
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
        elif kind in ("structure_tensor_coherence", "structure_tensor_orientation"):
            # Compute structure tensor from smoothed gradients
            # J = [ Jxx Jxy ]  where J_ij = smooth(∂x_i * ∂x_j)
            #     [ Jxy Jyy ]
            jxx = np.zeros(x.shape[:2], dtype=np.float32)
            jyy = np.zeros(x.shape[:2], dtype=np.float32)
            jxy = np.zeros(x.shape[:2], dtype=np.float32)
            
            for index in indices:
                smoothed = _masked_gaussian(x[..., index], valid, sigma_pixels)
                grad_y, grad_x = np.gradient(smoothed)
                gx = grad_x.astype(np.float32, copy=False)
                gy = grad_y.astype(np.float32, copy=False)
                
                # Smooth the gradient products with structure_tensor_window
                jxx += _masked_gaussian(gx * gx, valid, structure_tensor_window)
                jyy += _masked_gaussian(gy * gy, valid, structure_tensor_window)
                jxy += _masked_gaussian(gx * gy, valid, structure_tensor_window)
            
            # Average over channels
            jxx /= np.float32(len(indices))
            jyy /= np.float32(len(indices))
            jxy /= np.float32(len(indices))
            
            # Eigenvalues of 2x2 symmetric matrix:
            # λ = (tr ± sqrt(tr² - 4*det)) / 2
            # where tr = Jxx + Jyy, det = Jxx*Jyy - Jxy²
            trace = jxx + jyy
            det = jxx * jyy - jxy * jxy
            discriminant = np.maximum(trace * trace - 4 * det, 0.0)
            sqrt_disc = np.sqrt(discriminant)
            
            lambda1 = (trace + sqrt_disc) / 2.0  # larger eigenvalue
            lambda2 = (trace - sqrt_disc) / 2.0  # smaller eigenvalue
            
            if kind == "structure_tensor_coherence":
                # Coherence = (λ1 - λ2) / (λ1 + λ2 + ε)
                # Range [0, 1]: 0 = isotropic, 1 = perfectly directional
                epsilon = 1e-6
                derived = (lambda1 - lambda2) / (lambda1 + lambda2 + epsilon)
                derived = np.clip(derived, 0.0, 1.0).astype(np.float32, copy=False)
            else:  # structure_tensor_orientation
                # Orientation angle from eigenvector: θ = 0.5 * atan2(2*Jxy, Jxx - Jyy)
                # This gives the angle of the dominant direction (perpendicular to lineament)
                # Range [-π/2, π/2]
                derived = 0.5 * np.arctan2(2 * jxy, jxx - jyy).astype(np.float32, copy=False)
        elif kind == "ridge_valley_response":
            # Hessian eigenvalue analysis: fault-parallel troughs and crests
            # produce strongly asymmetric second derivatives — large curvature
            # across the lineament, near-zero curvature along it.
            scales = (
                tuple(ridge_valley_scales)
                if ridge_valley_scales is not None
                else (float(sigma_pixels),)
            )
            max_valley = np.zeros(x.shape[:2], dtype=np.float32)
            max_ridge = np.zeros(x.shape[:2], dtype=np.float32)

            for scale in scales:
                hxx = np.zeros(x.shape[:2], dtype=np.float32)
                hyy = np.zeros(x.shape[:2], dtype=np.float32)
                hxy = np.zeros(x.shape[:2], dtype=np.float32)

                for index in indices:
                    smoothed = _masked_gaussian(x[..., index], valid, scale)
                    grad_y, grad_x = np.gradient(smoothed)
                    # Second derivatives: d²/dx², d²/dy², d²/dxdy
                    dxx = np.gradient(grad_x.astype(np.float32, copy=False), axis=1)
                    dyy = np.gradient(grad_y.astype(np.float32, copy=False), axis=0)
                    dxy = np.gradient(grad_x.astype(np.float32, copy=False), axis=0)

                    hxx += _masked_gaussian(dxx, valid, structure_tensor_window)
                    hyy += _masked_gaussian(dyy, valid, structure_tensor_window)
                    hxy += _masked_gaussian(dxy, valid, structure_tensor_window)

                hxx /= np.float32(len(indices))
                hyy /= np.float32(len(indices))
                hxy /= np.float32(len(indices))

                lambda_max, lambda_min = _symmetric_eigenvalues(hxx, hyy, hxy)
                gap = np.maximum(lambda_max - lambda_min, 0.0)
                valley = gap * np.maximum(lambda_max, 0.0)
                ridge = gap * np.maximum(-lambda_min, 0.0)
                max_valley = np.maximum(max_valley, valley)
                max_ridge = np.maximum(max_ridge, ridge)

            signed = max_valley - max_ridge
            # Fold-pure robust scale: 99th percentile over the valid region only.
            valid_signed = np.abs(signed[valid]) if valid.any() else np.array([])
            denom = float(np.percentile(valid_signed, 99.0)) if valid_signed.size else 0.0
            if denom > 0:
                derived = np.clip(signed / denom, -1.0, 1.0).astype(np.float32, copy=False)
            else:
                derived = np.zeros(x.shape[:2], dtype=np.float32)
        elif kind == "steerable_filter":
            # Gabor-style directional filters at multiple orientations
            # Detects linear structures at any angle via max response
            wavelength = float(steerable_wavelength)
            n_orient = int(steerable_orientations)
            angles = np.linspace(0, np.pi, n_orient, endpoint=False)
            
            # Gabor kernel parameters
            gamma = 0.5  # spatial aspect ratio
            sigma = wavelength / (2 * np.pi * 0.56)  # standard Gabor bandwidth
            kernel_size = int(np.ceil(3 * sigma))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_half = kernel_size // 2
            
            # Generate coordinate grids for kernel
            y_k, x_k = np.meshgrid(
                np.arange(-kernel_half, kernel_half + 1, dtype=np.float32),
                np.arange(-kernel_half, kernel_half + 1, dtype=np.float32),
                indexing='ij'
            )
            
            # Accumulate max response across all orientations
            max_response = np.zeros(x.shape[:2], dtype=np.float32)
            
            for angle in angles:
                # Rotate coordinates
                x_rot = x_k * np.cos(angle) + y_k * np.sin(angle)
                y_rot = -x_k * np.sin(angle) + y_k * np.cos(angle)
                
                # Real Gabor kernel: Gaussian envelope × cosine carrier
                gaussian_envelope = np.exp(
                    -(x_rot**2 + (gamma * y_rot)**2) / (2 * sigma**2)
                )
                carrier = np.cos(2 * np.pi * x_rot / wavelength)
                gabor_kernel = (gaussian_envelope * carrier).astype(np.float32)
                # Zero-mean normalization
                gabor_kernel -= gabor_kernel.mean()
                
                # Convolve with each channel and accumulate
                response = np.zeros(x.shape[:2], dtype=np.float32)
                for index in indices:
                    smoothed = _masked_gaussian(x[..., index], valid, sigma_pixels)
                    # scipy.ndimage.convolve for consistency with other filters
                    from scipy.ndimage import convolve
                    band_response = convolve(
                        smoothed,
                        gabor_kernel,
                        mode='nearest',
                    )
                    response += np.abs(band_response)
                response /= float(len(indices))
                
                # Max across orientations for orientation-invariant response
                max_response = np.maximum(max_response, response)
            
            # Normalize by 99th percentile over valid region (fold-pure)
            valid_responses = max_response[valid] if valid.any() else np.array([])
            denom = float(np.percentile(valid_responses, 99.0)) if valid_responses.size else 0.0
            if denom > 1e-6:
                derived = np.clip(max_response / denom, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                # Constant or near-zero field: all responses below threshold, output zeros
                derived = np.zeros(x.shape[:2], dtype=np.float32)
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
    elif kind == "structure_tensor_coherence":
        interpretation = "eigenvalue_coherence_from_smoothed_structure_tensor"
    elif kind == "structure_tensor_orientation":
        interpretation = "dominant_orientation_angle_from_structure_tensor_eigenvector"
    elif kind == "ridge_valley_response":
        interpretation = "signed_multiscale_hessian_eigenvalue_ridge_valley_response"
    elif kind == "steerable_filter":
        interpretation = "gabor_style_directional_filter_max_response_across_orientations"
    else:
        interpretation = None
    
    metadata = {
        "method": "grouped_lineament_v1",
        "kind": kind,
        "sigma_pixels": float(sigma_pixels),
        "phase_epsilon_pixels": float(phase_epsilon_pixels),
        "groups_zero_based": normalized_groups,
        "output_channels": output_names,
        "phase_interpretation": interpretation,
    }
    
    if kind in ("structure_tensor_coherence", "structure_tensor_orientation"):
        metadata["structure_tensor_window"] = float(structure_tensor_window)

    if kind == "ridge_valley_response":
        metadata["structure_tensor_window"] = float(structure_tensor_window)
        metadata["multiscale_sigmas"] = (
            [float(s) for s in ridge_valley_scales]
            if ridge_valley_scales is not None
            else [float(sigma_pixels)]
        )

    if kind == "steerable_filter":
        metadata["steerable_wavelength"] = float(steerable_wavelength)
        metadata["steerable_orientations"] = int(steerable_orientations)

    return stacked, metadata
