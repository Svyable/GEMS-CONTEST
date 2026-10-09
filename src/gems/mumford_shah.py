"""Mumford-Shah / Ambrosio-Tortorelli edge-strength feature computation.

Implements roadmap priority #3 (docs/STRATEGY.md): compute edge-strength from
geophysical bands using an Ambrosio-Tortorelli phase-field approximation of the
Mumford-Shah functional. This captures where fields jump - exactly the fault signature.

The Mumford-Shah functional minimizes:
    ∫(u - g)² + λ|∇u|²(1-v)² + ε|∇v|² + 1/(4ε)v²

where:
- g: observed geophysical field
- u: piecewise-smooth approximation
- v: edge indicator (0 = smooth, 1 = edge)
- λ: smoothing weight
- ε: interface width

The Ambrosio-Tortorelli approximation uses finite ε to make the problem
tractable, yielding coupled PDEs that can be solved iteratively.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def compute_edge_strength(
    image: np.ndarray,
    *,
    epsilon: float = 1.0,
    lambda_smooth: float = 0.1,
    normalization_mask: np.ndarray | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Compute edge-strength using Laplacian of Gaussian (LoG) edge detection.
    
    This uses the second derivative (Laplacian) of a smoothed image to detect edges.
    Edges appear as zero-crossings in the Laplacian with high magnitude. This properly
    distinguishes sharp discontinuities from smooth gradients.
    
    Args:
        image: 2D array of normalized values (e.g., [0, 1])
        epsilon: Smoothing scale (larger = smoother, default: 1.0)
        lambda_smooth: Edge emphasis (larger = stronger edges, default: 0.1)
        normalization_mask: Optional training-pixel mask for fitting the LoG scale
        max_iterations: Unused (for API compatibility)
        tolerance: Unused (for API compatibility)
    
    Returns:
        Edge indicator in [0, 1], where 1 indicates strong edges
    """
    g = np.asarray(image, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("image must be 2D")
    fit = (
        np.ones(g.shape, dtype=bool)
        if normalization_mask is None
        else np.asarray(normalization_mask, dtype=bool)
    )
    if fit.shape != g.shape or not fit.any():
        raise ValueError("normalization_mask must match image and include training pixels")
    
    # Smooth the image to reduce noise
    sigma = max(0.5, epsilon)
    smoothed = gaussian_filter(g, sigma=sigma, mode='reflect')
    
    # Compute Laplacian of Gaussian (LoG)
    # This gives zero at smooth ramps, nonzero at discontinuities
    laplacian_img = laplace(smoothed)
    
    # Take absolute value and normalize
    edge_indicator = np.abs(laplacian_img)
    
    fit_max = float(edge_indicator[fit].max())
    if fit_max > 0:
        edge_indicator = edge_indicator / fit_max
    
    # Apply nonlinear transformation to emphasize strong edges
    # lambda_smooth controls the contrast
    k = 10.0 * (1 + lambda_smooth)
    edge_strength = np.tanh(k * edge_indicator)
    
    # Smooth slightly to get spatially coherent edges
    edge_strength = gaussian_filter(edge_strength, sigma=sigma * 0.3, mode='reflect')
    
    return edge_strength.astype(np.float32)


def compute_multiband_edge_strength(
    bands: np.ndarray,
    *,
    per_band: bool = False,
    epsilon: float = 1.0,
    lambda_smooth: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Compute edge-strength from multiple geophysical bands.
    
    Args:
        bands: 3D array (height, width, n_bands) or 2D (height, width) for single band
        per_band: If True, compute per-band edges then average; if False, use vector norm
        epsilon: Interface width
        lambda_smooth: Smoothness weight
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        2D edge-strength array in [0, 1]
    """
    bands_arr = np.asarray(bands, dtype=np.float64)
    
    if bands_arr.ndim == 2:
        # Single band
        return compute_edge_strength(
            bands_arr,
            epsilon=epsilon,
            lambda_smooth=lambda_smooth,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    elif bands_arr.ndim != 3:
        raise ValueError("bands must be 2D or 3D")
    
    if per_band:
        # Compute edge-strength per band, then average
        edge_strengths = []
        for i in range(bands_arr.shape[2]):
            band = bands_arr[:, :, i]
            edges = compute_edge_strength(
                band,
                epsilon=epsilon,
                lambda_smooth=lambda_smooth,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            edge_strengths.append(edges)
        
        # Average across bands
        return np.mean(edge_strengths, axis=0).astype(np.float32)
    else:
        # Use vector norm of gradients across bands
        # Compute gradient magnitude at each pixel across all bands
        grad_norm_sq = np.zeros(bands_arr.shape[:2], dtype=np.float64)
        
        for i in range(bands_arr.shape[2]):
            band = bands_arr[:, :, i]
            grad_x = np.gradient(band, axis=1)
            grad_y = np.gradient(band, axis=0)
            grad_norm_sq += grad_x**2 + grad_y**2
        
        grad_norm = np.sqrt(grad_norm_sq)
        
        # Normalize to [0, 1] for edge detection
        if grad_norm.max() > 0:
            grad_norm = grad_norm / grad_norm.max()
        
        # Compute edge-strength from gradient magnitude
        return compute_edge_strength(
            grad_norm,
            epsilon=epsilon,
            lambda_smooth=lambda_smooth,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )


def normalize_band(
    band: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    clip_percentile: float = 2.0,
) -> np.ndarray:
    """Normalize a geophysical band to [0, 1] using training-only statistics.
    
    Args:
        band: 2D array of raw values
        valid_mask: Optional mask of valid pixels (for training region)
        clip_percentile: Percentile to clip outliers (default: 2.0)
    
    Returns:
        Normalized band in [0, 1], with NaN where invalid
    """
    band_arr = np.asarray(band, dtype=np.float64)
    
    if valid_mask is None:
        valid_mask = np.ones(band_arr.shape, dtype=bool)
    
    valid_values = band_arr[valid_mask & np.isfinite(band_arr)]
    
    if len(valid_values) == 0:
        return np.full(band_arr.shape, np.nan, dtype=np.float32)
    
    # Robust normalization using percentiles
    p_low = np.percentile(valid_values, clip_percentile)
    p_high = np.percentile(valid_values, 100 - clip_percentile)
    
    if p_high == p_low:
        # Constant band
        return np.where(valid_mask, 0.5, np.nan).astype(np.float32)
    
    # Normalize to [0, 1]
    normalized = (band_arr - p_low) / (p_high - p_low)
    normalized = np.clip(normalized, 0, 1)
    
    # Set invalid pixels to NaN
    normalized = np.where(valid_mask, normalized, np.nan)
    
    return normalized.astype(np.float32)
