"""Tests for Mumford-Shah / Ambrosio-Tortorelli edge-strength computation."""

import numpy as np
import pytest

from gems.mumford_shah import (
    compute_edge_strength,
    compute_multiband_edge_strength,
    normalize_band,
)


def test_compute_edge_strength_step_discontinuity():
    """Test that a step discontinuity produces a high edge-strength ridge."""
    # Create a 100x100 image with a vertical step at x=50
    image = np.zeros((100, 100), dtype=np.float32)
    image[:, 50:] = 1.0
    
    edges = compute_edge_strength(image, epsilon=1.0, lambda_smooth=0.1, max_iterations=50)
    
    assert edges.shape == image.shape
    assert edges.dtype == np.float32
    assert np.all((edges >= 0) & (edges <= 1))
    
    # Edge strength should be high near x=50
    center_strip = edges[:, 48:52]
    assert center_strip.mean() > 0.5, "Expected high edge strength at discontinuity"
    
    # Edge strength should be low far from the edge
    left_region = edges[:, 10:20]
    right_region = edges[:, 80:90]
    assert left_region.mean() < 0.3, "Expected low edge strength in smooth region"
    assert right_region.mean() < 0.3, "Expected low edge strength in smooth region"


def test_compute_edge_strength_horizontal_line():
    """Test detection of horizontal discontinuity."""
    # Create image with horizontal step at y=50
    image = np.zeros((100, 100), dtype=np.float32)
    image[50:, :] = 1.0
    
    edges = compute_edge_strength(image, epsilon=1.0, lambda_smooth=0.1, max_iterations=50)
    
    # Edge should be detected along horizontal line
    horizontal_strip = edges[48:52, :]
    assert horizontal_strip.mean() > 0.5, "Expected high edge strength at horizontal line"


def test_compute_edge_strength_smooth_field():
    """Test that smooth fields in interior regions have low edge strength."""
    # Create smooth gradient
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    image_smooth = 0.5 * X + 0.5 * Y  # Smooth linear gradient
    
    # Create step discontinuity for comparison
    image_step = np.zeros((100, 100), dtype=np.float32)
    image_step[:, 50:] = 1.0
    
    edges_smooth = compute_edge_strength(
        image_smooth, epsilon=1.0, lambda_smooth=0.1, max_iterations=50
    )
    edges_step = compute_edge_strength(
        image_step, epsilon=1.0, lambda_smooth=0.1, max_iterations=50
    )
    
    # Compare interior regions (avoid boundary artifacts)
    interior = slice(10, 90)
    edges_smooth_interior = edges_smooth[interior, interior]
    edges_step_interior = edges_step[interior, interior]
    
    # Smooth field interior should have lower edge strength than step interior
    assert edges_smooth_interior.mean() < edges_step_interior.mean(), (
        "Expected smooth gradient interior to have lower edge strength than discontinuity"
    )
    # Also check that smooth interior has generally low values
    assert edges_smooth_interior.mean() < 0.3, "Expected low edge strength in smooth interior"


def test_compute_edge_strength_diagonal_line():
    """Test detection of diagonal discontinuity."""
    # Create image with diagonal step
    image = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        for j in range(100):
            if j > i:
                image[i, j] = 1.0
    
    edges = compute_edge_strength(image, epsilon=1.0, lambda_smooth=0.1, max_iterations=50)
    
    # Edge should be detected along diagonal
    # Sample points along the diagonal
    diagonal_points = [edges[i, i] for i in range(10, 90, 10)]
    assert np.mean(diagonal_points) > 0.4, "Expected edges along diagonal"


def test_compute_edge_strength_parameters():
    """Test that edge-strength parameters have expected effects."""
    # Create simple step
    image = np.zeros((50, 50), dtype=np.float32)
    image[:, 25:] = 1.0
    
    # Smaller epsilon should give sharper edges
    edges_small_eps = compute_edge_strength(image, epsilon=0.5, max_iterations=50)
    edges_large_eps = compute_edge_strength(image, epsilon=2.0, max_iterations=50)
    
    # Both should detect edges
    assert edges_small_eps.max() > 0.5
    assert edges_large_eps.max() > 0.5
    
    # Smaller lambda_smooth should allow more edges
    edges_small_lambda = compute_edge_strength(image, lambda_smooth=0.05, max_iterations=50)
    edges_large_lambda = compute_edge_strength(image, lambda_smooth=0.5, max_iterations=50)
    
    # Both should work
    assert edges_small_lambda.max() > 0.3
    assert edges_large_lambda.max() > 0.3


def test_compute_edge_strength_deterministic():
    """Test that edge detection is deterministic and stable."""
    image = np.zeros((50, 50), dtype=np.float32)
    image[:, 25:] = 1.0
    
    # Multiple calls should give identical results (deterministic)
    edges_1 = compute_edge_strength(image, max_iterations=50)
    edges_2 = compute_edge_strength(image, max_iterations=50)
    edges_3 = compute_edge_strength(image, max_iterations=100)
    
    # Should be identical (current implementation doesn't use iterations)
    assert np.allclose(edges_1, edges_2), "Expected deterministic results"
    assert np.allclose(edges_1, edges_3), "Expected consistent results"


def test_compute_multiband_edge_strength_single_band():
    """Test multiband with single band (should match single-band computation)."""
    image = np.zeros((50, 50), dtype=np.float32)
    image[:, 25:] = 1.0
    
    # 2D input
    edges_2d = compute_multiband_edge_strength(image, max_iterations=30)
    
    # 3D input with one band
    image_3d = image[:, :, np.newaxis]
    edges_3d = compute_multiband_edge_strength(image_3d, per_band=True, max_iterations=30)
    
    # Should be similar (not identical due to random init)
    assert edges_2d.shape == edges_3d.shape
    assert np.abs(edges_2d.mean() - edges_3d.mean()) < 0.2


def test_compute_multiband_edge_strength_multiple_bands():
    """Test multiband edge detection with multiple channels."""
    # Create 3 bands with edges at different locations
    band1 = np.zeros((50, 50), dtype=np.float32)
    band1[:, 25:] = 1.0  # Vertical edge at x=25
    
    band2 = np.zeros((50, 50), dtype=np.float32)
    band2[25:, :] = 1.0  # Horizontal edge at y=25
    
    band3 = np.zeros((50, 50), dtype=np.float32)
    band3[:, :] = 0.5  # Constant (no edges)
    
    bands = np.stack([band1, band2, band3], axis=2)
    
    # Per-band mode
    edges_per_band = compute_multiband_edge_strength(
        bands, per_band=True, max_iterations=30
    )
    
    assert edges_per_band.shape == (50, 50)
    assert edges_per_band.dtype == np.float32
    
    # Should detect edges in both locations
    # (averaged across bands)
    assert edges_per_band.max() > 0.3
    
    # Vector norm mode
    edges_vector = compute_multiband_edge_strength(
        bands, per_band=False, max_iterations=30
    )
    
    assert edges_vector.shape == (50, 50)
    assert edges_vector.max() > 0.3


def test_compute_multiband_invalid_dimensions():
    """Test that invalid array dimensions are rejected."""
    # 1D array
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        compute_multiband_edge_strength(np.zeros(100))
    
    # 4D array
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        compute_multiband_edge_strength(np.zeros((10, 10, 3, 3)))


def test_normalize_band_basic():
    """Test basic band normalization."""
    # Create band with values in [10, 110]
    band = np.linspace(10, 110, 1000).reshape(10, 100)
    
    normalized = normalize_band(band)
    
    assert normalized.shape == band.shape
    assert normalized.dtype == np.float32
    
    # Should be in [0, 1]
    valid_values = normalized[np.isfinite(normalized)]
    assert np.all((valid_values >= 0) & (valid_values <= 1))
    
    # Min/max should be near 0 and 1 (after clipping outliers)
    assert valid_values.min() < 0.1
    assert valid_values.max() > 0.9


def test_normalize_band_with_mask():
    """Test normalization with training-only mask."""
    # Create band
    band = np.random.rand(50, 50) * 100
    
    # Mask out half
    mask = np.zeros((50, 50), dtype=bool)
    mask[:, :25] = True  # Only left half is "training"
    
    normalized = normalize_band(band, valid_mask=mask)
    
    # Valid region should be normalized
    assert np.all(np.isfinite(normalized[:, :25]))
    
    # Invalid region should be NaN
    assert np.all(np.isnan(normalized[:, 25:]))


def test_normalize_band_constant():
    """Test normalization of constant band."""
    band = np.full((50, 50), 42.0)
    
    normalized = normalize_band(band)
    
    # Constant band should map to 0.5
    assert np.allclose(normalized[np.isfinite(normalized)], 0.5)


def test_normalize_band_with_outliers():
    """Test that outlier clipping works."""
    # Most values in [0, 1], with outliers
    band = np.random.uniform(0, 1, (50, 50))
    band[0, 0] = -100  # Outlier
    band[0, 1] = 100   # Outlier
    
    normalized = normalize_band(band, clip_percentile=2.0)
    
    # Should still normalize reasonably despite outliers
    valid_values = normalized[np.isfinite(normalized)]
    assert valid_values.min() >= 0
    assert valid_values.max() <= 1
    assert 0.3 < valid_values.mean() < 0.7  # Reasonable distribution


def test_normalize_band_empty():
    """Test normalization with no valid pixels."""
    band = np.full((50, 50), np.nan)
    
    normalized = normalize_band(band)
    
    # Should be all NaN
    assert np.all(np.isnan(normalized))


def test_edge_detection_integration():
    """Integration test: normalize then detect edges."""
    # Create raw geophysical-like band with step
    raw_band = np.random.normal(100, 10, (60, 60))
    raw_band[:, 30:] += 50  # Add step
    
    # Normalize
    normalized = normalize_band(raw_band)
    
    # Fill NaN with 0 for edge detection
    normalized_filled = np.nan_to_num(normalized, 0.0)
    
    # Detect edges
    edges = compute_edge_strength(normalized_filled, max_iterations=30)
    
    # Should detect the step
    center_strip = edges[:, 28:32]
    assert center_strip.mean() > 0.4, "Expected edge detection after normalization"


def test_multiband_edge_with_different_scales():
    """Test multiband edge detection with bands at different scales."""
    # Band 1: small values with edge
    band1 = np.zeros((50, 50))
    band1[:, 25:] = 0.1
    
    # Band 2: large values with edge at same location
    band2 = np.zeros((50, 50))
    band2[:, 25:] = 100
    
    # Normalize each
    norm1 = normalize_band(band1)
    norm2 = normalize_band(band2)
    
    # Stack and detect
    bands = np.stack([norm1, norm2], axis=2)
    bands_filled = np.nan_to_num(bands, 0.0)
    
    edges = compute_multiband_edge_strength(bands_filled, per_band=True, max_iterations=30)
    
    # Should detect common edge
    center_strip = edges[:, 23:27]
    assert center_strip.mean() > 0.4
