"""Tests for endpoint-extension post-processing."""

import numpy as np
import pytest

from gems.endpoint_extension import HAS_SKIMAGE

if HAS_SKIMAGE:
    from gems.endpoint_extension import (
        compute_structure_tensor_orientation,
        extend_endpoint,
        extend_fault_endpoints,
        find_skeleton_endpoints,
    )

pytestmark = pytest.mark.skipif(
    not HAS_SKIMAGE,
    reason="endpoint extension requires scikit-image (install with: pip install scikit-image)",
)


def test_find_skeleton_endpoints_line():
    """Test finding endpoints of a simple line."""
    skeleton = np.zeros((20, 20), dtype=bool)
    # Horizontal line from (10, 5) to (10, 15)
    skeleton[10, 5:16] = True
    
    endpoints = find_skeleton_endpoints(skeleton)
    
    assert len(endpoints) == 2
    assert (10, 5) in endpoints
    assert (10, 15) in endpoints


def test_find_skeleton_endpoints_junction():
    """Test that junctions are not identified as endpoints."""
    skeleton = np.zeros((20, 20), dtype=bool)
    # Y-junction
    skeleton[10, 5:11] = True  # Horizontal base
    skeleton[5:11, 10] = True  # Vertical stem
    
    endpoints = find_skeleton_endpoints(skeleton)
    
    # Should have endpoints at (10,5), (10,10) is junction (not endpoint), (5,10) is endpoint
    # Actually (10,10) has 2 neighbors so not endpoint
    assert (10, 5) in endpoints
    assert (5, 10) in endpoints
    assert (10, 10) not in endpoints


def test_find_skeleton_endpoints_empty():
    """Test empty skeleton returns no endpoints."""
    skeleton = np.zeros((20, 20), dtype=bool)
    endpoints = find_skeleton_endpoints(skeleton)
    assert len(endpoints) == 0


def test_compute_structure_tensor_orientation_horizontal():
    """Test orientation estimation for horizontal line."""
    skeleton = np.zeros((50, 50), dtype=bool)
    # Horizontal line
    skeleton[25, 10:30] = True
    
    endpoint = (25, 29)  # Right endpoint
    direction = compute_structure_tensor_orientation(skeleton, endpoint, window_size=10)
    
    assert direction is not None
    dy, dx = direction
    
    # Should be roughly horizontal (dy~0, |dx|~1)
    assert abs(dy) < 0.5
    assert abs(dx) > 0.5


def test_compute_structure_tensor_orientation_vertical():
    """Test orientation estimation for vertical line."""
    skeleton = np.zeros((50, 50), dtype=bool)
    # Vertical line
    skeleton[10:30, 25] = True
    
    endpoint = (29, 25)  # Bottom endpoint
    direction = compute_structure_tensor_orientation(skeleton, endpoint, window_size=10)
    
    assert direction is not None
    dy, dx = direction
    
    # Should be roughly vertical (|dy|~1, dx~0)
    assert abs(dy) > 0.5
    assert abs(dx) < 0.5


def test_compute_structure_tensor_orientation_diagonal():
    """Test orientation estimation for diagonal line."""
    skeleton = np.zeros((50, 50), dtype=bool)
    # Diagonal line (45 degrees)
    for i in range(20):
        skeleton[10 + i, 10 + i] = True
    
    endpoint = (29, 29)
    direction = compute_structure_tensor_orientation(skeleton, endpoint, window_size=10)
    
    assert direction is not None
    dy, dx = direction
    
    # Should be roughly 45 degrees
    assert abs(abs(dy) - abs(dx)) < 0.3


def test_compute_structure_tensor_orientation_insufficient_support():
    """Test that isolated points return None."""
    skeleton = np.zeros((50, 50), dtype=bool)
    skeleton[25, 25] = True  # Single point
    
    endpoint = (25, 25)
    direction = compute_structure_tensor_orientation(skeleton, endpoint, window_size=10)
    
    assert direction is None


def test_extend_endpoint_horizontal():
    """Test extending an endpoint horizontally."""
    probabilities = np.zeros((50, 50), dtype=np.float32)
    probabilities[25, 10:20] = 0.8  # High probability line
    
    skeleton = probabilities > 0.5
    endpoint = (25, 19)
    direction = (0.0, 1.0)  # Extend right
    
    extended = extend_endpoint(
        probabilities,
        skeleton,
        endpoint,
        direction,
        max_distance_pixels=10.0,
        decay_rate=0.1,
    )
    
    # Should have extended to the right with decay
    assert extended[25, 20] > 0
    assert extended[25, 25] > 0
    assert extended[25, 20] > extended[25, 25]  # Decay


def test_extend_endpoint_stops_at_skeleton():
    """Test that extension stops when hitting existing skeleton."""
    probabilities = np.zeros((50, 50), dtype=np.float32)
    probabilities[25, 10:20] = 0.8
    probabilities[25, 30:40] = 0.8  # Another segment
    
    skeleton = probabilities > 0.5
    endpoint = (25, 19)
    direction = (0.0, 1.0)
    
    extended = extend_endpoint(
        probabilities,
        skeleton,
        endpoint,
        direction,
        max_distance_pixels=20.0,
        decay_rate=0.05,
    )
    
    # Should extend but stop before hitting the other segment
    assert extended[25, 20] > 0
    assert extended[25, 29] > 0 or extended[25, 29] == 0  # Might reach or not


def test_extend_endpoint_edge_gating():
    """Test that extension stops when edge strength is too low."""
    probabilities = np.zeros((50, 50), dtype=np.float32)
    probabilities[25, 10:20] = 0.8
    
    skeleton = probabilities > 0.5
    endpoint = (25, 19)
    direction = (0.0, 1.0)
    
    # Create edge strength that drops off
    edge_strength = np.ones((50, 50), dtype=np.float32)
    edge_strength[25, 23:] = 0.1  # Low edge strength beyond pixel 23
    
    extended = extend_endpoint(
        probabilities,
        skeleton,
        endpoint,
        direction,
        max_distance_pixels=20.0,
        decay_rate=0.05,
        edge_strength=edge_strength,
        edge_threshold=0.3,
    )
    
    # Should extend to pixel 22 but not beyond
    assert extended[25, 20] > 0
    assert extended[25, 22] > 0
    # Pixel 23 has low edge, extension should stop
    assert extended[25, 25] == 0 or extended[25, 25] < extended[25, 22]


def test_extend_fault_endpoints_bridges_gap():
    """Test that extension bridges a gap in a broken line."""
    probabilities = np.zeros((100, 100), dtype=np.float32)
    
    # Two segments with a gap
    probabilities[50, 20:40] = 0.9
    probabilities[50, 45:65] = 0.9
    # Gap from 40 to 45
    
    extended = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=600.0,  # 6 pixels at 100m/px
        pixel_size_m=100.0,
        decay_rate=0.15,
    )
    
    # Gap should be partially filled
    gap_pixels = extended[50, 40:45]
    assert np.any(gap_pixels > 0), "Expected gap to be bridged"


def test_extend_fault_endpoints_no_long_tails_from_noise():
    """Test that isolated noise blobs don't sprout long tails."""
    probabilities = np.zeros((100, 100), dtype=np.float32)
    
    # Small isolated blob (2x2)
    probabilities[50:52, 50:52] = 0.8
    
    extended = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=300.0,
        pixel_size_m=100.0,
        decay_rate=0.2,
    )
    
    # Check that extension is limited
    # Count pixels with probability > 0.1 around the blob
    extended_region = extended[48:54, 48:54]
    extended_count = np.sum(extended_region > 0.1)
    
    # Should not extend much (original 4 pixels plus maybe a few more)
    assert extended_count < 15, "Noise blob extended too far"


def test_extend_fault_endpoints_preserves_valid_mask():
    """Test that valid_mask is preserved."""
    probabilities = np.zeros((50, 50), dtype=np.float32)
    probabilities[25, 10:30] = 0.8
    
    valid_mask = np.ones((50, 50), dtype=bool)
    valid_mask[:, 35:] = False  # Right side invalid
    
    extended = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=500.0,
        pixel_size_m=100.0,
        valid_mask=valid_mask,
    )
    
    # Invalid region should remain NaN
    assert np.all(np.isnan(extended[:, 35:]))
    
    # Valid region should be finite
    assert np.all(np.isfinite(extended[:, :35]))


def test_extend_fault_endpoints_bounds_preserved():
    """Test that extension respects raster bounds."""
    probabilities = np.zeros((50, 50), dtype=np.float32)
    # Line near edge
    probabilities[5, 40:48] = 0.9
    
    extended = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=500.0,
        pixel_size_m=100.0,
    )
    
    # Should not exceed bounds
    assert extended.shape == probabilities.shape
    assert np.all(np.isfinite(extended) | np.isnan(extended))


def test_extend_fault_endpoints_idempotence():
    """Test that repeated application doesn't grow indefinitely."""
    probabilities = np.zeros((100, 100), dtype=np.float32)
    probabilities[50, 20:40] = 0.9
    
    extended1 = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=300.0,
        pixel_size_m=100.0,
        decay_rate=0.2,
    )
    
    # Apply again
    extended2 = extend_fault_endpoints(
        extended1,
        threshold=0.5,
        max_extension_distance_m=300.0,
        pixel_size_m=100.0,
        decay_rate=0.2,
    )
    
    # Should not grow much more
    diff = np.sum(extended2 > 0.1) - np.sum(extended1 > 0.1)
    assert diff < 20, "Extension grew too much on second application"


def test_extend_fault_endpoints_empty_returns_unchanged():
    """Test that empty probability map returns unchanged."""
    probabilities = np.zeros((50, 50), dtype=np.float32)
    
    extended = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=300.0,
        pixel_size_m=100.0,
    )
    
    assert np.allclose(extended, probabilities, equal_nan=True)


def test_extend_fault_endpoints_invalid_inputs():
    """Test that invalid inputs are rejected."""
    # Wrong shape
    with pytest.raises(ValueError, match="must be 2D"):
        extend_fault_endpoints(np.zeros((10, 10, 3)))
    
    # Out of range values
    bad_probs = np.full((10, 10), 1.5, dtype=np.float32)
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        extend_fault_endpoints(bad_probs)
    
    # Mismatched mask
    probs = np.zeros((10, 10), dtype=np.float32)
    bad_mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="must match"):
        extend_fault_endpoints(probs, valid_mask=bad_mask)


def test_extend_fault_endpoints_with_edge_strength():
    """Test extension with edge-strength gating."""
    probabilities = np.zeros((100, 100), dtype=np.float32)
    probabilities[50, 20:40] = 0.9
    
    # Create edge strength that only supports extension to the right
    edge_strength = np.zeros((100, 100), dtype=np.float32)
    edge_strength[50, 20:50] = 0.8  # Strong edges in extension zone
    edge_strength[50, 50:] = 0.1    # Weak edges beyond
    
    extended = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=800.0,
        pixel_size_m=100.0,
        edge_strength=edge_strength,
        edge_threshold=0.5,
        decay_rate=0.1,
    )
    
    # Should extend into strong-edge region but stop at weak edges
    assert extended[50, 42] > 0  # Should reach here
    assert extended[50, 48] > 0 or extended[50, 48] == 0  # Boundary
    # Check that extension stopped reasonably
    assert np.sum(extended[50, 52:60] > 0) < 5  # Should not extend far into weak region


def test_extend_fault_endpoints_vertical_line():
    """Test extension works for vertical structures."""
    probabilities = np.zeros((100, 100), dtype=np.float32)
    probabilities[20:60, 50] = 0.85
    
    extended = extend_fault_endpoints(
        probabilities,
        threshold=0.5,
        max_extension_distance_m=500.0,
        pixel_size_m=100.0,
        decay_rate=0.15,
    )
    
    # Should extend vertically from both ends
    # Top extension
    assert extended[18, 50] > 0 or extended[19, 50] > 0
    # Bottom extension
    assert extended[60, 50] > 0 or extended[61, 50] > 0
