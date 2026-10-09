"""Endpoint-extension post-processing for fault trace tips and splays.

Roadmap priority #4: skeletonize probability maps, find arc endpoints, estimate
local orientation, and extend probability along that direction to recover plausible
continuations of known faults. With β=0.8, this targets cheap recall on newly
mapped continuations, splays, and parallel strands.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def compute_structure_tensor_orientation(
    skeleton: np.ndarray,
    endpoint: tuple[int, int],
    window_size: int = 15,
) -> tuple[float, float] | None:
    """Estimate local orientation at an endpoint using structure tensor.
    
    Args:
        skeleton: Binary skeleton array
        endpoint: (row, col) endpoint location
        window_size: Local window for orientation estimation
    
    Returns:
        (dy, dx) unit vector along dominant orientation, or None if insufficient support
    """
    row, col = endpoint
    h, w = skeleton.shape
    
    # Extract local window
    r_min = max(0, row - window_size)
    r_max = min(h, row + window_size + 1)
    c_min = max(0, col - window_size)
    c_max = min(w, col + window_size + 1)
    
    window = skeleton[r_min:r_max, c_min:c_max]
    
    if window.sum() < 3:
        return None  # Insufficient skeleton support
    
    # Get skeleton pixels in window
    skel_rows, skel_cols = np.nonzero(window)
    
    if len(skel_rows) < 3:
        return None
    
    # Use PCA on skeleton pixel coordinates
    # Convert to local coordinates relative to endpoint
    local_row = row - r_min
    local_col = col - c_min
    
    centered_rows = skel_rows - local_row
    centered_cols = skel_cols - local_col
    
    # Build covariance matrix
    cov = np.cov(np.vstack([centered_rows, centered_cols]))
    
    # Get principal eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_idx = np.argmax(eigenvalues)
    dy, dx = eigenvectors[:, principal_idx]
    
    # Normalize
    norm = np.sqrt(dy**2 + dx**2)
    if norm < 1e-6:
        return None
    
    dy, dx = dy / norm, dx / norm
    
    # Orient direction away from skeleton body
    # Sample a few pixels back along the skeleton
    test_positions = []
    for offset in [1, 2, 3]:
        test_r = int(row - offset * dy)
        test_c = int(col - offset * dx)
        if 0 <= test_r < h and 0 <= test_c < w:
            test_positions.append(skeleton[test_r, test_c])
    
    if test_positions and np.mean(test_positions) < 0.5:
        # Direction points away from skeleton; flip it
        dy, dx = -dy, -dx
    
    return (dy, dx)


def find_skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Find endpoints (pixels with exactly one neighbor) in a binary skeleton.
    
    Args:
        skeleton: Binary skeleton array
    
    Returns:
        List of (row, col) endpoint coordinates
    """
    endpoints = []
    rows, cols = np.nonzero(skeleton)
    
    for row, col in zip(rows, cols):
        # Count 8-connected neighbors
        neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if (0 <= r < skeleton.shape[0] and 0 <= c < skeleton.shape[1]
                        and skeleton[r, c]):
                    neighbors += 1
        
        # Endpoint has exactly 1 neighbor
        if neighbors == 1:
            endpoints.append((row, col))
    
    return endpoints


def extend_endpoint(
    probabilities: np.ndarray,
    skeleton: np.ndarray,
    endpoint: tuple[int, int],
    direction: tuple[float, float],
    max_distance_pixels: float,
    decay_rate: float = 0.1,
    edge_strength: np.ndarray | None = None,
    edge_threshold: float = 0.3,
    inplace: bool = False,
) -> np.ndarray:
    """Extend probability from an endpoint along a direction with decay.
    
    Args:
        probabilities: Original probability map
        skeleton: Binary skeleton
        endpoint: (row, col) starting point
        direction: (dy, dx) unit vector
        max_distance_pixels: Maximum extension distance
        decay_rate: Exponential decay rate (higher = faster decay)
        edge_strength: Optional edge indicator to gate extension
        edge_threshold: Minimum edge strength to allow extension
        inplace: If True, modify probabilities in-place (avoids O(H*W) copy per endpoint)
    
    Returns:
        Extended probability map (same shape as input)
    """
    # inplace=True avoids an O(H*W) copy per endpoint (identical result).
    extended = probabilities if inplace else probabilities.copy()
    row, col = endpoint
    dy, dx = direction
    h, w = probabilities.shape
    
    # Get base probability at endpoint
    base_prob = probabilities[row, col]
    if base_prob < 0.01:
        return extended  # Don't extend from very low probability
    
    # Extend along direction
    for step in range(1, int(max_distance_pixels) + 1):
        new_row = round(row + step * dy)
        new_col = round(col + step * dx)
        
        # Check bounds
        if not (0 <= new_row < h and 0 <= new_col < w):
            break
        
        # Check if we hit existing skeleton
        if skeleton[new_row, new_col]:
            break
        
        # Check edge strength if provided
        if edge_strength is not None and edge_strength[new_row, new_col] < edge_threshold:
            break  # No edge support; stop extending
        
        # Compute decayed probability
        decay = np.exp(-decay_rate * step)
        new_prob = base_prob * decay
        
        # Only extend if higher than existing probability
        extended[new_row, new_col] = max(extended[new_row, new_col], new_prob)
    
    return extended


def extend_fault_endpoints(
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    max_extension_distance_m: float = 300.0,
    pixel_size_m: float = 100.0,
    decay_rate: float = 0.1,
    edge_strength: np.ndarray | None = None,
    edge_threshold: float = 0.3,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extend fault trace endpoints to recover continuations and splays.
    
    Args:
        probabilities: Float32 probability map in [0, 1]
        threshold: Probability threshold for skeletonization
        max_extension_distance_m: Maximum extension distance in meters
        pixel_size_m: Pixel size in meters (default: 100m for GEMS)
        decay_rate: Exponential decay rate for extended probabilities
        edge_strength: Optional edge indicator (0-1) to gate extension
        edge_threshold: Minimum edge strength to allow extension
        valid_mask: Optional mask of valid pixels (True = valid)
    
    Returns:
        Extended probability map (float32, [0, 1])
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2D")
    
    if not np.all((probs >= 0) & (probs <= 1) | np.isnan(probs)):
        raise ValueError("probabilities must be in [0, 1] or NaN")
    
    # Apply valid mask
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != probs.shape:
            raise ValueError("valid_mask must match probabilities shape")
        probs = np.where(valid, probs, np.nan)
    
    # Threshold and skeletonize
    binary = np.nan_to_num(probs, nan=0.0) > threshold
    
    if not binary.any():
        return probs  # No structure to extend
    
    # Compute skeleton
    skeleton = skeletonize(binary)
    
    # Find endpoints
    endpoints = find_skeleton_endpoints(skeleton)
    
    if not endpoints:
        return probs  # No endpoints
    
    # Convert distance to pixels
    max_distance_pixels = max_extension_distance_m / pixel_size_m
    
    # Extend from each endpoint
    extended = probs.copy()
    
    for endpoint in endpoints:
        # Estimate orientation
        direction = compute_structure_tensor_orientation(skeleton, endpoint)
        
        if direction is None:
            continue  # Cannot estimate orientation
        
        # Extend
        extended = extend_endpoint(
            extended,
            skeleton,
            endpoint,
            direction,
            max_distance_pixels,
            decay_rate=decay_rate,
            edge_strength=edge_strength,
            inplace=True,
            edge_threshold=edge_threshold,
        )
    
    # Restore valid mask
    if valid_mask is not None:
        extended = np.where(valid_mask, extended, np.nan)
    
    return extended.astype(np.float32)
