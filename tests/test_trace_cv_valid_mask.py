"""Tests for trace CV valid mask handling.

Verifies that assign_trace_endpoints correctly intersects labels with the feature
valid mask before assigning trace bodies/endpoints, preventing withheld endpoints
from falling outside the valid mask with no retained trace body.
"""
from __future__ import annotations

import numpy as np
import pytest

from gems.cv import assign_trace_endpoints, trace_completion_fold


def test_trace_endpoints_respect_valid_mask():
    """Withheld endpoints must fall within the feature valid mask."""
    # Create a 50x50 raster with multiple horizontal traces
    labels = np.zeros((50, 50), dtype=bool)
    
    # Create 6 horizontal traces (enough for 5 folds)
    for row in [5, 10, 15, 20, 25, 30]:
        labels[row, 5:35] = True  # 30-pixel traces
    
    # Add one isolated label pixel outside valid mask at (45, 45)
    labels[45, 45] = True
    
    # Valid mask excludes bottom-right corner
    valid = np.ones((50, 50), dtype=bool)
    valid[40:, 40:] = False  # Corner pixels invalid
    
    # Assign endpoints with valid_mask
    pixel_folds = assign_trace_endpoints(
        labels,
        n_folds=5,
        seed=42,
        min_pixels=20,
        endpoint_fraction=0.3,
        valid_mask=valid,
    )
    
    # Verify: no fold assignments outside valid mask
    assert np.all(pixel_folds[~valid] == -1), \
        "Fold assignments must not appear outside valid mask"
    
    # Verify: withheld endpoints exist (fold >= 0)
    withheld = pixel_folds >= 0
    assert withheld.sum() > 0, "Should have some withheld endpoints"
    
    # Verify: all withheld endpoints are within valid mask
    assert np.all(valid[withheld]), \
        "All withheld endpoints must be within valid mask"


def test_trace_fold_with_invalid_label_pixel():
    """Reproduce the fold 1 crash: label pixel outside valid mask.
    
    The crash occurred when a withheld endpoint pixel lay outside the feature
    valid mask, leaving no retained trace body for training. The fix ensures
    assign_trace_endpoints only considers label pixels within the valid mask.
    """
    # Create scenario similar to the real crash:
    # - Multiple trace components, one with a pixel outside valid mask
    # - That pixel would be excluded from endpoint assignment
    # - trace_completion_fold should not crash
    
    labels = np.zeros((50, 50), dtype=bool)
    # Create 6 horizontal traces (enough for 5 folds)
    for row in [5, 10, 15, 20, 25, 30]:
        labels[row, 5:30] = True  # 25-pixel traces
    
    # Valid mask excludes the rightmost pixel of one trace
    valid = np.ones((50, 50), dtype=bool)
    valid[10, 29] = False  # Last pixel of one trace is invalid
    valid[15, 28:30] = False  # Last 2 pixels of another trace invalid
    
    # Without fix, if pixels at (10,29) or (15,28:30) became withheld endpoints,
    # the fold would have validation_truth but potentially insufficient train_truth.
    # With the fix, these invalid pixels are excluded from component detection.
    
    # Assign with valid_mask (the fix)
    pixel_folds = assign_trace_endpoints(
        labels,
        n_folds=5,
        seed=42,
        min_pixels=15,
        endpoint_fraction=0.3,
        valid_mask=valid,
    )
    
    # Verify: no fold assignments on invalid pixels
    assert np.all(pixel_folds[~valid] == -1), \
        "Fold assignments must not appear on invalid pixels"
    
    # Find which folds were assigned
    assigned_folds = np.unique(pixel_folds[pixel_folds >= 0])
    assert len(assigned_folds) > 0, "Should have assigned at least one fold"
    
    # Test each fold - none should crash
    for fold_id in assigned_folds:
        fold = trace_completion_fold(
            labels,
            pixel_folds,
            fold=fold_id,
            buffer_pixels=3,
            valid_mask=valid,
        )
        
        # Verify the fold is well-formed:
        # 1. Withheld endpoints exist and are within valid mask
        assert fold.validation_truth.sum() > 0, \
            f"Fold {fold_id}: should have withheld endpoints"
        assert np.all(valid[fold.validation_truth]), \
            f"Fold {fold_id}: withheld endpoints must be within valid mask"
        
        # 2. Training truth (retained trace body) exists
        assert fold.train_truth.sum() > 0, \
            f"Fold {fold_id}: should have retained trace body for training"
        
        # 3. Training truth is within valid mask
        assert np.all(valid[fold.train_truth]), \
            f"Fold {fold_id}: training truth must be within valid mask"
        
        # 4. No overlap between withheld and training truth
        assert not np.any(fold.validation_truth & fold.train_truth), \
            f"Fold {fold_id}: withheld and training truth must be disjoint"


def test_assign_trace_endpoints_without_valid_mask_backward_compat():
    """When valid_mask is not provided, behavior is unchanged (all labels considered)."""
    labels = np.zeros((50, 50), dtype=bool)
    # Create 6 horizontal traces
    for row in [5, 10, 15, 20, 25, 30]:
        labels[row, 5:35] = True  # 30-pixel traces
    
    # Without valid_mask (backward compatibility)
    pixel_folds = assign_trace_endpoints(
        labels,
        n_folds=5,
        seed=42,
        min_pixels=20,
        endpoint_fraction=0.3,
    )
    
    # Should still assign endpoints
    assert (pixel_folds >= 0).sum() > 0, "Should have withheld endpoints"


def test_assign_trace_endpoints_valid_mask_shape_mismatch():
    """valid_mask shape must match labels shape."""
    labels = np.zeros((20, 20), dtype=bool)
    labels[5:15, 5:15] = True
    
    valid = np.ones((20, 25), dtype=bool)  # Wrong shape
    
    with pytest.raises(ValueError, match="valid_mask must match labels shape"):
        assign_trace_endpoints(
            labels,
            n_folds=5,
            seed=42,
            min_pixels=10,
            endpoint_fraction=0.3,
            valid_mask=valid,
        )


def test_trace_endpoints_all_labels_outside_valid_mask():
    """When all label pixels are outside valid mask, no components remain."""
    labels = np.zeros((20, 20), dtype=bool)
    labels[10:15, 10:15] = True  # 5x5 block of labels
    
    # Valid mask excludes all label pixels
    valid = np.ones((20, 20), dtype=bool)
    valid[10:15, 10:15] = False
    
    # Should raise because no valid components remain for n_folds
    with pytest.raises(ValueError, match="need at least .* endpoint segments"):
        assign_trace_endpoints(
            labels,
            n_folds=5,
            seed=42,
            min_pixels=2,
            endpoint_fraction=0.3,
            valid_mask=valid,
        )
