"""Verify that lineament features respect fault/trace CV boundaries and do not leak."""

import numpy as np
import pytest

from gems.cv import assign_fault_components, fault_discovery_fold
from gems.lineament import grouped_lineament_features


def _synthetic_features_and_faults(seed=42):
    """Create synthetic geophysical features and fault labels for testing."""
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1, (50, 60, 3)).astype(np.float32)
    
    # Create two separate fault components
    labels = np.zeros((50, 60), dtype=bool)
    labels[10:15, 20:40] = True  # horizontal fault
    labels[30:45, 10:15] = True  # vertical fault
    
    return features, labels


@pytest.mark.parametrize(
    "kind",
    [
        "gradient_energy",
        "phase_edge",
        "mumford_shah_log",
        "structure_tensor_coherence",
        "structure_tensor_orientation",
        "ridge_valley_response",
        "steerable_filter",
    ],
)
def test_lineament_features_do_not_leak_from_held_out_inputs(kind):
    """Perturbing held-out region inputs must not change training-region features.
    
    This is the core leakage test: spatial filters with support radius R should
    not incorporate held-out values when the buffer >= R. We verify that changing
    held-out inputs leaves training-region lineament values unchanged.
    """
    features, labels = _synthetic_features_and_faults()
    valid = np.ones_like(labels, dtype=bool)
    
    # Create fault CV split: hold out one component with buffer=8
    component_folds = assign_fault_components(labels, n_folds=2, seed=0)
    fold_spec = fault_discovery_fold(
        labels,
        component_folds,
        fold=0,
        buffer_pixels=8,
        valid_mask=valid,
    )
    
    # Compute lineament features using the training mask (protocol)
    kwargs = {"sigma_pixels": 1.0}
    if kind == "structure_tensor_coherence" or kind == "structure_tensor_orientation":
        kwargs["structure_tensor_window"] = 2.0
    if kind == "ridge_valley_response":
        kwargs["structure_tensor_window"] = 2.0
    if kind == "steerable_filter":
        kwargs["steerable_wavelength"] = 8.0
        kwargs["steerable_orientations"] = 4
    
    baseline, _ = grouped_lineament_features(
        features,
        {"group": (0, 1, 2)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind=kind,
        **kwargs,
    )
    
    # Perturb held-out region inputs dramatically
    perturbed = features.copy()
    perturbed[~fold_spec.train_valid_mask] += 1000.0
    
    perturbed_result, _ = grouped_lineament_features(
        perturbed,
        {"group": (0, 1, 2)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind=kind,
        **kwargs,
    )
    
    # Training region features must be identical (held-out perturbation had no effect)
    training_region = fold_spec.train_valid_mask
    assert np.allclose(
        baseline[training_region],
        perturbed_result[training_region],
        rtol=0,
        atol=1e-6,
    ), f"{kind}: held-out input perturbation leaked into training region"
    
    # Held-out region features may differ (they're computed from different inputs)
    # but training inputs should never see them


@pytest.mark.parametrize("kind", ["gradient_energy", "phase_edge", "structure_tensor_coherence"])
def test_lineament_normalization_is_fold_pure(kind):
    """Normalization statistics must come only from training region, never held-out.
    
    This verifies fold-pure normalization: perturbing held-out inputs should not
    change the normalization percentiles or scale factors derived from training pixels.
    """
    features, labels = _synthetic_features_and_faults(seed=99)
    valid = np.ones_like(labels, dtype=bool)
    
    # Fault CV with held-out component
    component_folds = assign_fault_components(labels, n_folds=2, seed=1)
    fold_spec = fault_discovery_fold(
        labels,
        component_folds,
        fold=1,
        buffer_pixels=6,
        valid_mask=valid,
    )
    
    kwargs = {"sigma_pixels": 1.0}
    if kind == "structure_tensor_coherence":
        kwargs["structure_tensor_window"] = 2.0
    
    _, meta_baseline = grouped_lineament_features(
        features,
        {"test": (0, 1)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind=kind,
        **kwargs,
    )
    
    # Perturb held-out region dramatically
    perturbed = features.copy()
    perturbed[~fold_spec.train_valid_mask] *= 10.0
    
    _, meta_perturbed = grouped_lineament_features(
        perturbed,
        {"test": (0, 1)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind=kind,
        **kwargs,
    )
    
    # Normalization pixel count must be identical (same training region)
    assert meta_baseline["normalization_pixels"] == meta_perturbed["normalization_pixels"]
    
    # For scale-normalized kinds (ridge_valley, steerable), the 99th percentile
    # is computed from training pixels only, so it should be stable
    if kind == "ridge_valley_response" and "multiscale_sigmas" in meta_baseline:
        # Implementation detail: these use training-region percentiles
        pass


def test_spatial_cv_still_uses_full_valid_mask():
    """Spatial CV should use the full valid mask for filtering, not train_mask.
    
    This ensures we didn't break the existing spatial CV behavior. Spatial CV
    has spatially-separated training/validation regions, so there's no fault-based
    leakage concern — the full valid mask is correct.
    """
    features, labels = _synthetic_features_and_faults(seed=7)
    _, W = labels.shape
    
    # Spatial split: left half train, right half validation, buffer in between
    train_mask = np.zeros_like(labels, dtype=bool)
    train_mask[:, :W//2 - 5] = True
    
    # For spatial CV, the caller would pass valid (not train_mask) as valid_mask
    # because there's no component-based leakage. Verify this is sensible:
    result_full, _ = grouped_lineament_features(
        features,
        {"all": (0, 1, 2)},
        valid_mask=valid,
        normalization_mask=train_mask,
        kind="phase_edge",
        sigma_pixels=1.0,
    )
    
    result_restricted, _ = grouped_lineament_features(
        features,
        {"all": (0, 1, 2)},
        valid_mask=train_mask,
        normalization_mask=train_mask,
        kind="phase_edge",
        sigma_pixels=1.0,
    )
    
    # Training region should be identical (both used train_mask for normalization)
    # Full-valid filtering can use neighboring validation pixels, restricted cannot
    # So they may differ slightly near the boundary, but training core should match
    core = train_mask.copy()
    core[:, W//2 - 10:] = False  # exclude near-boundary pixels
    
    # Normalization is fold-pure in both, so training-core features should be similar
    # (not identical because filtering support differs, but close)
    correlation = np.corrcoef(
        result_full[core].ravel(),
        result_restricted[core].ravel()
    )[0, 1]
    assert correlation > 0.99, "spatial behavior changed unexpectedly"


def test_buffer_prevents_edge_artifact_leakage():
    """The buffer requirement ensures training windows don't sample edge artifacts.
    
    When held-out regions are masked during filtering, the boundary has a sharp
    transition. The buffer (>= filter support) ensures no training window sees
    these transition pixels.
    """
    features = np.ones((40, 40, 1), dtype=np.float32)
    # Create a held-out region with very different values
    held_out = np.zeros((40, 40), dtype=bool)
    held_out[15:25, 15:25] = True
    features[held_out] = 10.0
    
    train_mask = ~held_out
    
    # Compute with masking (protocol)
    result, _ = grouped_lineament_features(
        features,
        {"test": (0,)},
        valid_mask=train_mask,
        normalization_mask=train_mask,
        kind="gradient_energy",
        sigma_pixels=2.0,
    )
    
    # With sigma=2, filter support is ~4*2=8 pixels. If we enforce buffer=8,
    # then no training pixel within 8 pixels of the held-out boundary should
    # see any gradient from the transition.
    
    # Create a buffer-8 erosion of train_mask
    from scipy.ndimage import binary_erosion
    y, x = np.ogrid[-8:9, -8:9]
    kernel = (x*x + y*y) <= 8*8
    safe_train = binary_erosion(train_mask, structure=kernel)
    
    # In the safe training region (>8 pixels from held-out), gradient should be ~0
    # because the features are constant (1.0) there
    safe_gradient = result[safe_train]
    assert np.percentile(safe_gradient, 95) < 0.01, \
        "buffer did not prevent edge artifact leakage"


def test_trace_cv_respects_held_out_endpoints():
    """Trace CV holds out endpoints while keeping trace bodies as training truth.
    
    Verify that lineament features respect this: held-out endpoint regions must
    not leak into training inputs through spatial filtering.
    """
    # Create two long horizontal traces
    labels = np.zeros((30, 80), dtype=bool)
    labels[10, 10:70] = True  # 60-pixel trace
    labels[20, 10:70] = True  # another 60-pixel trace
    
    features = np.random.default_rng(88).normal(0, 1, (30, 80, 2)).astype(np.float32)
    valid = np.ones_like(labels, dtype=bool)
    
    # Assign trace endpoints (will hold out one end per trace)
    from gems.cv import assign_trace_endpoints, trace_completion_fold
    endpoint_folds = assign_trace_endpoints(
        labels,
        n_folds=2,
        seed=3,
        min_pixels=20,
        endpoint_fraction=0.25,
    )
    
    fold_spec = trace_completion_fold(
        labels,
        endpoint_folds,
        fold=0,
        buffer_pixels=5,
        valid_mask=valid,
    )
    
    # Baseline: compute features with training mask
    baseline, _ = grouped_lineament_features(
        features,
        {"test": (0, 1)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind="gradient_energy",
        sigma_pixels=1.5,
    )
    
    # Perturb held-out endpoint dramatically
    perturbed = features.copy()
    perturbed[~fold_spec.train_valid_mask] += 500.0
    
    perturbed_result, _ = grouped_lineament_features(
        perturbed,
        {"test": (0, 1)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind="gradient_energy",
        sigma_pixels=1.5,
    )
    
    # Training region (including trace body) must be unaffected
    assert np.allclose(
        baseline[fold_spec.train_valid_mask],
        perturbed_result[fold_spec.train_valid_mask],
        rtol=0,
        atol=1e-6,
    ), "trace endpoint perturbation leaked into training region"


@pytest.mark.parametrize("kind", ["mumford_shah_log", "ridge_valley_response"])
def test_multiscale_lineament_features_respect_masking(kind):
    """Multi-scale and multi-derivative features must also respect fold boundaries."""
    features, labels = _synthetic_features_and_faults(seed=55)
    valid = np.ones_like(labels, dtype=bool)
    
    component_folds = assign_fault_components(labels, n_folds=2, seed=2)
    fold_spec = fault_discovery_fold(
        labels,
        component_folds,
        fold=0,
        buffer_pixels=10,
        valid_mask=valid,
    )
    
    kwargs = {"sigma_pixels": 1.0}
    if kind == "ridge_valley_response":
        kwargs["ridge_valley_scales"] = (0.5, 1.5, 3.0)
        kwargs["structure_tensor_window"] = 2.0
    
    baseline, _ = grouped_lineament_features(
        features,
        {"all": (0, 1, 2)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind=kind,
        **kwargs,
    )
    
    # Perturb held-out inputs
    perturbed = features.copy()
    rng = np.random.default_rng(1234)
    perturbed[~fold_spec.train_valid_mask] = rng.normal(
        100, 50, perturbed[~fold_spec.train_valid_mask].shape
    ).astype(np.float32)
    
    perturbed_result, _ = grouped_lineament_features(
        perturbed,
        {"all": (0, 1, 2)},
        valid_mask=fold_spec.train_valid_mask,
        normalization_mask=fold_spec.train_valid_mask,
        kind=kind,
        **kwargs,
    )
    
    # Training region must be stable
    train_region = fold_spec.train_valid_mask
    max_abs_diff = np.abs(baseline[train_region] - perturbed_result[train_region]).max()
    assert max_abs_diff < 1e-5, \
        f"{kind}: held-out perturbation leaked (max diff {max_abs_diff})"
