"""Tests for the P2 wavelet-entropy lineament channel.

Covers docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md P2: an adaptation
(not reproduction) of Dwivedi (2026) wavelet-space entropy to potential-field
edge detection via the undecimated Haar transform.
"""

import numpy as np
import pytest

from gems.lineament import (
    _masked_gaussian,
    _undecimated_haar_detail_energies,
    grouped_lineament_features,
    lineament_kwargs,
    required_lineament_buffer,
)


def _step_features(size=64, edge_col=32, amplitude=10.0):
    features = np.zeros((size, size, 1), dtype=np.float32)
    features[:, edge_col:, 0] = amplitude
    return features


def test_constant_band_has_zero_entropy():
    features = np.full((32, 32, 2), 3.0, dtype=np.float32)
    valid = np.ones((32, 32), dtype=bool)
    derived, metadata = grouped_lineament_features(
        features,
        {"physical": (0, 1)},
        valid_mask=valid,
        kind="wavelet_entropy",
        sigma_pixels=1.0,
    )
    assert derived.shape == (32, 32, 1)
    assert np.isfinite(derived).all()
    assert not derived.any()
    assert metadata["kind"] == "wavelet_entropy"
    assert metadata["wavelet_levels"] == [1, 2, 3]
    assert metadata["phase_interpretation"] == (
        "undecimated_haar_detail_energy_entropy_across_scales"
    )
    assert "entropy_p99" in metadata["norm_params"]


def test_step_edge_produces_localized_positive_response():
    features = _step_features()
    valid = np.ones((64, 64), dtype=bool)
    derived, _ = grouped_lineament_features(
        features,
        {"signal": (0,)},
        valid_mask=valid,
        kind="wavelet_entropy",
        sigma_pixels=1.0,
    )
    assert np.isfinite(derived).all()
    assert np.all((derived >= 0.0) & (derived <= 1.0))
    edge_band = derived[:, 28:36, 0]
    flat_band = derived[:, 0:16, 0]
    assert edge_band.max() > 0.0
    # Flat interior far from the edge (and from reflect-padding border
    # effects) carries no detail energy.
    assert flat_band.max() == 0.0


def test_helper_returns_nonnegative_full_resolution_energies():
    rng = np.random.default_rng(7)
    image = rng.normal(size=(40, 48)).astype(np.float32)
    energies = _undecimated_haar_detail_energies(image, (1, 2, 3))
    assert len(energies) == 3
    for energy in energies:
        assert energy.shape == (40, 48)
        assert np.isfinite(energy).all()
        assert (energy >= 0.0).all()


def test_helper_constant_image_has_zero_detail_energy():
    energies = _undecimated_haar_detail_energies(np.full((24, 24), 2.5, dtype=np.float32), (1, 2))
    for energy in energies:
        assert not energy.any()


def test_helper_detects_step_at_each_level_scale():
    features = _step_features(size=64, edge_col=32)
    energies = _undecimated_haar_detail_energies(features[..., 0], (1, 2, 3))
    for level, energy in zip((1, 2, 3), energies):
        dilation = 2 ** (level - 1)
        # Detail energy must appear within the filter support of the edge.
        assert energy[:, 32 - dilation : 32 + dilation].max() > 0.0


def test_entropy_bounded_by_log_levels_on_two_scale_signal():
    # A step (broadband) plus a 1px spike train (fine scale) spreads detail
    # energy across scales; the per-pixel entropy must stay within [0, log L].
    features = _step_features(size=64, edge_col=32, amplitude=5.0)
    features[::2, ::2, 0] += 3.0
    energies = _undecimated_haar_detail_energies(features[..., 0], (1, 2, 3))
    stack = np.stack(energies, axis=0).astype(np.float64)
    total = stack.sum(axis=0)
    mask = total > 1e-12
    q = stack / (total + 1e-12)
    entropy = -(q * np.log(q + 1e-12)).sum(axis=0)
    entropy = np.maximum(entropy, 0.0)  # same float-dust clamp as the implementation
    assert entropy[mask].max() <= np.log(3) + 1e-9
    assert entropy[mask].min() >= 0.0


def test_levels_change_response():
    features = _step_features()
    valid = np.ones((64, 64), dtype=bool)
    one, _ = grouped_lineament_features(
        features,
        {"signal": (0,)},
        valid_mask=valid,
        kind="wavelet_entropy",
        sigma_pixels=1.0,
        wavelet_levels=(1, 2),
    )
    three, _ = grouped_lineament_features(
        features,
        {"signal": (0,)},
        valid_mask=valid,
        kind="wavelet_entropy",
        sigma_pixels=1.0,
        wavelet_levels=(1, 2, 3),
    )
    assert np.isfinite(one).all() and np.isfinite(three).all()
    assert one.max() > 0.0 and three.max() > 0.0
    assert not np.allclose(one, three)


def test_two_pass_fold_pure_reuse_matches_manual_scaling():
    # Real two-pass CV usage: pass A fits the fold-pure scale on the training
    # region; pass B reuses it on a disjoint evaluation region.
    rng = np.random.default_rng(11)
    features = rng.normal(size=(80, 80, 1)).astype(np.float32)
    training = np.zeros((80, 80), dtype=bool)
    training[10:40, 10:40] = True
    eval_valid = np.zeros((80, 80), dtype=bool)
    eval_valid[50:75, 50:75] = True

    _, meta_a = grouped_lineament_features(
        features,
        {"signal": (0,)},
        valid_mask=training,
        normalization_mask=training,
        kind="wavelet_entropy",
        sigma_pixels=1.0,
    )
    denom_a = meta_a["norm_params"]["entropy_p99"]["signal"]
    assert denom_a > 0

    # Reference: recompute the unnormalized entropy for the eval region with
    # the module's own helpers, then scale by the pass-A denominator. This
    # avoids the information loss that p99-clipping would introduce if we
    # rescaled an already-normalized reference.
    smoothed = _masked_gaussian(features[..., 0], eval_valid, 1.0)
    energies = _undecimated_haar_detail_energies(smoothed, (1, 2, 3))
    stack = np.stack(energies, axis=0).astype(np.float64)
    total = stack.sum(axis=0)
    q = stack / (total + 1e-12)
    entropy = np.maximum(-(q * np.log(q + 1e-12)).sum(axis=0), 0.0)
    entropy[total <= 1e-12] = 0.0

    # Pass B reuses the pass-A denominator on the disjoint eval region.
    pass_b, _ = grouped_lineament_features(
        features,
        {"signal": (0,)},
        valid_mask=eval_valid,
        normalization_mask=training,
        norm_params=meta_a["norm_params"],
        kind="wavelet_entropy",
        sigma_pixels=1.0,
    )
    expected = np.zeros((80, 80), dtype=np.float32)
    expected[eval_valid] = np.clip(entropy[eval_valid] / denom_a, 0.0, 1.0)
    np.testing.assert_allclose(pass_b[..., 0], expected, atol=1e-5, rtol=1e-5)


def test_invalid_pixels_are_zero_and_do_not_poison_result():
    features = _step_features()
    features[0, 0, 0] = np.nan
    valid = np.ones((64, 64), dtype=bool)
    valid[:4, :4] = False
    derived, _ = grouped_lineament_features(
        features,
        {"signal": (0,)},
        valid_mask=valid,
        kind="wavelet_entropy",
        sigma_pixels=1.0,
    )
    assert np.isfinite(derived).all()
    assert not derived[:4, :4].any()
    # The step edge away from the invalid corner still responds.
    assert derived[40:60, 28:36, 0].max() > 0.0


def test_required_buffer_covers_smoothing_plus_max_dilation():
    assert (
        required_lineament_buffer(
            {"kind": "wavelet_entropy", "sigma_pixels": 1.0, "wavelet_levels": None}
        )
        == 8
    )  # ceil(4*1) + 2**(3-1)
    assert (
        required_lineament_buffer(
            {"kind": "wavelet_entropy", "sigma_pixels": 1.0, "wavelet_levels": [1, 2]}
        )
        == 6
    )  # 4 + 2
    assert (
        required_lineament_buffer(
            {"kind": "wavelet_entropy", "sigma_pixels": 2.0, "wavelet_levels": [1, 2, 3]}
        )
        == 12
    )  # ceil(4*2) + 4


def test_lineament_kwargs_maps_wavelet_levels():
    options = lineament_kwargs({"kind": "wavelet_entropy", "wavelet_levels": [1, 2]})
    assert options["kind"] == "wavelet_entropy"
    assert options["wavelet_levels"] == [1, 2]
    options = lineament_kwargs({"kind": "wavelet_entropy"})
    assert options["wavelet_levels"] is None


@pytest.mark.parametrize(
    "levels",
    [[], [0], [7], [1.5], [1, "x"]],
)
def test_invalid_levels_rejected(levels):
    features = np.ones((16, 16, 1), dtype=np.float32)
    valid = np.ones((16, 16), dtype=bool)
    with pytest.raises(ValueError, match="wavelet_levels"):
        grouped_lineament_features(
            features,
            {"signal": (0,)},
            valid_mask=valid,
            kind="wavelet_entropy",
            wavelet_levels=levels,
        )


def test_norm_params_structure_validated():
    features = np.ones((16, 16, 1), dtype=np.float32)
    valid = np.ones((16, 16), dtype=bool)
    with pytest.raises(ValueError, match="entropy_p99"):
        grouped_lineament_features(
            features,
            {"signal": (0,)},
            valid_mask=valid,
            kind="wavelet_entropy",
            norm_params={"response_p99": {}},
        )


def test_groups_produce_independent_channels():
    features = np.stack(
        [
            _step_features(size=32, edge_col=10)[..., 0],
            _step_features(size=32, edge_col=22)[..., 0],
        ],
        axis=-1,
    ).astype(np.float32)
    valid = np.ones((32, 32), dtype=bool)
    derived, metadata = grouped_lineament_features(
        features,
        {"left": (0,), "right": (1,)},
        valid_mask=valid,
        kind="wavelet_entropy",
        sigma_pixels=1.0,
    )
    assert derived.shape == (32, 32, 2)
    assert metadata["output_channels"] == ["left:wavelet_entropy", "right:wavelet_entropy"]
    # Each channel peaks near its own edge.
    assert derived[:, 6:14, 0].max() > derived[:, 18:26, 0].max()
    assert derived[:, 18:26, 1].max() > derived[:, 6:14, 1].max()
