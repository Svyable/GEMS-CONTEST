"""Tests for the P4 curved-continuation post-processing (step 13).

All numpy-only: the module deliberately avoids skimage so these tests run on
CPU-only hosts. Pixel geometry: synthetic rasters stand in for fold
probability maps; thresholds/params mirror the queue config.
"""

from __future__ import annotations

import numpy as np
import pytest

from gems.curved_continuation import (
    CurvedContinuationParams,
    _zhang_suen_thinning,
    apply_curved_continuation,
    diagnostics,
    estimate_endpoint_orientation,
    find_skeleton_endpoints,
    grow_curved_path,
)


def _params(**kw) -> CurvedContinuationParams:
    base = {
        "max_extension_m": 500.0,
        "pixel_size_m": 100.0,  # 5 steps
        "curvature_penalty": 0.5,
        "beam_width": 3,
        "min_prob": 0.3,
        "support_threshold": 0.4,
        "min_path_length": 2,
        "max_paths": 200,
        "decay_rate": 0.15,
    }
    base.update(kw)
    return CurvedContinuationParams(**base)


def _line(shape=(20, 40), row=10, c0=5, c1=26, p=0.9) -> np.ndarray:
    """Strong corridor on cols [c0, c1)."""
    prob = np.zeros(shape)
    prob[row, c0:c1] = p
    return prob


def _line_with_tail(tail_p=0.4, tail_len=6, **kw) -> np.ndarray:
    """Strong corridor plus a weak-support tail (min_prob <= p < threshold)."""
    prob = _line(**kw)
    row, c1 = kw.get("row", 10), kw.get("c1", 26)
    prob[row, c1 : c1 + tail_len] = tail_p
    return prob


# ---- thinning ----

def test_thinning_keeps_line_connected():
    skel = _zhang_suen_thinning(np.zeros((15, 15), bool))
    assert skel.sum() == 0
    line = np.zeros((15, 15), bool)
    line[7, 2:13] = True
    skel = _zhang_suen_thinning(line)
    # one pixel wide, still spanning the line
    assert skel[7].sum() >= 9
    assert skel.sum() <= 12


def test_thinning_idempotent_on_thin_line():
    line = np.zeros((15, 15), bool)
    line[7, 2:13] = True
    once = _zhang_suen_thinning(line)
    assert _zhang_suen_thinning(once).sum() == once.sum()


def test_thinning_wide_bar_becomes_single_line():
    bar = np.zeros((15, 15), bool)
    bar[6:9, 2:13] = True  # 3 px wide
    skel = _zhang_suen_thinning(bar)
    assert skel.sum() <= 12
    assert skel.sum() >= 8


def test_thinning_deterministic():
    rng = np.random.default_rng(0)
    bw = rng.random((25, 25)) > 0.5
    assert (_zhang_suen_thinning(bw) == _zhang_suen_thinning(bw)).all()


# ---- endpoints & orientation ----

def test_endpoints_of_line():
    skel = np.zeros((20, 20), bool)
    skel[10, 5:16] = True
    eps = find_skeleton_endpoints(skel)
    assert len(eps) == 2
    assert (10, 5) in eps and (10, 15) in eps


def test_endpoints_loop_has_none():
    skel = np.zeros((20, 20), bool)
    skel[5:15, 5] = skel[5:15, 14] = True
    skel[5, 5:15] = skel[14, 5:15] = True
    assert find_skeleton_endpoints(skel) == []


def test_orientation_points_away_from_body():
    skel = np.zeros((20, 40), bool)
    skel[10, 5:26] = True  # body to the left of the right endpoint
    ori = estimate_endpoint_orientation(skel, (10, 25))
    assert ori is not None
    dy, dx = ori
    # away from body => eastward (dx > 0), roughly horizontal
    assert dx > 0.7
    assert abs(dy) < 0.5


def test_orientation_none_on_isolated_pixel():
    skel = np.zeros((10, 10), bool)
    skel[5, 5] = True
    assert estimate_endpoint_orientation(skel, (5, 5)) is None


# ---- path growing ----

def test_grow_straight_corridor():
    prob = _line_with_tail()
    params = _params(max_extension_m=500.0)  # 5 steps
    path = grow_curved_path(prob, (10, 25), (0.0, 1.0), params)
    assert len(path) == 5
    # straight east, no bending
    assert all(r == 10 for r, c in path)
    assert [c for r, c in path] == [26, 27, 28, 29, 30]


def test_grow_stops_at_support_gap():
    prob = _line_with_tail()
    prob[10, 28:32] = 0.0  # gap below min_prob
    params = _params()
    path = grow_curved_path(prob, (10, 25), (0.0, 1.0), params)
    # can only reach (10,26),(10,27) before the gap
    assert len(path) == 2


def test_grow_never_steps_on_below_min_prob():
    rng = np.random.default_rng(1)
    prob = rng.random((20, 40)) * 0.2  # all below min_prob=0.3
    params = _params()
    assert grow_curved_path(prob, (10, 20), (0.0, 1.0), params) == []


def test_grow_follows_curved_support_with_low_penalty():
    # corridor bends gently: east, then south-east diagonals
    prob = np.zeros((20, 40))
    cells = [(10, c) for c in range(5, 17)] + [(10 + k, 16 + k) for k in range(1, 8)]
    for r, c in cells:
        prob[r, c] = 0.9
    params = _params(curvature_penalty=0.0, beam_width=4, max_extension_m=800.0)
    path = grow_curved_path(prob, (10, 16), (0.0, 1.0), params)
    assert len(path) >= 4
    # followed the bend south-east, not straight east into the void
    assert any(r > 10 for r, c in path)


def test_high_penalty_prefers_straight_over_bend():
    # same bent corridor, huge penalty: the straight shot is preferred only
    # if support exists; with no support east, no path should grow.
    prob = np.zeros((20, 40))
    cells = [(10, c) for c in range(5, 15)] + [(10 + k, 15 + k) for k in range(1, 6)]
    for r, c in cells:
        prob[r, c] = 0.9
    params = _params(curvature_penalty=100.0, max_extension_m=300.0)
    path = grow_curved_path(prob, (10, 14), (0.0, 1.0), params)
    # east has no support at all (0.0 < min_prob), so nothing grows
    assert path == []


def test_grow_respects_valid_mask():
    prob = _line_with_tail()
    mask = np.ones_like(prob, dtype=bool)
    mask[10, 27:] = False
    params = _params(min_path_length=1)
    path = grow_curved_path(prob, (10, 25), (0.0, 1.0), params, valid_mask=mask)
    assert len(path) == 1
    assert path[0] == (10, 26)


def test_grow_deterministic():
    prob = _line_with_tail()
    params = _params()
    a = grow_curved_path(prob, (10, 25), (0.0, 1.0), params)
    b = grow_curved_path(prob, (10, 25), (0.0, 1.0), params)
    assert a == b


# ---- apply ----

def test_apply_extends_line_and_decays():
    prob = _line_with_tail()
    params = _params()
    out = apply_curved_continuation(prob, threshold=0.5, params=params)
    painted = [(r, c) for r in range(20) for c in range(26, 40) if out[r, c] > prob[r, c]]
    assert len(painted) >= 2
    # decay: painted values decrease away from the endpoint
    cols = sorted(c for r, c in painted)
    vals = [out[10, c] for c in cols]
    assert vals == sorted(vals, reverse=True)
    # paint never exceeds the skeleton threshold
    assert max(vals) <= 0.5 + 1e-9


def test_apply_never_overwrites_higher_probability():
    prob = _line_with_tail()
    prob[10, 26:32] = 0.8  # strong existing support ahead of the tip
    params = _params()
    out = apply_curved_continuation(prob, threshold=0.5, params=params)
    assert np.all(out >= prob - 1e-12)
    # the 0.8 pixels are untouched by the weaker paint
    assert (out[10, 26:32] == 0.8).all()


def test_apply_unsupported_bridge_rejected():
    # tip exists, but everything ahead is weak-but-above-min_prob noise that
    # fails the mean-support gate
    prob = _line_with_tail(tail_p=0.32, tail_len=8)
    params = _params()
    out = apply_curved_continuation(prob, threshold=0.5, params=params)
    assert np.allclose(out, prob)


def test_apply_empty_map_unchanged():
    prob = np.zeros((20, 20))
    params = _params()
    out = apply_curved_continuation(prob, threshold=0.5, params=params)
    assert np.array_equal(out, prob)


def test_apply_max_paths_budget():
    # two separate lines => 4 endpoints; budget 1 paints at most one tip
    prob = _line(shape=(30, 40), row=8, c0=5, c1=20, p=0.9)
    prob[20, 5:20] = 0.9
    params = _params(max_paths=1)
    out = apply_curved_continuation(prob, threshold=0.5, params=params)
    painted = int(((out > prob) & (out > 0)).sum())
    assert painted <= params.max_steps


def test_apply_min_path_length_gate():
    # a lone support pixel ahead of the tip: path too short to keep
    prob = _line(p=0.9)
    prob[10, 26] = 0.9
    params = _params(min_path_length=3, min_prob=0.3, support_threshold=0.0)
    out = apply_curved_continuation(prob, threshold=0.5, params=params)
    assert np.allclose(out, prob)


def test_apply_invalid_threshold_raises():
    params = _params()
    with pytest.raises(ValueError):
        apply_curved_continuation(np.zeros((5, 5)), threshold=1.5, params=params)


def test_diagnostics_report():
    prob = _line(p=0.9)
    params = _params()
    d = diagnostics(prob, threshold=0.5, params=params)
    assert d["skeleton_pixels"] > 0
    assert d["endpoints"] == 2
    assert d["max_steps"] == 5


# ---- param validation ----

@pytest.mark.parametrize(
    "kw",
    [
        {"max_extension_m": 0},
        {"pixel_size_m": -1},
        {"curvature_penalty": -0.1},
        {"beam_width": 0},
        {"min_prob": 1.0},
        {"min_prob": -0.1},
        {"support_threshold": 1.5},
        {"min_path_length": 0},
        {"max_paths": 0},
        {"decay_rate": 0.0},
        {"decay_rate": 1.0},
        {"window_size": 14},
        {"window_size": 2},
    ],
)
def test_params_validation(kw):
    with pytest.raises(ValueError):
        _params(**kw)


def test_max_steps_property():
    assert _params(max_extension_m=300.0, pixel_size_m=100.0).max_steps == 3
    assert _params(max_extension_m=50.0, pixel_size_m=100.0).max_steps == 1
