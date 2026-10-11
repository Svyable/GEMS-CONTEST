"""Tests for the distance-aware Tversky loss (P1).

NumPy tests run everywhere. Torch tests (gradient flow, numpy/torch agreement)
are skipped when torch is unavailable, following tests/test_training_ml.py.
"""

import math

import numpy as np
import pytest

from gems.distance_tversky import (
    build_training_loss,
    distance_tversky_components,
    distance_tversky_loss_value,
    kernel_offsets,
    truth_kernel_numpy,
)
from gems.metric import distance_weighted_tversky_components


def _patch_with_truth(rng, size=32, n_truth=20):
    pred = rng.random((size, size))
    truth = np.zeros((size, size), dtype=bool)
    idx = rng.choice(size * size, size=n_truth, replace=False)
    truth.flat[idx] = True
    return pred, truth


def test_kernel_offsets_triangular_weights():
    offsets = kernel_offsets(3.0)
    assert (0, 0, 1.0) in offsets
    for dy, dx, w in offsets:
        assert math.hypot(dy, dx) < 3.0
        assert w == pytest.approx(1.0 - math.hypot(dy, dx) / 3.0)
    assert kernel_offsets(1.0) == [(0, 0, 1.0)]


def test_kernel_offsets_reject_bad_radius():
    with pytest.raises(ValueError):
        kernel_offsets(0.0)


def test_perfect_prediction_scores_one():
    pred = np.zeros((16, 16))
    truth = np.zeros((16, 16), dtype=bool)
    truth[5:9, 5:9] = True
    pred[truth] = 1.0
    comp = distance_tversky_components(pred, truth)
    assert comp.tp_w == pytest.approx(truth.sum())
    assert comp.fn_w == pytest.approx(0.0)
    assert comp.fp_w == pytest.approx(0.0)
    assert distance_tversky_loss_value(pred, truth) == pytest.approx(0.0)


def test_one_pixel_shift_is_penalized_but_not_fatal():
    truth = np.zeros((20, 20), dtype=bool)
    truth[10, 10] = True
    exact = np.zeros((20, 20))
    exact[10, 10] = 1.0
    shifted = np.zeros((20, 20))
    shifted[10, 11] = 1.0  # one pixel away: kernel weight 1 - 1/3
    loss_exact = distance_tversky_loss_value(exact, truth)
    loss_shifted = distance_tversky_loss_value(shifted, truth)
    assert loss_exact == pytest.approx(0.0, abs=1e-9)
    # One pixel away still matches with weight 2/3; loss must be small but > 0.
    assert 0.0 < loss_shifted < 0.35


def test_distant_prediction_is_pure_false_positive():
    truth = np.zeros((20, 20), dtype=bool)
    truth[3, 3] = True
    pred = np.zeros((20, 20))
    pred[16, 16] = 1.0  # far outside the 3px kernel
    comp = distance_tversky_components(pred, truth)
    assert comp.tp_w == pytest.approx(0.0)
    assert comp.fn_w == pytest.approx(1.0)
    assert comp.fp_w == pytest.approx(1.0)


def test_empty_truth_gives_unit_loss_with_no_gradient_trap():
    rng = np.random.default_rng(0)
    pred = rng.random((16, 16)) * 0.1
    truth = np.zeros((16, 16), dtype=bool)
    comp = distance_tversky_components(pred, truth)
    assert comp.tp_w == pytest.approx(0.0)
    assert comp.fn_w == pytest.approx(0.0)
    assert comp.fp_w == pytest.approx(pred.sum())
    # With positives absent, any background mass still yields loss == 1.
    assert distance_tversky_loss_value(pred, truth) == pytest.approx(1.0)


def test_invalid_pixels_isolated():
    pred, truth = _patch_with_truth(np.random.default_rng(1))
    valid = np.ones_like(truth)
    valid[:4, :] = False  # top rows unsupervised
    pred_invalid = pred.copy()
    pred_invalid[~valid] = 0.9  # loud predictions where unsupervised
    comp_masked = distance_tversky_components(pred_invalid, truth, valid_mask=valid)
    comp_clean = distance_tversky_components(
        np.where(valid, pred_invalid, 0.0), truth, valid_mask=valid
    )
    assert comp_masked.tp_w == pytest.approx(comp_clean.tp_w)
    assert comp_masked.fp_w == pytest.approx(comp_clean.fp_w)
    # Truth pixels in the invalid region must not create a truth kernel.
    truth_only_invalid = np.zeros_like(truth)
    truth_only_invalid[1, 8] = True
    kernel = truth_kernel_numpy(truth_only_invalid, valid)
    assert kernel.max() == pytest.approx(0.0)


def _brute_components(pred, truth, core, radius=3.0):
    """Independent brute-force reference: explicit per-truth-pixel max over the
    kernel neighborhood with zero-fill at borders; FP field from the full-truth
    kernel so halo truth counts as context."""
    t = 0.0
    rows, cols = np.nonzero(core & truth)
    for r, c in zip(rows.tolist(), cols.tolist()):
        best = 0.0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                d = math.hypot(dy, dx)
                if d >= radius:
                    continue
                rr, cc = r + dy, c + dx
                if 0 <= rr < pred.shape[0] and 0 <= cc < pred.shape[1]:
                    best = max(best, pred[rr, cc] * (1.0 - d / radius))
        t += best
    fn = float((core & truth).sum()) - t
    kernel = truth_kernel_numpy(truth, None, radius_pixels=radius)
    f = float((pred * (1.0 - kernel) * core).sum())
    return t, fn, f


def test_halo_core_agrees_with_brute_force():
    # Core-only counting with a 3px halo: vectorized implementation must match
    # an independent brute-force reference, including halo context and the
    # no-wraparound rule at borders.
    rng = np.random.default_rng(2)
    full_pred, full_truth = _patch_with_truth(rng, size=40, n_truth=30)
    core = np.zeros((40, 40), dtype=bool)
    core[3:37, 3:37] = True
    comp = distance_tversky_components(full_pred, full_truth, core_mask=core)
    t, fn, f = _brute_components(full_pred, full_truth, core)
    assert comp.tp_w == pytest.approx(t, rel=1e-9)
    assert comp.fn_w == pytest.approx(fn, rel=1e-9)
    assert comp.fp_w == pytest.approx(f, rel=1e-9)


def test_halo_core_matches_metric_without_halo_truth():
    # When the halo band carries no truth, the core-only loss evidence agrees
    # with the published metric evaluated on the core region with the halo
    # zeroed (the metric has no halo-context concept).
    rng = np.random.default_rng(7)
    pred, truth = _patch_with_truth(rng, size=40, n_truth=30)
    truth[:3, :] = False
    truth[-3:, :] = False
    truth[:, :3] = False
    truth[:, -3:] = False
    core = np.zeros((40, 40), dtype=bool)
    core[3:37, 3:37] = True
    pred_core_only = np.where(core, pred, 0.0)
    comp = distance_tversky_components(pred_core_only, truth, core_mask=core)
    metric_comp = distance_weighted_tversky_components(pred_core_only, truth, valid_mask=core)
    assert comp.tp_w == pytest.approx(metric_comp.tp_w, rel=1e-9)
    assert comp.fn_w == pytest.approx(metric_comp.fn_w, rel=1e-9)
    assert comp.fp_w == pytest.approx(metric_comp.fp_w, rel=1e-9)


def test_border_has_no_wraparound():
    # A prediction at the right edge must not match truth at the left edge.
    pred = np.zeros((12, 12))
    truth = np.zeros((12, 12), dtype=bool)
    pred[6, 11] = 1.0
    truth[6, 0] = True
    comp = distance_tversky_components(pred, truth)
    assert comp.tp_w == pytest.approx(0.0)


def test_monotone_in_prediction_quality():
    # Scaling the matched mass up must not increase the loss.
    rng = np.random.default_rng(3)
    _, truth = _patch_with_truth(rng, size=24, n_truth=25)
    losses = []
    for scale in (0.2, 0.5, 0.8):
        pred = np.where(truth, scale, 0.0)
        losses.append(distance_tversky_loss_value(pred, truth))
    assert losses[0] > losses[1] > losses[2]


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        distance_tversky_components(np.zeros((8, 8)), np.zeros((9, 9), dtype=bool))
    with pytest.raises(ValueError):
        distance_tversky_components(
            np.zeros((8, 8)),
            np.zeros((8, 8), dtype=bool),
            valid_mask=np.zeros((8, 7), dtype=bool),
        )


def test_build_training_loss_unknown_name():
    with pytest.raises(ValueError, match="unknown training loss"):
        build_training_loss({"training": {"loss": "not_a_loss"}})


def _torch_tensors(rng, size=24):
    torch = pytest.importorskip("torch")
    pred, truth = _patch_with_truth(rng, size=size, n_truth=15)
    logits = np.log(pred / (1 - pred + 1e-9) + 1e-9).astype(np.float32)
    return (
        torch.from_numpy(logits).unsqueeze(0),
        torch.from_numpy(truth.astype(np.float32)).unsqueeze(0),
    )


def test_torch_agrees_with_numpy():
    torch = pytest.importorskip("torch")
    from gems.distance_tversky import build_distance_tversky_loss

    rng = np.random.default_rng(4)
    logits, targets = _torch_tensors(rng)
    loss_fn = build_distance_tversky_loss(blend_weight=1.0)
    torch_value = float(loss_fn(logits, targets).detach())
    prob = torch.sigmoid(logits).numpy()[0]
    numpy_value = distance_tversky_loss_value(prob, targets.numpy()[0] > 0.5)
    assert torch_value == pytest.approx(numpy_value, rel=1e-4, abs=1e-6)


def test_torch_gradient_flows():
    torch = pytest.importorskip("torch")
    from gems.distance_tversky import build_distance_tversky_loss

    rng = np.random.default_rng(5)
    logits, targets = _torch_tensors(rng)
    logits.requires_grad_(True)
    loss_fn = build_distance_tversky_loss(blend_weight=0.5)
    value = loss_fn(logits, targets)
    value.backward()
    grad = logits.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_torch_empty_truth_no_nan():
    torch = pytest.importorskip("torch")
    from gems.distance_tversky import build_distance_tversky_loss

    logits = torch.zeros(1, 20, 20)
    targets = torch.zeros(1, 20, 20)
    loss_fn = build_distance_tversky_loss(blend_weight=1.0)
    value = loss_fn(logits, targets)
    assert torch.isfinite(value)
    assert float(value) == pytest.approx(1.0)


def test_torch_masked_supervision_preserves_structure():
    torch = pytest.importorskip("torch")
    from gems.distance_tversky import build_distance_tversky_loss

    rng = np.random.default_rng(6)
    logits, targets = _torch_tensors(rng, size=24)
    valid = torch.ones(1, 24, 24, dtype=torch.bool)
    valid[:, :6, :] = False
    loss_fn = build_distance_tversky_loss(blend_weight=1.0)
    value = loss_fn(logits, targets, valid)
    assert torch.isfinite(value)
    assert getattr(loss_fn, "needs_spatial", False) is True


def test_build_training_loss_tversky_pairing():
    torch = pytest.importorskip("torch")  # noqa: F841
    from segmentation_models_pytorch.losses import TverskyLoss

    criterion, spatial = build_training_loss(
        {"training": {"loss": "tversky", "alpha": 0.2, "beta": 0.8}}
    )
    assert isinstance(criterion, TverskyLoss)
    assert spatial is False


def test_build_training_loss_distance_pairing():
    pytest.importorskip("torch")
    criterion, spatial = build_training_loss(
        {
            "training": {
                "loss": "distance_tversky",
                "alpha": 0.2,
                "beta": 0.8,
                "loss_distance_weight": 0.5,
                "loss_radius_pixels": 3.0,
                "loss_halo_pixels": 0,
            }
        }
    )
    assert getattr(criterion, "needs_spatial", False) is True
    assert spatial is True
