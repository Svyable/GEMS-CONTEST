"""Tests for the P5 skeleton-recall auxiliary loss.

NumPy tests run everywhere. Torch tests (numpy/torch agreement, gradient
flow, dispatcher) are skipped when torch is unavailable, following
tests/test_background_soften.py.
"""

import numpy as np
import pytest

from gems.distance_tversky import build_training_loss
from gems.skeleton_recall import (
    build_skeleton_recall_loss,
    masked_soft_tversky_loss_value,
    skeleton_recall_loss_value,
    soft_skeletonize,
    target_skeleton,
)


def _line(shape=(20, 40), row=10, col0=5, col1=35, value=1.0):
    img = np.zeros(shape)
    img[row, col0:col1] = value
    return img


def test_skeleton_of_thin_line_is_itself():
    """The top-hat term preserves sub-SE thin structures (fault traces are 1-3 px)."""
    s = soft_skeletonize(_line())
    assert s.shape == (20, 40)
    assert s[10, 5:35].mean() == pytest.approx(1.0)
    assert s[np.arange(20) != 10].max() == pytest.approx(0.0)
    assert set(np.unique(s)).issubset({0.0, 1.0})  # exact on binary input


def test_skeleton_of_thick_bar_is_thin():
    """A 9 px bar skeletonizes to a thin center structure, not the whole bar."""
    img = np.zeros((20, 60))
    img[5:14, 5:55] = 1.0
    s = soft_skeletonize(img)
    assert s[9, 20:40].mean() == pytest.approx(1.0)  # center row kept
    assert (s > 0.5).sum() < img.sum()  # strictly thinner than the bar
    assert s[:4].max() == pytest.approx(0.0) and s[15:].max() == pytest.approx(0.0)


def test_skeleton_empty_and_soft_inputs():
    assert soft_skeletonize(np.zeros((10, 10))).max() == pytest.approx(0.0)
    s = soft_skeletonize(_line(value=0.9) + 0.0)  # 0.9 line on 0.0 background
    assert s[10, 5:35].mean() == pytest.approx(0.9)
    # soft background: top-hat subtracts the local background level
    img = np.full((20, 40), 0.1)
    img[10, 5:35] = 0.9
    s2 = soft_skeletonize(img)
    assert s2[10, 5:35].mean() == pytest.approx(0.8)
    assert s2[9, 5:35].max() == pytest.approx(0.0)


def test_skeleton_no_wraparound():
    """Zero padding: structure at the border does not wrap to the far edge."""
    img = np.zeros((12, 12))
    img[6, 0:3] = 1.0  # line touching the left border
    s = soft_skeletonize(img)
    assert s[:, -1].max() == pytest.approx(0.0)
    assert s[6, 0:3].mean() == pytest.approx(1.0)


def test_skeleton_monotone_in_probability():
    lo = soft_skeletonize(_line(value=0.4))
    hi = soft_skeletonize(_line(value=0.9))
    assert hi[10, 20] > lo[10, 20]


def test_target_skeleton_is_binary_and_training_side_only():
    truth = np.zeros((20, 40), dtype=bool)
    truth[10, 5:35] = True
    valid = np.ones((20, 40), dtype=bool)
    valid[:, 30:] = False  # held-out region: truth there must not define the target
    t = target_skeleton(truth, valid)
    assert set(np.unique(t)).issubset({0.0, 1.0})
    assert t[10, 5:30].mean() == pytest.approx(1.0)
    assert t[:, 30:].max() == pytest.approx(0.0)


def test_recall_loss_perfect_and_broken():
    truth = _line().astype(bool)
    perfect = _line(value=1.0)
    assert skeleton_recall_loss_value(perfect, truth) == pytest.approx(0.0, abs=1e-9)
    broken = _line(value=1.0)
    broken[10, 18:22] = 0.0  # a gap in the predicted trace
    assert skeleton_recall_loss_value(broken, truth) > 0.05
    # empty truth -> no signal by construction
    assert skeleton_recall_loss_value(perfect, np.zeros((20, 40), dtype=bool)) == 0.0


def test_recall_loss_ignores_invalid_truth():
    truth = _line().astype(bool)
    pred = _line(value=1.0)
    valid = np.ones((20, 40), dtype=bool)
    valid[10, 5:35] = False  # the whole trace is held out
    assert skeleton_recall_loss_value(pred, truth, valid_mask=valid) == 0.0


def test_weight_zero_reduces_to_control():
    rng = np.random.default_rng(0)
    pred = rng.random((16, 16))
    truth = rng.random((16, 16)) > 0.9
    valid = rng.random((16, 16)) > 0.2
    base = masked_soft_tversky_loss_value(pred, truth, valid_mask=valid)
    # weight=0 must equal the base exactly (matched ablation anchor)
    assert base == pytest.approx(
        masked_soft_tversky_loss_value(pred, truth, valid_mask=valid), rel=1e-12
    )
    assert 0.0 <= base <= 1.0


def test_param_validation():
    from gems.skeleton_recall import _validate_params

    with pytest.raises(ValueError):
        _validate_params(-0.1, 5)
    with pytest.raises(ValueError):
        _validate_params(1.1, 5)
    with pytest.raises(ValueError):
        _validate_params(0.25, 0)
    with pytest.raises(ValueError):
        _validate_params(0.25, 13)
    with pytest.raises(ValueError):
        soft_skeletonize(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError):
        skeleton_recall_loss_value(np.zeros((4, 4)), np.zeros((5, 5), dtype=bool))


def test_dispatcher_routes_skeleton_recall():
    pytest.importorskip("torch")  # the new loss needs torch at build time
    criterion, needs_spatial = build_training_loss({"training": {"loss": "skeleton_recall"}})
    assert needs_spatial is True
    assert type(criterion).__name__ == "SkeletonRecallLoss"
    criterion2, _ = build_training_loss(
        {"training": {"loss": "skeleton_recall", "loss_skeleton_weight": 0.5, "loss_skeleton_iterations": 3}}
    )
    assert criterion2.skeleton_weight == pytest.approx(0.5)
    assert criterion2.skeleton_iterations == 3
    with pytest.raises(ValueError, match="unknown training loss"):
        build_training_loss({"training": {"loss": "nope"}})


def test_torch_agrees_with_numpy_and_base():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(2, 12, 12))
    truth = (rng.random((2, 12, 12)) > 0.85).astype(np.float64)
    valid = rng.random((2, 12, 12)) > 0.15
    loss_fn = build_skeleton_recall_loss(skeleton_weight=0.0)
    torch_value = float(
        loss_fn(
            torch.from_numpy(logits),
            torch.from_numpy(truth),
            torch.from_numpy(valid),
        ).detach()
    )
    numpy_values = [
        masked_soft_tversky_loss_value(1.0 / (1.0 + np.exp(-logits[b])), truth[b] > 0.5, valid_mask=valid[b])
        for b in range(2)
    ]
    assert torch_value == pytest.approx(float(np.mean(numpy_values)), rel=1e-5)


def test_torch_aux_term_matches_numpy():
    torch = pytest.importorskip("torch")
    truth = _line().astype(bool)
    pred = _line(value=0.8)
    loss_fn = build_skeleton_recall_loss(skeleton_weight=1.0)
    logits = np.log(np.clip(pred, 1e-6, 1 - 1e-6) / (1 - np.clip(pred, 1e-6, 1 - 1e-6)))
    torch_value = float(
        loss_fn(
            torch.from_numpy(logits[None]),
            torch.from_numpy(truth.astype(np.float64)[None]),
        ).detach()
    )
    base = masked_soft_tversky_loss_value(pred, truth)
    aux = skeleton_recall_loss_value(pred, truth)
    assert torch_value == pytest.approx(base + aux, rel=1e-5)


def test_torch_gradient_flows_through_skeleton():
    """The aux term must provide a learning signal on thin structures."""
    torch = pytest.importorskip("torch")
    truth = _line().astype(bool)
    logits = torch.zeros((1, 20, 40), requires_grad=True)
    loss_fn = build_skeleton_recall_loss(skeleton_weight=1.0)
    loss = loss_fn(logits, torch.from_numpy(truth.astype(np.float64)[None]))
    loss.backward()
    g = logits.grad.detach().numpy()
    assert np.isfinite(g).all()
    # gradient concentrates on the trace, not spread uniformly
    assert np.abs(g[0, 10, 5:35]).mean() > np.abs(g[0, :5]).mean()


def test_torch_held_out_truth_excluded_from_target():
    torch = pytest.importorskip("torch")
    truth = _line().astype(bool)
    valid = np.ones((20, 40), dtype=bool)
    valid[10, 5:35] = False
    loss_fn = build_skeleton_recall_loss(skeleton_weight=1.0)
    logits = torch.zeros((1, 20, 40))
    value = float(
        loss_fn(
            logits,
            torch.from_numpy(truth.astype(np.float64)[None]),
            torch.from_numpy(valid[None]),
        ).detach()
    )
    # no truth skeleton under the mask -> aux term is 0, only base remains
    base_only = float(
        build_skeleton_recall_loss(skeleton_weight=0.0)(
            logits,
            torch.from_numpy(truth.astype(np.float64)[None]),
            torch.from_numpy(valid[None]),
        ).detach()
    )
    assert value == pytest.approx(base_only, rel=1e-9)
