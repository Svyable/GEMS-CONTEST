"""Tests for the P3 background-softened Tversky loss.

NumPy tests run everywhere. Torch tests (numpy/torch agreement, dispatcher)
are skipped when torch is unavailable, following tests/test_training_ml.py.
"""

import numpy as np
import pytest

from gems.background_soften import (
    ambiguity_weights,
    build_background_softened_loss,
    softened_tversky_components,
    softened_tversky_loss_value,
)
from gems.distance_tversky import build_training_loss


def _plain_soft_tversky(pred, truth, valid=None, alpha=0.2, beta=0.8, eps=1e-12):
    """Reference plain soft Tversky (no background softening)."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth).astype(bool)
    v = np.ones_like(y, dtype=bool) if valid is None else np.asarray(valid).astype(bool)
    p = np.where(v, p, 0.0)
    y = y & v
    tp = float((p * y).sum())
    fp = float((p * (1.0 - y)).sum())
    fn = float(((1.0 - p) * y).sum())
    return 1.0 - tp / (tp + alpha * fp + beta * fn + eps)


def test_weights_band_geometry():
    truth = np.zeros((20, 20), dtype=bool)
    truth[10, 10] = True
    w = ambiguity_weights(truth, radius_pixels=3.0, soften_factor=0.5)
    assert w.shape == (20, 20)
    assert w[10, 10] == pytest.approx(1.0)  # truth pixel: not background
    # distance 2 < 3 -> softened; distance 5 >= 3 -> full
    assert w[10, 12] == pytest.approx(0.5)
    assert w[12, 10] == pytest.approx(0.5)
    assert w[10, 15] == pytest.approx(1.0)
    assert w[0, 0] == pytest.approx(1.0)
    # diagonal: hypot(2,2)=2.83 < 3 softened; hypot(3,0)=3 not < 3 full
    assert w[12, 12] == pytest.approx(0.5)
    assert w[10, 13] == pytest.approx(1.0)


def test_weights_empty_truth_all_ones():
    truth = np.zeros((8, 8), dtype=bool)
    w = ambiguity_weights(truth, radius_pixels=3.0, soften_factor=0.25)
    assert (w == 1.0).all()


def test_weights_training_side_only():
    # Truth outside the valid mask must not define an ambiguity band.
    truth = np.zeros((16, 16), dtype=bool)
    truth[4, 4] = True
    valid = np.ones((16, 16), dtype=bool)
    valid[:8, :] = False  # truth pixel is invalid
    w = ambiguity_weights(truth, valid, radius_pixels=3.0, soften_factor=0.5)
    assert (w[8:, :] == 1.0).all(), "invalid truth must not soften valid pixels"
    assert (w[:8, :] == 0.0).all(), "invalid pixels get zero weight"


def test_weights_no_wraparound():
    truth = np.zeros((12, 12), dtype=bool)
    truth[0, 0] = True
    w = ambiguity_weights(truth, radius_pixels=3.0, soften_factor=0.5)
    assert w[0, 2] == pytest.approx(0.5)
    assert w[0, 11] == pytest.approx(1.0), "band must not wrap across the border"
    assert w[11, 0] == pytest.approx(1.0), "band must not wrap across the border"


def test_weights_reject_bad_params():
    truth = np.zeros((8, 8), dtype=bool)
    with pytest.raises(ValueError):
        ambiguity_weights(truth, radius_pixels=0.0)
    with pytest.raises(ValueError):
        ambiguity_weights(truth, radius_pixels=-2.0)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            ambiguity_weights(truth, soften_factor=bad)
    with pytest.raises(ValueError):
        ambiguity_weights(np.zeros((4, 4, 4), dtype=bool))


def test_factor_one_equals_plain_tversky():
    rng = np.random.default_rng(7)
    pred = rng.random((24, 24))
    truth = np.zeros((24, 24), dtype=bool)
    truth[rng.random((24, 24)) < 0.1] = True
    comp = softened_tversky_components(pred, truth, radius_pixels=3.0, soften_factor=1.0)
    assert comp.loss() == pytest.approx(_plain_soft_tversky(pred, truth))


def test_perfect_prediction_loss_zero():
    truth = np.zeros((16, 16), dtype=bool)
    truth[5:9, 5:9] = True
    pred = truth.astype(float)
    assert softened_tversky_loss_value(pred, truth) == pytest.approx(0.0)


def test_softened_loss_never_exceeds_control():
    rng = np.random.default_rng(11)
    for _ in range(5):
        pred = rng.random((20, 20))
        truth = np.zeros((20, 20), dtype=bool)
        truth[rng.random((20, 20)) < 0.15] = True
        control = _plain_soft_tversky(pred, truth)
        soft = softened_tversky_loss_value(pred, truth, soften_factor=0.5)
        assert soft <= control + 1e-12
        if (pred * ~truth).sum() > 0 and (ambiguity_weights(truth) < 1.0).any():
            assert soft < control, "band FP mass must strictly reduce the loss"


def test_monotone_in_soften_factor():
    rng = np.random.default_rng(13)
    pred = rng.random((20, 20))
    truth = np.zeros((20, 20), dtype=bool)
    truth[8:12, 8:12] = True
    losses = [softened_tversky_loss_value(pred, truth, soften_factor=f) for f in (0.25, 0.5, 1.0)]
    assert losses[0] <= losses[1] <= losses[2]


def test_far_background_keeps_full_penalty():
    # FP mass far from truth must cost exactly the control cost.
    truth = np.zeros((30, 30), dtype=bool)
    truth[5, 5] = True
    pred = np.zeros((30, 30))
    pred[25, 25] = 0.9  # far from truth
    assert softened_tversky_loss_value(pred, truth, soften_factor=0.5) == pytest.approx(
        _plain_soft_tversky(pred, truth)
    )


def test_band_prediction_not_rewarded():
    # With prediction mass on truth, an extra band prediction must still
    # cost something (no free flooding), but less than far background.
    truth = np.zeros((20, 20), dtype=bool)
    truth[10, 10] = True
    base = np.zeros((20, 20))
    base[10, 10] = 0.9
    band = base.copy()
    band[10, 11] = 0.9
    far = base.copy()
    far[10, 18] = 0.9
    loss_base = softened_tversky_loss_value(base, truth, soften_factor=0.5)
    loss_band = softened_tversky_loss_value(band, truth, soften_factor=0.5)
    loss_far = softened_tversky_loss_value(far, truth, soften_factor=0.5)
    assert loss_band > loss_base, "band predictions still cost something"
    assert loss_band < loss_far, "band costs less than far background"


def test_components_score_consistency():
    rng = np.random.default_rng(17)
    pred = rng.random((12, 12))
    truth = rng.random((12, 12)) < 0.2
    comp = softened_tversky_components(pred, truth, soften_factor=0.5)
    assert comp.loss() == pytest.approx(1.0 - comp.score())


def test_dispatcher_background_softened():
    pytest.importorskip("torch")  # the new loss needs torch at build time
    loss_fn, needs_spatial = build_training_loss({"training": {"loss": "background_softened"}})
    assert needs_spatial is True
    assert getattr(loss_fn, "needs_spatial", False) is True
    assert loss_fn.ambiguity_radius_pixels == pytest.approx(3.0)
    assert loss_fn.background_soften_factor == pytest.approx(0.5)
    # Config keys flow through.
    loss_fn2, _ = build_training_loss(
        {
            "training": {
                "loss": "background_softened",
                "loss_ambiguity_radius_pixels": 4.0,
                "loss_background_soften_factor": 0.25,
            }
        }
    )
    assert loss_fn2.ambiguity_radius_pixels == pytest.approx(4.0)
    assert loss_fn2.background_soften_factor == pytest.approx(0.25)
    # Bad config values fail fast at build time, not mid-training.
    with pytest.raises(ValueError):
        build_training_loss(
            {"training": {"loss": "background_softened", "loss_background_soften_factor": 2.0}}
        )


def test_torch_agrees_with_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(19)
    logits = rng.normal(size=(2, 16, 16)).astype(np.float32)
    truth = (rng.random((2, 16, 16)) < 0.15).astype(np.float32)
    loss_fn = build_background_softened_loss(
        ambiguity_radius_pixels=3.0, background_soften_factor=0.5
    )
    torch_value = float(
        loss_fn(torch.from_numpy(logits), torch.from_numpy(truth)).detach()
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    numpy_values = [
        softened_tversky_loss_value(prob[b], truth[b] > 0.5, soften_factor=0.5)
        for b in range(2)
    ]
    assert torch_value == pytest.approx(float(np.mean(numpy_values)), rel=1e-5)


def test_torch_valid_mask_isolates_held_out_truth():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(23)
    logits = rng.normal(size=(1, 14, 14)).astype(np.float32)
    truth = np.zeros((1, 14, 14), dtype=np.float32)
    truth[0, 3, 3] = 1.0  # held-out truth: invalid
    valid = np.ones((1, 14, 14), dtype=bool)
    valid[0, :7, :] = False
    loss_fn = build_background_softened_loss(soften_factor=0.5)
    value = float(
        loss_fn(
            torch.from_numpy(logits), torch.from_numpy(truth), torch.from_numpy(valid)
        ).detach()
    )
    # With the truth pixel invalid, no band exists and the truth pixel is
    # excluded from all terms: equals the all-background control value.
    assert value == pytest.approx(1.0)
