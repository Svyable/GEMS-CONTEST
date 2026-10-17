"""Tests for the step-12 longer-schedule candidate and its compare script.

Step 12 doubles the training schedule (5 -> 10 epochs) with the incumbent
recipe otherwise unchanged. scripts/verify_candidate.py refuses epoch
mismatches by design, so scripts/compare_schedule.py performs the manual
paired comparison.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
CAND_CONFIG = REPO / "configs" / "candidates" / "cand_longer_schedule.yaml"
REF_CONFIG = REPO / "configs" / "reference_unet.yaml"
COMPARE = REPO / "scripts" / "compare_schedule.py"

VIEWS = ("spatial", "fault", "trace")


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_longer_schedule_config_epochs_and_loss():
    """The longer-schedule candidate doubles epochs, keeps the incumbent loss."""
    cfg = _load_config(CAND_CONFIG)
    assert cfg["training"]["epochs"] == 10
    assert cfg["training"]["loss"] == "tversky"
    assert cfg["training"]["alpha"] == 0.2
    assert cfg["training"]["beta"] == 0.8
    # No loss-specific extras leaked in from other candidate configs.
    for key in cfg["training"]:
        assert not key.startswith("loss_"), f"unexpected loss key {key}"


def test_longer_schedule_config_protocol_equal_except_epochs():
    """Everything the gate's training protocol checks (except epochs) matches
    the incumbent recipe, so the manual comparison isolates schedule length."""
    cand = _load_config(CAND_CONFIG)
    ref = _load_config(REF_CONFIG)
    assert ref["training"]["epochs"] == 5
    for section in ("data", "patches", "model", "augmentation"):
        assert cand[section] == ref[section], f"section {section} differs"
    for key in ("seed_sequence", "monte_carlo_iterations"):
        assert cand[key] == ref[key], f"key {key} differs"
    training = dict(cand["training"])
    ref_training = dict(ref["training"])
    del training["epochs"]
    del ref_training["epochs"]
    assert training == ref_training


# ---- compare_schedule.py ----

PROTO_BASE = {
    "schema_version": 2,
    "aggregation": "stitched_spatial_oof",
    "metric": "distance_weighted_tversky",
    "alpha": 0.2,
    "beta": 0.8,
    "radius_pixels": 3.0,
    "truth_sha256": "a" * 64,
    "fold_map_sha256": "b" * 64,
    "known_fault_exclusion_pixels": 0,
    "buffer_pixels": 16,
    "seed": 20260922,
    "train_step": 32,
}


def _gate_payload(view: str, scores: list[float], epochs: int, seed: int = 20260922) -> dict:
    proto = dict(PROTO_BASE)
    proto["epochs"] = epochs
    proto["seed"] = seed
    if view != "spatial":
        proto["aggregation"] = "overlapping_background_macro"
    return {
        "scheme": view,
        "evaluation_protocol": proto,
        "macro_mean": sum(scores) / len(scores),
        "macro_std": 0.0,
        "folds": [
            {"fold": i, "score": s, "valid_pixels": 100, "truth_pixels": 10}
            for i, s in enumerate(scores)
        ],
    }


def _lofo_payload(view: str, cal: list[float]) -> dict:
    return {
        "scheme": view,
        "folds": [
            {"fold": i, "raw_score": 0.1, "calibrated_score": c,
             "eval_pixels": 100, "truth_pixels": 10}
            for i, c in enumerate(cal)
        ],
        "calibrated_mean": sum(cal) / len(cal),
        "calibrated_std": 0.0,
    }


@pytest.fixture()
def comparison_dirs(tmp_path):
    inc_gate, cand_gate = tmp_path / "inc", tmp_path / "cand"
    inc_lofo, cand_lofo = tmp_path / "inc-lofo", tmp_path / "cand-lofo"
    for d in (inc_gate, cand_gate, inc_lofo, cand_lofo):
        d.mkdir()
    inc_scores = [0.10, 0.12, 0.09, 0.11, 0.10]
    cand_scores = [0.11, 0.13, 0.10, 0.12, 0.11]  # uniform +0.01
    for view in VIEWS:
        (inc_gate / f"{view}.json").write_text(
            json.dumps(_gate_payload(view, inc_scores, epochs=5)))
        (cand_gate / f"{view}.json").write_text(
            json.dumps(_gate_payload(view, cand_scores, epochs=10)))
        (inc_lofo / f"lofo-{view}.json").write_text(
            json.dumps(_lofo_payload(view, [s + 0.03 for s in inc_scores])))
        (cand_lofo / f"lofo-{view}.json").write_text(
            json.dumps(_lofo_payload(view, [s + 0.035 for s in cand_scores])))
    return inc_gate, cand_gate, inc_lofo, cand_lofo


def _run(inc_gate, cand_gate, *extra):
    return subprocess.run(
        [sys.executable, str(COMPARE), str(inc_gate), str(cand_gate), *extra],
        capture_output=True, text=True, timeout=30, check=False,
    )


def test_compare_reports_paired_deltas(comparison_dirs):
    inc_gate, cand_gate, inc_lofo, cand_lofo = comparison_dirs
    result = _run(inc_gate, cand_gate,
                  "--incumbent-lofo", str(inc_lofo),
                  "--candidate-lofo", str(cand_lofo))
    assert result.returncode == 0, result.stderr
    assert "incumbent 5 epochs vs candidate 10 epochs" in result.stdout
    assert "mean delta +0.01000" in result.stdout  # raw: uniform +0.01
    assert "LOFO-calibrated" in result.stdout
    assert "mean delta +0.01500" in result.stdout  # cal: +0.01 + 0.005


def test_compare_refuses_equal_epochs(comparison_dirs):
    """Equal epochs belong to the verify_candidate gate, not this script."""
    inc_gate, cand_gate, _, _ = comparison_dirs
    for view in VIEWS:
        payload = json.loads((cand_gate / f"{view}.json").read_text())
        payload["evaluation_protocol"]["epochs"] = 5
        (cand_gate / f"{view}.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "verify_candidate gate territory" in result.stderr


def test_compare_refuses_seed_mismatch(comparison_dirs):
    """Any training-protocol difference other than epochs is refused."""
    inc_gate, cand_gate, _, _ = comparison_dirs
    payload = json.loads((cand_gate / "spatial.json").read_text())
    payload["evaluation_protocol"]["seed"] = 20260923
    (cand_gate / "spatial.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "seed" in result.stderr


def test_compare_refuses_fold_mismatch(comparison_dirs):
    inc_gate, cand_gate, _, _ = comparison_dirs
    payload = json.loads((cand_gate / "fault.json").read_text())
    payload["folds"] = payload["folds"][:4]
    payload["macro_mean"] = sum(f["score"] for f in payload["folds"]) / 4
    (cand_gate / "fault.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "fold ids differ" in result.stderr


def test_compare_tolerates_missing_views(comparison_dirs):
    inc_gate, cand_gate, inc_lofo, cand_lofo = comparison_dirs
    (cand_gate / "trace.json").unlink()
    result = _run(inc_gate, cand_gate,
                  "--incumbent-lofo", str(inc_lofo),
                  "--candidate-lofo", str(cand_lofo))
    assert result.returncode == 0, result.stderr
    assert "trace: SKIPPED" in result.stdout
