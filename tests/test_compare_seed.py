"""Tests for the step-8/step-14 seed-robustness probe comparison tooling.

Step 8 re-trains the incumbent reference at a second seed (20260923); the
propose-and-verify gate refuses seed mismatches by design, so
scripts/compare_seed.py performs the manual paired comparison, tolerating
only a seed difference across the whole evaluation + training protocol.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COMPARE = REPO / "scripts" / "compare_seed.py"
COMPARE_SH = REPO / "scripts" / "queue" / "compare_seed.sh"

VIEWS = ("spatial", "fault", "trace")
INC_SEED = 20260922
CAND_SEED = 20260923

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
    "epochs": 5,
    "train_step": 32,
}


def _gate_payload(view: str, scores: list[float], seed: int) -> dict:
    proto = dict(PROTO_BASE)
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
    # spatial-only probe: fault/trace missing on the candidate side.
    inc_scores = [0.10, 0.12, 0.09, 0.11, 0.10]
    cand_scores = [0.105, 0.115, 0.095, 0.105, 0.10]  # small seed shifts
    (inc_gate / "spatial.json").write_text(
        json.dumps(_gate_payload("spatial", inc_scores, INC_SEED)))
    (cand_gate / "spatial.json").write_text(
        json.dumps(_gate_payload("spatial", cand_scores, CAND_SEED)))
    (inc_lofo / "lofo-spatial.json").write_text(
        json.dumps(_lofo_payload("spatial", [s + 0.03 for s in inc_scores])))
    (cand_lofo / "lofo-spatial.json").write_text(
        json.dumps(_lofo_payload("spatial", [s + 0.03 for s in cand_scores])))
    return inc_gate, cand_gate, inc_lofo, cand_lofo


def _run(inc_gate, cand_gate, *extra):
    return subprocess.run(
        [sys.executable, str(COMPARE), str(inc_gate), str(cand_gate), *extra],
        capture_output=True, text=True, timeout=30, check=False,
    )


def test_compare_reports_seed_paired_deltas(comparison_dirs):
    inc_gate, cand_gate, inc_lofo, cand_lofo = comparison_dirs
    result = _run(inc_gate, cand_gate,
                  "--incumbent-lofo", str(inc_lofo),
                  "--candidate-lofo", str(cand_lofo))
    assert result.returncode == 0, result.stderr
    assert f"incumbent seed {INC_SEED} vs candidate seed {CAND_SEED}" in result.stdout
    assert "mean delta +0.00000" in result.stdout  # net-zero seed shift
    assert "LOFO-calibrated" in result.stdout
    assert "seed-effect std" in result.stdout
    # fault/trace: no candidate JSON -> skipped, still exit 0
    assert "SKIPPED (missing gate JSON on at least one side)" in result.stdout


def test_compare_refuses_equal_seeds(comparison_dirs):
    """Equal seeds belong to the verify_candidate gate, not this script."""
    inc_gate, cand_gate, _, _ = comparison_dirs
    payload = json.loads((cand_gate / "spatial.json").read_text())
    payload["evaluation_protocol"]["seed"] = INC_SEED
    (cand_gate / "spatial.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "verify_candidate gate territory" in result.stderr


def test_compare_refuses_epoch_mismatch(comparison_dirs):
    """A second differing field (here epochs) is out of this script's scope."""
    inc_gate, cand_gate, _, _ = comparison_dirs
    payload = json.loads((cand_gate / "spatial.json").read_text())
    payload["evaluation_protocol"]["epochs"] = 10
    (cand_gate / "spatial.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "exactly ['seed']" in result.stderr


def test_compare_refuses_fold_mismatch(comparison_dirs):
    inc_gate, cand_gate, _, _ = comparison_dirs
    payload = json.loads((cand_gate / "spatial.json").read_text())
    payload["folds"] = payload["folds"][:-1]  # drop fold 4
    (cand_gate / "spatial.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "fold ids differ" in result.stderr


def test_compare_refuses_invalid_score(comparison_dirs):
    inc_gate, cand_gate, _, _ = comparison_dirs
    payload = json.loads((cand_gate / "spatial.json").read_text())
    payload["folds"][0]["score"] = 1.5
    (cand_gate / "spatial.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "invalid score" in result.stderr


def test_compare_missing_protocol_field(comparison_dirs):
    inc_gate, cand_gate, _, _ = comparison_dirs
    payload = json.loads((cand_gate / "spatial.json").read_text())
    del payload["evaluation_protocol"]["fold_map_sha256"]
    (cand_gate / "spatial.json").write_text(json.dumps(payload))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 2
    assert "missing" in result.stderr


def test_compare_without_lofo_skips_calibration(comparison_dirs):
    inc_gate, cand_gate, _, _ = comparison_dirs
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 0, result.stderr
    assert "LOFO-calibrated: SKIPPED" in result.stdout


def test_compare_seed_effect_std_magnitude(comparison_dirs):
    """A concentrated seed shift surfaces in the reported seed-effect std."""
    inc_gate, cand_gate, _, _ = comparison_dirs
    cand_scores = [0.10, 0.12, 0.09, 0.11, 0.16]  # +0.06 on fold 4 only
    (cand_gate / "spatial.json").write_text(
        json.dumps(_gate_payload("spatial", cand_scores, CAND_SEED)))
    result = _run(inc_gate, cand_gate)
    assert result.returncode == 0, result.stderr
    assert "mean delta +0.01200" in result.stdout  # 0.06/5
    assert "seed-effect std 0.02400" in result.stdout  # pstdev of [0,0,0,0,0.06]


def test_compare_sh_is_executable():
    assert COMPARE_SH.exists(), "scripts/queue/compare_seed.sh missing"
    assert COMPARE_SH.stat().st_mode & 0o111, "compare_seed.sh not executable"
