"""Tests for scripts/audit_results.py (committed-results integrity audit)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_results.py"

SCORES = [0.10, 0.11, 0.09, 0.12, 0.10]
CAL = [0.13, 0.14, 0.12, 0.15, 0.13]
THRESHOLDS = [0.95, 0.94, 0.96, 0.93, 0.95]


def run_audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_doc() -> dict:
    return {
        "evaluation_protocol": {
            "alpha": 0.2,
            "beta": 0.8,
            "buffer_pixels": 16,
            "metric": "distance_weighted_tversky",
            "schema_version": 2,
            "seed": 20260922,
        },
        "aggregate_score": 0.105,
        "folds": [
            {"fold": i, "score": s, "truth_pixels": 1000, "valid_pixels": 50000}
            for i, s in enumerate(SCORES)
        ],
    }


def _lofo_doc() -> dict:
    return {
        "scheme": "spatial",
        "folds": [
            {
                "fold": i,
                "threshold": t,
                "raw_score": r,
                "calibrated_score": c,
            }
            for i, (t, r, c) in enumerate(zip(THRESHOLDS, SCORES, CAL))
        ],
        "raw_mean": sum(SCORES) / len(SCORES),
        "calibrated_mean": sum(CAL) / len(CAL),
    }


def _gate_doc() -> dict:
    # REJECT: no view wins; margins use max(stds) * 1.5 (trials_before 1).
    views = []
    for view, cand, inc, cand_std, inc_std in (
        ("spatial", 0.100, 0.104, 0.010, 0.011),
        ("fault", 0.040, 0.045, 0.004, 0.005),
        ("trace", 0.008, 0.009, 0.001, 0.002),
    ):
        delta = cand - inc
        margin = max(cand_std, inc_std) * 1.5
        views.append(
            {
                "view": view,
                "comparable": True,
                "candidate_mean": cand,
                "incumbent_mean": inc,
                "candidate_std": cand_std,
                "incumbent_std": inc_std,
                "delta": delta,
                "required_margin": margin,
                "won": delta > margin,
                "fold_ids": [0, 1, 2, 3, 4],
                "reason": "",
            }
        )
    return {
        "candidate": "TEST-CAND",
        "commit": "abc123",
        "config": "configs/ref.yaml",
        "escalation_multiplier": 1.5,
        "hypothesis": "test",
        "timestamp_utc": "2026-10-27T00:00:00+00:00",
        "trials_after": 2,
        "trials_before": 1,
        "verdict": "REJECT",
        "views": views,
    }


def make_root(tmp_path: Path, *, with_ledger: bool = True) -> Path:
    root = tmp_path / "repo"
    run_dir = root / "results" / "runs" / "ref-cv-test"
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True)
    config_path = root / "configs" / "ref.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("epochs: 5\n")
    raw_path = scores_dir / "spatial.json"
    raw_path.write_text(json.dumps(_raw_doc()))
    lofo_path = scores_dir / "lofo-spatial.json"
    lofo_path.write_text(json.dumps(_lofo_doc()))
    notes_path = run_dir / "notes.txt"
    notes_path.write_text("box notes\n")
    manifest = {
        "schema_version": 1,
        "run_id": "ref-cv-test",
        "code": {"git_commit": "abc123", "git_dirty": False},
        "config": {"path": "configs/ref.yaml", "sha256": _sha(config_path)},
        "artifacts": [
            {
                "path": "runs/ref-cv-test/scores/spatial.json",
                "bytes": raw_path.stat().st_size,
                "sha256": _sha(raw_path),
            },
            {
                "path": "runs/ref-cv-test/scores/lofo-spatial.json",
                "bytes": lofo_path.stat().st_size,
                "sha256": _sha(lofo_path),
            },
            {
                "path": "runs/ref-cv-test/notes.txt",
                "bytes": notes_path.stat().st_size,
                "sha256": _sha(notes_path),
            },
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    gates_dir = root / "results" / "gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / "TEST-CAND.json").write_text(json.dumps(_gate_doc()))
    if with_ledger:
        ledger_dir = root / "docs"
        ledger_dir.mkdir(parents=True)
        (ledger_dir / "candidate-trials.jsonl").write_text(json.dumps(_gate_doc()) + "\n")
    return root


def test_real_repo_audit():
    """Against the committed repo: 33 findings, 0 hard errors."""
    result = run_audit(REPO)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Score files audited: 18 (7 LOFO)" in out
    assert "Summary: 33 findings, 0 hard errors — numbers check out." in out
    assert out.count("MISSING_ARTIFACT") == 25
    assert out.count("RELOCATED") == 6
    assert "ref-cv-fault-b32: MISSING_MANIFEST" in out
    assert "ENDPOINT-EXT-01: ORPHAN_LEDGER_ENTRY" in out
    # The one fully clean run: no finding mentions it as missing/relocated.
    assert "ref-cv-spatial-b32: MISSING" not in out
    assert "ref-cv-spatial-b32: RELOCATED" not in out


def test_clean_synthetic_repo(tmp_path: Path):
    root = make_root(tmp_path)
    result = run_audit(root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Findings: none" in result.stdout
    assert "0 hard errors" in result.stdout


def test_manifest_hash_mismatch_is_hard_error(tmp_path: Path):
    root = make_root(tmp_path)
    (root / "results" / "runs" / "ref-cv-test" / "notes.txt").write_text("box notex\n")
    result = run_audit(root)
    assert result.returncode == 2
    assert "SHA256 mismatch" in result.stdout


def test_lofo_raw_mismatch_is_hard_error(tmp_path: Path):
    root = make_root(tmp_path)
    doc = _lofo_doc()
    doc["folds"][0]["raw_score"] = 0.99
    doc["raw_mean"] = sum(f["raw_score"] for f in doc["folds"]) / 5
    (root / "results" / "runs" / "ref-cv-test" / "scores" / "lofo-spatial.json").write_text(
        json.dumps(doc)
    )
    result = run_audit(root)
    assert result.returncode == 2
    assert "disagrees with raw score file" in result.stdout


def test_gate_margin_mismatch_is_hard_error(tmp_path: Path):
    root = make_root(tmp_path)
    doc = _gate_doc()
    doc["views"][0]["required_margin"] = 0.0001
    (root / "results" / "gates" / "TEST-CAND.json").write_text(json.dumps(doc))
    result = run_audit(root)
    assert result.returncode == 2
    assert "required_margin" in result.stdout


def test_gate_ledger_mismatch_is_hard_error(tmp_path: Path):
    root = make_root(tmp_path)
    doc = _gate_doc()
    doc["verdict"] = "ACCEPT"
    (root / "docs" / "candidate-trials.jsonl").write_text(json.dumps(doc) + "\n")
    result = run_audit(root)
    assert result.returncode == 2
    assert "verdict" in result.stdout


def test_relocated_artifact_is_finding_not_error(tmp_path: Path):
    root = make_root(tmp_path)
    run_dir = root / "results" / "runs" / "ref-cv-test"
    raw = run_dir / "scores" / "spatial.json"
    raw.rename(run_dir / "spatial.json")
    result = run_audit(root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "RELOCATED" in result.stdout
    assert "0 hard errors" in result.stdout


def test_missing_raw_for_lofo_is_finding(tmp_path: Path):
    root = make_root(tmp_path)
    (root / "results" / "runs" / "ref-cv-test" / "scores" / "spatial.json").unlink()
    result = run_audit(root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "MISSING_RAW" in result.stdout


def test_malformed_score_json_is_hard_error(tmp_path: Path):
    root = make_root(tmp_path)
    (root / "results" / "runs" / "ref-cv-test" / "scores" / "spatial.json").write_text("{nope")
    result = run_audit(root)
    assert result.returncode == 2


def test_missing_manifest_is_finding(tmp_path: Path):
    root = make_root(tmp_path)
    (root / "results" / "runs" / "ref-cv-test" / "manifest.json").unlink()
    result = run_audit(root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "MISSING_MANIFEST" in result.stdout


def test_orphan_gate_is_finding(tmp_path: Path):
    root = make_root(tmp_path, with_ledger=False)
    result = run_audit(root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "ORPHAN_GATE" in result.stdout
