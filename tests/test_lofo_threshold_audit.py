"""Tests for scripts/lofo_threshold_audit.py (LOFO threshold audit)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "lofo_threshold_audit.py"


def run_audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def lofo_doc(scheme: str, thresholds: list[float], raw: list[float], cal: list[float]) -> dict:
    folds = [
        {"fold": i, "threshold": t, "raw_score": r, "calibrated_score": c}
        for i, (t, r, c) in enumerate(zip(thresholds, raw, cal))
    ]
    return {
        "scheme": scheme,
        "method": "LOFO exact threshold (test fixture)",
        "folds": folds,
        "raw_mean": sum(raw) / len(raw),
        "calibrated_mean": sum(cal) / len(cal),
    }


def write_lofo(root: Path, run: str, scheme: str, doc: dict, under_scores: bool = True) -> Path:
    run_dir = root / "results" / "runs" / run
    target = run_dir / "scores" if under_scores else run_dir
    target.mkdir(parents=True)
    path = target / f"lofo-{scheme}.json"
    path.write_text(json.dumps(doc))
    return path


def test_real_repo_audit():
    """Against the committed repo: 7 runs, every one trips at least one flag."""
    result = run_audit(REPO)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "7 LOFO runs audited, 7 flagged (2 ceiling, 3 unstable, 5 calibration-dominated)" in out
    # Trace: thresholds pinned at ~1.0 in both b16 runs, and most of the
    # headline trace score is threshold fitting.
    assert (
        "ref-cv-trace-b16 | trace | 1 [1, 1] range 5.4e-07 | 0.007832 -> 0.01856 "
        "gain +0.01072 (57.8%) | CEILING, CALIBRATION_DOMINATED" in out
    )
    assert (
        "ref-cv-trace-full | trace | 1 [0.9999, 1] range 5.2e-05 | 0.009317 -> 0.0162 "
        "gain +0.006881 (42.5%) | CEILING, CALIBRATION_DOMINATED" in out
    )
    # Spatial references: stable thresholds, but ~21% of the headline is fitting.
    assert (
        "ref-cv-spatial-b32 | spatial | 0.9585 [0.941, 0.9777] range 0.037 | "
        "0.09878 -> 0.1245 gain +0.02573 (20.7%) | CALIBRATION_DOMINATED" in out
    )
    # Fault b32: gain share under the line, but thresholds do not reproduce.
    assert (
        "ref-cv-fault-b32 | fault | 0.8897 [0.849, 0.952] range 0.1 | "
        "0.03661 -> 0.04503 gain +0.008413 (18.7%) | UNSTABLE" in out
    )
    # The one candidate with a LOFO file: widest spatial threshold spread.
    assert (
        "cand-mumford_shah_log-spatial-b16 | spatial | 0.9346 [0.899, 0.977] "
        "range 0.078 | 0.09144 -> 0.1146 gain +0.02317 (20.2%) | "
        "UNSTABLE, CALIBRATION_DOMINATED" in out
    )


def test_synthetic_clean_run(tmp_path: Path):
    root = tmp_path / "repo"
    write_lofo(
        root,
        "ref-clean",
        "spatial",
        lofo_doc(
            "spatial",
            [0.50, 0.52, 0.48, 0.51, 0.49],
            [0.10, 0.11, 0.09, 0.10, 0.10],
            [0.11, 0.12, 0.10, 0.11, 0.11],
        ),
        under_scores=False,
    )
    result = run_audit(root)
    assert result.returncode == 0, result.stderr
    assert (
        "1 LOFO runs audited, 0 flagged (0 ceiling, 0 unstable, 0 calibration-dominated)"
        in result.stdout
    )
    assert "| none" in result.stdout


def test_synthetic_ceiling_and_unstable_flags(tmp_path: Path):
    root = tmp_path / "repo"
    write_lofo(
        root,
        "cand-ceiling",
        "trace",
        lofo_doc(
            "trace",
            [0.9999, 0.9999, 0.9999, 0.9999, 0.9999],
            [0.010, 0.011, 0.009, 0.010, 0.010],
            [0.011, 0.012, 0.010, 0.011, 0.011],
        ),
    )
    write_lofo(
        root,
        "cand-unstable",
        "fault",
        lofo_doc(
            "fault",
            [0.60, 0.90, 0.65, 0.85, 0.70],
            [0.040, 0.041, 0.039, 0.040, 0.040],
            [0.045, 0.046, 0.044, 0.045, 0.045],
        ),
    )
    result = run_audit(root)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "2 LOFO runs audited, 2 flagged (1 ceiling, 1 unstable, 0 calibration-dominated)" in out
    assert "cand-ceiling | trace" in out and "CEILING" in out
    assert "cand-unstable | fault" in out and "UNSTABLE" in out


def test_malformed_lofo_is_hard_error(tmp_path: Path):
    root = tmp_path / "repo"
    run_dir = root / "results" / "runs" / "ref-bad" / "scores"
    run_dir.mkdir(parents=True)
    (run_dir / "lofo-spatial.json").write_text("{not json")
    result = run_audit(root)
    assert result.returncode == 2
    assert "lofo-spatial.json" in result.stderr


def test_mean_mismatch_is_hard_error(tmp_path: Path):
    root = tmp_path / "repo"
    doc = lofo_doc(
        "spatial",
        [0.50, 0.52, 0.48, 0.51, 0.49],
        [0.10, 0.11, 0.09, 0.10, 0.10],
        [0.11, 0.12, 0.10, 0.11, 0.11],
    )
    doc["raw_mean"] = 0.99  # disagrees with the fold raw scores
    write_lofo(root, "ref-mismatch", "spatial", doc)
    result = run_audit(root)
    assert result.returncode == 2
    assert "disagrees with fold mean" in result.stderr


def test_no_lofo_files_is_hard_error(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "results" / "runs").mkdir(parents=True)
    result = run_audit(root)
    assert result.returncode == 2
    assert "no LOFO files found" in result.stderr
