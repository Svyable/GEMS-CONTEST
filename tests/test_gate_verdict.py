"""Tests for scripts/render_gate_verdict.py (gate JSON -> EXPERIMENT_LOG block)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "render_gate_verdict.py"
REAL_GATE = REPO / "results" / "gates" / "ENDPOINT-EXT-02.json"


def run_render(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def make_gate(
    tmp_path: Path,
    name: str = "CAND-SYNTH",
    verdict: str = "ACCEPT",
    views: list[dict] | None = None,
) -> Path:
    views = (
        views
        if views is not None
        else [
            {
                "view": "spatial",
                "candidate_mean": 0.13,
                "candidate_std": 0.01,
                "incumbent_mean": 0.096,
                "incumbent_std": 0.01,
                "comparable": True,
                "delta": 0.034,
                "required_margin": 0.015,
                "won": True,
                "reason": "",
            },
            {
                "view": "fault",
                "candidate_mean": 0.05,
                "candidate_std": 0.008,
                "incumbent_mean": 0.036,
                "incumbent_std": 0.008,
                "comparable": True,
                "delta": 0.014,
                "required_margin": 0.012,
                "won": True,
                "reason": "",
            },
            {
                "view": "trace",
                "candidate_mean": 0.018,
                "candidate_std": 0.003,
                "incumbent_mean": 0.0186,
                "incumbent_std": 0.003,
                "comparable": True,
                "delta": -0.0006,
                "required_margin": 0.004,
                "won": False,
                "reason": "",
            },
        ]
    )
    report = {
        "candidate": name,
        "commit": "deadbeef",
        "config": "configs/synthetic.yaml",
        "hypothesis": "Synthetic candidate improves Tversky.",
        "timestamp_utc": "2026-10-23T10:00:00+00:00",
        "trials_after": 3,
        "trials_before": 2,
        "verdict": verdict,
        "views": views,
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_real_reject_gate_matches_committed_log_entry():
    """The renderer must reproduce the hand-written 2026-10-10
    ENDPOINT-EXT-02 entry content (golden test against the real artifact)."""
    proc = run_render(REAL_GATE)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "## 2026-10-10 — Gate decision: ENDPOINT-EXT-02 (REJECT)" in out
    assert (
        "**Candidate**: Endpoint extension (300 m, decay 0.15, "
        "skeleton threshold = fold LOFO threshold)" in out
    )
    assert "**Config**: `configs/resnet18_with_endpoint_extension.yaml`" in out
    # Exact numbers copied from docs/EXPERIMENT_LOG.md (2026-10-10 entry):
    assert "- **Spatial**: delta -3.3e-08 (below required margin 0.0148), won 0/3 views" in out
    # Hand-written 10-10 entry rounded margins ad hoc (0.0067 / 0.0026);
    # the renderer uses a consistent %.4g, preserving all content:
    assert "- **Fault**: delta -2e-06 (below required margin 0.006706), won 0/3 views" in out
    assert "- **Trace**: delta -4.9e-08 (below required margin 0.002632), won 0/3 views" in out
    assert "**Gate verdict**: **REJECT**. ENDPOINT-EXT-02 won on 0 of 3 views" in out
    assert "**Results committed**: `results/gates/ENDPOINT-EXT-02.json`" in out
    assert "docs/candidate-trials.jsonl" in out


def test_accept_gate_render(tmp_path):
    proc = run_render(make_gate(tmp_path))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Gate decision: CAND-SYNTH (ACCEPT)" in out
    assert "- **Spatial**: delta 0.034 (above required margin 0.015), won 2/3 views" in out
    assert "- **Trace**: delta -0.0006 (below required margin 0.004), won 2/3 views" in out
    assert "**Gate verdict**: **ACCEPT**. CAND-SYNTH won on 2 of 3 views" in out
    assert "2026-10-23" in out  # date comes from timestamp_utc


def test_inconclusive_gate_with_uncomparable_view(tmp_path):
    views = [
        {
            "view": "spatial",
            "comparable": True,
            "delta": 0.001,
            "required_margin": 0.015,
            "won": False,
            "reason": "",
        },
        {
            "view": "fault",
            "comparable": False,
            "delta": None,
            "required_margin": None,
            "won": False,
            "reason": "fault: expected a JSON object, got NoneType",
        },
    ]
    proc = run_render(make_gate(tmp_path, verdict="INCONCLUSIVE", views=views))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Gate decision: CAND-SYNTH (INCONCLUSIVE)" in out
    assert "- **Fault**: not comparable — fault: expected a JSON object, got NoneType" in out
    assert "**Gate verdict**: **INCONCLUSIVE**" in out
    # Non-comparable view excluded from the win denominator and called out.
    assert "could not be judged on 1 of 2 views" in out


def test_missing_keys_fail(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"candidate": "X"}), encoding="utf-8")
    proc = run_render(path)
    assert proc.returncode != 0
    assert "missing keys" in proc.stderr


def test_unknown_verdict_fails(tmp_path):
    proc = run_render(make_gate(tmp_path, verdict="MAYBE"))
    assert proc.returncode != 0
    assert "unknown verdict" in proc.stderr


def test_malformed_json_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    proc = run_render(path)
    assert proc.returncode != 0
    assert "cannot parse" in proc.stderr
