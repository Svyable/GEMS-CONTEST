"""Tests for scripts/gate_status.py (queued gate/compare inventory)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gate_status.py"


def run_status(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def make_root(tmp_path: Path, tasks: list[dict]) -> Path:
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text(json.dumps(tasks))
    (root / "scripts" / "queue" / "compare_seed.sh").write_text(
        "#!/bin/bash\nCMP=runs/compare/seed20260923-b16\n"
    )
    (root / "scripts" / "queue" / "compare_schedule.sh").write_text(
        "#!/bin/bash\nCMP=runs/compare/longer_schedule-b16\n"
    )
    return root


def gate_task(task_id: str, name: str, cand_dir: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/gate.sh {name} runs/gate/ref-b16 {cand_dir} "
        f"configs/cand.yaml \"hypothesis\"",
    }


def test_real_repo_inventory():
    """Against the committed repo: 1 decided gate, the rest pending, no orphans."""
    result = run_status(REPO)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "s3-gate-endpoint | gate | ENDPOINT-EXT-02 | DECIDED | REJECT" in out
    assert "10 gate decisions queued, 1 decided, 9 pending" in out
    assert "2 manual compares, 0 ran, 2 pending" in out
    assert "0 orphan gate JSONs" in out
    # The one partially-arrived candidate: spatial scored, fault/trace missing.
    assert (
        "s5b-gate-mumford_shah_log-b16 | gate | CAND-MUMFORD_SHAH_LOG-B16 | PENDING | "
        "candidate inputs present: spatial; missing: fault, trace" in out
    )


def test_synthetic_decided_pending_compare_and_orphan(tmp_path: Path):
    root = make_root(
        tmp_path,
        [
            gate_task("t-gate-ok", "CAND-OK", "runs/gate/cand-ok"),
            gate_task("t-gate-wait", "CAND-WAIT", "runs/gate/cand-wait"),
            {
                "id": "t-compare",
                "cmd": "/workspace/gems/queue/compare_seed.sh a b c d",
            },
        ],
    )
    gates = root / "results" / "gates"
    gates.mkdir(parents=True)
    (gates / "CAND-OK.json").write_text(
        json.dumps({"candidate": "CAND-OK", "verdict": "ACCEPT", "views": []})
    )
    (gates / "CAND-ORPHAN.json").write_text(
        json.dumps({"candidate": "CAND-ORPHAN", "verdict": "REJECT", "views": []})
    )
    (root / "results" / "runs" / "compare" / "seed20260923-b16").mkdir(parents=True)
    (root / "results" / "runs" / "compare" / "seed20260923-b16" / "compare.log").write_text("done\n")

    result = run_status(root)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "t-gate-ok | gate | CAND-OK | DECIDED | ACCEPT" in out
    assert "t-gate-wait | gate | CAND-WAIT | PENDING | no candidate gate inputs yet" in out
    assert "t-compare | compare | compare_seed | RAN |" in out
    assert "2 gate decisions queued, 1 decided, 1 pending" in out
    assert "1 manual compares, 1 ran, 0 pending" in out
    assert "1 orphan gate JSONs" in out
    assert "ORPHAN: results/gates/CAND-ORPHAN.json has no queued gate task" in out


def test_broken_gate_json_is_reported_not_skipped(tmp_path: Path):
    root = make_root(tmp_path, [gate_task("t-gate-bad", "CAND-BAD", "runs/gate/cand-bad")])
    gates = root / "results" / "gates"
    gates.mkdir(parents=True)
    (gates / "CAND-BAD.json").write_text("{not json")
    result = run_status(root)
    assert result.returncode == 0, result.stderr
    assert "t-gate-bad | gate | CAND-BAD | BROKEN | unparseable" in result.stdout

    (gates / "CAND-BAD.json").write_text(
        json.dumps({"candidate": "CAND-BAD", "verdict": "MAYBE", "views": []})
    )
    result = run_status(root)
    assert result.returncode == 0, result.stderr
    assert "t-gate-bad | gate | CAND-BAD | BROKEN |" in result.stdout
    assert "unknown verdict 'MAYBE'" in result.stdout


def test_missing_tasks_json_is_hard_error(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    result = run_status(root)
    assert result.returncode != 0
    assert "cannot read queue" in result.stderr
