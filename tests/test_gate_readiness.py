"""Tests for scripts/gate_readiness.py (gate-input validity audit)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gate_readiness.py"


def run_readiness(root: Path) -> subprocess.CompletedProcess[str]:
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
    return root


def write_gate_file(root: Path, gate_dir: str, filename: str, source: Path) -> None:
    path = root / "results" / gate_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, path)


GATE_TASK = {
    "id": "sX-gate-demo",
    "cmd": "/workspace/gems/queue/gate.sh DEMO runs/gate/ref-x runs/gate/cand-x "
    'configs/demo.yaml "Demo hypothesis."',
}
SCORE_TASK = {
    "id": "sX-score-cand-x-spatial",
    "cmd": "/workspace/gems/queue/score_view.sh runs/cand-x-spatial spatial "
    "data/processed/cv-spatial-v1.tif runs/gate/cand-x",
}
ASSEMBLE_TASK = {
    "id": "sX-assemble-ref-x",
    "cmd": "mkdir -p runs/gate/ref-x && cp runs/gate/ref-full/spatial.json "
    "runs/gate/ref-full/fault.json runs/gate/ref-x/ && ls runs/gate/ref-x",
}


def test_schema2_file_is_valid_and_schema1_is_invalid(tmp_path: Path) -> None:
    root = make_root(tmp_path, [GATE_TASK])
    write_gate_file(
        root, "runs/gate/ref-x", "spatial.json", REPO / "results/runs/gate/ref-b32/spatial.json"
    )
    write_gate_file(
        root,
        "runs/gate/cand-x",
        "spatial.json",
        REPO / "results/runs/ref-cv-spatial-full/spatial.json",
    )
    proc = run_readiness(root)
    assert proc.returncode == 0
    assert "VALID   incumbent  spatial" in proc.stdout
    assert "INVALID candidate  spatial" in proc.stdout
    assert "evaluation_protocol missing aggregation" in proc.stdout


def test_missing_input_names_producers(tmp_path: Path) -> None:
    root = make_root(tmp_path, [GATE_TASK, SCORE_TASK, ASSEMBLE_TASK])
    proc = run_readiness(root)
    assert proc.returncode == 0
    assert "MISSING incumbent  spatial" in proc.stdout
    assert "sX-assemble-ref-x" in proc.stdout
    assert "MISSING candidate  spatial" in proc.stdout
    assert "sX-score-cand-x-spatial" in proc.stdout


def test_decided_gate_is_excluded(tmp_path: Path) -> None:
    root = make_root(tmp_path, [GATE_TASK])
    verdict_dir = root / "results" / "gates"
    verdict_dir.mkdir(parents=True)
    (verdict_dir / "DEMO.json").write_text(json.dumps({"verdict": "REJECT"}))
    proc = run_readiness(root)
    assert proc.returncode == 0
    assert "PENDING GATES: 0 (decided: 1)" in proc.stdout
    assert "GATE DEMO" not in proc.stdout


def test_malformed_tasks_is_hard_error(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text("{not json")
    proc = run_readiness(root)
    assert proc.returncode == 2


def test_unparseable_gate_file_is_invalid(tmp_path: Path) -> None:
    root = make_root(tmp_path, [GATE_TASK])
    path = root / "results" / "runs" / "gate" / "ref-x"
    path.mkdir(parents=True)
    (path / "spatial.json").write_text("{broken")
    proc = run_readiness(root)
    assert proc.returncode == 0
    assert "INVALID incumbent  spatial" in proc.stdout
    assert "unparseable JSON" in proc.stdout


def test_real_repo_golden_counts() -> None:
    """Pin the committed state: 9 pending gates, 12 valid inputs, 0 invalid.

    The full-run schema-1 scores are evidence, not gate inputs: they must
    show up as INVALID evidence, never as usable gate inputs.
    """
    proc = run_readiness(REPO)
    assert proc.returncode == 0
    assert "PENDING GATES: 9 (decided: 1)" in proc.stdout
    assert "INPUTS: 12 VALID, 0 INVALID, 42 MISSING" in proc.stdout
    assert "INVALID schema=1 results/runs/ref-cv-spatial-full/spatial.json" in proc.stdout
    assert "INVALID schema=1 results/runs/ref-cv-fault-full/fault.json" in proc.stdout
