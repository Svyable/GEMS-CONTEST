"""Tests for scripts/queue_progress.py (live decision progress, step 53)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_progress.py"

NOW = "2026-11-28T00:00:00+00:00"


def run_progress(
    root: Path, queue_dir: Path, *extra: str, now: str = NOW
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--queue-dir",
            str(queue_dir),
            "--now",
            now,
            *extra,
        ],
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


def make_queue_dir(tmp_path: Path, tasks: list[dict]) -> Path:
    queue_dir = tmp_path / "queue"
    (queue_dir / "state").mkdir(parents=True)
    (queue_dir / "tasks.json").write_text(json.dumps(tasks))
    return queue_dir


def mark(queue_dir: Path, task_id: str, suffix: str) -> None:
    (queue_dir / "state" / f"{task_id}.{suffix}").write_text("2026-11-28T00:00:00\n")


def train_task(task_id: str, run: str, scheme: str = "spatial") -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/train_folds.sh {run} configs/ref.yaml "
        f"{scheme} data/processed/cv-{scheme}-v1.tif 16",
    }


def score_task(task_id: str, run: str, scheme: str, gate_dir: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/score_view.sh {run} {scheme} "
        f"data/processed/cv-{scheme}-v1.tif {gate_dir}",
    }


def gate_task(task_id: str, name: str, inc: str, cand: str) -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/gate.sh {name} {inc} {cand} configs/cand.yaml "hyp"',
    }


def two_gate_tasks() -> list[dict]:
    return [
        train_task("t1", "runs/cand-a-spatial"),
        score_task("s1", "runs/cand-a-spatial", "spatial", "runs/gate/cand-a"),
        train_task("t2", "runs/cand-b-spatial"),
        score_task("s2", "runs/cand-b-spatial", "spatial", "runs/gate/cand-b"),
        gate_task("g1", "GATE-A", "runs/gate/ref", "runs/gate/cand-a"),
        gate_task("g2", "GATE-B", "runs/gate/ref", "runs/gate/cand-b"),
    ]


def test_pending_and_in_flight_and_landed(tmp_path: Path) -> None:
    tasks = two_gate_tasks()
    root = make_root(tmp_path, tasks)
    queue_dir = make_queue_dir(tmp_path, tasks)
    result = run_progress(root, queue_dir, "--rate", "1.0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PROGRESS: 0 of 2 decisions landed; 0 of 2 train tasks done" in out
    assert "PENDING   gate    GATE-A" in out
    assert "PENDING   gate    GATE-B" in out
    assert "VERDICT: IN PROGRESS" in out

    # A feeder train done on the box moves only its gate to IN-FLIGHT.
    mark(queue_dir, "t1", "done")
    result = run_progress(root, queue_dir, "--rate", "1.0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "IN-FLIGHT gate    GATE-A" in out
    assert "PENDING   gate    GATE-B" in out

    # The gate task's own .done marker lands the decision.
    mark(queue_dir, "g1", "done")
    result = run_progress(root, queue_dir, "--rate", "1.0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PROGRESS: 1 of 2 decisions landed" in out
    assert "LANDED    gate    GATE-A" in out
    assert "done on box, not yet committed" in out


def test_failed_task_blocks_downstream_decisions(tmp_path: Path) -> None:
    tasks = two_gate_tasks()
    root = make_root(tmp_path, tasks)
    queue_dir = make_queue_dir(tmp_path, tasks)
    mark(queue_dir, "t1", "failed")
    result = run_progress(root, queue_dir, "--rate", "1.0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "FAILED: t1" in out
    assert "BLOCKED   gate    GATE-A" in out
    assert "blocked by failed t1" in out
    assert "BLOCKED   gate    GATE-B" in out
    assert "VERDICT: BLOCKED" in out


def test_failed_decision_task_reports_failed(tmp_path: Path) -> None:
    tasks = two_gate_tasks()
    root = make_root(tmp_path, tasks)
    queue_dir = make_queue_dir(tmp_path, tasks)
    mark(queue_dir, "g1", "failed")
    result = run_progress(root, queue_dir, "--rate", "1.0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "FAILED    gate    GATE-A" in out
    # GATE-B sits after the failed gate task in list order, so the
    # stopped runner blocks it too.
    assert "BLOCKED   gate    GATE-B" in out
    assert "VERDICT: BLOCKED" in out


def test_eta_on_track_and_at_risk(tmp_path: Path) -> None:
    tasks = two_gate_tasks()
    root = make_root(tmp_path, tasks)
    queue_dir = make_queue_dir(tmp_path, tasks)
    # 2 trains ahead of GATE-B at 1.0/day from Nov 28 -> ETA Nov 30.
    result = run_progress(root, queue_dir, "--rate", "1.0", "--cutoff", "2026-12-03T22:00:00+00:00")
    assert result.returncode == 0, result.stderr
    assert "ETA 2026-11-30 00:00 UTC — ON-TRACK" in result.stdout
    assert "PROJECTED BY CUTOFF: 2 of 2 decisions" in result.stdout

    result = run_progress(root, queue_dir, "--rate", "1.0", "--cutoff", "2026-11-29T00:00:00+00:00")
    assert result.returncode == 0, result.stderr
    assert "AT-RISK" in result.stdout
    assert "PROJECTED BY CUTOFF: 0 of 2 decisions" in result.stdout


def test_committed_verdict_lands_without_marker(tmp_path: Path) -> None:
    tasks = two_gate_tasks()
    root = make_root(tmp_path, tasks)
    queue_dir = make_queue_dir(tmp_path, tasks)
    gates_dir = root / "results" / "gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / "GATE-A.json").write_text(json.dumps({"verdict": "REJECT"}))
    result = run_progress(root, queue_dir, "--rate", "1.0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "LANDED    gate    GATE-A" in out
    assert "REJECT (committed)" in out
    assert "PROGRESS: 1 of 2 decisions landed" in out


def test_current_task_and_running_decision(tmp_path: Path) -> None:
    tasks = two_gate_tasks()
    root = make_root(tmp_path, tasks)
    queue_dir = make_queue_dir(tmp_path, tasks)
    (queue_dir / "status.json").write_text(json.dumps({"current": "g1", "note": "RUNNING"}))
    result = run_progress(root, queue_dir, "--rate", "1.0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "CURRENT: g1" in out
    assert "RUNNING   gate    GATE-A" in out


def test_unknown_rate_has_no_eta(tmp_path: Path) -> None:
    tasks = two_gate_tasks()
    root = make_root(tmp_path, tasks)
    queue_dir = make_queue_dir(tmp_path, tasks)
    result = run_progress(root, queue_dir)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "RATE: UNKNOWN" in out
    assert ", ETA " not in out


def test_missing_tasks_json_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, [])
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    result = run_progress(root, queue_dir)
    assert result.returncode == 2
    assert "cannot read" in result.stderr


def test_real_repo_sprint_progress(tmp_path: Path) -> None:
    """Locked real-repo case: the committed chained sprint, empty box state."""
    queue_dir = tmp_path / "queue"
    (queue_dir / "state").mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(REPO),
            "--queue-dir",
            str(queue_dir),
            "--tasks",
            str(REPO / "scripts" / "queue" / "tasks.sprint.json"),
            "--now",
            NOW,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # 10 gates + 2 compares; ENDPOINT-EXT-02 is decided in the
    # committed tree, the other 11 are pending (chain order).
    assert "PROGRESS: 1 of 12 decisions landed" in out
    assert "LANDED    gate    ENDPOINT-EXT-02" in out
    assert "REJECT (committed)" in out
    pending_lines = [line for line in out.splitlines() if "PENDING" in line]
    assert len(pending_lines) == 11
    assert "CAND-LOSS_DISTANCE_TVERSKY-B16" in pending_lines[0]
    assert "CAND-CURVED_CONTINUATION-B16" in pending_lines[1]
    assert "compare_seed" in pending_lines[2]
    assert "VERDICT: IN PROGRESS" in out


def test_real_repo_sprint_s7_lands(tmp_path: Path) -> None:
    """Marking the 21-task s7 prefix done lands s7; curved is next."""
    sprint = json.loads((REPO / "scripts" / "queue" / "tasks.sprint.json").read_text())
    queue_dir = tmp_path / "queue"
    (queue_dir / "state").mkdir(parents=True)
    for task in sprint[:21]:
        mark(queue_dir, task["id"], "done")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(REPO),
            "--queue-dir",
            str(queue_dir),
            "--tasks",
            str(REPO / "scripts" / "queue" / "tasks.sprint.json"),
            "--now",
            NOW,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PROGRESS: 2 of 12 decisions landed" in out
    assert "LANDED    gate    CAND-LOSS_DISTANCE_TVERSKY-B16" in out
    pending_lines = [line for line in out.splitlines() if "PENDING" in line or "IN-FLIGHT" in line]
    assert "CAND-CURVED_CONTINUATION-B16" in pending_lines[0]
