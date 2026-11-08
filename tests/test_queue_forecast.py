"""Tests for scripts/queue_forecast.py (queue drain vs safe cutoff)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_forecast.py"

NOW = "2026-11-08T13:39:00+00:00"


def run_forecast(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--now", NOW, *extra],
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


def write(root: Path, rel: str, text: str = "{}") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def manifest(root: Path, run: str, created_utc: str) -> None:
    write(
        root,
        f"results/runs/{run}/manifest.json",
        json.dumps({"created_utc": created_utc}),
    )


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


def three_manifests(root: Path) -> None:
    # 3 runs over a 2.00-day window -> 1.50 runs/day. Named m1..m3 so
    # they never collide with a train task's run dir.
    manifest(root, "m1", "2026-10-09T00:00:00+00:00")
    manifest(root, "m2", "2026-10-10T00:00:00+00:00")
    manifest(root, "m3", "2026-10-11T00:00:00+00:00")


def test_rate_and_feasible_verdict(tmp_path: Path) -> None:
    tasks = [train_task(f"t{i}", f"runs/new{i}") for i in range(1, 4)]
    root = make_root(tmp_path, tasks)
    three_manifests(root)
    result = run_forecast(root, "--cutoff", "2026-11-20T13:39:00+00:00")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "OBSERVED RATE: 1.50 runs/day (3 manifests, 2026-10-09 to 2026-10-11, 2.00 days)" in out
    assert "REMAINING: 3 train tasks" in out
    assert "DRAIN: 2.0 days at the observed rate" in out
    assert "CUTOFF: 2026-11-20 13:39 UTC (12.0 days left)" in out
    assert "VERDICT: FEASIBLE (margin +10.0 days" in out
    # Silence is measured from the last manifest to --now.
    assert "SILENCE: 28.6 days since the last manifest" in out


def test_infeasible_verdict(tmp_path: Path) -> None:
    tasks = [train_task(f"t{i}", f"runs/new{i}") for i in range(1, 4)]
    root = make_root(tmp_path, tasks)
    three_manifests(root)
    result = run_forecast(root, "--cutoff", "2026-11-09T13:39:00+00:00")
    assert result.returncode == 0, result.stderr
    assert "VERDICT: INFEASIBLE (margin -1.0 days" in result.stdout


def test_tight_verdict_inside_buffer(tmp_path: Path) -> None:
    tasks = [train_task(f"t{i}", f"runs/new{i}") for i in range(1, 4)]
    root = make_root(tmp_path, tasks)
    three_manifests(root)
    result = run_forecast(root, "--cutoff", "2026-11-12T13:39:00+00:00")
    assert result.returncode == 0, result.stderr
    assert "VERDICT: TIGHT (margin +2.0 days" in result.stdout


def test_unknown_rate_with_one_manifest(tmp_path: Path) -> None:
    root = make_root(tmp_path, [train_task("t1", "runs/new1")])
    manifest(root, "m1", "2026-10-09T00:00:00+00:00")
    result = run_forecast(root)
    assert result.returncode == 0, result.stderr
    assert "OBSERVED RATE: UNKNOWN (1 usable manifests; need >= 2)" in result.stdout
    assert "VERDICT: UNKNOWN (no observed rate)" in result.stdout


def test_malformed_tasks_json_exits_2(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text("{not json")
    result = run_forecast(root)
    assert result.returncode == 2
    assert "cannot read" in result.stderr


def test_scored_unrecorded_is_a_lower_bound_note(tmp_path: Path) -> None:
    root = make_root(tmp_path, [train_task("t1", "runs/new1")])
    three_manifests(root)
    write(root, "results/runs/r4/scores/spatial.json")
    result = run_forecast(root)
    assert result.returncode == 0, result.stderr
    assert "(+1 scored-but-unrecorded run(s): the rate above is a lower bound)" in result.stdout


def test_per_decision_train_counts(tmp_path: Path) -> None:
    # Incumbent gate dir fully present; candidate spatial needs one
    # pending train, fault/trace have no producer at all.
    tasks = [
        train_task("t1", "runs/r1"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
        gate_task("g1", "G-ONE", "runs/gate/inc", "runs/gate/cand"),
        gate_task("g2", "G-READY", "runs/gate/inc2", "runs/gate/cand2"),
    ]
    root = make_root(tmp_path, tasks)
    three_manifests(root)
    for view in ("spatial", "fault", "trace"):
        write(root, f"runs/gate/inc/{view}.json")
        write(root, f"runs/gate/inc2/{view}.json")
        write(root, f"runs/gate/cand2/{view}.json")
    result = run_forecast(root, "--cutoff", "2026-11-20T13:39:00+00:00")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "G-ONE" in out and "1 trains, ~0.7d, 3 missing inputs (t1)" in out
    assert "G-READY" in out and "0 trains, ~0.0d, 0 missing inputs" in out


def test_real_repo_locked_numbers(tmp_path: Path) -> None:
    # Locks the forecast against the committed tree at --now; update
    # when the box commits new manifests or the queue changes.
    result = run_forecast(REPO)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "OBSERVED RATE: 2.86 runs/day (6 manifests, 2026-10-09 to 2026-10-11, 2.10 days)" in out
    assert "(+1 scored-but-unrecorded run(s)" in out
    assert "SILENCE: 27.6 days since the last manifest" in out
    assert "REMAINING: 28 train tasks" in out
    assert "DRAIN: 9.8 days at the observed rate" in out
    assert "VERDICT: FEASIBLE (margin +15.6 days" in out
    assert "CAND-CURVED_CONTINUATION-B16" in out
    assert "0 trains, ~0.0d" in out
