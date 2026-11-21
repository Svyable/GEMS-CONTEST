"""Tests for scripts/queue_forecast.py (queue drain vs safe cutoff)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_forecast.py"

NOW = "2026-11-08T13:39:00+00:00"


def run_forecast(root: Path, *extra: str, now: str = NOW) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--now", now, *extra],
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


def test_resume_scenario_tight_and_reachable(tmp_path: Path) -> None:
    # Same fixture as test_per_decision_train_counts: rate 1.50/day,
    # drain 2.0d. Resuming 2026-11-28 leaves 5.9 days to the cutoff,
    # so the full drain fits but inside the 5-day buffer (TIGHT), and
    # both decisions are reachable.
    tasks = [
        train_task("t1", "runs/r1"),
        # Two extra pending trains with no decision feeders, so the
        # drain is 3 trains / 1.50 = 2.0d while G-ONE still needs 1.
        train_task("t2", "runs/new2"),
        train_task("t3", "runs/new3"),
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
    result = run_forecast(
        root,
        "--cutoff",
        "2026-12-03T22:00:00+00:00",
        "--resume-date",
        "2026-11-28",
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "SCENARIO: box resumes 2026-11-28 (5.9 days from resume to the cutoff)" in out
    assert "CUTOFF: 2026-12-03 22:00 UTC (5.9 days left from resume 2026-11-28)" in out
    assert "VERDICT: TIGHT (margin +3.9 days" in out
    assert "LAST RESUME for full drain (no buffer): 2026-12-01 22:00 UTC" in out
    assert "with 5-day buffer: 2026-11-26 22:00 UTC" in out
    assert "G-ONE" in out and "1 trains, ~0.7d, 3 missing inputs (t1) — REACHABLE" in out
    assert "G-READY" in out and "0 trains, ~0.0d, 0 missing inputs — REACHABLE" in out


def test_resume_scenario_infeasible_with_unreachable_decision(tmp_path: Path) -> None:
    # Resuming on the cutoff day itself: 0.9 days left, drain 1.3d,
    # so the full drain is INFEASIBLE and the 2-train gate is out of
    # reach while the 0-train gate still fits.
    tasks = [
        train_task("t1", "runs/r1"),
        train_task("t2", "runs/r2"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
        score_task("s2", "runs/r2", "fault", "runs/gate/cand"),
        gate_task("g1", "G-TWO", "runs/gate/inc", "runs/gate/cand"),
        gate_task("g2", "G-READY", "runs/gate/inc2", "runs/gate/cand2"),
    ]
    root = make_root(tmp_path, tasks)
    three_manifests(root)
    for view in ("spatial", "fault", "trace"):
        write(root, f"runs/gate/inc/{view}.json")
        write(root, f"runs/gate/inc2/{view}.json")
        write(root, f"runs/gate/cand2/{view}.json")
    result = run_forecast(
        root,
        "--cutoff",
        "2026-12-03T22:00:00+00:00",
        "--resume-date",
        "2026-12-03",
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "SCENARIO: box resumes 2026-12-03 (0.9 days from resume to the cutoff)" in out
    assert "VERDICT: INFEASIBLE (margin -0.4 days" in out
    assert (
        "G-TWO" in out
        and "2 trains, ~1.3d, 3 missing inputs (t1, t2) — OUT OF REACH (short by 0.4d)" in out
    )
    assert "G-READY" in out and "0 trains, ~0.0d, 0 missing inputs — REACHABLE" in out


def test_resume_date_in_the_past_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, [train_task("t1", "runs/new1")])
    three_manifests(root)
    result = run_forecast(root, "--resume-date", "2026-11-01")
    assert result.returncode == 2
    assert "in the past" in result.stderr


def test_resume_date_malformed_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, [train_task("t1", "runs/new1")])
    three_manifests(root)
    result = run_forecast(root, "--resume-date", "28-11-2026")
    assert result.returncode == 2
    assert "expected YYYY-MM-DD" in result.stderr


def test_real_repo_resume_nov_28(tmp_path: Path) -> None:
    # Locks the scenario against the committed tree: resuming
    # 2026-11-28 leaves 5.9 days, the 9.8-day drain no longer fits
    # (INFEASIBLE, -3.9d), but every decision's critical path does —
    # the curved gate needs 0 trains and the s7 loss gate 3 (~1.0d).
    # The last resume date for the full drain is 2026-11-24; the
    # buffered one (2026-11-19) has already passed at --now.
    result = run_forecast(
        REPO,
        "--resume-date",
        "2026-11-28",
        now="2026-11-21T13:39:00+00:00",
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "SCENARIO: box resumes 2026-11-28 (5.9 days from resume to the cutoff)" in out
    assert "VERDICT: INFEASIBLE (margin -3.9 days" in out
    assert (
        "LAST RESUME for full drain (no buffer): 2026-11-24 03:09 UTC; "
        "with 5-day buffer: 2026-11-19 03:09 UTC (already passed)" in out
    )
    assert "CAND-CURVED_CONTINUATION-B16" in out
    assert "0 trains, ~0.0d, 5 missing inputs — REACHABLE" in out
    assert "CAND-LOSS_DISTANCE_TVERSKY-B16" in out
    assert "3 trains, ~1.0d, 5 missing inputs" in out and "— REACHABLE" in out
