"""Tests for scripts/queue_status.py (queued production-task inventory)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_status.py"


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


def train_task(task_id: str, run: str, scheme: str = "spatial") -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/train_folds.sh {run} configs/ref.yaml "
        f"{scheme} data/processed/cv-{scheme}-v1.tif 16",
    }


def score_task(task_id: str, run: str, scheme: str = "spatial") -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/score_view.sh {run} {scheme} "
        f"data/processed/cv-{scheme}-v1.tif runs/gate/ref-b16",
    }


def record_task(task_id: str, run_id: str, run: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/record.sh {run_id} {run} "
        'configs/ref.yaml "hypothesis" "notes"',
    }


def test_real_repo_inventory():
    """Against the committed repo: 6 recorded runs, 1 scored, 34 pending."""
    result = run_status(REPO)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert (
        "Summary: 126 queued tasks (111 production, 10 gates, 2 compares); "
        "production 18 recorded, 3 scored, 0 trained, 90 pending; "
        "41 queued runs — 6 recorded, 1 scored, 0 trained, 34 pending." in out
    )
    assert (
        "Summary: 10 gate decisions queued, 1 decided, 9 pending; "
        "2 manual compares, 0 ran, 2 pending; 0 orphan gate JSONs; "
        "5 parked tasks." in out
    )
    assert "s3-gate-endpoint | gate | ENDPOINT-EXT-02 | DECIDED | REJECT" in out
    assert (
        "s5-record-ref-fault-b32 | record | ref-cv-fault-b32 | SCORED | "
        "score fault.json present, no manifest" in out
    )
    assert (
        "s5b-assemble-ref-b16 | assemble | runs/gate/ref-b16 | PENDING | "
        "no assembled files yet; missing: spatial.json, fault.json" in out
    )
    assert "runs/ref-cv-fault-b32 | SCORED | record+score+train | score present, no manifest" in out
    assert "runs/ref-cv-trace-b32 | PENDING | record+score+train | no committed evidence" in out
    assert (
        "PARKED: s6-gate-mumford_shah_log-trace, s6-gate-steerable_filter-trace, "
        "s21-rival-threshold-spatial-b16, s21-rival-threshold-fault-b16, "
        "s21-rival-threshold-trace-b16" in out
    )
    assert "ORPHAN" not in out


def test_synthetic_stages_and_rollup(tmp_path: Path):
    """Recorded/scored/trained/pending + relocated score + partial assemble."""
    tasks = [
        train_task("t1-train", "runs/run-a"),
        score_task("t1-score", "runs/run-a"),
        record_task("t1-record", "run-a", "runs/run-a"),
        train_task("t2-train", "runs/run-b"),
        score_task("t2-score", "runs/run-b"),
        record_task("t2-record", "run-b", "runs/run-b"),
        train_task("t3-train", "runs/run-c"),
        train_task("t4-train", "runs/run-d"),
        score_task("t4-score", "runs/run-d"),
        record_task("t4-record", "run-d", "runs/run-d"),
        train_task("t5-train", "runs/run-e", "fault"),
        {
            "id": "t6-view",
            "cmd": "/workspace/gems/queue/endpoint_view.sh runs/run-a "
            "runs/endpoint-a spatial && /workspace/gems/queue/score_view.sh "
            "runs/endpoint-a spatial data/processed/cv-spatial-v1.tif "
            "runs/gate/endpoint",
        },
        {
            "id": "t7-assemble",
            "cmd": "mkdir -p runs/gate/ref-b16 && cp runs/gate/ref-full/spatial.json "
            "runs/gate/ref-full/fault.json runs/gate/ref-b16/ && ls runs/gate/ref-b16",
        },
        {
            "id": "t8-gate",
            "cmd": "/workspace/gems/queue/gate.sh CAND-X runs/gate/ref-b16 "
            'runs/gate/cand-x configs/cand.yaml "hypothesis"',
        },
        {
            "id": "t9-compare",
            "cmd": "/workspace/gems/queue/compare_seed.sh runs/gate/ref-b16 "
            "runs/gate/ref-seed runs/ref-cv runs/ref-cv-seed",
        },
        {
            "id": "t10-wait",
            "cmd": "while kill -0 123 2>/dev/null; do sleep 30; done; echo done",
        },
    ]
    root = make_root(tmp_path, tasks)
    # run-a: fully recorded (manifest + relocated top-level score).
    (root / "results" / "runs" / "run-a").mkdir(parents=True)
    (root / "results" / "runs" / "run-a" / "manifest.json").write_text("{}")
    (root / "results" / "runs" / "run-a" / "spatial.json").write_text("{}")
    # run-b: scored only (scores/ dir, no manifest).
    (root / "results" / "runs" / "run-b" / "scores").mkdir(parents=True)
    (root / "results" / "runs" / "run-b" / "scores" / "spatial.json").write_text("{}")
    # run-c: trained only (fold files, no score).
    (root / "results" / "runs" / "run-c").mkdir(parents=True)
    for k in range(5):
        (root / "results" / "runs" / "run-c" / f"fold-{k}.json").write_text("{}")
    # endpoint-a: scored via its gate dir copy.
    (root / "results" / "runs" / "gate" / "endpoint").mkdir(parents=True)
    (root / "results" / "runs" / "gate" / "endpoint" / "spatial.json").write_text("{}")
    # assemble: only spatial.json landed in the gate dir.
    (root / "results" / "runs" / "gate" / "ref-b16").mkdir(parents=True)
    (root / "results" / "runs" / "gate" / "ref-b16" / "spatial.json").write_text("{}")
    # gate CAND-X decided; compare ran.
    (root / "results" / "gates").mkdir(parents=True)
    (root / "results" / "gates" / "CAND-X.json").write_text(json.dumps({"verdict": "ACCEPT"}))
    (root / "results" / "runs" / "compare" / "seed20260923-b16").mkdir(parents=True)
    (root / "results" / "runs" / "compare" / "seed20260923-b16" / "compare.log").write_text(
        "paired deltas"
    )
    # Orphans: an unqueued gate JSON and an unqueued run dir.
    (root / "results" / "gates" / "EXTRA.json").write_text(json.dumps({"verdict": "REJECT"}))
    (root / "results" / "runs" / "stray-run").mkdir(parents=True)

    result = run_status(root)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "t1-train | train | runs/run-a | RECORDED | manifest.json present" in out
    assert "t2-score | score | runs/run-b | SCORED | score spatial.json present, no manifest" in out
    # The record task has no scheme of its own; any view score counts.
    assert "t2-record | record | run-b | SCORED | score spatial.json present, no manifest" in out
    assert "t3-train | train | runs/run-c | TRAINED | 5/5 fold files present, no score" in out
    assert "t4-train | train | runs/run-d | PENDING | no committed evidence" in out
    assert (
        "t6-view | view | endpoint_view | SCORED | score spatial.json present, no manifest" in out
    )
    assert (
        "t7-assemble | assemble | runs/gate/ref-b16 | PARTIAL | "
        "present: spatial.json; missing: fault.json" in out
    )
    assert "t8-gate | gate | CAND-X | DECIDED | ACCEPT" in out
    assert "t9-compare | compare | compare_seed | RAN |" in out
    assert "t10-wait | housekeeping | while | HOUSEKEEPING |" in out
    assert "runs/run-a | RECORDED | record+score+train | manifest.json present" in out
    assert "runs/run-b | SCORED | record+score+train | score present, no manifest" in out
    assert "runs/run-c | TRAINED | train | fold files present, no score" in out
    assert "runs/endpoint-a | SCORED | view | score present, no manifest" in out
    assert "ORPHAN: results/gates/EXTRA.json has no queued gate task" in out
    assert "ORPHAN: results/runs/stray-run has no queued task referencing it" in out
    assert "0 parked tasks." in out


def test_broken_gate_and_unknown_compare(tmp_path: Path):
    """Unparseable gate JSON is BROKEN; a compare script without CMP= is UNKNOWN."""
    tasks = [
        {
            "id": "g1",
            "cmd": "/workspace/gems/queue/gate.sh BAD runs/gate/ref runs/gate/cand "
            'configs/c.yaml "h"',
        },
        {
            "id": "c1",
            "cmd": "/workspace/gems/queue/compare_seed.sh runs/gate/ref runs/gate/x "
            "runs/ref-cv runs/ref-cv-seed",
        },
    ]
    root = make_root(tmp_path, tasks)
    (root / "results" / "gates").mkdir(parents=True)
    (root / "results" / "gates" / "BAD.json").write_text("{not json")
    (root / "scripts" / "queue" / "compare_seed.sh").write_text("#!/bin/bash\n# no CMP here\n")
    result = run_status(root)
    assert result.returncode == 0, result.stderr
    assert "g1 | gate | BAD | BROKEN | unparseable BAD.json" in result.stdout
    assert "c1 | compare | compare_seed | UNKNOWN | could not read CMP=" in result.stdout


def test_malformed_and_missing_queue(tmp_path: Path):
    """A malformed or missing tasks.json is a hard error."""
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text("{broken")
    assert run_status(root).returncode != 0

    root2 = tmp_path / "repo2"
    (root2 / "scripts" / "queue").mkdir(parents=True)
    assert run_status(root2).returncode != 0
