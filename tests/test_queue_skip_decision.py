"""Tests for scripts/queue_skip_decision.py (step 52).

The runner stops the whole queue on the first failed task and the
doctor's fix retries the same task — right for a fixable cause, wrong
for a permanent one: after the 2026-11-28 resumption the chained
sprint (step 51) has ~5.9 days to land 7 of 11 decisions, and one
permanently failing train in the s7 segment would stall the chain at
decision 1. The skip tool removes one pending decision's non-shared
closure from a queue file so the chain continues at the next
decision, keeping shared feeders (s26 re-scores, ref assembles,
ref-trace-b32) that other decisions still need.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_skip_decision.py"
SPRINT = REPO / "scripts" / "queue_sprint.py"


def run_skip(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *extra],
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


def score_task(task_id: str, run: str, scheme: str, gate_dir: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/score_view.sh {run} {scheme} "
        f"data/processed/cv-{scheme}-v1.tif {gate_dir}",
    }


def record_task(task_id: str, name: str, run: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/record.sh {name} {run} configs/cand.yaml",
    }


def assemble_task(task_id: str, src_dir: str, dest: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"mkdir -p {dest} && cp {src_dir}/spatial.json {src_dir}/fault.json "
        f"{dest}/ && ls {dest}/",
    }


def gate_task(task_id: str, name: str, inc: str, cand: str) -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/gate.sh {name} {inc} {cand} configs/cand.yaml "hyp"',
    }


def synthetic_two_gate_queue() -> list[dict]:
    """Two gates sharing one incumbent; a filler train in no closure."""
    tasks: list[dict] = [
        train_task("ref-train-spatial", "runs/ref-spatial"),
        score_task("ref-score-spatial", "runs/ref-spatial", "spatial", "runs/gate/ref-full"),
        train_task("ref-train-fault", "runs/ref-fault"),
        score_task("ref-score-fault", "runs/ref-fault", "fault", "runs/gate/ref-full"),
        assemble_task("assemble-ref", "runs/gate/ref-full", "runs/gate/ref-b16"),
    ]
    for tag, name in (("a", "G-FIRST"), ("b", "G-SECOND")):
        tasks.append(train_task(f"{tag}-train", f"runs/cand{tag}-spatial"))
        tasks.append(
            score_task(f"{tag}-score", f"runs/cand{tag}-spatial", "spatial", f"runs/gate/cand{tag}")
        )
        tasks.append(record_task(f"{tag}-record", f"cand{tag}-spatial", f"runs/cand{tag}-spatial"))
        tasks.append(gate_task(f"{tag}-gate", name, "runs/gate/ref-b16", f"runs/gate/cand{tag}"))
    tasks.append(train_task("filler-train", "runs/filler"))
    return tasks


def test_skip_dry_run_removes_only_own_closure(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    queue = root / "scripts" / "queue" / "tasks.json"
    before = queue.read_text(encoding="utf-8")
    result = run_skip(root, "--decision", "G-FIRST")
    assert result.returncode == 0, result.stderr
    assert queue.read_text(encoding="utf-8") == before  # dry run writes nothing
    assert "SKIP gate G-FIRST: closure 9 tasks, remove 4, keep 5 shared" in result.stdout
    assert "1 pending train(s) saved" in result.stdout
    for task_id in ("a-train", "a-score", "a-record", "a-gate"):
        assert f"REMOVE {task_id}" in result.stdout
    assert "KEEP-SHARED assemble-ref (also needed by G-SECOND)" in result.stdout
    assert "QUEUE: 10 tasks remain, 1 pending decisions remain" in result.stdout
    assert "DRY RUN: nothing written" in result.stdout


def test_skip_write_applies_backs_up_and_cleans_markers(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    queue = root / "scripts" / "queue" / "tasks.json"
    state = queue.parent / "state"
    state.mkdir()
    (state / "a-train.failed").write_text("2026-11-29 rc=3")
    (state / "a-train.started").write_text("2026-11-29")
    (state / "b-train.failed").write_text("2026-11-29 rc=3")
    result = run_skip(root, "--decision", "G-FIRST", "--write")
    assert result.returncode == 0, result.stderr
    ids = [t["id"] for t in json.loads(queue.read_text(encoding="utf-8"))]
    assert ids == [
        "ref-train-spatial",
        "ref-score-spatial",
        "ref-train-fault",
        "ref-score-fault",
        "assemble-ref",
        "b-train",
        "b-score",
        "b-record",
        "b-gate",
        "filler-train",
    ]
    backups = list(queue.parent.glob("tasks.json.bak.*"))
    assert len(backups) == 1
    assert len(json.loads(backups[0].read_text(encoding="utf-8"))) == 14
    # Removed task's markers are gone; unrelated markers untouched.
    assert not (state / "a-train.failed").exists()
    assert not (state / "a-train.started").exists()
    assert (state / "b-train.failed").exists()
    assert "WROTE:" in result.stdout


def test_skip_keeps_done_tasks(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    state = root / "scripts" / "queue" / "state"
    state.mkdir()
    (state / "a-train.done").write_text("2026-11-29 rc=0")
    result = run_skip(root, "--decision", "G-FIRST")
    assert result.returncode == 0, result.stderr
    assert "keep 1 already-done" in result.stdout
    assert "KEEP-DONE a-train (.done marker exists)" in result.stdout
    assert "REMOVE a-train" not in result.stdout
    assert "0 pending train(s) saved" in result.stdout


def test_skip_unknown_decision_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    result = run_skip(root, "--decision", "G-NOPE", "--write")
    assert result.returncode == 2
    assert "unknown decision" in result.stderr
    queue = root / "scripts" / "queue" / "tasks.json"
    assert len(json.loads(queue.read_text(encoding="utf-8"))) == 14


def test_skip_decided_on_box_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    state = root / "scripts" / "queue" / "state"
    state.mkdir()
    (state / "a-gate.done").write_text("2026-11-29 rc=0")
    result = run_skip(root, "--decision", "G-FIRST")
    assert result.returncode == 2
    assert "already decided on the box" in result.stderr


def test_skip_orphan_guard_and_force(tmp_path: Path) -> None:
    """A view task on cand-a placed after a-score is in no closure but
    consumes cand-a's LOFO output: removing a-train/a-score would
    orphan it, so the skip refuses until --force."""
    tasks = synthetic_two_gate_queue()
    view = {
        "id": "a-view",
        "cmd": "/workspace/gems/queue/endpoint_view.sh runs/canda-spatial "
        "runs/endpoint-a spatial && /workspace/gems/queue/score_view.sh "
        "runs/endpoint-a spatial data/processed/cv-spatial-v1.tif runs/gate/endpoint-a",
    }
    # Insert after a-gate (index 8) so it is no writer-before of a-score.
    tasks.insert(9, view)
    root = make_root(tmp_path, tasks)
    result = run_skip(root, "--decision", "G-FIRST")
    assert result.returncode == 2, result.stdout
    assert "orphan" in result.stderr
    forced = run_skip(root, "--decision", "G-FIRST", "--force")
    assert forced.returncode == 0, forced.stderr
    assert "NEW NO-PRODUCER: a-view" in forced.stdout


def test_real_repo_skip_s7_locked(tmp_path: Path) -> None:
    """Skipping the s7 loss gate on the real queue removes exactly its
    own 10 tasks, keeps all 11 shared feeders, and the rebuilt chain
    lands 7 of the remaining 10 decisions from a Nov 28 resume —
    skeleton_recall, first past the cutoff in the full chain, now fits."""
    queue_copy = tmp_path / "tasks.json"
    queue_copy.write_text(
        (REPO / "scripts/queue/tasks.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(REPO),
            "--queue",
            str(queue_copy),
            "--decision",
            "CAND-LOSS_DISTANCE_TVERSKY-B16",
            "--write",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "SKIP gate CAND-LOSS_DISTANCE_TVERSKY-B16: closure 21 tasks, "
        "remove 10, keep 11 shared" in result.stdout
    )
    assert "3 pending train(s) saved" in result.stdout
    assert "QUEUE: 119 tasks remain, 10 pending decisions remain" in result.stdout
    assert "ORDER AUDIT after skip: 0 violations, 2 notes (0 new)" in result.stdout
    remaining_ids = {t["id"] for t in json.loads(queue_copy.read_text(encoding="utf-8"))}
    for task_id in (
        "s26-rescore-ref-spatial-full",
        "s26-rescore-ref-fault-full",
        "s26-reassemble-ref-b16",
        "s5b-assemble-ref-b16",
        "s7-assemble-ref-b16",
    ):
        assert task_id in remaining_ids
    assert "s7-train-loss_distance_tversky-spatial" not in remaining_ids
    assert "s7-gate-loss_distance_tversky-b16" not in remaining_ids

    # Rebuild the chain on the skipped queue: 7 of 10 fit from Nov 28,
    # and skeleton (first past the cutoff before the skip) now fits.
    chain_root = tmp_path / "chain-root"
    (chain_root / "scripts" / "queue").mkdir(parents=True)
    (chain_root / "scripts" / "queue" / "tasks.json").write_text(
        queue_copy.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for name in ("compare_seed.sh", "compare_schedule.sh"):
        (chain_root / "scripts" / "queue" / name).write_text(
            (REPO / "scripts" / "queue" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    # The chain builder reads run statuses and gate verdicts from the
    # committed tree; point it at the real one so ENDPOINT-EXT-02
    # counts as decided and recorded trains are not pending.
    (chain_root / "results").symlink_to(REPO / "results", target_is_directory=True)
    (chain_root / "runs").symlink_to(REPO / "runs", target_is_directory=True)
    chain = subprocess.run(
        [
            sys.executable,
            str(SPRINT),
            "--root",
            str(chain_root),
            "--chain",
            "--decision",
            "CAND-CURVED_CONTINUATION-B16",
            "--resume-date",
            "2026-11-28",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert chain.returncode == 0, chain.stderr
    assert "CHAIN: 10 pending decisions" in chain.stdout
    assert chain.stdout.count("— FITS") == 7
    assert chain.stdout.count("— PAST CUTOFF") == 3
    skeleton_line = next(
        line for line in chain.stdout.splitlines() if "CAND-SKELETON_RECALL-B16" in line
    )
    assert "— FITS" in skeleton_line
