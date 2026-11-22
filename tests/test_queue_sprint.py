"""Tests for scripts/queue_sprint.py (sprint-ordered queue)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_sprint.py"
ORDER_AUDIT = REPO / "scripts" / "queue_order_audit.py"
DUPS_AUDIT = REPO / "scripts" / "audit_queue_duplicates.py"


def run_sprint(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
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


def synthetic_s7_queue() -> list[dict]:
    """s7-shaped queue: full-run incumbents, s26 repair + assemble, s7
    triples + gate, plus a filler train and an unconsumed trace score."""
    tasks: list[dict] = [
        train_task("s1-train", "runs/ref-spatial"),
        score_task("s1-score", "runs/ref-spatial", "spatial", "runs/gate/ref-full"),
        train_task("s2-train", "runs/ref-fault"),
        score_task("s2-score", "runs/ref-fault", "fault", "runs/gate/ref-full"),
        train_task("s2-train-trace", "runs/ref-trace"),
        score_task("s2-score-trace", "runs/ref-trace", "trace", "runs/gate/ref-full"),
        train_task("filler-train", "runs/filler"),
        score_task("s26-rescore-spatial", "runs/ref-spatial", "spatial", "runs/gate/ref-full"),
        score_task("s26-rescore-fault", "runs/ref-fault", "fault", "runs/gate/ref-full"),
        assemble_task("s26-assemble", "runs/gate/ref-full", "runs/gate/ref-b16"),
    ]
    for view in ("spatial", "fault", "trace"):
        tasks.append(train_task(f"s7-train-{view}", f"runs/cand-{view}", view))
        tasks.append(score_task(f"s7-score-{view}", f"runs/cand-{view}", view, "runs/gate/cand"))
        tasks.append(record_task(f"s7-record-{view}", f"cand-{view}", f"runs/cand-{view}"))
    tasks.append(gate_task("s7-gate", "G-S7", "runs/gate/ref-b16", "runs/gate/cand"))
    tasks.append(train_task("tail-train", "runs/tail"))
    return tasks


def test_synthetic_prefix_is_positional_closure(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_s7_queue())
    out = tmp_path / "sprint.json"
    result = run_sprint(root, "--decision", "G-S7", "--out", str(out))
    assert result.returncode == 0, result.stderr
    sprint = json.loads(out.read_text(encoding="utf-8"))
    ids = [t["id"] for t in sprint]
    # Same task set, nothing lost or doubled.
    assert sorted(ids) == sorted(t["id"] for t in synthetic_s7_queue())
    expected_prefix = [
        "s1-train",
        "s1-score",
        "s2-train",
        "s2-score",
        # The trace full-run score is never consumed by the assemble
        # (it copies spatial + fault only), so it stays in the suffix.
        "s26-rescore-spatial",
        "s26-rescore-fault",
        "s26-assemble",
        "s7-train-spatial",
        "s7-score-spatial",
        "s7-record-spatial",
        "s7-train-fault",
        "s7-score-fault",
        "s7-record-fault",
        "s7-train-trace",
        "s7-score-trace",
        "s7-record-trace",
        "s7-gate",
    ]
    assert ids[: len(expected_prefix)] == expected_prefix
    assert "s2-train-trace" not in ids[: len(expected_prefix)]
    assert "s2-score-trace" not in ids[: len(expected_prefix)]
    assert "filler-train" not in ids[: len(expected_prefix)]
    # Suffix keeps the original relative order.
    assert ids[len(expected_prefix) :] == [
        "s2-train-trace",
        "s2-score-trace",
        "filler-train",
        "tail-train",
    ]
    assert "PREFIX: 17 tasks, 5 trains (5 still needing a run, no observed rate" in result.stdout
    printed = [line.strip() for line in result.stdout.splitlines() if line.startswith("  ")]
    assert printed == expected_prefix


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_s7_queue())
    out = tmp_path / "sprint.json"
    result = run_sprint(root, "--decision", "G-S7", "--dry-run", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert not out.exists()
    assert "DRY RUN: nothing written" in result.stdout


def test_unknown_decision_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_s7_queue())
    result = run_sprint(root, "--decision", "G-NOPE", "--out", str(tmp_path / "s.json"))
    assert result.returncode == 2
    assert "unknown decision 'G-NOPE'" in result.stderr
    assert "G-S7" in result.stderr


def test_out_equal_to_tasks_json_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_s7_queue())
    result = run_sprint(root, "--decision", "G-S7", "--out", str(root / "scripts/queue/tasks.json"))
    assert result.returncode == 2
    assert "must not be the live tasks.json" in result.stderr


def test_malformed_tasks_json_exits_2(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text("{not json")
    result = run_sprint(root)
    assert result.returncode == 2
    assert "cannot read" in result.stderr


def test_real_repo_default_sprint_locked(tmp_path: Path) -> None:
    # Locks the sprint against the committed tree; update when the
    # queue changes. The 21-task prefix is the s7 gate's positional
    # closure: full-run producers, the s5b trace writer, the s26
    # repair + assemble, the three s7 assembles, the s7 triples, gate.
    out = tmp_path / "tasks.sprint.json"
    result = run_sprint(REPO, "--out", str(out))
    assert result.returncode == 0, result.stderr
    sprint = json.loads(out.read_text(encoding="utf-8"))
    ids = [t["id"] for t in sprint]
    live = json.loads((REPO / "scripts/queue/tasks.json").read_text(encoding="utf-8"))
    assert sorted(ids) == sorted(t["id"] for t in live)
    assert len(ids) == 129
    expected_prefix = [
        "s1-train-ref-spatial",
        "s1-score-ref-spatial",
        "s2-train-ref-fault",
        "s2-score-ref-fault",
        "s5b-train-ref-trace-b16",
        "s5b-score-ref-trace-b16",
        "s26-rescore-ref-spatial-full",
        "s26-rescore-ref-fault-full",
        "s26-reassemble-ref-b16",
        "s5b-assemble-ref-b16",
        "s7-train-loss_distance_tversky-spatial",
        "s7-score-loss_distance_tversky-spatial",
        "s7-record-loss_distance_tversky-spatial",
        "s7-train-loss_distance_tversky-fault",
        "s7-score-loss_distance_tversky-fault",
        "s7-record-loss_distance_tversky-fault",
        "s7-train-loss_distance_tversky-trace",
        "s7-score-loss_distance_tversky-trace",
        "s7-record-loss_distance_tversky-trace",
        "s7-assemble-ref-b16",
        "s7-gate-loss_distance_tversky-b16",
    ]
    assert ids[: len(expected_prefix)] == expected_prefix
    assert "PREFIX: 21 tasks, 6 trains (3 still needing a run, ~1.0d at 2.86 runs/day)" in (
        result.stdout
    )
    # The real audit scripts accept the sprint file as a queue.
    audit_root = tmp_path / "audit-root"
    (audit_root / "scripts" / "queue").mkdir(parents=True)
    (audit_root / "scripts" / "queue" / "tasks.json").write_text(out.read_text(encoding="utf-8"))
    (audit_root / "scripts" / "queue" / "tasks.parked.json").write_text(
        (REPO / "scripts/queue/tasks.parked.json").read_text(encoding="utf-8")
    )
    order = subprocess.run(
        [sys.executable, str(ORDER_AUDIT), "--root", str(audit_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert order.returncode == 0, order.stdout + order.stderr
    assert "0 ORDER-VIOLATION(s)" in order.stdout
    dups = subprocess.run(
        [sys.executable, str(DUPS_AUDIT), "--root", str(audit_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dups.returncode == 0, dups.stdout + dups.stderr
    assert "0 finding(s)" in dups.stdout


def test_real_repo_decided_gate_exits_2(tmp_path: Path) -> None:
    result = run_sprint(REPO, "--decision", "ENDPOINT-EXT-02", "--out", str(tmp_path / "s.json"))
    assert result.returncode == 2
    assert "already decided: REJECT" in result.stderr


def test_real_repo_compare_decision(tmp_path: Path) -> None:
    out = tmp_path / "sprint.json"
    result = run_sprint(REPO, "--decision", "compare_seed", "--out", str(out))
    assert result.returncode == 0, result.stderr
    sprint = json.loads(out.read_text(encoding="utf-8"))
    ids = [t["id"] for t in sprint]
    prefix = ids[: ids.index("s14-compare-seed20260923-b16") + 1]
    assert prefix[-1] == "s14-compare-seed20260923-b16"
    assert "s8-train-seed20260923-spatial" in prefix
    assert "s26-reassemble-ref-b16" in prefix
