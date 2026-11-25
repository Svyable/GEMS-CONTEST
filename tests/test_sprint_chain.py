"""Tests for the chained sprint in scripts/queue_sprint.py (step 51).

The single-decision sprint (step 48) leaves the suffix in original
queue order, which scheduled the 0-train curved-continuation gate at
position 127 of 129 — after the s7 prefix the box would grind through
the original order and never reach it before the cutoff. The chain
keeps the first decision's closure first, then appends every remaining
pending decision's closure in ascending standalone cost, shared
feeders placed once.
"""

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


def test_chain_orders_first_then_cheapest_shared_once(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    out = tmp_path / "chain.json"
    result = run_sprint(root, "--chain", "--decision", "G-FIRST", "--out", str(out))
    assert result.returncode == 0, result.stderr
    ids = [t["id"] for t in json.loads(out.read_text(encoding="utf-8"))]
    assert sorted(ids) == sorted(t["id"] for t in synthetic_two_gate_queue())
    # First segment is G-FIRST's closure; the shared incumbent producers
    # and assemble appear once, inside it.
    first = [
        "ref-train-spatial",
        "ref-score-spatial",
        "ref-train-fault",
        "ref-score-fault",
        "assemble-ref",
        "a-train",
        "a-score",
        "a-record",
        "a-gate",
    ]
    assert ids[: len(first)] == first
    # Second segment is only G-SECOND's own triple + gate.
    assert ids[len(first) : len(first) + 4] == ["b-train", "b-score", "b-record", "b-gate"]
    assert ids[-1] == "filler-train"
    assert "CHAIN: 2 pending decisions" in result.stdout
    # Standalone cost counts the shared incumbent trains (2 ref trains
    # + the candidate's own); the second segment's marginal cost drops
    # to 1 because the shared trains were placed in the first segment.
    assert "1. gate G-FIRST: 9 tasks, +3 trains (standalone 3); cumulative 3 trains" in (
        result.stdout
    )
    assert "2. gate G-SECOND: 4 tasks, +1 trains (standalone 3); cumulative 4 trains" in (
        result.stdout
    )
    assert "REMAINDER: 1 tasks in original order" in result.stdout


def test_chain_first_decision_is_configurable(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    out = tmp_path / "chain.json"
    result = run_sprint(root, "--chain", "--decision", "G-SECOND", "--out", str(out))
    assert result.returncode == 0, result.stderr
    ids = [t["id"] for t in json.loads(out.read_text(encoding="utf-8"))]
    assert ids.index("b-gate") < ids.index("a-gate")
    assert "1. gate G-SECOND" in result.stdout


def test_resume_date_without_chain_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    result = run_sprint(root, "--resume-date", "2026-11-28", "--out", str(tmp_path / "c.json"))
    assert result.returncode == 2
    assert "--resume-date only applies with --chain" in result.stderr


def test_bad_resume_date_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    result = run_sprint(
        root, "--chain", "--resume-date", "28-11-2026", "--out", str(tmp_path / "c.json")
    )
    assert result.returncode == 2
    assert "must be YYYY-MM-DD" in result.stderr


def test_resume_date_after_cutoff_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    result = run_sprint(
        root, "--chain", "--resume-date", "2026-12-04", "--out", str(tmp_path / "c.json")
    )
    assert result.returncode == 2
    assert "after the safe cutoff" in result.stderr


def test_chain_unknown_first_decision_exits_2(tmp_path: Path) -> None:
    root = make_root(tmp_path, synthetic_two_gate_queue())
    result = run_sprint(root, "--chain", "--decision", "G-NOPE", "--out", str(tmp_path / "c.json"))
    assert result.returncode == 2
    assert "unknown decision 'G-NOPE'" in result.stderr


def test_real_repo_chain_locked(tmp_path: Path) -> None:
    # Locks the chained order against the committed tree; update when
    # the queue changes. The first segment is byte-identical to the
    # step-48 s7 sprint prefix; the rest is cheapest-first.
    out = tmp_path / "tasks.sprint.json"
    result = run_sprint(REPO, "--chain", "--resume-date", "2026-11-28", "--out", str(out))
    assert result.returncode == 0, result.stderr
    ids = [t["id"] for t in json.loads(out.read_text(encoding="utf-8"))]
    live = json.loads((REPO / "scripts/queue/tasks.json").read_text(encoding="utf-8"))
    assert sorted(ids) == sorted(t["id"] for t in live)
    assert len(ids) == 129
    s7_prefix = [
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
    assert ids[: len(s7_prefix)] == s7_prefix
    # Decision order: s7 first, then ascending standalone cost. The
    # 0-train curved gate — last in the step-48 suffix — decides second.
    decision_order = [
        "s7-gate-loss_distance_tversky-b16",
        "s13-gate-curved_continuation-b16",
        "s14-compare-seed20260923-b16",
        "s5b-gate-mumford_shah_log-b16",
        "s5b-gate-steerable_filter-b16",
        "s9-gate-wavelet_entropy-b16",
        "s10-gate-loss_background_soften-b16",
        "s11-gate-skeleton_recall-b16",
        "s12-compare-longer_schedule-b16",
        "s6-gate-structure_tensor_coherence-b32",
        "s6-gate-ridge_valley_response-b32",
    ]
    positions = [ids.index(d) for d in decision_order]
    assert positions == sorted(positions)
    assert "CHAIN: 11 pending decisions" in result.stdout
    # From a Nov 28 resume at the observed rate, 7 decisions fit before
    # the Dec 3 22:00 UTC cutoff; skeleton is the first past it.
    assert result.stdout.count("— FITS") == 7
    assert result.stdout.count("— PAST CUTOFF") == 4
    assert (
        "2. gate CAND-CURVED_CONTINUATION-B16: 6 tasks, +0 trains (standalone 0); "
        "cumulative 3 trains, ~1.0d" in result.stdout
    )
    # Ridge shares the ref-trace-b32 train with structure-tensor, so
    # its marginal cost (3) is below its standalone cost (4).
    assert (
        "11. gate CAND-RIDGE_VALLEY_RESPONSE-B32: 10 tasks, +3 trains (standalone 4); "
        "cumulative 28 trains, ~9.8d" in result.stdout
    )
    assert "REMAINDER: 12 tasks in original order" in result.stdout
    # The real audit scripts accept the chained file as a queue.
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
