"""Tests for scripts/queue_critical_path.py (resume-order ranking)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_critical_path.py"


def run_critical_path(root: Path) -> subprocess.CompletedProcess[str]:
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


def write(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


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


def assemble_task(task_id: str, dest: str, *sources: str) -> dict:
    return {
        "id": task_id,
        "cmd": "mkdir -p " + dest + " && cp " + " ".join(sources) + " " + dest + "/",
    }


def gate_task(task_id: str, name: str, inc: str, cand: str) -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/gate.sh {name} {inc} {cand} configs/cand.yaml "hyp"',
    }


def test_chain_train_score_assemble_gate(tmp_path: Path) -> None:
    tasks = [
        train_task("t1", "runs/r1"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
        assemble_task("a1", "runs/gate/inc", "runs/gate/full/spatial.json"),
        gate_task("g1", "G-NAME", "runs/gate/inc", "runs/gate/cand"),
        {"id": "r1", "cmd": '/workspace/gems/queue/record.sh r1 runs/r1 configs/ref.yaml "x"'},
    ]
    root = make_root(tmp_path, tasks)
    write(root, "runs/gate/full/spatial.json")
    result = run_critical_path(root)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PENDING DECISIONS: 1 gates, 0 compares" in out
    assert "MISSING incumbent runs/gate/inc/spatial.json <- a1" in out
    assert "MISSING candidate runs/gate/cand/spatial.json <- s1" in out
    assert "a1 [assemble] unblocks 1: G-NAME (runnable now)" in out
    assert "s1 [score] unblocks 1: G-NAME (after t1)" in out
    assert "t1 [train] unblocks 1: G-NAME (runnable now)" in out
    assert "NO DECISION IMPACT (1 pending tasks" in out
    assert "r1" in out.split("NO DECISION IMPACT")[1]


def test_evidence_elsewhere_and_no_producer(tmp_path: Path) -> None:
    tasks = [
        score_task("s2", "runs/r2", "spatial", "runs/gate/full2"),
        assemble_task("a2", "runs/gate/inc2", "runs/gate/full2/spatial.json"),
        gate_task("g2", "G2", "runs/gate/inc2", "runs/gate/cand2"),
    ]
    root = make_root(tmp_path, tasks)
    # s2 already ran (manifest landed) but its gate-dir copy never did.
    write(root, "runs/r2/manifest.json")
    # cand2/trace has no queued producer at all.
    result = run_critical_path(root)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "MISSING incumbent runs/gate/inc2/spatial.json <- a2 [EVIDENCE-ELSEWHERE]" in out
    assert "MISSING candidate runs/gate/cand2/trace.json <- - [NO-PRODUCER]" in out
    assert "a2 [assemble] unblocks 1: G2 (blocked: runs/gate/full2/spatial.json)" in out


def test_decided_gate_excluded(tmp_path: Path) -> None:
    tasks = [gate_task("g3", "DONE", "runs/gate/inc", "runs/gate/cand")]
    root = make_root(tmp_path, tasks)
    (root / "results" / "gates").mkdir(parents=True)
    (root / "results" / "gates" / "DONE.json").write_text('{"verdict": "ACCEPT"}')
    result = run_critical_path(root)
    assert result.returncode == 0, result.stderr
    assert "PENDING DECISIONS: 0 gates, 0 compares (1 gates decided" in result.stdout
    assert "GATE DONE" not in result.stdout


def test_view_runnable_uses_input_run(tmp_path: Path) -> None:
    def view_task(task_id: str, input_run: str, output_run: str, gate_dir: str) -> dict:
        return {
            "id": task_id,
            "cmd": f"/workspace/gems/queue/curved_view.sh {input_run} {output_run} spatial "
            f"&& /workspace/gems/queue/score_view.sh {output_run} spatial "
            f"data/x {gate_dir}",
        }

    tasks = [
        view_task("v1", "runs/in1", "runs/out1", "runs/gate/cand1"),
        view_task("v2", "runs/in2", "runs/out2", "runs/gate/cand2"),
        gate_task("g1", "G1", "runs/gate/inc", "runs/gate/cand1"),
        gate_task("g2", "G2", "runs/gate/inc", "runs/gate/cand2"),
    ]
    root = make_root(tmp_path, tasks)
    for view in ("spatial", "fault", "trace"):
        write(root, f"runs/gate/inc/{view}.json")
    write(root, "runs/in1/manifest.json")
    result = run_critical_path(root)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "v1 [view] unblocks 1: G1 (runnable now)" in out
    assert "v2 [view] unblocks 1: G2 (after training runs/in2)" in out


def test_compare_lofo_no_producer(tmp_path: Path) -> None:
    tasks = [
        {
            "id": "c1",
            "cmd": "/workspace/gems/queue/compare_schedule.sh runs/gate/inc "
            'runs/gate/cand runs/base runs/candbase "hyp"',
        },
    ]
    root = make_root(tmp_path, tasks)
    result = run_critical_path(root)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PENDING DECISIONS: 0 gates, 1 compares" in out
    assert (
        "MISSING incumbent LOFO runs/base-spatial-b16/scores/lofo-spatial.json "
        "(required) <- - [NO-PRODUCER]" in out
    )
    assert (
        "MISSING candidate LOFO runs/candbase-spatial-b16/scores/lofo-spatial.json "
        "(required) <- - [NO-PRODUCER]" in out
    )


def test_hard_errors(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    result = run_critical_path(root)
    assert result.returncode == 2
    (root / "scripts" / "queue" / "tasks.json").write_text("{not json")
    result = run_critical_path(root)
    assert result.returncode == 2


def test_real_repo_golden() -> None:
    result = run_critical_path(REPO)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PENDING DECISIONS: 9 gates, 2 compares (1 gates decided" in out
    assert "s5b-assemble-ref-b16 [assemble] unblocks 9" in out
    assert "s7-assemble-ref-b16 [assemble] unblocks 9" in out
    assert "s6-train-ref-trace-b32 [train] unblocks 2" in out
    assert "EVIDENCE-ELSEWHERE" in out
    assert "NO-PRODUCER" in out
    # The schedule compare's incumbent LOFO inputs have no queued producer:
    # no score task writes runs/ref-cv-spatial-b16 or runs/ref-cv-fault-b16.
    assert (
        "MISSING incumbent LOFO runs/ref-cv-spatial-b16/scores/lofo-spatial.json "
        "(required) <- - [NO-PRODUCER]" in out
    )
