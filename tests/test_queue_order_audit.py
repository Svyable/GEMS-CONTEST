"""Tests for scripts/queue_order_audit.py (step 33).

The runner executes pending tasks strictly in tasks.json list order, so a
consumer placed before its producer runs against stale or missing inputs
— the failure mode step 26 fixed by hand-inserting re-scores before the
assemble they feed. These tests verify the audit catches each dependency
shape out of order, stays silent on the committed queue except for the
two known by-design notes, and exits 0/1/2 as documented.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from queue_order_audit import OrderAudit, main, render

REPO = Path(__file__).resolve().parents[1]


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


def record_task(task_id: str, run_id: str, run: str) -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/record.sh {run_id} {run} configs/ref.yaml "hyp" "notes"',
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


def view_task(task_id: str, base: str, out: str, scheme: str, gate: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/endpoint_view.sh {base} {out} {scheme} "
        f"&& /workspace/gems/queue/score_view.sh {out} {scheme} "
        f"data/processed/cv-{scheme}-v1.tif {gate}",
    }


def compare_schedule_task(task_id: str, inc: str, cand: str, base: str) -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/compare_schedule.sh {inc} {cand} runs/ref-cv {base} "hyp"',
    }


def compare_seed_task(task_id: str, inc: str, cand: str, run: str) -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/compare_seed.sh {inc} {cand} runs/ref-cv {run} "hyp"',
    }


def clean_chain() -> list[dict]:
    """train -> score -> assemble -> gate -> record, all in order."""
    return [
        train_task("t1", "runs/r1"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
        score_task("s1f", "runs/r1", "fault", "runs/gate/cand"),
        score_task("s1t", "runs/r1", "trace", "runs/gate/cand"),
        assemble_task(
            "a1",
            "runs/gate/inc",
            "runs/gate/full/spatial.json",
            "runs/gate/full/fault.json",
            "runs/gate/full/trace.json",
        ),
        gate_task("g1", "G-NAME", "runs/gate/inc", "runs/gate/cand"),
        record_task("r1", "r1", "runs/r1"),
    ]


def test_clean_chain_no_violations() -> None:
    result = OrderAudit(clean_chain()).audit()
    assert result["violations"] == []
    # The assemble's source dir (runs/gate/full) has no queued producer:
    # grandfathered incumbent scores, reported as notes, not violations.
    assert len(result["notes"]) == 3
    assert "OK" in render(result)


def test_gate_before_score_producer_is_violation() -> None:
    tasks = [
        train_task("t1", "runs/r1"),
        gate_task("g1", "G-NAME", "runs/gate/inc", "runs/gate/cand"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
    ]
    result = OrderAudit(tasks).audit()
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["consumer"]["task_id"] == "g1"
    assert finding["producer"]["task_id"] == "s1"
    assert "runs/gate/cand/spatial.json" in finding["input"]


def test_score_before_train_is_violation() -> None:
    tasks = [
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
        train_task("t1", "runs/r1"),
    ]
    result = OrderAudit(tasks).audit()
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["consumer"]["task_id"] == "s1"
    assert finding["producer"]["task_id"] == "t1"


def test_record_before_score_is_violation() -> None:
    tasks = [
        train_task("t1", "runs/r1"),
        record_task("r1", "r1", "runs/r1"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
    ]
    result = OrderAudit(tasks).audit()
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["consumer"]["task_id"] == "r1"
    assert finding["producer"]["task_id"] == "s1"


def test_assemble_before_source_score_is_violation() -> None:
    tasks = [
        train_task("t1", "runs/r1"),
        assemble_task("a1", "runs/gate/inc", "runs/gate/full/spatial.json"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/full"),
    ]
    result = OrderAudit(tasks).audit()
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["consumer"]["task_id"] == "a1"
    assert finding["producer"]["task_id"] == "s1"


def test_view_before_base_score_is_violation() -> None:
    tasks = [
        train_task("t1", "runs/r1"),
        view_task("v1", "runs/r1", "runs/r1-view", "spatial", "runs/gate/cand"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/full"),
    ]
    result = OrderAudit(tasks).audit()
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["consumer"]["task_id"] == "v1"
    assert finding["producer"]["task_id"] == "s1"
    assert "lofo-spatial.json" in finding["input"]


def test_compare_schedule_before_candidate_train_is_violation() -> None:
    tasks = [
        train_task("t1", "runs/r1"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/inc"),
        score_task("s1f", "runs/r1", "fault", "runs/gate/inc"),
        score_task("s1t", "runs/r1", "trace", "runs/gate/inc"),
        compare_schedule_task("c1", "runs/gate/inc", "runs/gate/cand", "runs/cand-x"),
        train_task("t2", "runs/cand-x-spatial-b16"),
    ]
    result = OrderAudit(tasks).audit()
    # Candidate gate dir (all 3 views) + 3 candidate run dirs, all produced
    # late or never: 3 run-dir violations; the gate-dir reads are notes
    # (no queued producer at all — presence is the invariant tests' job).
    run_violations = [v for v in result["violations"] if "fold predictions" in v["input"]]
    assert len(run_violations) == 1
    assert run_violations[0]["producer"]["task_id"] == "t2"
    gate_notes = [n for n in result["notes"] if "runs/gate/cand/" in n["input"]]
    assert len(gate_notes) == 3


def test_compare_seed_before_candidate_train_is_violation() -> None:
    tasks = [
        compare_seed_task(
            "c1",
            "runs/gate/inc",
            "runs/gate/cand",
            "runs/r-seed20260923",
        ),
        train_task("t1", "runs/r-seed20260923"),
    ]
    result = OrderAudit(tasks).audit()
    assert any(
        v["consumer"]["task_id"] == "c1" and v["producer"]["task_id"] == "t1"
        for v in result["violations"]
    )


def test_triple_out_of_order_is_violation() -> None:
    tasks = [
        {
            "id": "s1-record-x",
            "cmd": '/workspace/gems/queue/record.sh x runs/x configs/ref.yaml "h" "n"',
        },
        {
            "id": "s1-train-x",
            "cmd": "/workspace/gems/queue/train_folds.sh runs/x configs/ref.yaml spatial data/processed/cv-spatial-v1.tif 16",
        },
        {
            "id": "s1-score-x",
            "cmd": "/workspace/gems/queue/score_view.sh runs/x spatial data/processed/cv-spatial-v1.tif runs/gate/cand",
        },
    ]
    result = OrderAudit(tasks).audit()
    triple = [v for v in result["violations"] if "triple" in v["input"]]
    assert len(triple) == 2  # record before train, record before score
    assert {v["consumer"]["task_id"] for v in triple} == {"s1-record-x"}


def test_no_producer_is_note_not_violation() -> None:
    tasks = [
        gate_task("g1", "G-NAME", "runs/gate/inc", "runs/gate/cand"),
    ]
    result = OrderAudit(tasks).audit()
    assert result["violations"] == []
    assert len(result["notes"]) == 6  # 3 views x 2 gate dirs
    assert "NOTE" in render(result)


def test_wait_task_is_ignored() -> None:
    tasks = [
        {"id": "s1-wait", "cmd": "while kill -0 1234 2>/dev/null; do sleep 30; done"},
        train_task("t1", "runs/r1"),
    ]
    result = OrderAudit(tasks).audit()
    assert result["unparsed"] == 1
    assert result["violations"] == []


def test_committed_queue_is_order_clean(tmp_path: Path) -> None:
    """Regression lock: the real 129-task queue must have zero violations.

    This is the step-26 generalization — the re-scores inserted before the
    ref-b16 assemble must stay before it, and every gate/compare must
    follow its producers. Only the two known by-design notes remain: the
    s14 seed probe is spatial-only, so its candidate gate dir never gets
    fault/trace JSONs (compare_seed.sh skips missing views).
    """
    tasks = json.loads((REPO / "scripts" / "queue" / "tasks.json").read_text())
    result = OrderAudit(tasks).audit()
    assert result["tasks"] == 129
    assert result["violations"] == []
    assert len(result["notes"]) == 2
    assert {n["consumer"]["task_id"] for n in result["notes"]} == {"s14-compare-seed20260923-b16"}
    assert {n["input"] for n in result["notes"]} == {
        "runs/gate/ref-seed20260923-b16/fault.json",
        "runs/gate/ref-seed20260923-b16/trace.json",
    }


def test_main_exit_codes(tmp_path: Path) -> None:
    # Clean chain on a fabricated root -> exit 0.
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text(json.dumps(clean_chain()))
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "queue_order_audit.py"), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "0 ORDER-VIOLATION(s)" in proc.stdout
    # Violating queue -> exit 1.
    bad = [
        gate_task("g1", "G-NAME", "runs/gate/inc", "runs/gate/cand"),
        train_task("t1", "runs/r1"),
        score_task("s1", "runs/r1", "spatial", "runs/gate/cand"),
    ]
    (root / "scripts" / "queue" / "tasks.json").write_text(json.dumps(bad))
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "queue_order_audit.py"), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "VIOLATION g1" in proc.stdout
    # Malformed tasks.json -> exit 2.
    (root / "scripts" / "queue" / "tasks.json").write_text("{not json")
    assert main(["--root", str(root)]) == 2


def test_main_on_real_repo_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "queue_order_audit.py"), "--root", str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "129 tasks" in proc.stdout
    assert "0 ORDER-VIOLATION(s), 2 NO-PRODUCER note(s)" in proc.stdout
