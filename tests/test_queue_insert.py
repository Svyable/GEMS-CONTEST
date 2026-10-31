"""Tests for scripts/queue_insert.py (mid-queue insertion, step 26).

The runner executes pending tasks in list order, so a task that must run
before an already-queued consumer cannot be appended — it has to be
inserted before the consumer. These tests pin the insert tool's contract
on synthetic queues and the real repo state it just changed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "queue_insert.py"


def run_insert(root: Path, anchor: str, new_tasks: list[dict], extra: list[str] | None = None):
    insert_file = root / "insert.json"
    insert_file.write_text(json.dumps(new_tasks))
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(root),
        "--anchor",
        anchor,
        "--insert-file",
        str(insert_file),
    ] + (extra or [])
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def make_root(tmp_path: Path, tasks: list[dict]) -> Path:
    root = tmp_path / "repo"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return root


def base_tasks() -> list[dict]:
    return [
        {"id": "a-train", "step": 1, "cmd": "train a"},
        {"id": "a-assemble", "step": 1, "cmd": "mkdir x && cp f x/"},
        {"id": "a-gate", "step": 1, "cmd": "gate a"},
    ]


def test_insert_before_anchor(tmp_path: Path):
    root = make_root(tmp_path, base_tasks())
    new = [
        {"id": "b-rescore", "step": 2, "cmd": "score b"},
        {"id": "b-reassemble", "step": 2, "cmd": "mkdir x && cp f x/"},
    ]
    result = run_insert(root, "a-assemble", new)
    assert result.returncode == 0, result.stderr
    tasks = json.loads((root / "scripts" / "queue" / "tasks.json").read_text())
    assert [t["id"] for t in tasks] == [
        "a-train",
        "b-rescore",
        "b-reassemble",
        "a-assemble",
        "a-gate",
    ]
    assert "queue 3 -> 5 tasks" in result.stdout


def test_dry_run_writes_nothing(tmp_path: Path):
    root = make_root(tmp_path, base_tasks())
    before = (root / "scripts" / "queue" / "tasks.json").read_text()
    result = run_insert(root, "a-assemble", [{"id": "b-x", "cmd": "echo b"}], ["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout
    assert (root / "scripts" / "queue" / "tasks.json").read_text() == before


def test_missing_anchor_fails_without_write(tmp_path: Path):
    root = make_root(tmp_path, base_tasks())
    before = (root / "scripts" / "queue" / "tasks.json").read_text()
    result = run_insert(root, "no-such-task", [{"id": "b-x", "cmd": "echo b"}])
    assert result.returncode == 2
    assert (root / "scripts" / "queue" / "tasks.json").read_text() == before


def test_duplicate_new_ids_fail_without_write(tmp_path: Path):
    root = make_root(tmp_path, base_tasks())
    before = (root / "scripts" / "queue" / "tasks.json").read_text()
    result = run_insert(
        root, "a-assemble", [{"id": "b-x", "cmd": "echo 1"}, {"id": "b-x", "cmd": "echo 2"}]
    )
    assert result.returncode == 2
    assert (root / "scripts" / "queue" / "tasks.json").read_text() == before


def test_collision_with_existing_id_fails_without_write(tmp_path: Path):
    root = make_root(tmp_path, base_tasks())
    before = (root / "scripts" / "queue" / "tasks.json").read_text()
    result = run_insert(root, "a-assemble", [{"id": "a-gate", "cmd": "echo b"}])
    assert result.returncode == 2
    assert (root / "scripts" / "queue" / "tasks.json").read_text() == before


def test_missing_cmd_fails_without_write(tmp_path: Path):
    root = make_root(tmp_path, base_tasks())
    before = (root / "scripts" / "queue" / "tasks.json").read_text()
    result = run_insert(root, "a-assemble", [{"id": "b-x"}])
    assert result.returncode == 2
    assert (root / "scripts" / "queue" / "tasks.json").read_text() == before


def test_real_repo_step26_block():
    """Step 26: the re-score + re-assemble block sits directly before s5b-assemble."""
    tasks = json.loads((REPO / "scripts" / "queue" / "tasks.json").read_text())
    assert len(tasks) == 129
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids after insert"
    anchor = ids.index("s5b-assemble-ref-b16")
    assert ids[anchor - 3 : anchor] == [
        "s26-rescore-ref-spatial-full",
        "s26-rescore-ref-fault-full",
        "s26-reassemble-ref-b16",
    ]
    by_id = {t["id"]: t for t in tasks}
    # Re-scores replay the original s1/s2 score commands verbatim: same run,
    # scheme, fold map, and gate dir, so the current score_cv.py emits schema-2.
    assert by_id["s26-rescore-ref-spatial-full"]["cmd"] == by_id["s1-score-ref-spatial"]["cmd"]
    assert by_id["s26-rescore-ref-fault-full"]["cmd"] == by_id["s2-score-ref-fault"]["cmd"]
    # The re-assemble must repair, not preserve: unconditional cp, unlike
    # s7-assemble-ref-b16's cp -n, which would keep a stale schema-1 copy.
    reassemble = by_id["s26-reassemble-ref-b16"]["cmd"]
    assert "cp runs/gate/ref-full/spatial.json runs/gate/ref-full/fault.json" in reassemble
    assert "cp -n" not in reassemble
    assert "runs/gate/ref-b16" in reassemble
    assert all(t.get("step") == 26 for t in by_id.values() if t["id"].startswith("s26-"))
