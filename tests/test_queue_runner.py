"""Tests for the queue runner contract (step 42).

Every queue audit so far (order, refs, args, duplicates, config drift,
forecast) checks the queue's *contents*, and the doctor diagnoses the
runner's *state dir* — but the runner itself, the one engine that turns
tasks.json into .done/.failed markers, had never been executed on this
VM. These tests run the real scripts/queue/runner.py in a sandboxed
QUEUE_DIR + GEMS_WORKSPACE with trivial shell commands and pin its
contract: in-order execution, skip-done, stop-on-failure with a .failed
marker (the exit-3 WAIT-MEM-TIMEOUT path step 41 depends on), resume
after the marker is deleted, live-appended tasks picked up, and the
single-instance flock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "queue" / "runner.py"


def _queue(tmp_path: Path, tasks: list[dict]) -> tuple[Path, Path]:
    """A sandboxed queue dir + workspace holding the given tasks."""
    q = tmp_path / "queue"
    q.mkdir()
    (q / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    return q, ws


def _env(q: Path, ws: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["QUEUE_DIR"] = str(q)
    env["GEMS_WORKSPACE"] = str(ws)
    return env


def _run(q: Path, ws: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER)],
        env=_env(q, ws),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _status(q: Path) -> dict:
    return json.loads((q / "status.json").read_text(encoding="utf-8"))


def _wait_for(marker: Path, what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not marker.exists():
        assert time.monotonic() < deadline, f"{what} never appeared"
        time.sleep(0.05)


def test_runner_runs_in_order_and_marks_done(tmp_path: Path) -> None:
    q, ws = _queue(
        tmp_path,
        [
            {"id": "t1", "step": 1, "cmd": "echo t1 >> run.log"},
            {"id": "t2", "step": 1, "cmd": "echo t2 >> run.log"},
            {"id": "t3", "step": 1, "cmd": "echo t3 >> run.log"},
        ],
    )
    proc = _run(q, ws)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (ws / "run.log").read_text().splitlines() == ["t1", "t2", "t3"]
    state = q / "state"
    for tid in ("t1", "t2", "t3"):
        marker = state / f"{tid}.done"
        assert marker.exists(), f"{tid} has no .done marker"
        assert "rc=0" in marker.read_text()
    status = _status(q)
    assert status["note"].startswith("IDLE")
    assert status["current"] is None
    assert [r["state"] for r in status["tasks"]] == ["done"] * 3
    assert "t1" in (q / "STATUS.txt").read_text()


def test_runner_stops_on_first_failure(tmp_path: Path) -> None:
    """A non-zero exit (here rc=3, the step-41 WAIT-MEM-TIMEOUT code)
    writes .failed, stops the queue, and leaves later tasks unstarted."""
    q, ws = _queue(
        tmp_path,
        [
            {"id": "ok", "step": 1, "cmd": "echo ok >> run.log"},
            {"id": "fail", "step": 1, "cmd": "exit 3"},
            {"id": "never", "step": 1, "cmd": "echo never >> run.log"},
        ],
    )
    proc = _run(q, ws)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (ws / "run.log").read_text().splitlines() == ["ok"]
    failed = q / "state" / "fail.failed"
    assert failed.exists()
    assert "rc=3" in failed.read_text()
    assert not (q / "state" / "never.done").exists()
    assert not (q / "state" / "never.started").exists()
    status = _status(q)
    assert status["note"].startswith("STOPPED")
    assert [r["state"] for r in status["tasks"]] == ["done", "FAILED", "pending"]


def test_runner_resumes_after_failed_marker_deleted(tmp_path: Path) -> None:
    """The doctor's STOPPED resume: fix the cause, delete .failed, and
    the queue re-runs the failed task (not from scratch) and continues."""
    q, ws = _queue(
        tmp_path,
        [
            {
                "id": "flaky",
                "step": 1,
                "cmd": "test -f fixed.txt || exit 1; echo flaky >> run.log",
            },
            {"id": "after", "step": 1, "cmd": "echo after >> run.log"},
        ],
    )
    first = _run(q, ws)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (q / "state" / "flaky.failed").exists()
    assert not (ws / "run.log").exists()
    (ws / "fixed.txt").write_text("fixed", encoding="utf-8")
    (q / "state" / "flaky.failed").unlink()
    second = _run(q, ws)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (ws / "run.log").read_text().splitlines() == ["flaky", "after"]
    assert (q / "state" / "flaky.done").exists()
    assert _status(q)["note"].startswith("IDLE")


def test_runner_skips_done_tasks_on_restart(tmp_path: Path) -> None:
    """A task with a .done marker is never re-executed, even though its
    command would happily run again."""
    q, ws = _queue(
        tmp_path,
        [
            {"id": "done-one", "step": 1, "cmd": "echo done-one >> run.log"},
            {"id": "todo", "step": 1, "cmd": "echo todo >> run.log"},
        ],
    )
    state = q / "state"
    state.mkdir()
    (state / "done-one.done").write_text("2026-11-16 00:00:00 rc=0 elapsed_s=0", encoding="utf-8")
    proc = _run(q, ws)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (ws / "run.log").read_text().splitlines() == ["todo"]
    assert _status(q)["note"].startswith("IDLE")


def test_runner_second_instance_exits_quietly(tmp_path: Path) -> None:
    """The flock holds while the first runner is mid-task: a second
    instance exits 0 at once and runs nothing."""
    q, ws = _queue(
        tmp_path,
        [
            {"id": "slow", "step": 1, "cmd": "sleep 3; echo slow >> run.log"},
            {"id": "later", "step": 1, "cmd": "echo later >> run.log"},
        ],
    )
    first = subprocess.Popen(
        [sys.executable, str(RUNNER)],
        env=_env(q, ws),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = ""
    try:
        _wait_for(q / "state" / "slow.started", "first runner's task")
        second = _run(q, ws, timeout=30)
        assert second.returncode == 0, second.stdout + second.stderr
        assert "runner already active" in second.stdout
        assert not (ws / "run.log").exists()
    finally:
        out, _ = first.communicate(timeout=30)
    assert first.returncode == 0, out
    assert (ws / "run.log").read_text().splitlines() == ["slow", "later"]


def test_runner_picks_up_live_appended_tasks(tmp_path: Path) -> None:
    """tasks.json is re-read before every task, so a task appended while
    the runner is mid-queue runs without a restart."""
    q, ws = _queue(
        tmp_path,
        [{"id": "slow", "step": 1, "cmd": "sleep 2; echo slow >> run.log"}],
    )
    proc = subprocess.Popen(
        [sys.executable, str(RUNNER)],
        env=_env(q, ws),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = ""
    try:
        _wait_for(q / "state" / "slow.started", "runner's task")
        tasks = json.loads((q / "tasks.json").read_text(encoding="utf-8"))
        tasks.append({"id": "appended", "step": 1, "cmd": "echo appended >> run.log"})
        (q / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        proc.wait(timeout=30)
        out, _ = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
            out, _ = proc.communicate()
    assert proc.returncode == 0, out
    assert (ws / "run.log").read_text().splitlines() == ["slow", "appended"]
