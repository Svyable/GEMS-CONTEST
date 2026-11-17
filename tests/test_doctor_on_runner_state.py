"""Tests for the doctor against runner-produced state (step 43).

The doctor (steps 29/30/32) was verified only against hand-built
simulated queue dirs, and the runner (step 42) was verified only
against its own markers — but on the box the doctor reads state the
runner actually wrote. A hand-written fixture can drift from what
the runner writes (marker stamps, status.json shape, queue.log
lines, the lock pid), and the doctor is the tool Sven runs on the
box to decide what to do next, so a misread there is the most
expensive failure mode left in the queue tooling. These tests close
that loop: they run the real ``scripts/queue/runner.py`` to produce
the state, then run the real doctor (``diagnose``) against it, and
pin the verdict for each runner outcome — STOPPED, ALL-DONE,
RUNNING, INTERRUPTED, RUNNING-STALLED, and RUNNING-WAIT-MEM.

The one verdict still simulated-only is NOT-RUNNING, which needs no
runner at all (a tasks.json with no markers is the production input,
and ``test_queue_doctor.py`` already pins it against the committed
queue). The RUNNING-WAIT-MEM case uses a stub ``train_folds.sh``
that prints one WAIT-MEM line and sleeps: the doctor's trigger is
the command naming the wrapper plus WAIT-MEM in the log tail, and
the stub exercises exactly that path through the real runner.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from queue_doctor import diagnose

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


def _run_foreground(q: Path, ws: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run the runner to completion, stdout mirroring the box's nohup redirect."""
    with open(q / "queue.log", "w", encoding="utf-8") as log:
        return subprocess.run(
            [sys.executable, str(RUNNER)],
            env=_env(q, ws),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )


def _start_runner(q: Path, ws: Path) -> tuple[subprocess.Popen, object]:
    """Start the runner in the background, in its own process group.

    Returns the process and the open queue.log handle; the caller
    owns both and must kill the group (see _kill_runner) so no
    sleeping child outlives the test.
    """
    log = open(q / "queue.log", "w", encoding="utf-8")  # noqa: SIM115 — closed by _kill_runner
    proc = subprocess.Popen(
        [sys.executable, str(RUNNER)],
        env=_env(q, ws),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, log


def _kill_runner(proc: subprocess.Popen, log: object) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait(timeout=10)
    log.close()  # type: ignore[attr-defined]


def _wait_for(marker: Path, what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not marker.exists():
        assert time.monotonic() < deadline, f"{what} never appeared"
        time.sleep(0.05)


def _wait_for_current(q: Path, task_id: str, timeout: float = 10.0) -> None:
    """Wait until status.json names task_id as the runner's current task."""
    status_path = q / "status.json"
    deadline = time.monotonic() + timeout
    while True:
        if status_path.exists():
            try:
                if json.loads(status_path.read_text(encoding="utf-8")).get("current") == task_id:
                    return
            except json.JSONDecodeError:
                pass
        assert time.monotonic() < deadline, f"runner never started {task_id}"
        time.sleep(0.05)


def _wait_for_log_text(q: Path, text: str, timeout: float = 10.0) -> None:
    log_path = q / "queue.log"
    deadline = time.monotonic() + timeout
    while text not in log_path.read_text(encoding="utf-8", errors="replace"):
        assert time.monotonic() < deadline, f"{text!r} never appeared in queue.log"
        time.sleep(0.05)


def _diagnose(q: Path, ws: Path) -> dict:
    return diagnose(q, q / "tasks.json", ws)


def test_doctor_reads_stopped_from_runner_failure(tmp_path: Path) -> None:
    """The step-41 exit-3 path, end to end: runner writes .failed and
    STOPPED status; the doctor must say STOPPED, name the task, and
    print the clear-marker resume."""
    q, ws = _queue(
        tmp_path,
        [
            {"id": "ok", "step": 1, "cmd": "echo ok >> run.log"},
            {"id": "fail", "step": 1, "cmd": "exit 3"},
            {"id": "never", "step": 1, "cmd": "echo never >> run.log"},
        ],
    )
    proc = _run_foreground(q, ws)
    assert proc.returncode == 0, proc.stderr
    result = _diagnose(q, ws)
    assert result["verdict"] == "STOPPED"
    assert result["counts"] == {"done": 1, "failed": 1, "pending": 1}
    assert result["first_pending"] == "never"
    assert any("rc=3" in note for note in result["notes"])
    assert "fail.failed" in result["resume"]
    assert "runner.py" in result["resume"]


def test_doctor_reads_all_done_from_runner_completion(tmp_path: Path) -> None:
    q, ws = _queue(
        tmp_path,
        [
            {"id": "t1", "step": 1, "cmd": "echo t1 >> run.log"},
            {"id": "t2", "step": 1, "cmd": "echo t2 >> run.log"},
        ],
    )
    proc = _run_foreground(q, ws)
    assert proc.returncode == 0, proc.stderr
    result = _diagnose(q, ws)
    assert result["verdict"] == "ALL-DONE"
    assert result["counts"] == {"done": 2, "failed": 0, "pending": 0}
    assert result["first_pending"] is None


def test_doctor_reads_running_from_live_runner(tmp_path: Path) -> None:
    q, ws = _queue(
        tmp_path,
        [
            {"id": "slow", "step": 1, "cmd": "sleep 120; echo slow >> run.log"},
            {"id": "later", "step": 1, "cmd": "echo later >> run.log"},
        ],
    )
    proc, log = _start_runner(q, ws)
    try:
        _wait_for(q / "state" / "slow.started", "runner's task")
        _wait_for_current(q, "slow")
        result = _diagnose(q, ws)
        assert result["verdict"] == "RUNNING"
        assert result["current"] == "slow"
        assert any(f"pid file: {proc.pid}" in note for note in result["notes"])
        assert result["counts"] == {"done": 0, "failed": 0, "pending": 2}
    finally:
        _kill_runner(proc, log)


def test_doctor_reads_interrupted_from_killed_runner(tmp_path: Path) -> None:
    """SIGKILL the runner mid-task (the box-reboot shape): the lock is
    free, .started remains with no .done/.failed, and the doctor must
    say INTERRUPTED with a safe-restart resume."""
    q, ws = _queue(
        tmp_path,
        [
            {"id": "slow", "step": 1, "cmd": "sleep 120; echo slow >> run.log"},
            {"id": "later", "step": 1, "cmd": "echo later >> run.log"},
        ],
    )
    proc, log = _start_runner(q, ws)
    try:
        _wait_for(q / "state" / "slow.started", "runner's task")
        _wait_for_current(q, "slow")
    finally:
        _kill_runner(proc, log)
    assert not (q / "state" / "slow.done").exists()
    assert not (q / "state" / "slow.failed").exists()
    result = _diagnose(q, ws)
    assert result["verdict"] == "INTERRUPTED"
    assert result["first_pending"] == "slow"
    assert any("died mid-task" in note for note in result["notes"])
    assert "nohup python3" in result["resume"]


def test_doctor_reads_stalled_from_aged_runner_state(tmp_path: Path) -> None:
    """A live runner whose task started 49h ago and whose log has
    been silent 49h must read RUNNING-STALLED. The state is
    runner-produced and then aged (marker stamp, status.json stamp,
    log mtime), not hand-built — this is the step-30 verdict against
    the step-42 runner."""
    q, ws = _queue(
        tmp_path,
        [{"id": "slow", "step": 1, "cmd": "sleep 120; echo slow >> run.log"}],
    )
    proc, log = _start_runner(q, ws)
    try:
        _wait_for(q / "state" / "slow.started", "runner's task")
        _wait_for_current(q, "slow")
        # Control: the same live runner state reads RUNNING before aging —
        # the 49h stamps are what flip the verdict, not the setup.
        assert _diagnose(q, ws)["verdict"] == "RUNNING"
        old = datetime.now(UTC) - timedelta(hours=49)
        stamp = old.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        (q / "state" / "slow.started").write_text(stamp, encoding="utf-8")
        status_path = q / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["updated"] = stamp
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        old_ts = old.timestamp()
        os.utime(q / "queue.log", (old_ts, old_ts))
        result = _diagnose(q, ws)
        assert result["verdict"] == "RUNNING-STALLED"
        assert any("hung" in note for note in result["notes"])
        assert "kill the child" in result["resume"]
    finally:
        _kill_runner(proc, log)


def test_doctor_reads_wait_mem_from_runner_train_task(tmp_path: Path) -> None:
    """A train task whose wrapper prints WAIT-MEM and then sleeps
    must read RUNNING-WAIT-MEM through the real runner, with the
    step-41 6h bound in the note."""
    q, ws = _queue(
        tmp_path,
        [
            {
                "id": "t1-train",
                "step": 1,
                "cmd": "bash train_folds.sh runs/a configs/x.yaml spatial fm.tif 16",
            },
        ],
    )
    stub = ws / "train_folds.sh"
    stub.write_text(
        '#!/bin/sh\necho "WAIT-MEM fold 0: MemAvailable=1200MB < 5000MB"\nsleep 120\n',
        encoding="utf-8",
    )
    proc, log = _start_runner(q, ws)
    try:
        _wait_for(q / "state" / "t1-train.started", "runner's task")
        _wait_for_current(q, "t1-train")
        _wait_for_log_text(q, "WAIT-MEM")
        result = _diagnose(q, ws)
        assert result["verdict"] == "RUNNING-WAIT-MEM"
        assert result["current"] == "t1-train"
        assert any("memory gate" in note for note in result["notes"])
        assert any("6h" in note for note in result["notes"])
    finally:
        _kill_runner(proc, log)
