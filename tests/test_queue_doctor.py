"""Tests for the queue doctor (step 29).

The box has been silent since 2026-10-12, and the VM cannot see the
box-side queue dir — so the doctor must be verified against simulated
queue dirs here, covering every verdict it can return. Each test builds
a minimal queue dir (tasks.json + state/ markers + optional lock,
status.json, queue.log) and runs the doctor in-process.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from queue_doctor import diagnose, main, render


def _write_queue(
    tmp_path: Path,
    tasks: list[dict],
    *,
    done: tuple[str, ...] = (),
    failed: tuple[str, ...] = (),
    started: tuple[str, ...] = (),
    status: dict | None = None,
    log: str | None = None,
) -> Path:
    queue_dir = tmp_path / "queue"
    state = queue_dir / "state"
    state.mkdir(parents=True)
    (queue_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    for task_id in done:
        (state / f"{task_id}.done").write_text(
            "2026-10-12 10:00:00 UTC rc=0 elapsed_s=60", encoding="utf-8"
        )
    for task_id in failed:
        (state / f"{task_id}.failed").write_text(
            "2026-10-20 09:00:00 UTC rc=1 elapsed_s=30", encoding="utf-8"
        )
    for task_id in started:
        (state / f"{task_id}.started").write_text("2026-10-20 09:00:00 UTC", encoding="utf-8")
    if status is not None:
        (queue_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    if log is not None:
        (queue_dir / "queue.log").write_text(log, encoding="utf-8")
    return queue_dir


TASKS = [
    {"id": "t1-train", "step": 1, "cmd": "train_folds.sh runs/a configs/x.yaml spatial fm.tif 16"},
    {"id": "t2-score", "step": 1, "cmd": "score_view.sh runs/a spatial fm.tif runs/gate/g"},
    {"id": "t3-gate", "step": 1, "cmd": 'gate.sh G runs/gate/ref runs/gate/g configs/x.yaml "h"'},
]


def test_stopped_names_first_failed_and_clear_command(tmp_path: Path) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",), failed=("t2-score",))
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "STOPPED"
    assert result["counts"] == {"done": 1, "failed": 1, "pending": 1}
    assert result["first_pending"] == "t3-gate"
    assert "t2-score.failed" in result["resume"]
    assert "runner.py" in result["resume"]


def test_all_done(tmp_path: Path) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train", "t2-score", "t3-gate"))
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "ALL-DONE"
    assert result["first_pending"] is None


def test_not_running_when_lock_free_and_pending(tmp_path: Path) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",))
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "NOT-RUNNING"
    assert result["first_pending"] == "t2-score"
    assert "runner.py" in result["resume"]


def test_not_running_resume_points_at_autostart_installer(tmp_path: Path) -> None:
    """Step 31: the NOT-RUNNING resume must name the persistent fix.

    The doctor used to say "nothing auto-starts it after a reboot"
    and only print the one-off nohup command. Now that
    install_autostart.sh exists, the resume must point at it, so a
    reboot is not diagnosed the same way twice.
    """
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",))
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "NOT-RUNNING"
    assert "install_autostart.sh" in result["resume"]
    assert "nohup python3" in result["resume"]


def test_interrupted_when_started_but_no_done_or_failed(tmp_path: Path) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",), started=("t2-score",))
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "INTERRUPTED"
    assert any("died mid-task" in note for note in result["notes"])


def test_running_when_lock_held(tmp_path: Path) -> None:
    queue_dir = _write_queue(
        tmp_path,
        TASKS,
        done=("t1-train",),
        started=("t2-score",),
        status={"updated": "2026-11-03 07:00:00 UTC", "current": "t2-score"},
    )
    lock_path = queue_dir / "runner.lock"
    with open(lock_path, "w") as handle:
        handle.write("12345")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "RUNNING"
    assert result["current"] == "t2-score"


def test_running_wait_mem_when_train_task_and_wait_mem_in_log(tmp_path: Path) -> None:
    log = (
        "2026-11-03 07:00:00 START t1-train: train_folds.sh runs/a\n"
        "2026-11-03 07:00:01 WAIT-MEM fold 0: MemAvailable=1200MB < 5000MB\n"
    )
    queue_dir = _write_queue(
        tmp_path,
        TASKS,
        started=("t1-train",),
        status={"updated": "2026-11-03 07:00:00 UTC", "current": "t1-train"},
        log=log,
    )
    lock_path = queue_dir / "runner.lock"
    with open(lock_path, "w") as handle:
        handle.write("12345")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "RUNNING-WAIT-MEM"
    assert any("memory gate" in note for note in result["notes"])


def test_wait_loop_task_flagged(tmp_path: Path) -> None:
    tasks = [
        {
            "id": "t0-wait",
            "step": 1,
            "cmd": "while kill -0 1769192 2>/dev/null; do sleep 30; done; echo done",
        }
    ]
    queue_dir = _write_queue(tmp_path, tasks)
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert result["verdict"] == "NOT-RUNNING"
    assert any("wait loop on external pid 1769192" in note for note in result["notes"])


def test_missing_tasks_json_is_hard_error(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    assert main(["--queue-dir", str(queue_dir)]) == 2


def test_malformed_tasks_json_is_hard_error(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    (queue_dir / "tasks.json").write_text("{not json", encoding="utf-8")
    assert main(["--queue-dir", str(queue_dir)]) == 2


def test_main_exit_zero_on_diagnosis(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",))
    assert main(["--queue-dir", str(queue_dir)]) == 0
    assert "VERDICT: NOT-RUNNING" in capsys.readouterr().out


def test_real_queue_dir_diagnoses_without_crashing(tmp_path: Path) -> None:
    """The committed tasks.json parses and the doctor reports on it.

    The VM has no box state dir, so every task reads pending — the point
    is the doctor runs end-to-end against the real queue shape, not that
    the verdict matches box reality (which the VM cannot see).
    """
    repo = Path(__file__).resolve().parent.parent
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    result = diagnose(queue_dir, repo / "scripts" / "queue" / "tasks.json")
    assert result["verdict"] == "NOT-RUNNING"
    assert result["counts"]["pending"] == 129
    assert result["first_pending"] == "s1-wait-existing-spatial-loop"


# --- Step 30: RUNNING-STALLED, wait-pid liveness, queue-vs-checkout SYNC ---


def _hold_lock_and_diagnose(queue_dir: Path, workspace: Path | None = None) -> dict:
    lock_path = queue_dir / "runner.lock"
    with open(lock_path, "w") as handle:
        handle.write("12345")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return diagnose(queue_dir, queue_dir / "tasks.json", workspace)


def _age_log(queue_dir: Path, when: datetime) -> None:
    stamp = when.timestamp()
    os.utime(queue_dir / "queue.log", (stamp, stamp))


def test_running_stalled_when_lock_held_and_log_silent_48h(tmp_path: Path) -> None:
    queue_dir = _write_queue(
        tmp_path,
        TASKS,
        done=("t1-train",),
        started=("t2-score",),
        status={"updated": "2026-10-20 09:00:00 UTC", "current": "t2-score"},
        log="2026-10-20 09:00:00 START t2-score: score_view.sh runs/a\n",
    )
    _age_log(queue_dir, datetime(2026, 10, 21, tzinfo=UTC))
    result = _hold_lock_and_diagnose(queue_dir)
    assert result["verdict"] == "RUNNING-STALLED"
    assert any("hung" in note for note in result["notes"])
    assert "kill the child" in result["resume"]


def test_wait_mem_takes_precedence_over_stalled(tmp_path: Path) -> None:
    log = (
        "2026-10-20 09:00:00 START t1-train: train_folds.sh runs/a\n"
        "2026-10-20 09:00:01 WAIT-MEM fold 0: MemAvailable=1200MB < 5000MB\n"
    )
    queue_dir = _write_queue(
        tmp_path,
        TASKS,
        started=("t1-train",),
        status={"updated": "2026-10-20 09:00:00 UTC", "current": "t1-train"},
        log=log,
    )
    _age_log(queue_dir, datetime(2026, 10, 21, tzinfo=UTC))
    result = _hold_lock_and_diagnose(queue_dir)
    assert result["verdict"] == "RUNNING-WAIT-MEM"


def test_wait_loop_pid_alive_is_genuine_wait(tmp_path: Path) -> None:
    tasks = [
        {
            "id": "t0-wait",
            "step": 1,
            "cmd": f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 30; done; echo done",
        }
    ]
    queue_dir = _write_queue(tmp_path, tasks)
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert any("still running on this host" in note for note in result["notes"])


def test_wait_loop_pid_dead_is_not_the_stall(tmp_path: Path) -> None:
    proc = subprocess.Popen(["true"])
    proc.wait()
    dead_pid = proc.pid
    tasks = [
        {
            "id": "t0-wait",
            "step": 1,
            "cmd": f"while kill -0 {dead_pid} 2>/dev/null; do sleep 30; done; echo done",
        }
    ]
    queue_dir = _write_queue(tmp_path, tasks)
    result = diagnose(queue_dir, queue_dir / "tasks.json")
    assert any("not running on this host" in note for note in result["notes"])
    assert any("exit immediately once the runner starts" in note for note in result["notes"])


def _write_workspace(tmp_path: Path, tasks: list[dict], wrappers: dict[str, str]) -> Path:
    ws_queue = tmp_path / "ws" / "scripts" / "queue"
    ws_queue.mkdir(parents=True)
    (ws_queue / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    for name, content in wrappers.items():
        (ws_queue / name).write_text(content, encoding="utf-8")
    return tmp_path / "ws"


def test_sync_diverged_tasks_and_wrapper(tmp_path: Path) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",))
    (queue_dir / "train_folds.sh").write_text("old wrapper\n", encoding="utf-8")
    extra = {"id": "t4-new", "step": 1, "cmd": "score_view.sh runs/b spatial fm.tif runs/gate/g"}
    workspace = _write_workspace(tmp_path, [*TASKS, extra], {"train_folds.sh": "new wrapper\n"})
    result = diagnose(queue_dir, queue_dir / "tasks.json", workspace)
    assert result["sync"] is not None
    assert result["sync"]["tasks_diverged"] is True
    assert result["sync"]["only_in_repo"] == ["t4-new"]
    assert result["sync"]["wrappers_diverged"] == ["train_folds.sh"]
    assert any("SYNC DIVERGED" in note for note in result["notes"])
    assert "SYNC: DIVERGED" in render(result)


def test_sync_ok_and_rendered(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",))
    (queue_dir / "train_folds.sh").write_text("same wrapper\n", encoding="utf-8")
    workspace = _write_workspace(tmp_path, TASKS, {"train_folds.sh": "same wrapper\n"})
    assert main(["--queue-dir", str(queue_dir), "--workspace", str(workspace)]) == 0
    out = capsys.readouterr().out
    assert "SYNC: OK" in out
    assert "3 tasks" in out


def test_sync_skipped_without_checkout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    queue_dir = _write_queue(tmp_path, TASKS, done=("t1-train",))
    result = diagnose(queue_dir, queue_dir / "tasks.json", tmp_path / "no-such-checkout")
    assert result["sync"] is None
    assert result["verdict"] == "NOT-RUNNING"
    assert (
        main(["--queue-dir", str(queue_dir), "--workspace", str(tmp_path / "no-such-checkout")])
        == 0
    )
    assert "SYNC: skipped" in capsys.readouterr().out
