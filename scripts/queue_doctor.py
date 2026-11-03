#!/usr/bin/env python3
# ruff: noqa: DTZ005, DTZ006, DTZ007 — runner stamps are naive local
# times by construction (runner.py writes datetime.now() + %Z label), so
# ages are computed naive-to-naive and reported approximately.
"""Diagnose a silent GEMS queue runner from its state dir (step 29).

`scripts/queue_status.py` (step 23) answers which queued tasks have
landed in the committed tree, and `scripts/queue_critical_path.py`
answers what should run first. Neither answers the question the box
has posed since 2026-10-12: *why has nothing landed for weeks?* This
tool answers that from the box-side queue dir alone — the one place
the VM cannot see. It reads only ``tasks.json``, ``state/*.done`` /
``*.failed`` / ``*.started``, ``runner.lock``, ``status.json``, and
the tail of ``queue.log``; it never opens rasters, predictions, or
checkpoints, and it changes nothing, so running it is safe while a
runner is live.

Stall signatures, in the order the runner itself would hit them:

- FAILED: a ``state/<id>.failed`` marker exists. The runner checks
  for any failed marker before every task and exits immediately, so
  one failed task stops the whole queue on every restart until the
  marker is cleared. The fix is human: inspect ``queue.log`` for that
  task, fix the cause, delete the marker, restart the runner.
- RUNNING: ``runner.lock`` is flock-held. The current task comes from
  ``status.json``; if its command is a ``train_folds.sh`` invocation
  and the ``queue.log`` tail shows ``WAIT-MEM`` lines, the runner is
  not training — it is parked in the train wrapper's memory gate
  (added 2026-10-10 after OOM kills), waiting for ``MemAvailable``
  to reach ``GEMS_MIN_AVAIL_MB``. That wait has no timeout, so a
  memory-hogging job elsewhere on the shared box stalls the queue
  silently and indefinitely.
- INTERRUPTED: the first pending task has a ``.started`` marker but
  no ``.done`` / ``.failed`` and the lock is free — the runner died
  mid-task (box reboot, kill, crash). Safe to restart: the runner
  re-runs the task and the wrappers skip finished folds.
- NOT-RUNNING: pending tasks remain, no failed marker, lock free —
  the runner is simply not started. This is the expected verdict
  after a box reboot, since nothing auto-starts the runner.
- WAIT: the next task is a ``while kill -0 <pid>`` wait loop (the
  s1 housekeeping task) — the queue is waiting on an external
  process, not on GPU work.
- ALL-DONE: no pending tasks.

Output is a verdict line, the evidence behind it (counts, first
pending task in list order — the task the runner will run next —
current task and its age, ``status.json`` / ``queue.log`` ages),
and the exact resume command for the verdict. Exit 0 whenever a
diagnosis was produced; exit 2 when tasks.json is missing or
malformed, because without the queue the diagnosis is meaningless.

Usage:
    python scripts/queue_doctor.py [--queue-dir PATH] [--tasks PATH]
    QUEUE_DIR=/workspace/gems/queue python scripts/queue_doctor.py
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_QUEUE_DIR = "/workspace/gems/queue"
STAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
WAIT_RE = re.compile(r"^\s*while\s+kill\s+-0\s+(\d+)")
LOG_TAIL_BYTES = 200_000


def _parse_stamp(text: str) -> datetime | None:
    match = STAMP_RE.search(text.strip())
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _age_text(stamp: datetime | None, now: datetime) -> str:
    if stamp is None:
        return "unknown age"
    delta = now - stamp
    if delta.total_seconds() < 0:
        return "in the future (clock skew?)"
    days, rem = divmod(int(delta.total_seconds()), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h ago"
    if hours:
        return f"{hours}h {minutes}m ago"
    return f"{minutes}m ago"


def _lock_held(queue_dir: Path) -> tuple[bool, str | None]:
    """True when another process holds the runner lock; also its PID line."""
    lock_path = queue_dir / "runner.lock"
    if not lock_path.exists():
        return False, None
    pid = lock_path.read_text(encoding="utf-8", errors="replace").strip() or None
    # Append mode: flock works on it, and unlike "w" it never truncates
    # the pid the live runner wrote — the doctor changes nothing.
    with open(lock_path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True, pid
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False, pid


def _log_tail_flags(queue_dir: Path) -> dict[str, bool | str | None]:
    """WAIT-MEM / last-START evidence from the tail of queue.log."""
    log_path = queue_dir / "queue.log"
    if not log_path.exists():
        return {"exists": False, "wait_mem": False, "last_start": None}
    with open(log_path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - LOG_TAIL_BYTES))
        tail = handle.read().decode("utf-8", errors="replace")
    starts = re.findall(r"START (\S+):", tail)
    return {
        "exists": True,
        "wait_mem": "WAIT-MEM" in tail,
        "last_start": starts[-1] if starts else None,
    }


def diagnose(queue_dir: Path, tasks_path: Path) -> dict:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise TypeError("tasks.json is not a list")
    now = datetime.now()
    state = queue_dir / "state"
    rows = []
    for task in tasks:
        task_id = task.get("id", "?")
        markers = {}
        for suffix in ("done", "failed", "started"):
            path = state / f"{task_id}.{suffix}"
            if path.exists():
                markers[suffix] = path.read_text(encoding="utf-8", errors="replace").strip()
        if "done" in markers:
            status = "done"
        elif "failed" in markers:
            status = "FAILED"
        else:
            status = "pending"
        rows.append({"id": task_id, "cmd": task.get("cmd", ""), "status": status, **markers})
    done = [r for r in rows if r["status"] == "done"]
    failed = [r for r in rows if r["status"] == "FAILED"]
    pending = [r for r in rows if r["status"] == "pending"]
    lock_held, lock_pid = _lock_held(queue_dir)
    status_json: dict = {}
    status_path = queue_dir / "status.json"
    if status_path.exists():
        try:
            status_json = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status_json = {}
    status_age = _age_text(_parse_stamp(str(status_json.get("updated", ""))), now)
    log_flags = _log_tail_flags(queue_dir)
    log_age = "missing"
    if log_flags["exists"]:
        mtime = datetime.fromtimestamp((queue_dir / "queue.log").stat().st_mtime)
        log_age = _age_text(mtime, now)
    first_pending = pending[0] if pending else None
    current_id = status_json.get("current")
    current = next((r for r in rows if r["id"] == current_id), None)
    current_age = _age_text(_parse_stamp(current.get("started", "")) if current else None, now)

    verdict = "ALL-DONE"
    notes: list[str] = []
    resume = "Nothing to resume: every queued task is done."
    if failed:
        verdict = "STOPPED"
        blocker = failed[0]
        notes.append(
            f"runner exits before every task while state/{blocker['id']}.failed exists; "
            f"first failed in list order is {blocker['id']} ({blocker.get('failed', '')})."
        )
        resume = (
            f"Inspect queue.log for {blocker['id']}, fix the cause, then:\n"
            f"  rm {state / (blocker['id'] + '.failed')}\n"
            f"  nohup python3 {queue_dir}/runner.py >> {queue_dir}/queue.log 2>&1 &"
        )
    elif not pending:
        notes.append("every queued task has a .done marker.")
    elif lock_held:
        verdict = "RUNNING"
        notes.append(f"runner.lock is held (pid file: {lock_pid or 'empty'}).")
        if current is not None:
            notes.append(f"current task {current['id']} started {current_age}.")
            if "train_folds.sh" in current["cmd"] and log_flags["wait_mem"]:
                verdict = "RUNNING-WAIT-MEM"
                notes.append(
                    "queue.log tail shows WAIT-MEM: the train wrapper's memory gate "
                    "is waiting for MemAvailable >= GEMS_MIN_AVAIL_MB (default 5000MB). "
                    "This wait has no timeout — check what else on the box holds memory."
                )
        elif log_flags["last_start"]:
            notes.append(
                f"status.json names no current task; last START in queue.log: {log_flags['last_start']}."
            )
        resume = "Runner is live; no action unless the WAIT-MEM note applies."
    elif first_pending is not None and "started" in first_pending:
        verdict = "INTERRUPTED"
        notes.append(
            f"first pending task {first_pending['id']} has a .started marker "
            f"({first_pending['started']}) but no .done/.failed and the lock is free: "
            "the runner died mid-task."
        )
        resume = (
            "Safe to restart — the runner re-runs the task and wrappers skip finished folds:\n"
            f"  nohup python3 {queue_dir}/runner.py >> {queue_dir}/queue.log 2>&1 &"
        )
    else:
        verdict = "NOT-RUNNING"
        notes.append(
            "pending tasks remain, no failed marker, lock free: the runner is not started."
        )
        resume = (
            "Start it (nothing auto-starts it after a reboot):\n"
            f"  nohup python3 {queue_dir}/runner.py >> {queue_dir}/queue.log 2>&1 &"
        )
    if first_pending is not None:
        wait_match = WAIT_RE.match(first_pending["cmd"])
        if wait_match:
            notes.append(
                f"next task {first_pending['id']} is a wait loop on external pid "
                f"{wait_match.group(1)} — the queue waits on that process, not on GPU work."
            )
    return {
        "verdict": verdict,
        "counts": {"done": len(done), "failed": len(failed), "pending": len(pending)},
        "first_pending": first_pending["id"] if first_pending else None,
        "current": current_id,
        "current_age": current_age,
        "status_age": status_age,
        "log_age": log_age,
        "notes": notes,
        "resume": resume,
    }


def render(result: dict) -> str:
    counts = result["counts"]
    lines = [
        f"VERDICT: {result['verdict']}",
        f"TASKS: {counts['done']} done, {counts['failed']} failed, {counts['pending']} pending",
        f"FIRST PENDING (list order, runs next): {result['first_pending'] or 'none'}",
        f"CURRENT: {result['current'] or 'none'} ({result['current_age']})",
        f"STATUS.JSON AGE: {result['status_age']}; QUEUE.LOG AGE: {result['log_age']}",
    ]
    for note in result["notes"]:
        lines.append(f"NOTE: {note}")
    lines.append(f"RESUME:\n{result['resume']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=Path(os.environ.get("QUEUE_DIR", DEFAULT_QUEUE_DIR)),
        help="box queue dir holding tasks.json, state/, runner.lock (default: $QUEUE_DIR or "
        "/workspace/gems/queue)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="tasks.json path (default: <queue-dir>/tasks.json)",
    )
    args = parser.parse_args(argv)
    tasks_path = args.tasks or (args.queue_dir / "tasks.json")
    try:
        result = diagnose(args.queue_dir, tasks_path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"error: cannot diagnose from {tasks_path}: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
