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

Step 30 adds two blind spots the step-29 doctor could not see:

- SYNC: queued commands invoke ``/workspace/gems/queue/*.sh`` — the
  queue dir's own copies, not the repo checkout — and the runner
  re-reads the queue dir's ``tasks.json`` before every task. Pushed
  fixes (steps 27–28 wrapper fixes, step-26 task insertions) reach
  the runner only if the queue dir was re-synced from the checkout.
  The doctor therefore hash-compares the queue dir's ``tasks.json``
  and wrapper files against ``<workspace>/scripts/queue/`` (default
  ``$GEMS_WORKSPACE`` or ``/workspace/gems/GEMS-CONTEST``) and
  reports DIVERGED with the exact ids/files that differ. The check
  is skipped, not failed, when no checkout is present (the VM case).
- RUNNING-STALLED: a held lock alone does not mean healthy work —
  a hung subprocess holds the lock forever while writing nothing.
  No step in this repo runs silently for 48 hours (full-map folds
  log per fold and take minutes to hours), so a lock held with the
  current task started 48+ hours ago *and* ``queue.log`` unwritten
  for 48+ hours is refined to RUNNING-STALLED. WAIT-MEM takes
  precedence: its wait prints once and then sleeps silently, so an
  old log mtime there is expected, not a stall.
- WAIT pid liveness: a wait-loop first-pending task is only a real
  wait while its pid is alive on this host; a dead pid means the
  task exits immediately once the runner starts, so it is not the
  stall.

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
- RUNNING-STALLED: lock held, current task 48+ hours old, and
  ``queue.log`` unwritten for 48+ hours (and not WAIT-MEM).
- INTERRUPTED: the first pending task has a ``.started`` marker but
  no ``.done`` / ``.failed`` and the lock is free — the runner died
  mid-task (box reboot, kill, crash). Safe to restart: the runner
  re-runs the task and the wrappers skip finished folds.
- NOT-RUNNING: pending tasks remain, no failed marker, lock free —
  the runner is simply not started. This is the expected verdict
  after a box reboot, unless ``install_autostart.sh`` (step 31: a
  systemd user unit, or an @reboot cron entry as fallback) was run —
  in which case a NOT-RUNNING verdict after a reboot means the
  auto-start itself failed and its logs are the next thing to read.
- WAIT: the next task is a ``while kill -0 <pid>`` wait loop (the
  s1 housekeeping task) — the queue is waiting on an external
  process, not on GPU work, unless that pid is already dead.
- ALL-DONE: no pending tasks.

Output is a verdict line, the evidence behind it (counts, first
pending task in list order — the task the runner will run next —
current task and its age, ``status.json`` / ``queue.log`` ages),
the queue-vs-checkout SYNC line, and the exact resume command for
the verdict. Exit 0 whenever a diagnosis was produced; exit 2 when
tasks.json is missing or malformed, because without the queue the
diagnosis is meaningless.

Usage:
    python scripts/queue_doctor.py [--queue-dir PATH] [--tasks PATH]
        [--workspace PATH]
    QUEUE_DIR=/workspace/gems/queue python scripts/queue_doctor.py
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_QUEUE_DIR = "/workspace/gems/queue"
DEFAULT_WORKSPACE = "/workspace/gems/GEMS-CONTEST"
STAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
WAIT_RE = re.compile(r"^\s*while\s+kill\s+-0\s+(\d+)")
LOG_TAIL_BYTES = 200_000
# No repo step runs silently this long: full-map folds log per fold
# and take minutes to hours, so 48h of lock-held log silence is a
# hung subprocess, not slow training.
STALLED_AFTER_SECONDS = 48 * 3600
# Queue-dir files the runner actually executes (tasks invoke these
# absolute queue-dir paths, not the repo checkout — see module doc).
WRAPPER_FILES = (
    "train_folds.sh",
    "score_view.sh",
    "record.sh",
    "gate.sh",
    "endpoint_view.sh",
    "curved_view.sh",
    "compare_seed.sh",
    "compare_schedule.sh",
    "rival_threshold.sh",
    "runner.py",
    "lofo_calibrate_view.py",
)


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


def _resolve_workspace(workspace: Path | None) -> Path:
    if workspace is not None:
        return workspace
    return Path(os.environ.get("GEMS_WORKSPACE", DEFAULT_WORKSPACE))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync_check(queue_dir: Path, queue_tasks: list, workspace: Path) -> dict | None:
    """Queue-dir copies vs the repo checkout, or None if no checkout.

    Returns None (skip, not failure) when the checkout's tasks.json
    is absent — the VM case, where no box checkout exists. A present
    but malformed checkout tasks.json is reported, not raised: the
    queue dir's tasks are authoritative for the verdict.
    """
    repo_queue = workspace / "scripts" / "queue"
    repo_tasks_path = repo_queue / "tasks.json"
    if not repo_tasks_path.exists():
        return None
    try:
        repo_tasks = json.loads(repo_tasks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "error": str(exc)}
    if not isinstance(repo_tasks, list):
        return {"available": False, "error": "checkout tasks.json is not a list"}
    queue_ids = [t.get("id", "?") for t in queue_tasks]
    repo_ids = [t.get("id", "?") for t in repo_tasks]
    queue_set, repo_set = set(queue_ids), set(repo_ids)
    wrappers_diverged: list[str] = []
    wrappers_compared = 0
    for name in WRAPPER_FILES:
        queue_file, repo_file = queue_dir / name, repo_queue / name
        if queue_file.exists() and repo_file.exists():
            wrappers_compared += 1
            if _file_sha256(queue_file) != _file_sha256(repo_file):
                wrappers_diverged.append(name)
    return {
        "available": True,
        "queue_count": len(queue_ids),
        "repo_count": len(repo_ids),
        "only_in_queue": [i for i in queue_ids if i not in repo_set],
        "only_in_repo": [i for i in repo_ids if i not in queue_set],
        "tasks_diverged": queue_set != repo_set,
        "wrappers_compared": wrappers_compared,
        "wrappers_diverged": wrappers_diverged,
    }


def diagnose(queue_dir: Path, tasks_path: Path, workspace: Path | None = None) -> dict:
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
    current_started = _parse_stamp(current.get("started", "")) if current else None
    if current_started is None:
        current_started = _parse_stamp(str(status_json.get("updated", "")))
    current_age = _age_text(current_started, now)
    log_mtime = None
    if log_flags["exists"]:
        log_mtime = datetime.fromtimestamp((queue_dir / "queue.log").stat().st_mtime)
    # Hung-subprocess signal: lock held, current task old, log silent.
    # A missing log cannot prove silence (it may have been rotated),
    # so it is reported as "missing", not promoted to STALLED.
    stalled = (
        current_started is not None
        and log_mtime is not None
        and (now - current_started).total_seconds() >= STALLED_AFTER_SECONDS
        and (now - log_mtime).total_seconds() >= STALLED_AFTER_SECONDS
    )
    workspace_path = _resolve_workspace(workspace)
    sync = _sync_check(queue_dir, tasks, workspace_path)

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
            elif stalled:
                verdict = "RUNNING-STALLED"
                notes.append(
                    f"current task started {current_age} and queue.log was last written "
                    f"{_age_text(log_mtime, now)}: no repo step runs silently for 48h, "
                    "so the runner's child subprocess is hung (or its output is lost) "
                    "while still holding the lock."
                )
                resume = (
                    "Do not delete runner.lock while its holder lives (flock blocks a "
                    "second runner anyway). Inspect the runner's child process tree "
                    f"(ps --ppid {lock_pid or 'the lock pid'}), and if the child is "
                    "genuinely hung with no output for 48h+, kill the child: the runner "
                    "will then record the task as failed and stop for inspection."
                )
        elif log_flags["last_start"]:
            notes.append(
                f"status.json names no current task; last START in queue.log: {log_flags['last_start']}."
            )
        if verdict != "RUNNING-STALLED":
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
            "Start it now:\n"
            f"  nohup python3 {queue_dir}/runner.py >> {queue_dir}/queue.log 2>&1 &\n"
            "Make it survive reboots (systemd user unit, or an @reboot cron\n"
            "entry when no user systemd manager answers):\n"
            f"  bash {workspace_path}/scripts/queue/install_autostart.sh"
        )
    if first_pending is not None:
        wait_match = WAIT_RE.match(first_pending["cmd"])
        if wait_match:
            pid = int(wait_match.group(1))
            if _pid_alive(pid):
                liveness = f"pid {pid} is still running on this host, so the wait is genuine."
            else:
                liveness = (
                    f"pid {pid} is not running on this host, so this task will exit "
                    "immediately once the runner starts; it is not the stall."
                )
            notes.append(
                f"next task {first_pending['id']} is a wait loop on external pid "
                f"{pid} — the queue waits on that process, not on GPU work. {liveness}"
            )
    if (
        sync is not None
        and sync.get("available")
        and (sync["tasks_diverged"] or sync["wrappers_diverged"])
    ):
        repo_queue = workspace_path / "scripts" / "queue"
        notes.append(
            "SYNC DIVERGED: the queue dir's tasks.json / wrappers differ from the "
            f"repo checkout at {workspace_path} (see SYNC line) — the runner executes "
            "the queue-dir copies, so pushed fixes do not reach it until re-synced: "
            f"cp {repo_queue}/tasks.json {repo_queue}/*.sh {repo_queue}/*.py {queue_dir}/ "
            "(tasks.json is re-read before each task; wrappers are read at task start, "
            "so a sync only affects tasks not yet started)."
        )
    elif sync is not None and not sync.get("available"):
        notes.append(f"SYNC CHECK FAILED: {sync.get('error', 'unknown error')}")
    return {
        "verdict": verdict,
        "counts": {"done": len(done), "failed": len(failed), "pending": len(pending)},
        "first_pending": first_pending["id"] if first_pending else None,
        "current": current_id,
        "current_age": current_age,
        "status_age": status_age,
        "log_age": log_age,
        "workspace": str(workspace_path),
        "sync": sync,
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
    sync = result.get("sync")
    if sync is None:
        lines.append(
            "SYNC: skipped — no repo checkout tasks.json at "
            f"{result.get('workspace')}/scripts/queue/tasks.json"
        )
    elif not sync.get("available"):
        lines.append(f"SYNC: check failed — {sync.get('error', 'unknown error')}")
    elif sync["tasks_diverged"] or sync["wrappers_diverged"]:
        lines.append(
            f"SYNC: DIVERGED — queue {sync['queue_count']} tasks vs repo "
            f"{sync['repo_count']} tasks; only in queue: {sync['only_in_queue']}; "
            f"only in repo: {sync['only_in_repo']}; wrappers diverged: "
            f"{sync['wrappers_diverged']} (compared {sync['wrappers_compared']})"
        )
    elif sync["wrappers_compared"] == 0:
        lines.append(
            f"SYNC: OK — queue tasks.json matches repo checkout "
            f"({sync['queue_count']} tasks); no wrapper files present in the "
            "queue dir to compare"
        )
    else:
        lines.append(
            f"SYNC: OK — queue tasks.json matches repo checkout "
            f"({sync['queue_count']} tasks); {sync['wrappers_compared']} wrapper "
            "files compared, all identical"
        )
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
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="repo checkout to compare the queue dir against for the SYNC check "
        "(default: $GEMS_WORKSPACE or /workspace/gems/GEMS-CONTEST)",
    )
    args = parser.parse_args(argv)
    tasks_path = args.tasks or (args.queue_dir / "tasks.json")
    try:
        result = diagnose(args.queue_dir, tasks_path, args.workspace)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"error: cannot diagnose from {tasks_path}: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
