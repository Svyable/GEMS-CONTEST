"""Tests for the one-command box resumption (step 49).

Step 48 built the sprint-ordered queue for a late resumption, but the
box-side handoff was still five manual steps in the right order (git
pull, autostart install, sprint-queue install, runner start, doctor
check) — and a mistake in the order wastes a day of the window that
closes 2026-12-03 22:00 UTC. ``scripts/queue/resume_box.sh`` runs
them in order with guards. These tests run the real script in a
sandboxed workspace + queue dir with a stub runner (holds the flock,
sleeps, exits) and a stub installer (exit code via env), and the real
doctor for the final verdict — the real installer is never run (it
would touch systemd/crontab) and the real runner is never started
(it would execute queue commands).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "queue"
RESUME = QUEUE / "resume_box.sh"
DOCTOR = REPO / "scripts" / "queue_doctor.py"
BASH = shutil.which("bash") or "/bin/bash"

# Files the doctor's SYNC check compares between queue dir and checkout.
WRAPPERS = (
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

STUB_RUNNER = """\
import fcntl, os, sys, time
q = os.environ["QUEUE_DIR"]
lock = open(os.path.join(q, "runner.lock"), "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(0)
open(os.path.join(q, "stub-runner.log"), "a").write("started\\n")
time.sleep(8)
"""

STUB_INSTALLER = """\
#!/usr/bin/env bash
echo "stub-installer $*" >> "$QUEUE_DIR/stub-installer.log"
exit "${GEMS_STUB_INSTALLER_RC:-0}"
"""


def _sandbox(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Workspace (git checkout, no upstream) + queue dir + stubs.

    Returns (workspace, queue_dir, stub_runner, stub_installer).
    """
    ws = tmp_path / "ws"
    qdir = tmp_path / "queue"
    (ws / "scripts" / "queue").mkdir(parents=True)
    (qdir / "state").mkdir(parents=True)
    for name in ("tasks.json", "tasks.sprint.json"):
        shutil.copy2(QUEUE / name, ws / "scripts" / "queue" / name)
    for name in WRAPPERS:
        shutil.copy2(QUEUE / name, ws / "scripts" / "queue" / name)
        shutil.copy2(QUEUE / name, qdir / name)
    shutil.copy2(QUEUE / "tasks.json", qdir / "tasks.json")
    subprocess.run(["git", "-C", str(ws), "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(ws),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "sandbox",
        ],
        check=True,
    )
    runner = tmp_path / "stub_runner.py"
    runner.write_text(STUB_RUNNER, encoding="utf-8")
    installer = tmp_path / "stub_installer.sh"
    installer.write_text(STUB_INSTALLER, encoding="utf-8")
    return ws, qdir, runner, installer


def _env(ws: Path, qdir: Path, runner: Path, installer: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "GEMS_WORKSPACE": str(ws),
            "QUEUE_DIR": str(qdir),
            "GEMS_RESUME_RUNNER": str(runner),
            "GEMS_RESUME_INSTALLER": str(installer),
            "GEMS_RESUME_DOCTOR": str(DOCTOR),
        }
    )
    env.update(extra)
    return env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(RESUME), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _wait_lock_free(qdir: Path, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["flock", "-n", str(qdir / "runner.lock"), "-c", "true"],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return
        time.sleep(0.2)
    raise AssertionError("runner.lock still held after stub runner should have exited")


def test_script_syntax() -> None:
    proc = subprocess.run([BASH, "-n", str(RESUME)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_resume_happy_path(tmp_path: Path) -> None:
    ws, qdir, runner, installer = _sandbox(tmp_path)
    proc = _run(_env(ws, qdir, runner, installer))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Sprint queue installed over the live one, live one backed up.
    live = json.loads((qdir / "tasks.json").read_text(encoding="utf-8"))
    sprint = json.loads((QUEUE / "tasks.sprint.json").read_text(encoding="utf-8"))
    assert [t["id"] for t in live] == [t["id"] for t in sprint]
    backups = list(qdir.glob("tasks.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) != live
    # Installer ran, runner started, real doctor reported RUNNING + SYNC OK.
    assert (qdir / "stub-installer.log").exists()
    assert (qdir / "stub-runner.log").read_text().strip() == "started"
    assert "VERDICT: RUNNING" in proc.stdout
    assert "SYNC: OK" in proc.stdout
    assert "resume complete" in proc.stdout
    _wait_lock_free(qdir)


def test_resume_dry_run_changes_nothing(tmp_path: Path) -> None:
    ws, qdir, runner, installer = _sandbox(tmp_path)
    before = (qdir / "tasks.json").read_bytes()
    proc = _run(_env(ws, qdir, runner, installer), "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (qdir / "tasks.json").read_bytes() == before
    assert list(qdir.glob("tasks.json.bak.*")) == []
    assert not (qdir / "stub-installer.log").exists()
    assert not (qdir / "stub-runner.log").exists()
    assert "[dry-run]" in proc.stdout
    assert "dry run complete" in proc.stdout


def test_resume_refuses_failed_marker(tmp_path: Path) -> None:
    ws, qdir, runner, installer = _sandbox(tmp_path)
    tasks = json.loads((qdir / "tasks.json").read_text(encoding="utf-8"))
    failed_id = tasks[0]["id"]
    (qdir / "state" / f"{failed_id}.failed").write_text("2026-11-23 rc=3", encoding="utf-8")
    before = (qdir / "tasks.json").read_bytes()
    proc = _run(_env(ws, qdir, runner, installer))
    assert proc.returncode != 0
    assert failed_id in proc.stdout + proc.stderr
    assert (qdir / "tasks.json").read_bytes() == before
    assert not (qdir / "stub-runner.log").exists()


def test_resume_refuses_when_runner_active(tmp_path: Path) -> None:
    ws, qdir, runner, installer = _sandbox(tmp_path)
    holder = subprocess.Popen(
        ["flock", str(qdir / "runner.lock"), "-c", "sleep 10"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)  # let the holder take the lock
        before = (qdir / "tasks.json").read_bytes()
        proc = _run(_env(ws, qdir, runner, installer))
        assert proc.returncode != 0
        assert "already active" in proc.stdout + proc.stderr
        assert (qdir / "tasks.json").read_bytes() == before
        assert not (qdir / "stub-runner.log").exists()
    finally:
        holder.kill()
        holder.wait()


def test_resume_warns_and_continues_when_installer_has_no_mechanism(tmp_path: Path) -> None:
    ws, qdir, runner, installer = _sandbox(tmp_path)
    proc = _run(_env(ws, qdir, runner, installer, GEMS_STUB_INSTALLER_RC="2"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WARNING: no auto-start mechanism" in proc.stdout
    assert "will NOT survive a reboot" in proc.stdout
    live = json.loads((qdir / "tasks.json").read_text(encoding="utf-8"))
    sprint = json.loads((QUEUE / "tasks.sprint.json").read_text(encoding="utf-8"))
    assert [t["id"] for t in live] == [t["id"] for t in sprint]
    assert (qdir / "stub-runner.log").exists()
    _wait_lock_free(qdir)


def test_resume_refuses_stale_sprint(tmp_path: Path) -> None:
    ws, qdir, runner, installer = _sandbox(tmp_path)
    tasks = json.loads((qdir / "tasks.json").read_text(encoding="utf-8"))
    tasks.append({"id": "not-in-sprint", "step": 99, "cmd": "true"})
    (qdir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    proc = _run(_env(ws, qdir, runner, installer))
    assert proc.returncode != 0
    assert "not a permutation" in proc.stdout + proc.stderr
    assert not (qdir / "stub-runner.log").exists()


def test_resume_second_run_is_idempotent(tmp_path: Path) -> None:
    ws, qdir, runner, installer = _sandbox(tmp_path)
    env = _env(ws, qdir, runner, installer)
    first = _run(env)
    assert first.returncode == 0, first.stdout + first.stderr
    _wait_lock_free(qdir)
    second = _run(env)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "sprint queue already installed" in second.stdout
    # No second backup: the live queue was already the sprint queue.
    assert len(list(qdir.glob("tasks.json.bak.*"))) == 1
    _wait_lock_free(qdir)
