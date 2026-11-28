"""End-to-end resumption rehearsal (step 54).

Steps 42/43/49/53 each proved one link of the Nov 28 resumption chain
in isolation: the runner contract (42), the doctor reading
runner-produced state (43), resume_box.sh with a *stub* runner (49),
and queue_progress.py against hand-built markers (53). Nobody had run
the composed chain Sven executes today: resume_box.sh installs the
sprint queue, the *real* runner drains it in sprint order, and
queue_progress + the doctor agree on what landed.

These tests compose all four real tools in one sandbox. The queue is
synthetic (two gate decisions, trivial stub wrappers that exit 0 —
the real wrappers need GPU/data), but every tool under test is the
real one: resume_box.sh, runner.py, queue_progress.py,
queue_doctor.py. The live queue lists decision B first; the sprint
file lists decision A first; the rehearsal proves the reorder — not
just the file copy — governed execution, via the runner's own
queue.log START order.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "queue"
RESUME = QUEUE / "resume_box.sh"
RUNNER = QUEUE / "runner.py"
DOCTOR = REPO / "scripts" / "queue_doctor.py"
PROGRESS = REPO / "scripts" / "queue_progress.py"
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

STUB_INSTALLER = """\
#!/usr/bin/env bash
exit 0
"""

NOW = "2026-11-28T12:00:00+00:00"


def _tasks(qdir: Path, fail_a_train: bool = False) -> tuple[list[dict], list[dict]]:
    """(live order, sprint order) for two gate decisions A and B.

    Live order runs B's closure first; sprint order runs A's first —
    the same permutation relationship resume_box.sh installs on the
    box. Commands point at stub wrappers in the queue dir with the
    real argument shapes, so queue_progress's critical-path parser
    sees genuine train/score/gate tasks.
    """
    w = str(qdir)

    def train(task_id: str, run: str) -> dict:
        return {
            "id": task_id,
            "step": 7,
            "cmd": f"{w}/train_folds.sh {run} configs/ref.yaml spatial "
            "data/processed/cv-spatial-v1.tif 16",
        }

    def score(task_id: str, run: str, gate_dir: str) -> dict:
        return {
            "id": task_id,
            "step": 7,
            "cmd": f"{w}/score_view.sh {run} spatial data/processed/cv-spatial-v1.tif {gate_dir}",
        }

    def gate(task_id: str, name: str, cand: str) -> dict:
        return {
            "id": task_id,
            "step": 7,
            "cmd": f'{w}/gate.sh {name} runs/gate/ref {cand} configs/cand.yaml "hyp"',
        }

    a = [
        train("t-a", "runs/cand-a-spatial"),
        score("s-a", "runs/cand-a-spatial", "runs/gate/cand-a"),
        gate("g-a", "GATE-A", "runs/gate/cand-a"),
    ]
    b = [
        train("t-b", "runs/cand-b-spatial"),
        score("s-b", "runs/cand-b-spatial", "runs/gate/cand-b"),
        gate("g-b", "GATE-B", "runs/gate/cand-b"),
    ]
    return b + a, a + b


def _sandbox(tmp_path: Path, fail_a_train: bool = False) -> tuple[Path, Path, Path]:
    """Workspace (git, no upstream) + queue dir + stub installer."""
    ws = tmp_path / "ws"
    qdir = tmp_path / "queue"
    (ws / "scripts" / "queue").mkdir(parents=True)
    (qdir / "state").mkdir(parents=True)

    live, sprint = _tasks(qdir, fail_a_train)
    (ws / "scripts" / "queue" / "tasks.json").write_text(json.dumps(live), encoding="utf-8")
    (ws / "scripts" / "queue" / "tasks.sprint.json").write_text(
        json.dumps(sprint), encoding="utf-8"
    )
    (qdir / "tasks.json").write_text(json.dumps(live), encoding="utf-8")

    # Stub wrappers: exit 0, except the A train wrapper when the
    # failure variant is requested. Identical copies in workspace and
    # queue dir keep the doctor's SYNC check clean. runner.py and
    # lofo_calibrate_view.py are copied real (SYNC compares them; the
    # runner itself is started from the repo path via env override).
    for name in WRAPPERS:
        if name in ("runner.py", "lofo_calibrate_view.py"):
            shutil.copy2(QUEUE / name, ws / "scripts" / "queue" / name)
            shutil.copy2(QUEUE / name, qdir / name)
            continue
        body = "#!/usr/bin/env bash\nexit 0\n"
        if fail_a_train and name == "train_folds.sh":
            body = '#!/usr/bin/env bash\ncase "$1" in *cand-a*) exit 3;; *) exit 0;; esac\n'
        for dest in (ws / "scripts" / "queue" / name, qdir / name):
            dest.write_text(body, encoding="utf-8")
            dest.chmod(0o755)

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
    installer = tmp_path / "stub_installer.sh"
    installer.write_text(STUB_INSTALLER, encoding="utf-8")
    return ws, qdir, installer


def _env(ws: Path, qdir: Path, installer: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "GEMS_WORKSPACE": str(ws),
            "QUEUE_DIR": str(qdir),
            "GEMS_RESUME_RUNNER": str(RUNNER),
            "GEMS_RESUME_INSTALLER": str(installer),
            "GEMS_RESUME_DOCTOR": str(DOCTOR),
        }
    )
    return env


def _wait_drained(qdir: Path, task_ids: list[str], timeout: float = 30.0) -> None:
    """Wait until every task has a .done or .failed marker and the
    runner has released the flock."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        settled = all(
            (qdir / "state" / f"{tid}.done").exists() or (qdir / "state" / f"{tid}.failed").exists()
            for tid in task_ids
        )
        probe = subprocess.run(
            ["flock", "-n", str(qdir / "runner.lock"), "-c", "true"],
            capture_output=True,
            check=False,
        )
        if settled and probe.returncode == 0:
            return
        time.sleep(0.2)
    raise AssertionError("runner did not drain the queue in time")


def _progress(ws: Path, qdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROGRESS),
            "--root",
            str(ws),
            "--queue-dir",
            str(qdir),
            "--now",
            NOW,
            "--rate",
            "2.86",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _doctor(ws: Path, qdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DOCTOR), "--queue-dir", str(qdir), "--workspace", str(ws)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_resumption_chain_lands_both_decisions_in_sprint_order(tmp_path: Path) -> None:
    ws, qdir, installer = _sandbox(tmp_path)
    proc = subprocess.run(
        [BASH, str(RESUME)],
        env=_env(ws, qdir, installer),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "resume complete" in proc.stdout

    sprint_ids = ["t-a", "s-a", "g-a", "t-b", "s-b", "g-b"]
    _wait_drained(qdir, sprint_ids)

    # The reorder governed execution, not just the file copy: the real
    # runner's queue.log START order is the sprint order (A first),
    # even though the live queue listed B first.
    log = (qdir / "queue.log").read_text(encoding="utf-8")
    starts = re.findall(r"START (\S+):", log)
    assert starts == sprint_ids, starts

    # Progress reads the runner-produced markers: both decisions LANDED.
    prog = _progress(ws, qdir)
    assert prog.returncode == 0, prog.stderr
    assert "PROGRESS: 2 of 2 decisions landed" in prog.stdout
    assert re.search(r"LANDED\s+gate\s+GATE-A", prog.stdout)
    assert re.search(r"LANDED\s+gate\s+GATE-B", prog.stdout)
    assert "VERDICT: ALL LANDED" in prog.stdout or "2 of 2" in prog.stdout

    # The doctor agrees on the same state dir: nothing left to run.
    doc = _doctor(ws, qdir)
    assert "VERDICT: ALL-DONE" in doc.stdout, doc.stdout + doc.stderr


def test_resumption_chain_failure_blocks_decisions_and_stops(tmp_path: Path) -> None:
    ws, qdir, installer = _sandbox(tmp_path, fail_a_train=True)
    proc = subprocess.run(
        [BASH, str(RESUME)],
        env=_env(ws, qdir, installer),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The first task fails inside resume's 3s startup window: resume must
    # report the runner's STOPPED note (not "failed to start") and still
    # run the doctor, whose verdict names the failed task (step 54 fix).
    assert "runner started and stopped at once" in proc.stdout
    assert "VERDICT: STOPPED" in proc.stdout

    _wait_drained(qdir, ["t-a"])
    assert (qdir / "state" / "t-a.failed").exists()
    # The runner STOPs at the failure: nothing after t-a ever started.
    assert not (qdir / "state" / "s-a.started").exists()
    assert not (qdir / "state" / "t-b.started").exists()

    # Progress names the blocker for both decisions (A's gate is after
    # the failed train in sprint order; so is B's whole closure).
    prog = _progress(ws, qdir)
    assert prog.returncode == 0, prog.stderr
    assert "PROGRESS: 0 of 2 decisions landed" in prog.stdout
    assert "blocked by failed t-a" in prog.stdout
    assert "VERDICT: BLOCKED" in prog.stdout

    # The doctor agrees: STOPPED, naming the same task.
    doc = _doctor(ws, qdir)
    assert "VERDICT: STOPPED" in doc.stdout, doc.stdout + doc.stderr
    assert "t-a" in doc.stdout
