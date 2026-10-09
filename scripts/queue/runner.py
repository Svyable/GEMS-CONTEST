#!/usr/bin/env python3
"""Resumable GEMS experiment queue.

- Tasks live in tasks.json (re-read before every task, so tasks can be appended live).
- A task is done when state/<id>.done exists; it is skipped on restart.
- On a non-zero exit the queue STOPS (state/<id>.failed) so a human/agent can inspect.
- status.json + STATUS.txt are rewritten on every transition; queue.log has all output.
Start/resume:  nohup python3 ${QUEUE_DIR:-/workspace/gems/queue}/runner.py >> ${QUEUE_DIR:-/workspace/gems/queue}/queue.log 2>&1 &
Single instance enforced with flock on runner.lock.
"""
import fcntl, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

Q = Path(os.environ.get("QUEUE_DIR", "/workspace/gems/queue")); S = Q / "state"; S.mkdir(exist_ok=True)
WORKSPACE = Path(os.environ.get("GEMS_WORKSPACE", "/workspace/gems/GEMS-CONTEST"))
lock = open(Q / "runner.lock", "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("runner already active; exiting"); sys.exit(0)
lock.write(str(os.getpid())); lock.flush()

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()

def write_status(current=None, note=""):
    tasks = json.loads((Q / "tasks.json").read_text())
    rows = []
    for t in tasks:
        st = "done" if (S / f"{t['id']}.done").exists() else "FAILED" if (S / f"{t['id']}.failed").exists() else "running" if t["id"] == current else "pending"
        info = {}
        for suffix in ("done", "failed", "started"):
            p = S / f"{t['id']}.{suffix}"
            if p.exists(): info[suffix] = p.read_text().strip()
        rows.append({"id": t["id"], "step": t.get("step"), "state": st, **info})
    status = {"updated": now(), "runner_pid": os.getpid(), "current": current, "note": note, "tasks": rows}
    (Q / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    lines = [f"GEMS queue status @ {status['updated']} (runner pid {os.getpid()}) {note}"]
    for r in rows:
        lines.append(f"[{r['state']:>7}] step {r['step']}  {r['id']}  {r.get('started','')} -> {r.get('done', r.get('failed',''))}")
    (Q / "STATUS.txt").write_text("\n".join(lines) + "\n")

while True:
    tasks = json.loads((Q / "tasks.json").read_text())
    if any((S / f"{t['id']}.failed").exists() for t in tasks):
        write_status(note="STOPPED: a task failed; fix and delete state/<id>.failed to resume"); break
    pending = [t for t in tasks if not (S / f"{t['id']}.done").exists()]
    if not pending:
        write_status(note="IDLE: all tasks done"); break
    t = pending[0]
    (S / f"{t['id']}.started").write_text(now())
    write_status(current=t["id"])
    print(f"\n===== {now()} START {t['id']}: {t['cmd']}", flush=True)
    t0 = time.time()
    rc = subprocess.call(["bash", "-c", t["cmd"]], cwd=str(WORKSPACE))
    el = int(time.time() - t0)
    print(f"===== {now()} END {t['id']} rc={rc} elapsed_s={el}", flush=True)
    (S / f"{t['id']}.{'done' if rc == 0 else 'failed'}").write_text(f"{now()} rc={rc} elapsed_s={el}")
    write_status()
