#!/usr/bin/env bash
# One-command box resumption for a late GEMS return (step 49).
#
# After a long silence the box needs five manual steps in the right order:
#   1. git pull (fresh checkout)
#   2. install auto-start (step 31) so the runner survives a reboot
#   3. install the sprint-ordered queue (steps 48/51) so the s7
#      promotion decision runs first, then every remaining decision
#      cheapest-first, instead of draining in list order
#   4. start the runner
#   5. run the doctor (step 29) and print its verdict
# A mistake in the order — starting the runner before the sprint copy, or
# copying over a STOPPED queue — wastes a day of the window that closes
# 2026-12-03 22:00 UTC. This script runs them in order with guards and
# refuses instead of guessing when the state is not what it expects.
#
# Usage (on the box, after the Nov 28 return):
#   bash scripts/queue/resume_box.sh [--dry-run]
#
# Guards (all refuse with exit 1 and a named reason, changing nothing):
#   - a runner already holds the flock on runner.lock
#   - any state/*.failed marker exists (the runner would STOP at once;
#     clear it first — the doctor names the task)
#   - the sprint file is not an exact permutation of the live tasks.json
#     (stale sprint file: regenerate it with scripts/queue_sprint.py)
# Auto-start install is best-effort: exit 2 from install_autostart.sh
# (no systemd user manager and no crontab) warns and continues — the
# runner still starts by nohup, it just will not survive a reboot.
#
# Environment overrides (same idiom as the other queue scripts):
#   GEMS_WORKSPACE        repo checkout (default /workspace/gems/GEMS-CONTEST)
#   QUEUE_DIR             box queue dir (default /workspace/gems/queue)
#   GEMS_RESUME_RUNNER    runner to start (default $GEMS_WORKSPACE/scripts/queue/runner.py)
#   GEMS_RESUME_INSTALLER auto-start installer (default $GEMS_WORKSPACE/scripts/queue/install_autostart.sh)
#   GEMS_RESUME_DOCTOR    doctor to run at the end (default $GEMS_WORKSPACE/scripts/queue_doctor.py)
#   GEMS_RESUME_SPRINT    sprint-ordered queue (default $GEMS_WORKSPACE/scripts/queue/tasks.sprint.json)
set -euo pipefail

WORKSPACE="${GEMS_WORKSPACE:-/workspace/gems/GEMS-CONTEST}"
QUEUE_DIR="${QUEUE_DIR:-/workspace/gems/queue}"
RUNNER="${GEMS_RESUME_RUNNER:-$WORKSPACE/scripts/queue/runner.py}"
INSTALLER="${GEMS_RESUME_INSTALLER:-$WORKSPACE/scripts/queue/install_autostart.sh}"
DOCTOR="${GEMS_RESUME_DOCTOR:-$WORKSPACE/scripts/queue_doctor.py}"
SPRINT="${GEMS_RESUME_SPRINT:-$WORKSPACE/scripts/queue/tasks.sprint.json}"
DRY_RUN=0

say() { printf '%s\n' "$*"; }
die() { printf 'resume_box: ERROR: %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
elif [ $# -gt 0 ]; then
    die "usage: bash scripts/queue/resume_box.sh [--dry-run]"
fi

# --- pre-flight: read-only checks, safe to run under --dry-run too ---------
command -v flock >/dev/null 2>&1 || die "flock not found on PATH"
command -v python3 >/dev/null 2>&1 || die "python3 not found on PATH"
[ -f "$SPRINT" ] || die "sprint file not found: $SPRINT (regenerate with scripts/queue_sprint.py)"
[ -f "$QUEUE_DIR/tasks.json" ] || die "live queue not found: $QUEUE_DIR/tasks.json"
git -C "$WORKSPACE" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "$WORKSPACE is not a git checkout"

# A live runner holds the flock; a second one would exit at once, so refuse
# rather than silently doing nothing after installing the sprint queue.
if [ -f "$QUEUE_DIR/runner.lock" ] && ! flock -n "$QUEUE_DIR/runner.lock" -c true 2>/dev/null; then
    die "runner already active (runner.lock is held) — nothing to resume"
fi

# A .failed marker stops the runner on its first loop iteration; installing
# a new tasks.json over it would just stop again on the same task.
FAILED_MARKER="$(ls "$QUEUE_DIR"/state/*.failed 2>/dev/null | head -n 1 || true)"
if [ -n "$FAILED_MARKER" ]; then
    FAILED_ID="$(basename "$FAILED_MARKER" .failed)"
    die "queue is STOPPED on failed task $FAILED_ID — inspect queue.log, fix the cause, then: rm $FAILED_MARKER"
fi

# The sprint file must be an exact permutation of the live queue; if the
# live queue changed since step 48 generated it, the reorder is stale.
python3 - "$QUEUE_DIR/tasks.json" "$SPRINT" <<'EOF' || die "sprint file $SPRINT is not a permutation of the live queue — regenerate it with scripts/queue_sprint.py"
import json, sys
live = json.load(open(sys.argv[1], encoding="utf-8"))
sprint = json.load(open(sys.argv[2], encoding="utf-8"))
if sorted(t["id"] for t in live) != sorted(t["id"] for t in sprint):
    sys.exit(1)
EOF

# --- 1. fresh checkout ------------------------------------------------------
if git -C "$WORKSPACE" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    if [ "$DRY_RUN" = 1 ]; then
        say "[dry-run] would run: git -C $WORKSPACE pull --ff-only"
    else
        say "step 1/5: git pull --ff-only"
        git -C "$WORKSPACE" pull --ff-only \
            || die "git pull failed — resolve the checkout by hand, then re-run"
    fi
else
    say "step 1/5: no upstream configured — skipping pull (checkout must already be current)"
fi

# --- 2. auto-start (best-effort: exit 2 means no mechanism, warn + go on) --
if [ "$DRY_RUN" = 1 ]; then
    say "[dry-run] would run: bash $INSTALLER"
else
    say "step 2/5: install auto-start"
    rc=0
    bash "$INSTALLER" || rc=$?
    if [ "$rc" -eq 2 ]; then
        say "WARNING: no auto-start mechanism on this host (installer exit 2) — the runner below starts by nohup and will NOT survive a reboot"
    elif [ "$rc" -ne 0 ]; then
        die "install_autostart.sh failed with exit $rc"
    fi
fi

# --- 3. install the sprint queue (back up the live one first) --------------
if cmp -s "$QUEUE_DIR/tasks.json" "$SPRINT"; then
    say "step 3/5: sprint queue already installed — no copy needed"
elif [ "$DRY_RUN" = 1 ]; then
    say "[dry-run] would back up $QUEUE_DIR/tasks.json and copy $SPRINT over it"
else
    BACKUP="$QUEUE_DIR/tasks.json.bak.$(date +%Y%m%d-%H%M%S)"
    say "step 3/5: install sprint queue (live queue backed up to $BACKUP)"
    cp "$QUEUE_DIR/tasks.json" "$BACKUP"
    cp "$SPRINT" "$QUEUE_DIR/tasks.json"
fi

# --- 4. start the runner ----------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
    say "[dry-run] would run: nohup python3 $RUNNER >> $QUEUE_DIR/queue.log 2>&1 &"
else
    say "step 4/5: start runner"
    mkdir -p "$QUEUE_DIR/state"
    nohup python3 "$RUNNER" >>"$QUEUE_DIR/queue.log" 2>&1 &
    sleep 3
    if [ -f "$QUEUE_DIR/runner.lock" ] && ! flock -n "$QUEUE_DIR/runner.lock" -c true 2>/dev/null; then
        say "runner is up (runner.lock held)"
    else
        # The runner exits 0 at once when every task is done (IDLE) or when
        # its first task fails within the startup window (STOPPED — it
        # started, ran, and stopped on a task failure; step 54 rehearsal).
        # Both are queue-state exits, not startup failures: report the note
        # and continue to the doctor, which names the failed task and the
        # fix. Anything else is a failed start.
        NOTE="$(python3 -c "import json; print(json.load(open('$QUEUE_DIR/status.json'))['note'])" 2>/dev/null || true)"
        if [ -n "$NOTE" ] && [ "${NOTE#IDLE}" != "$NOTE" ]; then
            say "runner exited at once: $NOTE"
        elif [ -n "$NOTE" ] && [ "${NOTE#STOPPED}" != "$NOTE" ]; then
            say "runner started and stopped at once: $NOTE"
        else
            say "runner did not stay up — last queue.log lines:"
            tail -n 20 "$QUEUE_DIR/queue.log" 2>/dev/null || true
            die "runner failed to start (see queue.log)"
        fi
    fi
fi

# --- 5. doctor verdict ------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
    say "[dry-run] would run: python3 $DOCTOR --queue-dir $QUEUE_DIR --workspace $WORKSPACE"
    say "dry run complete — nothing was changed"
    exit 0
fi
say "step 5/5: doctor verdict"
python3 "$DOCTOR" --queue-dir "$QUEUE_DIR" --workspace "$WORKSPACE"
say "resume complete — the sprint queue is running; the s7 loss-gate closure is first in list order"
