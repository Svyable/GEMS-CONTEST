#!/usr/bin/env python3
"""Live chain progress from the box queue state (step 53).

The forecast (step 34) answers "can the queue drain before the
cutoff?" from committed manifests, and the chained sprint (step 51)
answers "in what order should decisions land?" at build time. Neither
answers the question the Nov 28 resumption actually asks, hour by
hour: *which decisions have landed on the box, which are in flight,
and are we still on track for the cutoff?* The doctor (steps 29-32)
reports runner health (RUNNING / STOPPED / ...), not decision
progress, and the committed-tree inventories (steps 19/23) are blind
to box-side work until the box commits it.

This script reads the box queue dir directly — the installed
``tasks.json`` (sprint order after ``resume_box.sh``), the
``state/*.{done,failed,started}`` markers, and ``status.json`` — and
maps that state onto every gate/compare decision in the queue:

- LANDED: the decision task has a ``.done`` marker, or the decision
  is already decided in the committed tree (a gate verdict JSON with
  a known verdict, or a compare log) even if the box marker is gone.
- FAILED: the decision task itself has a ``.failed`` marker.
- BLOCKED: an earlier task in list order has a ``.failed`` marker —
  the runner stops there, so this decision cannot land until the
  marker is cleared (fix the cause, or skip the failed decision with
  ``queue_skip_decision.py``, step 52).
- RUNNING: the decision task is the runner's current task.
- IN-FLIGHT: at least one feeder task (``queue_critical_path.py``)
  already has a ``.done`` marker.
- PENDING: none of the above.

Remaining trains before a decision = train tasks earlier in list
order without a ``.done`` marker (the runner is sequential, so those
must finish first; score/view/record/assemble tasks take minutes and
are excluded from the day count, matching the forecast). The ETA is
``now + remaining trains / observed rate`` (rate from
``queue_forecast.py``, or ``--rate``); ON-TRACK means the ETA lands
before the 2026-12-03 22:00 UTC safe cutoff, AT-RISK past it. Box
truth wins over committed truth: a train recorded in the committed
tree but without a ``.done`` marker still counts as remaining,
because the runner keys on markers, not manifests.

Exit 0 whenever the report could be produced; exit 2 on a missing or
malformed queue ``tasks.json``, matching the sibling inventories.

Usage:
    uv run python scripts/queue_progress.py [--queue-dir PATH]
        [--root PATH] [--tasks PATH] [--now ISO] [--cutoff ISO]
        [--rate RUNS_PER_DAY]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from queue_critical_path import CriticalPath, _compare_ran, _gate_decided
from queue_forecast import SAFE_CUTOFF, _parse_time
from queue_forecast import analyze as forecast_analyze

DEFAULT_QUEUE_DIR = "/workspace/gems/queue"


def _markers(state_dir: Path) -> dict[str, set[str]]:
    """Task ids by marker suffix from the box state dir ({} if absent)."""
    found: dict[str, set[str]] = {"done": set(), "failed": set(), "started": set()}
    if not state_dir.is_dir():
        return found
    for suffix, ids in found.items():
        for path in state_dir.glob(f"*.{suffix}"):
            ids.add(path.name[: -len(suffix) - 1])
    return found


def _current_task(queue_dir: Path) -> str | None:
    status_path = queue_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    current = status.get("current") if isinstance(status, dict) else None
    return current if isinstance(current, str) and current else None


def analyze_progress(
    root: Path,
    tasks: list[dict],
    markers: dict[str, set[str]],
    current: str | None,
    now: datetime,
    cutoff: datetime,
    rate: float | None,
) -> dict:
    cp = CriticalPath(root, tasks)
    analysis = cp.analyze()
    feeders_by_name = {d["name"]: set(d["feeders"]) for d in analysis["decisions"]}
    index_by_id = {t.get("id", "?"): i for i, t in enumerate(tasks)}
    done = markers["done"]
    failed = markers["failed"]

    train_ids = [t["task_id"] for t in cp.production if t["kind"] == "train"]
    trains_done = sum(1 for task_id in train_ids if task_id in done)
    failed_positions = sorted(
        (index_by_id[task_id], task_id) for task_id in failed if task_id in index_by_id
    )

    decisions: list[dict] = []
    parsed_decisions = [(g, "gate") for g in cp.gates] + [(c, "compare") for c in cp.compares]
    for parsed, kind in parsed_decisions:
        task_id = parsed["task_id"]
        name = parsed["name"]
        position = index_by_id.get(task_id, len(tasks))
        if kind == "gate":
            committed_decided, verdict = _gate_decided(root, parsed)
        else:
            committed_decided, verdict = _compare_ran(root, parsed), ""
        landed = task_id in done or committed_decided
        blockers = [fid for pos, fid in failed_positions if pos < position]
        remaining_trains = sum(
            1
            for t in cp.production
            if t["kind"] == "train" and t["index"] < position and t["task_id"] not in done
        )
        feeders = feeders_by_name.get(name, set())
        feeders_done = sum(1 for fid in feeders if fid in done)
        if landed:
            status = "LANDED"
        elif task_id in failed:
            status = "FAILED"
        elif blockers:
            status = "BLOCKED"
        elif task_id == current:
            status = "RUNNING"
        elif feeders_done:
            status = "IN-FLIGHT"
        else:
            status = "PENDING"
        eta = None
        on_track = None
        if not landed and rate:
            eta = now + timedelta(days=remaining_trains / rate)
            on_track = eta <= cutoff
        decisions.append(
            {
                "kind": kind,
                "name": name,
                "task_id": task_id,
                "position": position,
                "status": status,
                "committed_decided": committed_decided,
                "verdict": verdict,
                "blocker": blockers[0] if blockers else None,
                "remaining_trains": remaining_trains,
                "feeders": len(feeders),
                "feeders_done": feeders_done,
                "eta": eta,
                "on_track": on_track,
            }
        )
    decisions.sort(key=lambda d: d["position"])

    landed_count = sum(1 for d in decisions if d["status"] == "LANDED")
    projected = landed_count + sum(1 for d in decisions if d["on_track"])
    if decisions and landed_count == len(decisions):
        verdict = "COMPLETE"
    elif failed_positions:
        verdict = "BLOCKED"
    else:
        verdict = "IN PROGRESS"
    return {
        "decisions": decisions,
        "landed": landed_count,
        "projected_by_cutoff": projected,
        "verdict": verdict,
        "trains_done": trains_done,
        "trains_total": len(train_ids),
        "tasks_done": sum(1 for t in tasks if t.get("id") in done),
        "tasks_total": len(tasks),
        "failed": [fid for _pos, fid in failed_positions],
        "current": current,
        "rate": rate,
        "now": now,
        "cutoff": cutoff,
        "unparsed": cp.unparsed,
    }


def render(progress: dict) -> str:
    lines = [
        (
            f"PROGRESS: {progress['landed']} of {len(progress['decisions'])} decisions landed; "
            f"{progress['trains_done']} of {progress['trains_total']} train tasks done; "
            f"{progress['tasks_done']} of {progress['tasks_total']} tasks done"
        )
    ]
    if progress["current"]:
        lines.append(f"CURRENT: {progress['current']}")
    if progress["failed"]:
        lines.append(
            "FAILED: "
            + ", ".join(progress["failed"])
            + " (runner STOPs here — fix the cause and clear the marker, "
            "or skip the decision with queue_skip_decision.py)"
        )
    if progress["rate"]:
        lines.append(f"RATE: {progress['rate']:.2f} runs/day (observed; ETAs are lower bounds)")
    else:
        lines.append("RATE: UNKNOWN (no observed rate — no ETAs)")
    lines.append(
        f"CUTOFF: {progress['cutoff'].strftime('%Y-%m-%d %H:%M UTC')} "
        f"({(progress['cutoff'] - progress['now']).total_seconds() / 86400:.1f} days left)"
    )
    lines.append("\nDECISIONS (in queue order):")
    for decision in progress["decisions"]:
        line = (
            f"  {decision['status']:9} {decision['kind']:7} {decision['name']:38} "
            f"({decision['task_id']})"
        )
        if decision["status"] == "LANDED":
            if decision["verdict"]:
                line += f" — {decision['verdict']} (committed)"
            elif decision["committed_decided"]:
                line += " — decided (committed)"
            else:
                line += " — done on box, not yet committed"
        elif decision["status"] == "BLOCKED":
            line += f" — blocked by failed {decision['blocker']}"
        elif decision["status"] == "FAILED":
            line += " — decision task failed; see queue.log"
        else:
            line += (
                f" — {decision['remaining_trains']} trains ahead, "
                f"{decision['feeders_done']}/{decision['feeders']} feeders done"
            )
            if decision["eta"] is not None:
                track = "ON-TRACK" if decision["on_track"] else "AT-RISK"
                line += f", ETA {decision['eta'].strftime('%Y-%m-%d %H:%M UTC')} — {track}"
        lines.append(line)
    lines.append(
        f"\nPROJECTED BY CUTOFF: {progress['projected_by_cutoff']} of "
        f"{len(progress['decisions'])} decisions at the observed rate"
    )
    lines.append(f"VERDICT: {progress['verdict']}")
    if progress["unparsed"]:
        lines.append(f"NOTE: {progress['unparsed']} unparsed task(s) ignored")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live decision progress from box queue state.")
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=Path(os.environ.get("QUEUE_DIR", DEFAULT_QUEUE_DIR)),
        help="box queue dir holding tasks.json, state/, status.json",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repo checkout")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="tasks.json to read (default: <queue-dir>/tasks.json)",
    )
    parser.add_argument("--now", type=str, default=None, help="ISO timestamp for 'now' (tests)")
    parser.add_argument("--cutoff", type=str, default=None, help="ISO safe cutoff")
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="runs/day override (default: observed rate from committed manifests)",
    )
    args = parser.parse_args(argv)
    now = _parse_time(args.now) if args.now else datetime.now(UTC)
    if now is None:
        print(f"error: cannot parse --now {args.now!r}", file=sys.stderr)
        return 2
    cutoff = _parse_time(args.cutoff) if args.cutoff else SAFE_CUTOFF
    if cutoff is None:
        print(f"error: cannot parse --cutoff {args.cutoff!r}", file=sys.stderr)
        return 2
    tasks_path = args.tasks if args.tasks is not None else args.queue_dir / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise TypeError("tasks.json is not a list")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"error: cannot read {tasks_path}: {exc}", file=sys.stderr)
        return 2
    rate = args.rate
    if rate is None:
        try:
            rate = forecast_analyze(args.root, tasks, now, cutoff)["rate"]
        except OSError:
            rate = None  # e.g. a checkout without results/runs yet
    markers = _markers(args.queue_dir / "state")
    current = _current_task(args.queue_dir)
    print(render(analyze_progress(args.root, tasks, markers, current, now, cutoff, rate)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
