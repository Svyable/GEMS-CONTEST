#!/usr/bin/env python3
"""Forecast whether the experiment queue can drain before the safe cutoff.

`scripts/queue_status.py` (step 23) answers which queued tasks have
landed and `scripts/queue_critical_path.py` answers what should run
first. Neither answers the question the contest close forces: *is
there enough time left?* With 25 days to the 2026-12-03 close and a
box silent for weeks, the queue's feasibility is a number, not a
feeling — this script computes it from committed evidence only.

Method
------
- Observed throughput: committed run manifests
  (``results/runs/*/manifest.json``) carry ``created_utc``. The rate
  is recorded runs per day over the window from the first to the last
  manifest. It is a lower bound on the true train rate: a run whose
  scores were committed without a manifest (e.g. ref-cv-fault-b32)
  finished training inside the same window, so the count below
  reports scored-but-unrecorded runs alongside the rate.
- Remaining work: pending train tasks, from the same committed-tree
  evidence ``queue_status.py`` uses (manifest, score JSON, or fold
  files move a run out of PENDING). Score, view, assemble, gate, and
  compare tasks run in minutes on CPU and are excluded from the day
  count; the 5-fold train tasks dominate wall-clock, and the runner
  is sequential, so drain days = pending trains / runs per day.
- Per-decision earliest: for each pending gate/compare, the
  ``queue_critical_path`` feeders (including transitive train edges)
  give the train tasks on its critical path; days = trains / rate.
  These are per-decision lower bounds assuming the runner works on
  that decision's feeders first — shared trains (e.g. the ref-b16
  assemble feeding seven gates) count once per decision, not once
  overall, so the per-decision days do not sum to the drain days.
- Deadline: the safe planning cutoff 2026-12-03 22:00 UTC from the
  2026-10-28 prize preflight (live page says 23:59 UTC; Appendix A.1
  says 5:00 p.m. ET — plan to the earlier one). Verdict bands at the
  observed rate: FEASIBLE when the drain leaves a 5-day buffer for
  scoring, gating, and submission; TIGHT inside that buffer;
  INFEASIBLE past the cutoff. A second, independent line reports the
  current rate: 0 runs/day across the silence window means the queue
  never drains until the box resumes, whatever the observed rate was.

Output: the rate, the silence, remaining trains, drain days, days to
the cutoff, the verdict, and the per-decision earliest days. Exit 0
whenever the forecast could be produced (including an UNKNOWN rate
with fewer than two manifests); exit 2 on an unreadable or malformed
tasks.json, matching the sibling inventories.

Usage:
    uv run python scripts/queue_forecast.py [--root PATH]
        [--now 2026-11-08T13:39:00+00:00] [--cutoff 2026-12-03T22:00:00+00:00]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from queue_critical_path import CriticalPath

SAFE_CUTOFF = datetime(2026, 12, 3, 22, 0, 0, tzinfo=UTC)
# Planning buffer, in days, for scoring/gating/submission after the
# last train finishes. Below this the verdict is TIGHT, not FEASIBLE.
BUFFER_DAYS = 5.0


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _manifest_times(root: Path) -> tuple[list[datetime], int]:
    """created_utc of every committed run manifest (plus skip count)."""
    times: list[datetime] = []
    skipped = 0
    for path in sorted((root / "results" / "runs").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        parsed = _parse_time(manifest.get("created_utc"))
        if parsed is None:
            skipped += 1
            continue
        times.append(parsed)
    return sorted(times), skipped


def _scored_unrecorded(root: Path) -> int:
    """Runs with committed scores but no manifest (rate lower bound)."""
    count = 0
    for run_dir in (root / "results" / "runs").iterdir():
        if not run_dir.is_dir() or (run_dir / "manifest.json").exists():
            continue
        if list((run_dir / "scores").glob("*.json")) or list(run_dir.glob("*.json")):
            count += 1
    return count


def analyze(
    root: Path,
    tasks: list[dict],
    now: datetime,
    cutoff: datetime,
) -> dict:
    times, skipped = _manifest_times(root)
    critical = CriticalPath(root, tasks)
    analysis = critical.analyze()
    by_id = analysis["by_id"]

    pending_trains = [
        t for t in critical.production if t["kind"] == "train" and critical._needs_run(t)
    ]
    pending_other = {
        kind: sum(1 for t in critical.production if t["kind"] == kind and critical._needs_run(t))
        for kind in ("score", "view", "record")
    }
    pending_other["assemble"] = sum(1 for t in critical.assembles if critical._needs_run(t))

    rate = None
    window_days = None
    if len(times) >= 2:
        window_days = (times[-1] - times[0]).total_seconds() / 86400
        if window_days > 0:
            rate = len(times) / window_days

    silence_days = None
    if times:
        silence_days = (now - times[-1]).total_seconds() / 86400

    drain_days = None
    if rate:
        drain_days = len(pending_trains) / rate
    days_left = (cutoff - now).total_seconds() / 86400

    margin = None
    verdict = "UNKNOWN"
    if drain_days is not None:
        margin = days_left - drain_days
        if margin >= BUFFER_DAYS:
            verdict = "FEASIBLE"
        elif margin >= 0:
            verdict = "TIGHT"
        else:
            verdict = "INFEASIBLE"

    decisions = []
    for decision in analysis["decisions"]:
        trains = sorted(
            task_id
            for task_id in decision["feeders"]
            if by_id.get(task_id, {}).get("kind") == "train"
        )
        days = len(trains) / rate if rate else None
        decisions.append(
            {
                "kind": decision["kind"],
                "name": decision["name"],
                "trains": trains,
                "days": days,
                "missing": len(decision["missing"]),
            }
        )
    decisions.sort(key=lambda d: (len(d["trains"]), d["name"]))

    return {
        "rate": rate,
        "window_days": window_days,
        "manifests": len(times),
        "manifests_skipped": skipped,
        "first_manifest": times[0] if times else None,
        "last_manifest": times[-1] if times else None,
        "scored_unrecorded": _scored_unrecorded(root),
        "silence_days": silence_days,
        "pending_trains": len(pending_trains),
        "pending_other": pending_other,
        "drain_days": drain_days,
        "days_left": days_left,
        "margin_days": margin,
        "verdict": verdict,
        "cutoff": cutoff,
        "decisions": decisions,
        "unparsed": analysis["unparsed"],
    }


def render(forecast: dict) -> str:
    lines = []
    if forecast["rate"] is None:
        lines.append(
            f"OBSERVED RATE: UNKNOWN ({forecast['manifests']} usable manifests; need >= 2)"
        )
    else:
        lines.append(
            "OBSERVED RATE: "
            f"{forecast['rate']:.2f} runs/day "
            f"({forecast['manifests']} manifests, "
            f"{forecast['first_manifest'].date()} to "
            f"{forecast['last_manifest'].date()}, "
            f"{forecast['window_days']:.2f} days)"
        )
    if forecast["scored_unrecorded"]:
        lines.append(
            f"  (+{forecast['scored_unrecorded']} scored-but-unrecorded "
            "run(s): the rate above is a lower bound)"
        )
    if forecast["silence_days"] is not None:
        lines.append(
            f"SILENCE: {forecast['silence_days']:.1f} days since the last "
            "manifest (0 runs/day in that window — the queue never drains "
            "until the box resumes)"
        )
    other = ", ".join(f"{n} {kind}" for kind, n in forecast["pending_other"].items() if n)
    lines.append(
        f"REMAINING: {forecast['pending_trains']} train tasks"
        + (f" (+ {other}, minutes each, excluded from day count)" if other else "")
    )
    if forecast["drain_days"] is not None:
        lines.append(f"DRAIN: {forecast['drain_days']:.1f} days at the observed rate")
    lines.append(
        f"CUTOFF: {forecast['cutoff'].strftime('%Y-%m-%d %H:%M UTC')} "
        f"({forecast['days_left']:.1f} days left)"
    )
    if forecast["margin_days"] is not None:
        lines.append(
            f"VERDICT: {forecast['verdict']} "
            f"(margin {forecast['margin_days']:+.1f} days; "
            f"{BUFFER_DAYS:.0f}-day buffer for scoring/gating/submission)"
        )
    else:
        lines.append("VERDICT: UNKNOWN (no observed rate)")
    lines.append("\nEARLIEST DECISION (trains on its critical path):")
    for decision in forecast["decisions"]:
        days = f"{decision['days']:.1f}d" if decision["days"] is not None else "?"
        lines.append(
            f"  {decision['kind']:7} {decision['name']:38} "
            f"{len(decision['trains'])} trains, ~{days}, "
            f"{decision['missing']} missing inputs"
            + (f" ({', '.join(decision['trains'])})" if decision["trains"] else "")
        )
    if forecast["unparsed"]:
        lines.append(f"\nNOTE: {forecast['unparsed']} unparsed task(s) ignored")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecast queue drain vs the safe cutoff.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="ISO timestamp for 'now' (tests); default: current UTC time",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="ISO timestamp for the safe cutoff; default 2026-12-03T22:00:00+00:00",
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
    tasks_path = args.root / "scripts" / "queue" / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise TypeError("tasks.json is not a list")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"error: cannot read {tasks_path}: {exc}", file=sys.stderr)
        return 2
    print(render(analyze(args.root, tasks, now, cutoff)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
