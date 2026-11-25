#!/usr/bin/env python3
"""Build a sprint-ordered queue for a late box resumption.

Step 47 showed the full 9.8-day drain only fits if the box resumes by
2026-11-24, but every pending decision stays reachable even on a Dec 1
resume — so past Nov 24 the plan switches from draining the queue in
list order to sprinting the cheapest promotion decision first (the s7
loss gate: 3 trains, ~1.0d). The runner executes strictly in list
order, so that switch needs a reordered queue, not just a restart.

This script produces one: the full live queue, reordered so the target
decision's closure runs first. The sprint prefix is the decision's
*positional closure*, not just its pending feeders:

- the decision's feeder tasks from ``queue_critical_path.py`` (trains,
  scores, assembles that still need a run), in queue order;
- every earlier task that writes a file a prefix task consumes
  (``_writers_before``), because the box runner keys on ``.done``
  markers, not on the committed tree: a task the committed tree calls
  RECORDED may still be un-run on the box, and the re-scores that
  repair a gate copy must run before the assemble that copies it
  (step 26). Score/view writers are all included (a re-score repairs
  content); assemble writers only the earliest (copies are idempotent
  given the same sources, so the first assemble suffices). Already-run
  producers stay in the prefix: the order audit is positional, so a
  consumer may not precede its earliest producer even when that
  producer's output is committed, and on the box they are skipped in
  seconds via ``.done`` markers (or re-run if box state was lost);
- the record tasks for runs trained in the prefix whose run has no
  manifest yet (completing the new train/score/record triples; records
  for already-recorded runs are skipped as no-ops);
- the decision's gate/compare task itself, last in the prefix.

The suffix is every other live task in its original relative order, so
nothing is dropped: the sprint file is a permutation of tasks.json.
That matters twice: the doctor's SYNC check is id-set based, so a pure
reorder stays SYNC-clean, and ``.done`` markers key on task ids, so
already-run prefix tasks are skipped in seconds on the box.

The output is validated before it is written: the task-id multiset
must match tasks.json exactly, and ``queue_order_audit.py`` must report
zero ORDER-VIOLATIONs on the sprint order (a reorder that moved a
consumer ahead of its earliest producer would fail here instead of on
the box). The output is a *separate* file (default
``scripts/queue/tasks.sprint.json``) and never overwrites the live
``tasks.json`` — the swap is a deliberate box-side step:

    cp scripts/queue/tasks.sprint.json <QUEUE_DIR>/tasks.json

after ``git pull``. Exit 0 on success, 2 on an unreadable tasks.json,
an unknown or already-decided target, or ``--out`` pointing at the
live tasks.json.

Step 51 adds ``--chain``: the single-decision sprint leaves the suffix
in original queue order, which schedules some cheap decisions last —
the 0-train curved-continuation gate sat at position 127 of 129, so
after the s7 prefix the box would grind through the original order and
never reach it before the cutoff. The chained sprint keeps the first
decision's closure first (default s7, the highest-expected-gain gate),
then appends every remaining pending decision's closure in ascending
standalone cost (pending trains on its critical path; ties by the
decision task's queue position), shared feeders placed once, and the
remainder in original order. The report prints each decision's
marginal and cumulative pending trains (and days at the observed
rate), so the resumption plan shows exactly which decisions land
before the cutoff — with ``--resume-date`` each decision is marked
FITS or PAST CUTOFF against the safe cutoff.

Usage:
    uv run python scripts/queue_sprint.py [--root PATH]
        [--decision NAME] [--out PATH] [--dry-run]
    uv run python scripts/queue_sprint.py --chain [--decision NAME]
        [--resume-date YYYY-MM-DD] [--out PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from queue_critical_path import VIEWS, CriticalPath, _gate_decided
from queue_forecast import SAFE_CUTOFF
from queue_forecast import analyze as forecast_analyze
from queue_order_audit import OrderAudit

DEFAULT_DECISION = "CAND-LOSS_DISTANCE_TVERSKY-B16"
DEFAULT_OUT = "tasks.sprint.json"


def _writers_before(cp: CriticalPath, parsed: dict) -> set[str]:
    """Task ids writing a file ``parsed`` consumes, earlier in the queue.

    Score/view writers are all included (a re-score repairs content);
    assemble writers only the earliest (a copy is idempotent given the
    same sources). Train tasks consume no queued file.
    """
    index = parsed["index"]
    kind = parsed["kind"]
    writers: set[str] = set()
    if kind == "train":
        return writers
    if kind == "score":
        run = parsed["run"]
        for train in cp.train_by_run.get(run, []):
            if train["index"] < index:
                writers.add(train["task_id"])
        for view in cp.production:
            if view["kind"] == "view" and view["run"] == run and view["index"] < index:
                writers.add(view["task_id"])
        return writers
    if kind == "view":
        input_run = parsed.get("input_run", parsed["run"])
        for train in cp.train_by_run.get(input_run, []):
            if train["index"] < index:
                writers.add(train["task_id"])
        for view in cp.production:
            if view["kind"] == "view" and view["run"] == input_run and view["index"] < index:
                writers.add(view["task_id"])
        for score in cp.score_by_run.get((input_run, parsed["scheme"]), []):
            if score["index"] < index:
                writers.add(score["task_id"])
        return writers
    if kind == "record":
        for scheme in VIEWS:
            for score in cp.score_by_run.get((parsed["run"], scheme), []):
                if score["index"] < index:
                    writers.add(score["task_id"])
        return writers
    if kind == "assemble":
        for src_dir, src_view in parsed["sources"]:
            for score in cp.score_by_gate.get((src_dir, src_view), []):
                if score["index"] < index:
                    writers.add(score["task_id"])
        return writers
    if kind in ("gate", "compare"):
        dirs = [parsed["incumbent_dir"], parsed["candidate_dir"]]
        for gate_dir in dirs:
            for view in VIEWS:
                for score in cp.score_by_gate.get((gate_dir, view), []):
                    if score["index"] < index:
                        writers.add(score["task_id"])
                assembles = [
                    a
                    for a in cp.assembles
                    if a["dest"] == gate_dir and view + ".json" in a["files"]
                ]
                if assembles:
                    earliest = min(assembles, key=lambda a: a["index"])
                    if earliest["index"] < index:
                        writers.add(earliest["task_id"])
        if kind == "compare":
            if parsed.get("seed"):
                for train in cp.train_by_run.get(parsed["candidate_base"], []):
                    if train["index"] < index:
                        writers.add(train["task_id"])
            else:
                for view in VIEWS:
                    run = f"{parsed['candidate_base']}-{view}-b16"
                    for train in cp.train_by_run.get(run, []):
                        if train["index"] < index:
                            writers.add(train["task_id"])
        return writers
    return writers


def _parsed_index(cp: CriticalPath, tasks: list[dict]) -> tuple[dict, dict, dict]:
    by_id = {t.get("id", "?"): t for t in tasks}
    index_by_id = {t.get("id", "?"): i for i, t in enumerate(tasks)}
    parsed_by_id: dict[str, dict] = {}
    for parsed in cp.production + cp.assembles + cp.gates + cp.compares:
        parsed_by_id[parsed["task_id"]] = parsed
    return by_id, index_by_id, parsed_by_id


def _find_decision(
    root: Path, cp: CriticalPath, analysis: dict, decision_name: str
) -> tuple[dict, dict]:
    """Return (decision, decision_task) for a pending decision name."""
    decision = next((d for d in analysis["decisions"] if d["name"] == decision_name), None)
    if decision is None:
        decided = None
        for gate in cp.gates:
            if gate["name"] == decision_name:
                is_decided, verdict = _gate_decided(root, gate)
                if is_decided:
                    decided = verdict
        if decided is not None:
            raise ValueError(f"decision {decision_name} already decided: {decided}")
        pending = sorted(d["name"] for d in analysis["decisions"])
        raise ValueError(
            f"unknown decision {decision_name!r}; pending decisions: {', '.join(pending)}"
        )
    if decision["kind"] == "gate":
        decision_task = next(g for g in cp.gates if g["name"] == decision_name)
    else:
        decision_task = next(c for c in cp.compares if c["name"] == decision_name)
    return decision, decision_task


def _closure_ids(
    cp: CriticalPath,
    decision: dict,
    decision_task: dict,
    parsed_by_id: dict,
    index_by_id: dict,
) -> list[str]:
    """Ordered positional closure for one decision, decision task last.

    Seed: pending feeders + the decision task; then close under
    "every earlier writer of a consumed file" plus pending records
    for runs trained inside the prefix, to a fixpoint.
    """
    prefix: set[str] = set(decision["feeders"]) | {decision_task["task_id"]}
    record_by_run = {p["run"]: p for p in cp.production if p["kind"] == "record"}
    changed = True
    while changed:
        changed = False
        for task_id in sorted(prefix, key=lambda i: index_by_id[i]):
            parsed = parsed_by_id.get(task_id)
            if parsed is None:
                continue
            for writer in _writers_before(cp, parsed):
                if writer not in prefix:
                    prefix.add(writer)
                    changed = True
        for task_id in sorted(prefix, key=lambda i: index_by_id[i]):
            parsed = parsed_by_id.get(task_id)
            if parsed is None or parsed["kind"] != "train":
                continue
            record = record_by_run.get(parsed["run"])
            if (
                record is not None
                and record["status"] == "PENDING"
                and record["task_id"] not in prefix
            ):
                prefix.add(record["task_id"])
                changed = True

    ordered = sorted(prefix, key=lambda i: index_by_id[i])
    ordered.remove(decision_task["task_id"])
    ordered.append(decision_task["task_id"])
    return ordered


def _validate_order(tasks: list[dict], ordered_ids: list[str], by_id: dict) -> dict:
    """Permutation + order-audit validation shared by sprint and chain."""
    if Counter(ordered_ids) != Counter(t.get("id", "?") for t in tasks):
        raise ValueError("sprint is not a permutation of tasks.json (ids lost or doubled)")
    audit = OrderAudit([by_id[i] for i in ordered_ids]).audit()
    violations = audit["violations"]
    if violations:
        detail = "; ".join(
            f"{v['consumer']['task_id']} after {v['producer']['task_id']} ({v['input']})"
            for v in violations
        )
        raise ValueError(f"sprint order fails the order audit: {detail}")
    return audit


def _pending_trains(ids: list[str], parsed_by_id: dict) -> int:
    return sum(
        1
        for i in ids
        if parsed_by_id.get(i, {}).get("kind") == "train"
        and parsed_by_id.get(i, {}).get("status") == "PENDING"
    )


def build_sprint(root: Path, tasks: list[dict], decision_name: str) -> dict:
    """Return the sprint queue (prefix + suffix) for one pending decision."""
    cp = CriticalPath(root, tasks)
    analysis = cp.analyze()
    by_id, index_by_id, parsed_by_id = _parsed_index(cp, tasks)
    decision, decision_task = _find_decision(root, cp, analysis, decision_name)

    ordered_prefix = _closure_ids(cp, decision, decision_task, parsed_by_id, index_by_id)
    prefix = set(ordered_prefix)
    suffix = [t.get("id", "?") for t in tasks if t.get("id", "?") not in prefix]
    sprint_ids = ordered_prefix + suffix
    audit = _validate_order(tasks, sprint_ids, by_id)
    sprint_tasks = [by_id[i] for i in sprint_ids]

    prefix_trains = sum(1 for i in ordered_prefix if parsed_by_id.get(i, {}).get("kind") == "train")
    # The day estimate counts only trains that still need a run per the
    # committed tree; already-run producers in the prefix are skipped on
    # the box via .done markers and cost no GPU time.
    prefix_trains_pending = sum(
        1
        for i in ordered_prefix
        if parsed_by_id.get(i, {}).get("kind") == "train"
        and parsed_by_id.get(i, {}).get("status") == "PENDING"
    )
    return {
        "tasks": sprint_tasks,
        "prefix": ordered_prefix,
        "decision": decision,
        "prefix_trains": prefix_trains,
        "prefix_trains_pending": prefix_trains_pending,
        "audit": audit,
    }


def render(result: dict, out: Path | None, root: Path, rate: float | None) -> str:
    decision = result["decision"]
    prefix = result["prefix"]
    trains = result["prefix_trains"]
    pending_trains = result["prefix_trains_pending"]
    if rate:
        est = (
            f"{pending_trains} still needing a run, "
            f"~{pending_trains / rate:.1f}d at {rate:.2f} runs/day"
        )
    else:
        est = f"{pending_trains} still needing a run, no observed rate for a day estimate"
    lines = [
        (
            f"SPRINT: {decision['kind']} {decision['name']} "
            f"({len(decision['missing'])} missing inputs)"
        ),
        f"PREFIX: {len(prefix)} tasks, {trains} trains ({est})",
        "  " + "\n  ".join(prefix),
        f"SUFFIX: {len(result['tasks']) - len(prefix)} tasks in original order",
        (
            f"ORDER AUDIT: {len(result['audit']['violations'])} violations, "
            f"{len(result['audit']['notes'])} notes, {result['audit']['unparsed']} unparsed"
        ),
    ]
    if out is not None:
        lines.append(f"WROTE: {out}")
        try:
            display = str(out.relative_to(root))
        except ValueError:
            display = str(out)
        lines.append(
            "BOX HANDOFF (after git pull): "
            f"cp {display} <QUEUE_DIR>/tasks.json — then restart the runner. "
            "Already-run prefix tasks are skipped via .done markers, and the "
            "doctor's SYNC check is id-set based, so the reorder stays SYNC-clean."
        )
    else:
        lines.append("DRY RUN: nothing written")
    return "\n".join(lines) + "\n"


def build_chain(root: Path, tasks: list[dict], first_name: str) -> dict:
    """Return the chained sprint: every pending decision, cheapest-first.

    The first decision's closure runs first (default s7 — the
    highest-expected-gain gate keeps the head of the queue); each
    remaining pending decision's closure follows in ascending
    standalone cost (pending trains on its critical path, ties by the
    decision task's queue position). Shared feeders are placed once,
    so later segments show their *marginal* train cost. Tasks in no
    closure keep their original relative order at the tail.
    """
    cp = CriticalPath(root, tasks)
    analysis = cp.analyze()
    by_id, index_by_id, parsed_by_id = _parsed_index(cp, tasks)
    first_decision, _ = _find_decision(root, cp, analysis, first_name)

    def decision_task_for(decision: dict) -> dict:
        if decision["kind"] == "gate":
            return next(g for g in cp.gates if g["name"] == decision["name"])
        return next(c for c in cp.compares if c["name"] == decision["name"])

    def standalone_cost(decision: dict) -> int:
        return _pending_trains(sorted(decision["feeders"]), parsed_by_id)

    rest = [d for d in analysis["decisions"] if d["name"] != first_name]
    rest.sort(key=lambda d: (standalone_cost(d), index_by_id[decision_task_for(d)["task_id"]]))
    ordered_decisions = [first_decision] + rest

    placed: list[str] = []
    placed_set: set[str] = set()
    segments: list[dict] = []
    cumulative = 0
    for decision in ordered_decisions:
        task = decision_task_for(decision)
        closure = _closure_ids(cp, decision, task, parsed_by_id, index_by_id)
        segment = [i for i in closure if i not in placed_set]
        marginal = _pending_trains(segment, parsed_by_id)
        cumulative += marginal
        segments.append(
            {
                "kind": decision["kind"],
                "name": decision["name"],
                "task_id": task["task_id"],
                "ids": segment,
                "standalone_pending_trains": standalone_cost(decision),
                "marginal_pending_trains": marginal,
                "cumulative_pending_trains": cumulative,
            }
        )
        placed.extend(segment)
        placed_set.update(segment)

    remainder = [t.get("id", "?") for t in tasks if t.get("id", "?") not in placed_set]
    chain_ids = placed + remainder
    audit = _validate_order(tasks, chain_ids, by_id)
    return {
        "tasks": [by_id[i] for i in chain_ids],
        "segments": segments,
        "remainder": remainder,
        "audit": audit,
    }


def render_chain(
    result: dict,
    out: Path | None,
    root: Path,
    rate: float | None,
    resume_date: datetime | None,
) -> str:
    segments = result["segments"]
    lines = [
        (
            f"CHAIN: {len(segments)} pending decisions, first decision first, "
            "then ascending standalone cost (shared feeders placed once)"
        )
    ]
    for n, seg in enumerate(segments, 1):
        cumulative = seg["cumulative_pending_trains"]
        if rate:
            days = cumulative / rate
            when = f"cumulative {cumulative} trains, ~{days:.1f}d"
            if resume_date is not None:
                from datetime import timedelta

                finish = resume_date + timedelta(days=days)
                fit = "FITS" if finish <= SAFE_CUTOFF else "PAST CUTOFF"
                when += f" (decides ~{finish.date().isoformat()} from resume — {fit})"
        else:
            when = f"cumulative {cumulative} trains, no observed rate for a day estimate"
        lines.append(
            f"  {n}. {seg['kind']} {seg['name']}: {len(seg['ids'])} tasks, "
            f"+{seg['marginal_pending_trains']} trains (standalone "
            f"{seg['standalone_pending_trains']}); {when}"
        )
    lines.append(
        f"REMAINDER: {len(result['remainder'])} tasks in original order "
        "(in no pending decision's closure)"
    )
    lines.append(
        f"ORDER AUDIT: {len(result['audit']['violations'])} violations, "
        f"{len(result['audit']['notes'])} notes, {result['audit']['unparsed']} unparsed"
    )
    if out is not None:
        lines.append(f"WROTE: {out}")
        try:
            display = str(out.relative_to(root))
        except ValueError:
            display = str(out)
        lines.append(
            "BOX HANDOFF (after git pull): "
            f"cp {display} <QUEUE_DIR>/tasks.json — then restart the runner. "
            "Already-run tasks are skipped via .done markers, and the "
            "doctor's SYNC check is id-set based, so the reorder stays SYNC-clean."
        )
    else:
        lines.append("DRY RUN: nothing written")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a sprint-ordered queue.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--decision",
        default=None,
        help=(
            f"pending gate/compare to sprint to (default {DEFAULT_DECISION}); "
            "with --chain, the decision whose closure runs first"
        ),
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help="order every pending decision cheapest-first after the first",
    )
    parser.add_argument(
        "--resume-date",
        default=None,
        help="with --chain: mark each decision FITS/PAST CUTOFF from this YYYY-MM-DD",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default scripts/queue/tasks.sprint.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the order, write nothing")
    args = parser.parse_args(argv)

    tasks_path = args.root / "scripts" / "queue" / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise TypeError("tasks.json is not a list")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"error: cannot read {tasks_path}: {exc}", file=sys.stderr)
        return 2

    out = None if args.dry_run else (args.out or args.root / "scripts" / "queue" / DEFAULT_OUT)
    if out is not None and out.resolve() == tasks_path.resolve():
        print("error: --out must not be the live tasks.json", file=sys.stderr)
        return 2

    decision_name = args.decision or DEFAULT_DECISION
    resume_date = None
    if args.resume_date is not None:
        if not args.chain:
            print("error: --resume-date only applies with --chain", file=sys.stderr)
            return 2
        try:
            resume_date = datetime.strptime(args.resume_date, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            print(
                f"error: --resume-date must be YYYY-MM-DD, got {args.resume_date!r}",
                file=sys.stderr,
            )
            return 2
        if resume_date > SAFE_CUTOFF:
            print(
                f"error: --resume-date {args.resume_date} is after the safe cutoff "
                f"{SAFE_CUTOFF.date().isoformat()}",
                file=sys.stderr,
            )
            return 2

    try:
        if args.chain:
            result = build_chain(args.root, tasks, decision_name)
        else:
            result = build_sprint(args.root, tasks, decision_name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rate = None
    try:
        forecast = forecast_analyze(args.root, tasks, datetime.now(UTC), SAFE_CUTOFF)
        rate = forecast["rate"]
    except Exception:  # noqa: BLE001 - forecast failure must not block the sprint
        rate = None
    if out is not None:
        out.write_text(json.dumps(result["tasks"], indent=2) + "\n", encoding="utf-8")
    if args.chain:
        print(render_chain(result, out, args.root, rate, resume_date), end="")
    else:
        print(render(result, out, args.root, rate), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
