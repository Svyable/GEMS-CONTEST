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

Usage:
    uv run python scripts/queue_sprint.py [--root PATH]
        [--decision NAME] [--out PATH] [--dry-run]
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


def build_sprint(root: Path, tasks: list[dict], decision_name: str) -> dict:
    """Return the sprint queue (prefix + suffix) for one pending decision."""
    cp = CriticalPath(root, tasks)
    analysis = cp.analyze()
    by_id = {t.get("id", "?"): t for t in tasks}
    index_by_id = {t.get("id", "?"): i for i, t in enumerate(tasks)}
    parsed_by_id: dict[str, dict] = {}
    for parsed in cp.production + cp.assembles + cp.gates + cp.compares:
        parsed_by_id[parsed["task_id"]] = parsed

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

    # Seed: pending feeders + the decision task; then close under
    # "every earlier writer of a consumed file" plus pending records
    # for runs trained inside the prefix, to a fixpoint.
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

    ordered_prefix = sorted(prefix, key=lambda i: index_by_id[i])
    ordered_prefix.remove(decision_task["task_id"])
    ordered_prefix.append(decision_task["task_id"])
    suffix = [t.get("id", "?") for t in tasks if t.get("id", "?") not in prefix]
    sprint_ids = ordered_prefix + suffix

    # Validation: same task set (a permutation, nothing lost or doubled)
    # and the repo's own order audit clean on the sprint order.
    if Counter(sprint_ids) != Counter(t.get("id", "?") for t in tasks):
        raise ValueError("sprint is not a permutation of tasks.json (ids lost or doubled)")
    sprint_tasks = [by_id[i] for i in sprint_ids]
    audit = OrderAudit(sprint_tasks).audit()
    violations = audit["violations"]
    if violations:
        detail = "; ".join(
            f"{v['consumer']['task_id']} after {v['producer']['task_id']} ({v['input']})"
            for v in violations
        )
        raise ValueError(f"sprint order fails the order audit: {detail}")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a sprint-ordered queue.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--decision",
        default=DEFAULT_DECISION,
        help=f"pending gate/compare to sprint to (default {DEFAULT_DECISION})",
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

    try:
        result = build_sprint(args.root, tasks, args.decision)
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
    print(render(result, out, args.root, rate), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
