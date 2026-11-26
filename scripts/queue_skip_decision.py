#!/usr/bin/env python3
"""Skip one pending decision in a live queue so the chain can continue.

The runner stops the whole queue on the first failed task
(``state/<id>.failed``) and the doctor's fix is "fix the cause, clear
the marker, restart" — which retries the same task. That is right
when the cause is fixable, but wrong when it is not: after the
2026-11-28 resumption the chained sprint (step 51) has ~5.9 days to
land 7 of 11 decisions before the 2026-12-03 22:00 UTC safe cutoff,
and one permanently failing train in the s7 segment would stall the
chain at decision 1 with the other 10 never attempted.

This tool removes one pending decision (a gate or compare) from a
queue file: the decision task plus every task in its positional
closure (``queue_sprint._closure_ids``) that no other pending
decision's closure needs. Shared feeders stay — skipping the s7 loss
gate keeps the s26 re-scores and the ref-b16 assembles that every
other b16 gate consumes, and skipping one b32 gate keeps the
ref-trace-b32 train the other b32 gate shares. Relative order of the
remaining tasks is preserved, so a chained sprint stays chained.

Guards (all refuse with exit 2, changing nothing):

- unknown or already-decided decision (same errors as queue_sprint);
- the decision task already has a ``.done`` marker (decided on the
  box, whatever the committed tree says);
- the removal would create an ORDER-VIOLATION (a remaining consumer
  ahead of its earliest producer);
- the removal would create a new NO-PRODUCER note — a remaining task
  outside every pending closure consumes a file only a removed task
  writes. ``--force`` overrides this one guard (the notes are printed
  either way): the consumers can be historical tasks whose outputs
  are already committed (e.g. skipping the curved-continuation gate
  orphans the decided ENDPOINT-EXT-02 gate's trace input on paper),
  but the tool does not guess that — the operator decides.

With ``--state-dir`` (default: ``<queue-dir>/state`` when it exists),
tasks that already have a ``.done`` marker are kept in the queue
(they cost seconds to skip and rewriting history helps no one), and
on ``--write`` the ``.failed``/``.started`` markers of removed tasks
are deleted so the doctor does not report a stale STOPPED. The live
queue is backed up to ``<queue>.bak.<timestamp>`` before writing.
Default is a dry run: report only, write nothing.

After a skip the doctor's SYNC check reports DIVERGED naming exactly
the removed ids — that is expected, not damage: the queue-dir file no
longer matches the repo checkout by design. Restart the runner to
continue the chain at the next decision.

Usage:
    uv run python scripts/queue_skip_decision.py --decision NAME
        [--root PATH] [--queue PATH] [--state-dir PATH] [--force]
        [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from queue_critical_path import CriticalPath
from queue_order_audit import OrderAudit
from queue_sprint import _closure_ids, _find_decision, _parsed_index, _pending_trains


def _note_keys(audit: dict) -> set[tuple[str, str]]:
    return {(n["consumer"]["task_id"], n["input"]) for n in audit["notes"]}


def _violation_keys(audit: dict) -> set[tuple[str, str, str]]:
    return {
        (v["consumer"]["task_id"], v["producer"]["task_id"], v["input"])
        for v in audit["violations"]
    }


def build_skip(
    root: Path,
    tasks: list[dict],
    decision_name: str,
    done_ids: set[str],
) -> dict:
    """Compute the skip: removable / kept-shared / kept-done task ids."""
    cp = CriticalPath(root, tasks)
    analysis = cp.analyze()
    _by_id, index_by_id, parsed_by_id = _parsed_index(cp, tasks)
    decision, decision_task = _find_decision(root, cp, analysis, decision_name)

    def task_for(d: dict) -> dict:
        if d["kind"] == "gate":
            return next(g for g in cp.gates if g["name"] == d["name"])
        return next(c for c in cp.compares if c["name"] == d["name"])

    target_closure = set(_closure_ids(cp, decision, decision_task, parsed_by_id, index_by_id))
    shared_with: dict[str, list[str]] = {}
    for other in analysis["decisions"]:
        if other["name"] == decision_name:
            continue
        other_closure = set(_closure_ids(cp, other, task_for(other), parsed_by_id, index_by_id))
        for task_id in target_closure & other_closure:
            shared_with.setdefault(task_id, []).append(other["name"])

    kept_shared = sorted(shared_with, key=lambda i: index_by_id[i])
    removable_all = target_closure - set(kept_shared)
    kept_done = sorted((i for i in removable_all if i in done_ids), key=lambda i: index_by_id[i])
    removable = sorted(
        (i for i in removable_all if i not in done_ids), key=lambda i: index_by_id[i]
    )
    removable_set = set(removable)
    remaining = [t for t in tasks if t.get("id") not in removable_set]

    audit_before = OrderAudit(tasks).audit()
    audit_after = OrderAudit(remaining).audit()
    new_violations = _violation_keys(audit_after) - _violation_keys(audit_before)
    new_notes = _note_keys(audit_after) - _note_keys(audit_before)

    return {
        "decision": decision,
        "decision_task_id": decision_task["task_id"],
        "closure": sorted(target_closure, key=lambda i: index_by_id[i]),
        "removable": removable,
        "kept_shared": kept_shared,
        "shared_with": shared_with,
        "kept_done": kept_done,
        "remaining": remaining,
        "trains_saved": _pending_trains(removable, parsed_by_id),
        "parsed_by_id": parsed_by_id,
        "new_violations": sorted(new_violations),
        "new_notes": sorted(new_notes),
        "audit_after": audit_after,
        "pending_decisions": [d["name"] for d in analysis["decisions"]],
    }


def render(result: dict, queue_path: Path, wrote: bool, backup: Path | None) -> str:
    parsed_by_id = result["parsed_by_id"]
    lines = [
        (
            f"SKIP {result['decision']['kind']} {result['decision']['name']}: "
            f"closure {len(result['closure'])} tasks, "
            f"remove {len(result['removable'])}, "
            f"keep {len(result['kept_shared'])} shared, "
            f"keep {len(result['kept_done'])} already-done; "
            f"{result['trains_saved']} pending train(s) saved"
        )
    ]
    for task_id in result["removable"]:
        kind = parsed_by_id.get(task_id, {}).get("kind", "?")
        status = parsed_by_id.get(task_id, {}).get("status", "?")
        lines.append(f"  REMOVE {task_id} ({kind}, {status})")
    for task_id in result["kept_shared"]:
        who = ", ".join(result["shared_with"][task_id])
        lines.append(f"  KEEP-SHARED {task_id} (also needed by {who})")
    for task_id in result["kept_done"]:
        lines.append(f"  KEEP-DONE {task_id} (.done marker exists)")
    lines.append(
        f"QUEUE: {len(result['remaining'])} tasks remain, "
        f"{len(result['pending_decisions']) - 1} pending decisions remain"
    )
    lines.append(
        f"ORDER AUDIT after skip: {len(result['audit_after']['violations'])} violations, "
        f"{len(result['audit_after']['notes'])} notes "
        f"({len(result['new_notes'])} new)"
    )
    for consumer, inp in result["new_notes"]:
        lines.append(f"  NEW NO-PRODUCER: {consumer} consumes {inp}")
    if wrote:
        lines.append(f"WROTE: {queue_path} (backup: {backup})")
        lines.append(
            "NOTE: the doctor's SYNC check will report DIVERGED naming the "
            "removed ids — expected after a skip. Restart the runner to "
            "continue at the next decision."
        )
    else:
        lines.append("DRY RUN: nothing written (re-run with --write to apply)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Skip one pending decision in a queue.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--queue",
        type=Path,
        default=None,
        help="queue file (default <root>/scripts/queue/tasks.json; on the box, <QUEUE_DIR>/tasks.json)",
    )
    parser.add_argument("--decision", required=True, help="pending gate/compare name to skip")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="runner state dir (default <queue-dir>/state when it exists)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow the skip even if it creates new NO-PRODUCER notes (printed either way)",
    )
    parser.add_argument("--write", action="store_true", help="apply the skip (default: dry run)")
    args = parser.parse_args(argv)

    queue_path = args.queue or args.root / "scripts" / "queue" / "tasks.json"
    try:
        tasks = json.loads(queue_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise TypeError("queue file is not a list")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"error: cannot read {queue_path}: {exc}", file=sys.stderr)
        return 2

    state_dir = args.state_dir
    if state_dir is None and (queue_path.parent / "state").is_dir():
        state_dir = queue_path.parent / "state"
    done_ids: set[str] = set()
    if state_dir is not None:
        done_ids = {p.name[: -len(".done")] for p in state_dir.glob("*.done")}

    try:
        result = build_skip(args.root, tasks, args.decision, done_ids)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if result["decision_task_id"] in done_ids:
        print(
            f"error: decision {args.decision} already decided on the box "
            f"({result['decision_task_id']} has a .done marker)",
            file=sys.stderr,
        )
        return 2
    if result["new_violations"]:
        detail = "; ".join(f"{c} after {p} ({i})" for c, p, i in result["new_violations"])
        print(f"error: skip would create ORDER-VIOLATIONs: {detail}", file=sys.stderr)
        return 2
    if result["new_notes"] and not args.force:
        detail = "; ".join(f"{c} consumes {i}" for c, i in result["new_notes"])
        print(
            f"error: skip would orphan remaining tasks (new NO-PRODUCER notes): {detail}; "
            "re-run with --force if those consumers are historical/already-produced",
            file=sys.stderr,
        )
        return 2

    backup = None
    if args.write:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = queue_path.with_name(f"{queue_path.name}.bak.{stamp}")
        backup.write_text(queue_path.read_text(encoding="utf-8"), encoding="utf-8")
        queue_path.write_text(json.dumps(result["remaining"], indent=2) + "\n", encoding="utf-8")
        if state_dir is not None:
            for task_id in result["removable"]:
                for suffix in ("failed", "started"):
                    marker = state_dir / f"{task_id}.{suffix}"
                    if marker.exists():
                        marker.unlink()
    print(render(result, queue_path, args.write, backup), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
