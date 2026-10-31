#!/usr/bin/env python3
"""Insert new tasks into scripts/queue/tasks.json before a named anchor task.

The runner (scripts/queue/runner.py) re-reads tasks.json before every task
and runs the first task with no state/<id>.done marker, so list order IS
execution order among pending tasks. Appending a task that must precede an
already-queued consumer is therefore wrong: it lands after the consumer
and, for a copy-style assemble task, after the copy it was meant to fix
(step 25: schema-2 re-scores of the s1/s2 full runs must run BEFORE
s5b-assemble-ref-b16, or the assemble copies schema-1 files into ref-b16
and the later s7-assemble's `cp -n` never repairs them).

This tool is the only supported way to insert mid-queue. It validates
everything before writing anything: the anchor must exist exactly once,
every new id must be a non-empty string unique across both the new tasks
and the existing queue, and every new task must carry a non-empty cmd.
On any failure it exits 2 and leaves tasks.json untouched.

Usage:
    python scripts/queue_insert.py --anchor ANCHOR_ID --insert-file NEW.json
    python scripts/queue_insert.py --anchor ANCHOR_ID --insert-file NEW.json --dry-run

NEW.json is a JSON list of task objects, each with "id" and "cmd" plus the
usual optional "step" and "note" fields. --root defaults to the repo root.
Exit 0 on success (or on a clean dry run); exit 2 on any validation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_tasks(path: Path) -> list[dict]:
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read {path}: {exc}") from exc
    if not isinstance(tasks, list):
        raise SystemExit(f"error: queue {path} is not a task list")
    return tasks


def _validate_new(new: object, existing_ids: set[str]) -> list[dict]:
    if not isinstance(new, list) or not new:
        raise SystemExit("error: insert file must hold a non-empty list of tasks")
    seen: set[str] = set()
    for task in new:
        if not isinstance(task, dict):
            raise SystemExit(f"error: inserted task is not an object: {task!r}")
        task_id = task.get("id")
        cmd = task.get("cmd")
        if not isinstance(task_id, str) or not task_id:
            raise SystemExit(f"error: inserted task missing a string id: {task!r}")
        if not isinstance(cmd, str) or not cmd:
            raise SystemExit(f"error: inserted task {task_id!r} missing a cmd")
        if task_id in seen:
            raise SystemExit(f"error: duplicate id among inserted tasks: {task_id!r}")
        if task_id in existing_ids:
            raise SystemExit(f"error: id already in queue: {task_id!r}")
        seen.add(task_id)
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--anchor", required=True, help="existing task id to insert before")
    parser.add_argument("--insert-file", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.root or Path(__file__).resolve().parents[1]
    tasks_path = root / "scripts" / "queue" / "tasks.json"
    tasks = _load_tasks(tasks_path)
    existing_ids = {t.get("id") for t in tasks if isinstance(t, dict)}

    try:
        new_raw = json.loads(args.insert_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {args.insert_file}: {exc}", file=sys.stderr)
        return 2
    try:
        new = _validate_new(new_raw, existing_ids)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    anchors = [i for i, t in enumerate(tasks) if t.get("id") == args.anchor]
    if len(anchors) != 1:
        print(
            f"error: anchor {args.anchor!r} found {len(anchors)} times, expected exactly 1",
            file=sys.stderr,
        )
        return 2
    index = anchors[0]

    inserted_ids = [t["id"] for t in new]
    result = tasks[:index] + new + tasks[index:]
    # Post-condition: ids still unique, anchor immediately follows the block.
    result_ids = [t.get("id") for t in result]
    assert len(result_ids) == len(set(result_ids)), "insert broke id uniqueness"
    assert result[index + len(new)]["id"] == args.anchor, "anchor moved"

    if args.dry_run:
        print(
            f"dry run: would insert {', '.join(inserted_ids)} at index {index}, "
            f"before {args.anchor!r} (queue {len(tasks)} -> {len(result)} tasks)"
        )
        return 0
    tasks_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"inserted {', '.join(inserted_ids)} at index {index}, "
        f"before {args.anchor!r} (queue {len(tasks)} -> {len(result)} tasks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
