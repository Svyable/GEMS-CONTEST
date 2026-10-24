#!/usr/bin/env python3
"""Inventory queued gate/compare decisions against committed results.

The box runs the queue in scripts/queue/tasks.json and commits gate verdicts
to results/gates/<NAME>.json (via scripts/queue/gate.sh) and manual-compare
logs to runs/compare/<name>/compare.log (via compare_seed.sh /
compare_schedule.sh). Nothing in this repo previously answered, in one place,
"which queued decisions are already decided and which are still waiting" —
the daily run hand-grepped tasks.json against results/gates/.

This script prints that inventory: one row per queued gate/compare task, the
status of its committed output, and — for pending gates — which per-view
score files already sit in the candidate gate dir (partial progress toward
the gate). Gate JSONs in results/gates/ with no queued gate task are listed
as orphans, since they are decisions nothing in the queue asked for.

Path mapping: queue commands name box-relative dirs (runs/gate/...,
runs/compare/...); when the box commits them they land under results/
(results/runs/gate/..., results/gates/...), so the inventory checks the
results/-prefixed path first and falls back to the bare queue path.

Usage:
    uv run python scripts/gate_status.py [--root PATH]

Exit code is 0 whenever the inventory could be produced, even when every
decision is still pending; a missing or malformed tasks.json is a hard
error (exit 2) because without the queue the inventory is meaningless.
A gate JSON that exists but does not parse, or carries an unknown verdict,
is reported BROKEN, not silently skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VIEWS = ("spatial", "fault", "trace")
KNOWN_VERDICTS = ("ACCEPT", "REJECT", "INCONCLUSIVE")
CMP_RE = re.compile(r"^CMP=([^\s#]+)", re.MULTILINE)


def _tokens(cmd: str) -> list[str]:
    # Only the leading tokens matter (script, NAME, dirs, config); the
    # trailing quoted hypothesis may contain spaces and escaped quotes, so
    # a plain split is both sufficient and safer than shlex here.
    return cmd.split()


def _find_script_token(tokens: list[str], basename: str) -> int | None:
    for i, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1] == basename:
            return i
    return None


def _parse_gate_task(task: dict) -> dict | None:
    tokens = _tokens(task.get("cmd", ""))
    idx = _find_script_token(tokens, "gate.sh")
    if idx is None or len(tokens) < idx + 4:
        return None
    return {
        "kind": "gate",
        "task_id": task.get("id", "?"),
        "name": tokens[idx + 1],
        "incumbent_dir": tokens[idx + 2],
        "candidate_dir": tokens[idx + 3],
    }


def _parse_compare_task(task: dict, queue_dir: Path) -> dict | None:
    tokens = _tokens(task.get("cmd", ""))
    for basename in ("compare_seed.sh", "compare_schedule.sh"):
        idx = _find_script_token(tokens, basename)
        if idx is not None:
            script = queue_dir / basename
            match = CMP_RE.search(script.read_text(encoding="utf-8")) if script.exists() else None
            return {
                "kind": "compare",
                "task_id": task.get("id", "?"),
                "name": basename.replace(".sh", ""),
                "compare_dir": match.group(1) if match else None,
            }
    return None


def _committed_path(root: Path, queue_dir: str) -> Path:
    """Where a box queue dir lands once committed: results/<dir> if present,
    else the bare queue path (for local/uncommitted layouts)."""
    prefixed = root / "results" / queue_dir
    if prefixed.exists():
        return prefixed
    return root / queue_dir


def _gate_status(root: Path, gate: dict) -> tuple[str, str]:
    """Return (status, detail) for one queued gate task."""
    path = root / "results" / "gates" / f"{gate['name']}.json"
    if path.exists():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return "BROKEN", f"unparseable {path.name}: {exc}"
        verdict = report.get("verdict") if isinstance(report, dict) else None
        if verdict in KNOWN_VERDICTS:
            return "DECIDED", verdict
        return "BROKEN", f"{path.name} has unknown verdict {verdict!r}"
    gate_dir = _committed_path(root, gate["candidate_dir"])
    present = [v for v in VIEWS if (gate_dir / f"{v}.json").exists()]
    missing = [v for v in VIEWS if v not in present]
    if present:
        return "PENDING", f"candidate inputs present: {', '.join(present)}; missing: {', '.join(missing)}"
    return "PENDING", "no candidate gate inputs yet"


def _compare_status(root: Path, compare: dict) -> tuple[str, str]:
    if compare["compare_dir"] is None:
        return "UNKNOWN", "could not read CMP= from compare script"
    compare_dir = _committed_path(root, compare["compare_dir"])
    log = compare_dir / "compare.log"
    try:
        shown = str(log.relative_to(root))
    except ValueError:
        shown = str(log)
    if log.exists():
        return "RAN", shown
    return "PENDING", f"no compare.log in {shown.rsplit('/', 1)[0]} yet"


def inventory(root: Path) -> dict:
    tasks_path = root / "scripts" / "queue" / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read queue {tasks_path}: {exc}") from exc
    if not isinstance(tasks, list):
        raise SystemExit(f"error: queue {tasks_path} is not a task list")

    rows: list[dict] = []
    queued_gate_names: set[str] = set()
    for task in tasks:
        gate = _parse_gate_task(task)
        if gate is not None:
            status, detail = _gate_status(root, gate)
            queued_gate_names.add(gate["name"])
            rows.append({**gate, "status": status, "detail": detail})
            continue
        compare = _parse_compare_task(task, root / "scripts" / "queue")
        if compare is not None:
            status, detail = _compare_status(root, compare)
            rows.append({**compare, "status": status, "detail": detail})

    gates_dir = root / "results" / "gates"
    orphans = sorted(
        p.stem
        for p in gates_dir.glob("*.json")
        if p.stem not in queued_gate_names
    ) if gates_dir.exists() else []
    return {"rows": rows, "orphans": orphans}


def render(inv: dict) -> str:
    lines = ["TASK | KIND | NAME | STATUS | DETAIL"]
    for row in inv["rows"]:
        lines.append(
            f"{row['task_id']} | {row['kind']} | {row['name']} | "
            f"{row['status']} | {row['detail']}"
        )
    gates = [r for r in inv["rows"] if r["kind"] == "gate"]
    compares = [r for r in inv["rows"] if r["kind"] == "compare"]
    decided = sum(1 for r in gates if r["status"] == "DECIDED")
    ran = sum(1 for r in compares if r["status"] == "RAN")
    lines.append("")
    lines.append(
        f"Summary: {len(gates)} gate decisions queued, {decided} decided, "
        f"{len(gates) - decided} pending; {len(compares)} manual compares, "
        f"{ran} ran, {len(compares) - ran} pending; "
        f"{len(inv['orphans'])} orphan gate JSONs."
    )
    for orphan in inv["orphans"]:
        lines.append(f"ORPHAN: results/gates/{orphan}.json has no queued gate task")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    sys.stdout.write(render(inventory(args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
