#!/usr/bin/env python3
"""Inventory queued production tasks against the committed results tree.

The box runs the queue in scripts/queue/tasks.json and commits run outputs
to results/runs/<run>/ (via scripts/queue/score_view.sh + record.sh) and
gate verdicts to results/gates/<NAME>.json. scripts/gate_status.py already
answers which queued *decisions* are decided; nothing answered, in one
place, which queued *production* tasks (train/score/record/view) have
actually landed in the committed tree and which are still outstanding —
the daily run hand-grepped tasks.json against results/runs/.

This script prints that inventory: one row per queued task, the furthest
pipeline stage with committed evidence (RECORDED > SCORED > TRAINED >
PENDING), a per-run rollup, and a summary. Gate/compare rows reuse the
gate_status.py checks (it remains the authoritative gate detail); parked
tasks in scripts/queue/tasks.parked.json are counted, not inventoried.

Path mapping: queue commands name box-relative dirs (runs/...,
runs/gate/...); when the box commits them they land under results/
(results/runs/..., results/runs/gate/...), so the inventory checks the
results/-prefixed path first and falls back to the bare queue path.
Score files also appear at the run top level instead of scores/ for the
"-full" runs (see the step-22 audit's RELOCATED findings), so both
locations count as SCORED evidence.

Usage:
    uv run python scripts/queue_status.py [--root PATH]

Exit code is 0 whenever the inventory could be produced, even when every
task is still pending; a missing or malformed tasks.json is a hard error
(exit 1) because without the queue the inventory is meaningless.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VIEWS = ("spatial", "fault", "trace")
KNOWN_VERDICTS = ("ACCEPT", "REJECT", "INCONCLUSIVE")
# Furthest-stage ordering: later stages imply the earlier ones ran.
STAGES = ("PENDING", "TRAINED", "SCORED", "RECORDED")


def _tokens(cmd: str) -> list[str]:
    # Only the leading tokens matter (script, dirs, config); the trailing
    # quoted hypotheses may contain spaces and escaped quotes, so a plain
    # split is both sufficient and safer than shlex here.
    return cmd.split()


def _find_script_token(tokens: list[str], basename: str) -> int | None:
    for i, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1] == basename:
            return i
    return None


def _parse_production_task(task: dict) -> dict | None:
    cmd = task.get("cmd", "")
    tokens = _tokens(cmd)
    task_id = task.get("id", "?")
    # View tasks chain a post-processing step and a score step with &&.
    for basename, kind in (
        ("endpoint_view.sh", "view"),
        ("curved_view.sh", "view"),
    ):
        idx = _find_script_token(tokens, basename)
        if idx is not None and len(tokens) >= idx + 4:
            score_idx = _find_script_token(tokens, "score_view.sh")
            gate = tokens[score_idx + 4] if score_idx is not None else None
            return {
                "kind": kind,
                "task_id": task_id,
                "name": basename.replace(".sh", ""),
                "run": tokens[idx + 2],
                "scheme": tokens[idx + 3],
                "gate_dir": gate,
            }
    idx = _find_script_token(tokens, "train_folds.sh")
    if idx is not None and len(tokens) >= idx + 6:
        return {
            "kind": "train",
            "task_id": task_id,
            "name": tokens[idx + 1],
            "run": tokens[idx + 1],
            "scheme": tokens[idx + 3],
            "gate_dir": None,
        }
    idx = _find_script_token(tokens, "score_view.sh")
    if idx is not None and len(tokens) >= idx + 5:
        return {
            "kind": "score",
            "task_id": task_id,
            "name": tokens[idx + 1],
            "run": tokens[idx + 1],
            "scheme": tokens[idx + 2],
            "gate_dir": tokens[idx + 4],
        }
    idx = _find_script_token(tokens, "record.sh")
    if idx is not None and len(tokens) >= idx + 3:
        return {
            "kind": "record",
            "task_id": task_id,
            "name": tokens[idx + 1],
            "run": tokens[idx + 2],
            "scheme": None,
            "gate_dir": None,
        }
    return None


def _parse_assemble_task(task: dict) -> dict | None:
    cmd = task.get("cmd", "")
    if "mkdir" not in cmd or "cp" not in cmd:
        return None
    # The command chains segments with && and ; — isolate the cp segment.
    segments = [seg for seg in cmd.replace(";", "&&").split("&&")]
    cp_segment = next((seg for seg in segments if seg.strip().split()[:1] == ["cp"]), None)
    if cp_segment is None:
        return None
    parts = [t for t in cp_segment.strip().split()[1:] if not t.startswith("-") and ">" not in t]
    if len(parts) < 2:
        return None
    sources, dest = parts[:-1], parts[-1].rstrip("/")
    return {
        "kind": "assemble",
        "task_id": task.get("id", "?"),
        "name": dest,
        "files": [Path(s).name for s in sources],
        "dest": dest,
    }


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
            match = (
                re.search(r"^CMP=([^\s#]+)", script.read_text(encoding="utf-8"), re.MULTILINE)
                if script.exists()
                else None
            )
            return {
                "kind": "compare",
                "task_id": task.get("id", "?"),
                "name": basename.replace(".sh", ""),
                "compare_dir": match.group(1) if match else None,
            }
    return None


def _candidate_dirs(root: Path, queue_dir: str) -> list[Path]:
    """Where a box queue dir may live once committed, results/ first."""
    return [root / "results" / queue_dir, root / queue_dir]


def _exists_in(root: Path, queue_dir: str, filename: str) -> bool:
    return any((d / filename).exists() for d in _candidate_dirs(root, queue_dir))


def _fold_count(root: Path, run: str) -> int:
    return sum(1 for k in range(5) if _exists_in(root, run, f"fold-{k}.json"))


def _score_present(root: Path, prod: dict) -> str | None:
    """Return the scored view name if a score file is committed, else None."""
    run, scheme = prod["run"], prod["scheme"]
    # Record tasks carry no scheme: any view score counts as SCORED.
    schemes = [scheme] if scheme is not None else list(VIEWS)
    for scheme_name in schemes:
        if _exists_in(root, run, f"scores/{scheme_name}.json"):
            return scheme_name
        # The "-full" runs were committed with the score at the run top level.
        if _exists_in(root, run, f"{scheme_name}.json"):
            return scheme_name
        gate_dir = prod.get("gate_dir")
        if gate_dir and _exists_in(root, gate_dir, f"{scheme_name}.json"):
            return scheme_name
    return None


def _production_status(root: Path, prod: dict) -> tuple[str, str]:
    run = prod["run"]
    if _exists_in(root, run, "manifest.json"):
        return "RECORDED", "manifest.json present"
    scored = _score_present(root, prod)
    if scored:
        return "SCORED", f"score {scored}.json present, no manifest"
    folds = _fold_count(root, run)
    if folds:
        return "TRAINED", f"{folds}/5 fold files present, no score"
    return "PENDING", "no committed evidence"


def _assemble_status(root: Path, task: dict) -> tuple[str, str]:
    present = [f for f in task["files"] if _exists_in(root, task["dest"], f)]
    missing = [f for f in task["files"] if f not in present]
    if not missing:
        return "DONE", f"assembled: {', '.join(present)}"
    if present:
        return "PARTIAL", f"present: {', '.join(present)}; missing: {', '.join(missing)}"
    return "PENDING", f"no assembled files yet; missing: {', '.join(missing)}"


def _gate_status(root: Path, gate: dict) -> tuple[str, str]:
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
    gate_dir = _candidate_dirs(root, gate["candidate_dir"])[0]
    present = [v for v in VIEWS if (gate_dir / f"{v}.json").exists()]
    missing = [v for v in VIEWS if v not in present]
    if present:
        return (
            "PENDING",
            f"candidate inputs present: {', '.join(present)}; missing: {', '.join(missing)}",
        )
    return "PENDING", "no candidate gate inputs yet"


def _compare_status(root: Path, compare: dict) -> tuple[str, str]:
    if compare["compare_dir"] is None:
        return "UNKNOWN", "could not read CMP= from compare script"
    compare_dir = _candidate_dirs(root, compare["compare_dir"])[0]
    log = compare_dir / "compare.log"
    if log.exists():
        return "RAN", str(log.relative_to(root))
    return "PENDING", f"no compare.log in {compare['compare_dir']} yet"


def _referenced_runs(tasks: list[dict]) -> set[str]:
    runs: set[str] = set()
    for task in tasks:
        prod = _parse_production_task(task)
        if prod is not None:
            runs.add(prod["run"])
            if prod.get("gate_dir"):
                runs.add(prod["gate_dir"])
        asm = _parse_assemble_task(task)
        if asm is not None:
            runs.add(asm["dest"])
        gate = _parse_gate_task(task)
        if gate is not None:
            runs.add(gate["incumbent_dir"])
            runs.add(gate["candidate_dir"])
        cmp_task = _parse_compare_task(task, Path())
        if cmp_task is not None and cmp_task["compare_dir"]:
            runs.add(cmp_task["compare_dir"])
    return runs


def inventory(root: Path) -> dict:
    queue_dir = root / "scripts" / "queue"
    tasks_path = queue_dir / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read queue {tasks_path}: {exc}") from exc
    if not isinstance(tasks, list):
        raise SystemExit(f"error: queue {tasks_path} is not a task list")

    parked: list[str] = []
    parked_path = queue_dir / "tasks.parked.json"
    if parked_path.exists():
        try:
            parked_tasks = json.loads(parked_path.read_text(encoding="utf-8"))
            parked = [t.get("id", "?") for t in parked_tasks if isinstance(t, dict)]
        except (OSError, json.JSONDecodeError):
            parked = ["<unreadable>"]

    rows: list[dict] = []
    queued_gate_names: set[str] = set()
    for task in tasks:
        prod = _parse_production_task(task)
        if prod is not None:
            status, detail = _production_status(root, prod)
            rows.append({**prod, "status": status, "detail": detail})
            continue
        asm = _parse_assemble_task(task)
        if asm is not None:
            status, detail = _assemble_status(root, asm)
            rows.append({**asm, "status": status, "detail": detail})
            continue
        gate = _parse_gate_task(task)
        if gate is not None:
            status, detail = _gate_status(root, gate)
            queued_gate_names.add(gate["name"])
            rows.append({**gate, "status": status, "detail": detail})
            continue
        compare = _parse_compare_task(task, queue_dir)
        if compare is not None:
            status, detail = _compare_status(root, compare)
            rows.append({**compare, "status": status, "detail": detail})
            continue
        rows.append(
            {
                "kind": "housekeeping",
                "task_id": task.get("id", "?"),
                "name": _tokens(task.get("cmd", ""))[0][:40] if task.get("cmd") else "?",
                "status": "HOUSEKEEPING",
                "detail": "box-side setup, no committed output expected",
            }
        )

    # Per-run rollup over production tasks (train/score/record/view).
    rollup: dict[str, dict] = {}
    for row in rows:
        if row["kind"] not in ("train", "score", "record", "view"):
            continue
        run = row["run"]
        entry = rollup.setdefault(run, {"stage": "PENDING", "kinds": set()})
        entry["kinds"].add(row["kind"])
        if STAGES.index(row["status"]) > STAGES.index(entry["stage"]):
            entry["stage"] = row["status"]

    gates_dir = root / "results" / "gates"
    orphans = (
        sorted(p.stem for p in gates_dir.glob("*.json") if p.stem not in queued_gate_names)
        if gates_dir.exists()
        else []
    )
    referenced = _referenced_runs(tasks)
    committed_runs = root / "results" / "runs"
    orphan_runs: list[str] = []
    if committed_runs.exists():
        for child in sorted(committed_runs.iterdir()):
            if not child.is_dir() or child.name == "gate":
                continue
            if f"runs/{child.name}" not in referenced:
                orphan_runs.append(child.name)
        gate_root = committed_runs / "gate"
        if gate_root.exists():
            for child in sorted(gate_root.iterdir()):
                if child.is_dir() and f"runs/gate/{child.name}" not in referenced:
                    orphan_runs.append(f"gate/{child.name}")
    return {
        "rows": rows,
        "rollup": rollup,
        "orphans": orphans,
        "orphan_runs": orphan_runs,
        "parked": parked,
    }


def render(inv: dict) -> str:
    lines = ["TASK | KIND | NAME | STATUS | DETAIL"]
    for row in inv["rows"]:
        lines.append(
            f"{row['task_id']} | {row['kind']} | {row['name']} | {row['status']} | {row['detail']}"
        )
    lines.append("")
    lines.append("RUN | STAGE | TASKS | EVIDENCE")
    for run in sorted(inv["rollup"]):
        entry = inv["rollup"][run]
        kinds = "+".join(sorted(entry["kinds"]))
        detail = {
            "RECORDED": "manifest.json present",
            "SCORED": "score present, no manifest",
            "TRAINED": "fold files present, no score",
            "PENDING": "no committed evidence",
        }[entry["stage"]]
        lines.append(f"{run} | {entry['stage']} | {kinds} | {detail}")
    lines.append("")
    prod = [r for r in inv["rows"] if r["kind"] in ("train", "score", "record", "view")]
    counts = {s: sum(1 for r in prod if r["status"] == s) for s in STAGES}
    gates = [r for r in inv["rows"] if r["kind"] == "gate"]
    compares = [r for r in inv["rows"] if r["kind"] == "compare"]
    decided = sum(1 for r in gates if r["status"] == "DECIDED")
    ran = sum(1 for r in compares if r["status"] == "RAN")
    rollup = inv["rollup"]
    rcounts = {s: sum(1 for e in rollup.values() if e["stage"] == s) for s in STAGES}
    lines.append(
        f"Summary: {len(inv['rows'])} queued tasks ({len(prod)} production, "
        f"{len(gates)} gates, {len(compares)} compares); "
        f"production {counts['RECORDED']} recorded, {counts['SCORED']} scored, "
        f"{counts['TRAINED']} trained, {counts['PENDING']} pending; "
        f"{len(rollup)} queued runs — {rcounts['RECORDED']} recorded, "
        f"{rcounts['SCORED']} scored, {rcounts['TRAINED']} trained, "
        f"{rcounts['PENDING']} pending."
    )
    lines.append(
        f"Summary: {len(gates)} gate decisions queued, {decided} decided, "
        f"{len(gates) - decided} pending; {len(compares)} manual compares, "
        f"{ran} ran, {len(compares) - ran} pending; "
        f"{len(inv['orphans'])} orphan gate JSONs; "
        f"{len(inv['parked'])} parked tasks."
    )
    for orphan in inv["orphans"]:
        lines.append(f"ORPHAN: results/gates/{orphan}.json has no queued gate task")
    for orphan in inv["orphan_runs"]:
        lines.append(f"ORPHAN: results/runs/{orphan} has no queued task referencing it")
    if inv["parked"]:
        lines.append(f"PARKED: {', '.join(inv['parked'])}")
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
