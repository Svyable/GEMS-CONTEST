#!/usr/bin/env python3
"""Audit whether pending gate inputs are actually usable, not just present.

scripts/queue_critical_path.py ranks which queued tasks unblock which
decisions, but it treats a gate input as satisfied once the file exists.
That is not sufficient: scripts/verify_candidate.py parses every gate-dir
JSON through gems.verification._parse_view_result, which enforces the
schema-2 protocol (schema_version == 2, stitched aggregate_components on
the spatial view, digest formats, fold/macro consistency). A schema-1
score file in a gate dir fails at parse time, so a gate whose inputs are
merely copied — not re-scored — still cannot run.

This script closes that gap. For each pending gate (queued gate.sh task
with no committed verdict in results/gates/), it classifies every
incumbent/candidate view input as:

  VALID    — parses under the real gate parser;
  INVALID  — present but rejected, with the parser's exact reason
             (schema-1 files land here: re-score, do not copy);
  MISSING  — absent, with the queued producer task ids and any
             committed same-view score evidence elsewhere in
             results/runs/ (with that evidence's own validity, so a
             copy-vs-rescore decision is visible).

Path mapping: queue commands name box-relative dirs (runs/gate/...);
when the box commits them they land under results/ (results/runs/
gate/...), so the audit checks the results/-prefixed path first and
falls back to the bare queue path.

Usage:
    uv run python scripts/gate_readiness.py [--root PATH]

Exit code is 0 whenever the audit could be produced, even when every
gate is blocked; a missing or malformed tasks.json is a hard error
(exit 2) because without the queue the audit is meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems.verification import _parse_view_result

VIEWS = ("spatial", "fault", "trace")
KNOWN_VERDICTS = ("ACCEPT", "REJECT", "INCONCLUSIVE")


def _tokens(cmd: str) -> list[str]:
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
        "task_id": task.get("id", "?"),
        "name": tokens[idx + 1],
        "incumbent_dir": tokens[idx + 2],
        "candidate_dir": tokens[idx + 3],
    }


def _parse_score_task(task: dict) -> dict | None:
    """score_view.sh RUN_DIR SCHEME FOLD_MAP GATE_DIR -> (gate dir, view)."""
    tokens = _tokens(task.get("cmd", ""))
    idx = _find_script_token(tokens, "score_view.sh")
    if idx is None or len(tokens) < idx + 5:
        return None
    return {"task_id": task.get("id", "?"), "gate_dir": tokens[idx + 4], "view": tokens[idx + 2]}


def _parse_assemble_task(task: dict) -> dict | None:
    """mkdir ... && cp ... DEST -> destination gate dir (sibling convention)."""
    cmd = task.get("cmd", "")
    if "mkdir" not in cmd or "cp" not in cmd:
        return None
    segments = [seg for seg in cmd.replace(";", "&&").split("&&")]
    cp_segment = next((seg for seg in segments if seg.strip().split()[:1] == ["cp"]), None)
    if cp_segment is None:
        return None
    parts = [t for t in cp_segment.strip().split()[1:] if not t.startswith("-") and ">" not in t]
    if len(parts) < 2:
        return None
    return {"task_id": task.get("id", "?"), "dest": parts[-1].rstrip("/")}


def _resolve(root: Path, queue_dir: str, filename: str) -> Path | None:
    for candidate in (root / "results" / queue_dir / filename, root / queue_dir / filename):
        if candidate.exists():
            return candidate
    return None


def _classify(path: Path, view: str) -> tuple[str, str]:
    """VALID/INVALID for one score JSON, using the real gate parser."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "INVALID", f"unparseable JSON: {exc}"
    try:
        _parse_view_result(view, data)
    except Exception as exc:  # noqa: BLE001 - the parser's reason is the report
        return "INVALID", f"{type(exc).__name__}: {exc}"
    return "VALID", ""


def _gate_decided(root: Path, gate: dict) -> tuple[bool, str]:
    path = root / "results" / "gates" / f"{gate['name']}.json"
    if not path.exists():
        return False, ""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "BROKEN"
    verdict = report.get("verdict") if isinstance(report, dict) else None
    if verdict in KNOWN_VERDICTS:
        return True, verdict
    return False, "BROKEN"


def _evidence(root: Path, view: str) -> list[dict]:
    """Committed same-view score files outside gate dirs, with validity."""
    found: list[dict] = []
    for base in (root / "results" / "runs", root / "runs"):
        if not base.exists():
            continue
        for path in sorted(base.rglob(f"{view}.json")):
            if "gate" in path.parts:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or data.get("scheme") != view:
                continue
            status, reason = _classify(path, view)
            found.append(
                {
                    "path": str(path.relative_to(root)),
                    "status": status,
                    "reason": reason,
                    "schema": (data.get("evaluation_protocol") or {}).get("schema_version"),
                }
            )
    return found


def analyze(root: Path, tasks: list[dict]) -> dict:
    gates = [g for t in tasks if (g := _parse_gate_task(t)) is not None]
    scores = [s for t in tasks if (s := _parse_score_task(t)) is not None]
    assembles = [a for t in tasks if (a := _parse_assemble_task(t)) is not None]
    evidence = {view: _evidence(root, view) for view in VIEWS}

    pending: list[dict] = []
    decided: list[dict] = []
    for gate in gates:
        is_decided, verdict = _gate_decided(root, gate)
        if is_decided:
            decided.append({**gate, "verdict": verdict})
            continue
        inputs: list[dict] = []
        for side, gate_dir in (
            ("incumbent", gate["incumbent_dir"]),
            ("candidate", gate["candidate_dir"]),
        ):
            for view in VIEWS:
                filename = f"{view}.json"
                path = _resolve(root, gate_dir, filename)
                if path is not None:
                    status, reason = _classify(path, view)
                    inputs.append(
                        {
                            "side": side,
                            "view": view,
                            "status": status,
                            "reason": reason,
                            "path": str(path.relative_to(root)),
                            "producers": [],
                        }
                    )
                else:
                    producers = [
                        s["task_id"]
                        for s in scores
                        if s["gate_dir"] == gate_dir and s["view"] == view
                    ]
                    producers += [a["task_id"] for a in assembles if a["dest"] == gate_dir]
                    inputs.append(
                        {
                            "side": side,
                            "view": view,
                            "status": "MISSING",
                            "reason": "",
                            "path": f"{gate_dir}/{filename}",
                            "producers": producers,
                        }
                    )
        pending.append({**gate, "inputs": inputs})
    return {
        "pending_gates": pending,
        "decided_gates": decided,
        "evidence": evidence,
    }


def render(analysis: dict) -> str:
    lines: list[str] = []
    pending = analysis["pending_gates"]
    decided = analysis["decided_gates"]
    lines.append(f"PENDING GATES: {len(pending)} (decided: {len(decided)})")
    n_valid = n_invalid = n_missing = 0
    rescore_blockers: list[str] = []
    for gate in pending:
        lines.append(f"\nGATE {gate['name']} (task {gate['task_id']}):")
        for inp in gate["inputs"]:
            status = inp["status"]
            if status == "VALID":
                n_valid += 1
            elif status == "INVALID":
                n_invalid += 1
                if "re-score" in inp["reason"]:
                    rescore_blockers.append(f"{gate['name']} {inp['side']} {inp['view']}")
            else:
                n_missing += 1
            detail = inp["path"]
            if inp["reason"]:
                detail += f" -- {inp['reason']}"
            if inp["producers"]:
                detail += f" <- {', '.join(inp['producers'])}"
            lines.append(f"  {inp['status']:7} {inp['side']:10} {inp['view']:8} {detail}")
    lines.append(f"\nINPUTS: {n_valid} VALID, {n_invalid} INVALID, {n_missing} MISSING")
    if rescore_blockers:
        lines.append(
            "RE-SCORE REQUIRED (present-but-invalid, copy cannot fix): "
            + "; ".join(sorted(set(rescore_blockers)))
        )
    lines.append("\nCOMMITTED EVIDENCE (same-view scores outside gate dirs):")
    for view in VIEWS:
        for ev in analysis["evidence"][view]:
            extra = f" -- {ev['reason']}" if ev["reason"] else ""
            lines.append(f"  {view:8} {ev['status']:7} schema={ev['schema']} {ev['path']}{extra}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root or Path(__file__).resolve().parents[1]
    tasks_path = root / "scripts" / "queue" / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {tasks_path}: {exc}", file=sys.stderr)
        return 2
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])
    print(render(analyze(root, tasks)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
