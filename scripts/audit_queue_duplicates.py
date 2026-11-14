#!/usr/bin/env python3
"""Audit duplication in the queued experiment queue: same work twice.

The refs audit (step 38) checks that files a queued command names exist;
the args audit (step 39) checks that a command's arguments are consistent
with each other. Neither checks whether the queue asks for the same work
twice — and the queue's execution model makes duplication silently
destructive rather than merely wasteful:

- The runner keys on task ids with `.done` markers (step 26): a second
  task with an id that already ran is skipped without a trace, so a
  duplicated id silently drops a whole train/score/record triple.
- `gate.sh` writes its verdict to `phase2-outputs/gates/<NAME>.json`: two
  gate tasks with the same name clobber each other's verdict, and the
  ledger keeps only the later one.
- `score_view.sh` writes `<GATE_DIR>/<VIEW>.json`: two different runs
  scored into the same gate dir and view land in one file, and the later
  score silently wins — the gate then compares the wrong candidate.
- `train_folds.sh` trains into a run dir and `record.sh` writes a manifest
  keyed by run id: a repeated run dir or run id redoes (or overwrites)
  folds and manifests the first task already produced.
- `endpoint_view.sh` / `curved_view.sh` write a derived run dir: two views
  from different base runs into the same output dir clobber each other.

One deliberate exception: the step-26 replays. Step 26 inserted three
byte-identical replays into the live queue — `s26-rescore-ref-spatial-full`,
`s26-rescore-ref-fault-full` (schema-2 re-scores of the s1/s2 full runs)
and `s26-reassemble-ref-b16` (unconditional re-assemble into
`runs/gate/ref-b16`, repairing a stale schema-1 copy even if the s5b
assemble already ran). They are the same work on purpose, so a duplicate
whose extra tasks are all step-26 replays (id starts with `s26-`) is a
REPLAY note, not a finding. Any other duplication is a finding.

Deliberately out of scope: whether named files exist (refs audit, step 38),
argument consistency (args audit, step 39), task order (order audit,
step 33), recorded hashes (config-drift audit, step 35), and box-side
state — this audit reads the committed queue files only.

Findings:

- DUP-TASK-ID / DUP-COMMAND / DUP-GATE-NAME / DUP-SCORE-COVERAGE /
  DUP-TRAIN-RUN / DUP-RECORD-ID / DUP-VIEW-OUT: exit 1.
- UNREADABLE-QUEUE: a queue file cannot be read or parsed. Exit 2.
- REPLAY (note, exit 0): a step-26 replay of an existing task.

Usage:
    uv run python scripts/audit_queue_duplicates.py [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

_QUEUE_FILES = (("live", "tasks.json"), ("parked", "tasks.parked.json"))

# Step-26 replay tasks are byte-identical replays inserted on purpose
# (EXPERIMENT_LOG 2026-10-31): re-scores repair a stale schema-1 gate copy
# and re-assemble repairs a stale assemble. Any other duplication is a bug.
_REPLAY_PREFIX = "s26-"

_KNOWN_WRAPPERS = (
    "train_folds.sh",
    "score_view.sh",
    "record.sh",
    "gate.sh",
    "endpoint_view.sh",
    "curved_view.sh",
    "rival_threshold.sh",
    "compare_seed.sh",
    "compare_schedule.sh",
)


def _note(notes: list[dict], kind: str, source: str, ref: str) -> None:
    notes.append({"kind": kind, "source": source, "ref": ref})


def _finding(findings: list[dict], kind: str, source: str, ref: str) -> None:
    findings.append({"kind": kind, "source": source, "ref": ref})


def _segments(cmd: str) -> list[list[str]]:
    """Tokenize each `&&` segment of a queue command (unparseable skipped)."""
    out: list[list[str]] = []
    for segment in cmd.split("&&"):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if tokens:
            out.append(tokens)
    return out


def _wrapper_args(tokens: list[str]) -> tuple[str, list[str]] | None:
    wrapper = Path(tokens[0]).name
    if wrapper not in _KNOWN_WRAPPERS:
        return None
    return wrapper, tokens[1:]


def audit(root: Path) -> dict:
    findings: list[dict] = []
    notes: list[dict] = []
    tasks_checked = 0
    per_file: dict[str, dict] = {}
    unreadable = 0

    queue_dir = root / "scripts" / "queue"
    entries: list[tuple[str, str, str]] = []  # (file label, task id, cmd)
    for label, filename in _QUEUE_FILES:
        path = queue_dir / filename
        try:
            tasks = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(tasks, list):
                raise TypeError(f"{filename} is not a list")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            findings.append({"source": filename, "kind": "UNREADABLE-QUEUE", "ref": str(exc)})
            unreadable += 1
            per_file[label] = {"tasks": 0}
            continue
        per_file[label] = {"tasks": len(tasks)}
        tasks_checked += len(tasks)
        for task in tasks:
            entries.append((label, task.get("id", "?"), task.get("cmd", "")))

    # DUP-TASK-ID: the runner keys on ids with .done markers; a second task
    # with an id that already ran is silently skipped.
    by_id: dict[str, list[str]] = {}
    for label, task_id, _ in entries:
        by_id.setdefault(task_id, []).append(f"{label}:{task_id}")
    for task_id, sources in sorted(by_id.items()):
        if len(sources) > 1:
            _finding(
                findings,
                "DUP-TASK-ID",
                sources[0],
                f"task id {task_id!r} appears {len(sources)} times: "
                + ", ".join(sources)
                + " — the runner skips the later one on its .done marker",
            )

    # DUP-COMMAND: identical command text. Extra tasks that are step-26
    # replays are documented intent (see module docstring), not findings.
    by_cmd: dict[str, list[tuple[str, str]]] = {}
    for label, task_id, cmd in entries:
        by_cmd.setdefault(cmd, []).append((f"{label}:{task_id}", task_id))
    for cmd, owners in sorted(by_cmd.items()):
        if len(owners) < 2:
            continue
        replays = [o for o in owners if o[1].startswith(_REPLAY_PREFIX)]
        others = [o for o in owners if not o[1].startswith(_REPLAY_PREFIX)]
        original = others[0][0] if others else owners[0][0]
        for source, _ in replays:
            _note(
                notes,
                "REPLAY",
                source,
                f"step-26 replay of {original}: {cmd[:100]}",
            )
        if len(others) > 1:
            _finding(
                findings,
                "DUP-COMMAND",
                others[0][0],
                "identical command in " + ", ".join(s for s, _ in others) + f": {cmd[:100]}",
            )

    # Per-wrapper groupings: gate name, score coverage, train run dir,
    # record run id, view output dir.
    gate_names: dict[str, list[str]] = {}
    score_cov: dict[tuple[str, str], list[tuple[str, str]]] = {}
    train_runs: dict[str, list[str]] = {}
    record_ids: dict[str, list[str]] = {}
    view_outs: dict[str, list[tuple[str, str]]] = {}
    for label, task_id, cmd in entries:
        source = f"{label}:{task_id}"
        for tokens in _segments(cmd):
            parsed = _wrapper_args(tokens)
            if parsed is None:
                continue
            wrapper, args = parsed
            if wrapper == "gate.sh" and len(args) >= 1:
                gate_names.setdefault(args[0], []).append(source)
            elif wrapper == "score_view.sh" and len(args) >= 4:
                score_cov.setdefault((args[3], args[1]), []).append((source, args[0]))
            elif wrapper == "train_folds.sh" and len(args) >= 1:
                train_runs.setdefault(args[0], []).append(source)
            elif wrapper == "record.sh" and len(args) >= 1:
                record_ids.setdefault(args[0], []).append(source)
            elif wrapper in ("endpoint_view.sh", "curved_view.sh") and len(args) >= 3:
                view_outs.setdefault(args[1], []).append((source, args[0]))

    for name, sources in sorted(gate_names.items()):
        if len(sources) > 1:
            _finding(
                findings,
                "DUP-GATE-NAME",
                sources[0],
                f"gate name {name!r} in "
                + ", ".join(sources)
                + " — both write phase2-outputs/gates/"
                + f"{name}.json, the later verdict clobbers the first",
            )

    for (gate_dir, view), scored in sorted(score_cov.items()):
        run_dirs = {run for _, run in scored}
        if len(run_dirs) > 1:
            _finding(
                findings,
                "DUP-SCORE-COVERAGE",
                scored[0][0],
                f"{gate_dir}/{view}.json scored from {len(run_dirs)} runs: "
                + ", ".join(f"{s} ({r})" for s, r in scored)
                + " — the later score silently wins",
            )
        elif len(scored) > 1:
            # Same run scored into the same view file twice: a replay,
            # not a collision — but only the documented s26 replays may
            # do this (identical commands are already DUP-COMMAND-checked).
            for source, _ in scored[1:]:
                task_id = source.split(":", 1)[1]
                if task_id.startswith(_REPLAY_PREFIX):
                    _note(
                        notes,
                        "REPLAY",
                        source,
                        f"step-26 re-score of {gate_dir}/{view}.json "
                        f"from {scored[0][1]} (same run, byte-identical replay)",
                    )

    for run_dir, sources in sorted(train_runs.items()):
        if len(sources) > 1:
            _finding(
                findings,
                "DUP-TRAIN-RUN",
                sources[0],
                f"run dir {run_dir} trained by "
                + ", ".join(sources)
                + " — the later train redoes (or overwrites) the first's folds",
            )

    for run_id, sources in sorted(record_ids.items()):
        if len(sources) > 1:
            _finding(
                findings,
                "DUP-RECORD-ID",
                sources[0],
                f"run id {run_id!r} recorded by "
                + ", ".join(sources)
                + " — both write the same run manifest",
            )

    for out_run, derived in sorted(view_outs.items()):
        base_runs = {base for _, base in derived}
        if len(base_runs) > 1:
            _finding(
                findings,
                "DUP-VIEW-OUT",
                derived[0][0],
                f"view output {out_run} derived from {len(base_runs)} base runs: "
                + ", ".join(f"{s} ({b})" for s, b in derived)
                + " — the later view clobbers the first",
            )

    return {
        "tasks": tasks_checked,
        "per_file": per_file,
        "unreadable": unreadable,
        "findings": findings,
        "notes": notes,
    }


def render(result: dict) -> str:
    per = result["per_file"]
    lines = [
        (
            f"QUEUE DUPLICATION AUDIT: {result['tasks']} tasks "
            f"({per.get('live', {}).get('tasks', 0)} live, "
            f"{per.get('parked', {}).get('tasks', 0)} parked), "
            f"{len(result['notes'])} replay note(s)"
        )
    ]
    findings = result["findings"]
    other = [f for f in findings if f["kind"] != "UNREADABLE-QUEUE"]
    lines.append(f"{len(other)} finding(s), {result['unreadable']} unreadable queue file(s)")
    for finding in findings:
        lines.append(f"  {finding['kind']} {finding['source']}: {finding['ref']}")
    for note in result["notes"]:
        lines.append(f"  NOTE {note['kind']} {note['source']}: {note['ref']}")
    if not findings:
        lines.append(
            "  OK: no task id, command, gate name, score file, train run, "
            "record id, or view output is claimed twice."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit duplication in the queued commands.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = audit(args.root)
    print(render(result), end="")
    if result["unreadable"]:
        return 2
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
