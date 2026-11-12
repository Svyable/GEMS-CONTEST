#!/usr/bin/env python3
"""Audit static repo references in the queue: wrappers, configs, scripts.

The box runner executes `tasks.json` strictly in list order and stops on
the first failure, so a task that names a file the repo no longer has —
a config renamed after the task was queued, a wrapper moved, a script
deleted — fails on the box, not here. With the box silent for a month,
that failure would surface only after resumption, burning the drain
margin the forecast measures (step 34). The order audit (step 33) checks
that consumers follow producers; the config-drift audit (step 35) checks
that recorded hashes still match the files on disk. Neither checks that
the files a queued command *names* still exist at all. This audit closes
that gap.

Two layers, both from committed evidence only:

1. Queue tasks (`scripts/queue/tasks.json` live, `tasks.parked.json`
   parked — parked tasks go live later, so they are checked the same
   way and reported separately):
   - wrapper scripts: tokens naming `queue/<name>.sh` (the box path is
     `/workspace/gems/queue/<name>.sh`) must exist under `scripts/queue/`.
   - configs: tokens naming `configs/**/*.yaml` must exist in the repo.
   - python scripts: tokens naming `scripts/**/*.py` must exist.
2. Wrappers themselves (`scripts/queue/*.sh`): a wrapper that calls
   `uv run python scripts/<name>.py` breaks the same way if that script
   is renamed, so every `scripts/**/*.py` and `configs/**/*.yaml` token
   in a wrapper is checked too.

Deliberately out of scope (box-side state, not repo references):
`data/*` (competition data is never committed), `runs/*` (run dirs and
gate dirs are train/score outputs or grandfathered state — the order
audit and the doctor own those), and gate/candidate names (identifiers,
not files). Double-quoted segments are stripped before extraction: the
trailing hypothesis/notes arguments of `record.sh` / `gate.sh` are free
text and must never be read as paths.

Findings:

- MISSING-REF: a named file does not exist in the repo. Exit 1.
- UNREADABLE-QUEUE: a queue file cannot be read or parsed. Exit 2.
- UNPARSED (note, exit 0): a task whose command names no static repo
  reference at all (e.g. the s1 wait-for-existing-loop task).

Usage:
    uv run python scripts/audit_queue_refs.py [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_WRAPPER_RE = re.compile(r"(?:^|[\s\"'])(?:/workspace/gems/)?queue/([A-Za-z0-9_.\-]+\.sh)")
_CONFIG_RE = re.compile(r"(?:^|[\s\"'])configs/([A-Za-z0-9_/.\-]+\.yaml)")
_SCRIPT_RE = re.compile(r"(?:^|[\s\"'])scripts/([A-Za-z0-9_/.\-]+\.py)")
_QUOTED_RE = re.compile(r'"[^"]*"')

_QUEUE_FILES = (("live", "tasks.json"), ("parked", "tasks.parked.json"))


def _strip_quoted(cmd: str) -> str:
    return _QUOTED_RE.sub(" ", cmd)


def extract_refs(cmd: str) -> dict[str, list[str]]:
    """Static repo references named in one queue command (or wrapper text)."""
    text = _strip_quoted(cmd)
    return {
        "wrappers": sorted(set(_WRAPPER_RE.findall(text))),
        "configs": sorted(set(_CONFIG_RE.findall(text))),
        "scripts": sorted(set(_SCRIPT_RE.findall(text))),
    }


def _check_refs(
    refs: dict[str, list[str]],
    root: Path,
    source: str,
    findings: list[dict],
) -> int:
    checked = 0
    for name in refs["wrappers"]:
        checked += 1
        if not (root / "scripts" / "queue" / name).is_file():
            findings.append({"source": source, "kind": "MISSING-REF", "ref": f"queue/{name}"})
    for name in refs["configs"]:
        checked += 1
        if not (root / "configs" / name).is_file():
            findings.append({"source": source, "kind": "MISSING-REF", "ref": f"configs/{name}"})
    for name in refs["scripts"]:
        checked += 1
        if not (root / "scripts" / name).is_file():
            findings.append({"source": source, "kind": "MISSING-REF", "ref": f"scripts/{name}"})
    return checked


def audit(root: Path) -> dict:
    findings: list[dict] = []
    tasks_checked = 0
    refs_checked = 0
    unparsed: list[str] = []
    per_file: dict[str, dict] = {}
    unreadable = 0

    queue_dir = root / "scripts" / "queue"
    for label, filename in _QUEUE_FILES:
        path = queue_dir / filename
        try:
            tasks = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(tasks, list):
                raise TypeError(f"{filename} is not a list")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            findings.append({"source": filename, "kind": "UNREADABLE-QUEUE", "ref": str(exc)})
            unreadable += 1
            per_file[label] = {"tasks": 0, "refs": 0, "unparsed": 0}
            continue
        file_refs = 0
        file_unparsed = 0
        for task in tasks:
            task_id = task.get("id", "?")
            refs = extract_refs(task.get("cmd", ""))
            total = sum(len(v) for v in refs.values())
            if total == 0:
                file_unparsed += 1
                unparsed.append(f"{label}:{task_id}")
                continue
            file_refs += _check_refs(refs, root, f"{label}:{task_id}", findings)
        per_file[label] = {
            "tasks": len(tasks),
            "refs": file_refs,
            "unparsed": file_unparsed,
        }
        tasks_checked += len(tasks)
        refs_checked += file_refs

    wrappers_checked = 0
    wrapper_refs = 0
    for wrapper in sorted(queue_dir.glob("*.sh")):
        wrappers_checked += 1
        text = wrapper.read_text(encoding="utf-8")
        refs = extract_refs(text)
        wrapper_refs += _check_refs(refs, root, f"wrapper:{wrapper.name}", findings)

    return {
        "tasks": tasks_checked,
        "refs": refs_checked,
        "wrappers": wrappers_checked,
        "wrapper_refs": wrapper_refs,
        "per_file": per_file,
        "unparsed": unparsed,
        "unreadable": unreadable,
        "findings": findings,
    }


def render(result: dict) -> str:
    per = result["per_file"]
    lines = [
        (
            f"QUEUE REF AUDIT: {result['tasks']} tasks "
            f"({per.get('live', {}).get('tasks', 0)} live, "
            f"{per.get('parked', {}).get('tasks', 0)} parked), "
            f"{result['refs']} task refs + {result['wrapper_refs']} wrapper refs "
            f"checked across {result['wrappers']} wrappers "
            f"({len(result['unparsed'])} unparsed)"
        )
    ]
    findings = result["findings"]
    missing = [f for f in findings if f["kind"] == "MISSING-REF"]
    other = [f for f in findings if f["kind"] != "MISSING-REF"]
    lines.append(f"{len(missing)} MISSING-REF(s), {len(other)} other finding(s)")
    for finding in findings:
        lines.append(f"  {finding['kind']} {finding['source']}: {finding['ref']}")
    for task in result["unparsed"]:
        lines.append(f"  NOTE {task}: no static repo reference (wait/shell task)")
    if not missing and not other:
        lines.append(
            "  OK: every wrapper, config, and script a queued command or "
            "wrapper names still exists in the repo."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit static repo references named by queue tasks and wrappers."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = audit(args.root)
    print(render(result), end="")
    if result["unreadable"]:
        return 2
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
