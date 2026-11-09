#!/usr/bin/env python3
"""Audit committed results for config drift: does the config file on disk
still hash to what the committed record says it was?

A gate refuses a protocol-unequal comparison, and the protocol a run
scores under is fixed by the config file at train time — every score
JSON and run manifest records its ``config_sha256``. But nothing today
checks that the file still hashes to that value. A config edited after
a run was scored (a loss tweak, a channel change, a buffer fix) silently
changes what the queue trains next: the pending trains for that config
score under a new protocol, their gate side is protocol-unequal to the
recorded side, and the gate refuses — after the GPU days are already
spent. This audit closes that gap by comparing, per run, the recorded
hashes against the current file.

Evidence, per run under ``results/runs/<run>/`` (the ``gate/`` copies
are excluded — they duplicate their source run's protocol, and the
step-22 audit already verifies their content by SHA256):

- ``manifest.json``: ``config.path`` + ``config.sha256`` — the
  authoritative run-to-config mapping and the hash at record time.
- ``scores/*.json`` and run-top-level ``*.json``:
  ``evaluation_protocol.config_sha256`` (schema 2 only; schema-1
  scores carry no hash).
- ``fold-*.json``: top-level ``config_sha256``.

The config a re-train would use comes from the queue: the
``train_folds.sh`` task for the same run dir names its config as the
second argument. A manifest path that disagrees with the queue task is
a CONFIG-MISMATCH — the queue would re-train that run under a
different config than the one the record says it used.

Findings
--------
- DRIFT: the config file's current hash differs from the recorded
  hash. The finding lists the pending train tasks using that config:
  those are the trains that would score under the new protocol.
- CONFIG-MISSING: the config file no longer exists in the repo.
- CONFIG-MISMATCH: manifest and queue disagree on the run's config.
- MIXED: the committed records of one run disagree with each other
  on the config hash — the provenance is self-contradictory.
- NO-RECORDED-HASH / NO-TRAIN-TASK: notes, not findings — a scored
  run with no recorded hash (schema-1 scores with no manifest), or a
  scored run with no train task left in the queue (grandfathered).

Exit code is 0 when clean (notes allowed), 1 on any DRIFT,
CONFIG-MISSING, CONFIG-MISMATCH, or MIXED finding, 2 on an unreadable
or malformed ``tasks.json`` — matching the sibling queue audits.

Usage:
    uv run python scripts/audit_config_drift.py [--root PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

VIEWS = ("spatial", "fault", "trace")


def _load_tasks(root: Path) -> list[dict]:
    tasks_path = root / "scripts" / "queue" / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read tasks.json: {exc}") from exc
    if not isinstance(tasks, list):
        raise TypeError("tasks.json must be a JSON list")
    return tasks


def _train_configs(tasks: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map run dir -> config path, and config path -> train task ids.

    Parses ``train_folds.sh <run_dir> <config> ...`` commands. A task
    whose command does not parse is skipped (the queue-order audit
    owns command-shape problems); its run simply has no queue config.
    """
    run_config: dict[str, str] = {}
    config_trains: dict[str, list[str]] = {}
    for task in tasks:
        cmd = task.get("cmd", "")
        if "train_folds.sh" not in cmd:
            continue
        try:
            parts = shlex.split(cmd)
        except ValueError:
            continue
        try:
            idx = next(i for i, p in enumerate(parts) if p.endswith("train_folds.sh"))
        except StopIteration:
            continue
        if len(parts) < idx + 3:
            continue
        run_dir = parts[idx + 1].removeprefix("runs/")
        config = parts[idx + 2]
        run_config[run_dir] = config
        config_trains.setdefault(config, []).append(task.get("id", "?"))
    return run_config, config_trains


def _recorded_hashes(run_dir: Path) -> tuple[set[str], set[str], int]:
    """Recorded (config path, config hash) pairs for one run dir.

    Returns (paths, hashes, skipped): manifest paths, every recorded
    hash, and the count of JSON files that could not be read.
    """
    paths: set[str] = set()
    hashes: set[str] = set()
    skipped = 0
    candidates = list(run_dir.glob("*.json"))
    candidates += list((run_dir / "scores").glob("*.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        if not isinstance(data, dict):
            continue
        if path.name == "manifest.json":
            config = data.get("config") or {}
            if isinstance(config.get("path"), str):
                paths.add(config["path"])
            if isinstance(config.get("sha256"), str):
                hashes.add(config["sha256"])
            continue
        protocol = data.get("evaluation_protocol") or {}
        if isinstance(protocol.get("config_sha256"), str):
            hashes.add(protocol["config_sha256"])
        if isinstance(data.get("config_sha256"), str):
            hashes.add(data["config_sha256"])
    return paths, hashes, skipped


def audit(root: Path) -> dict:
    """Run the drift audit. Returns a report dict; never raises on
    bad result files (those are counted as skipped)."""
    tasks = _load_tasks(root)
    run_config, config_trains = _train_configs(tasks)

    runs: list[dict] = []
    findings = 0
    notes = 0
    skipped_files = 0
    runs_dir = root / "results" / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name != "gate"):
            paths, hashes, skipped = _recorded_hashes(run_dir)
            skipped_files += skipped
            run = run_dir.name
            queue_config = run_config.get(run)
            entry: dict = {"run": run, "status": "OK", "detail": ""}
            if paths and queue_config and queue_config not in paths:
                entry["status"] = "CONFIG-MISMATCH"
                entry["detail"] = (
                    f"manifest says {sorted(paths)} but queue trains with {queue_config}"
                )
                findings += 1
            elif not hashes:
                entry["status"] = "NO-RECORDED-HASH"
                entry["detail"] = "no committed record carries a config hash"
                notes += 1
            elif len(hashes) > 1:
                entry["status"] = "MIXED"
                entry["detail"] = "committed records disagree: " + ", ".join(
                    sorted(h[:12] for h in hashes)
                )
                findings += 1
            else:
                config = next(iter(paths)) if paths else queue_config
                if config is None:
                    entry["status"] = "NO-TRAIN-TASK"
                    entry["detail"] = "no manifest path and no queue train task"
                    notes += 1
                else:
                    entry["config"] = config
                    config_path = root / config
                    if not config_path.is_file():
                        entry["status"] = "CONFIG-MISSING"
                        entry["detail"] = f"{config} no longer exists in the repo"
                        findings += 1
                    else:
                        current = hashlib.sha256(config_path.read_bytes()).hexdigest()
                        recorded = next(iter(hashes))
                        if current == recorded:
                            entry["detail"] = f"{config} {current[:12]} matches"
                        else:
                            entry["status"] = "DRIFT"
                            users = config_trains.get(config, [])
                            entry["detail"] = (
                                f"{config} recorded {recorded[:12]} "
                                f"but now {current[:12]}"
                                + (f"; trains using it: {', '.join(users)}" if users else "")
                            )
                            findings += 1
            runs.append(entry)
    return {
        "runs": runs,
        "findings": findings,
        "notes": notes,
        "skipped_files": skipped_files,
        "checked": len(runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit committed results for config drift.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    try:
        report = audit(args.root)
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for entry in report["runs"]:
        print(f"{entry['status']:16} {entry['run']:38} {entry['detail']}")
    print(
        f"\n{report['checked']} runs checked, "
        f"{report['findings']} finding(s), {report['notes']} note(s), "
        f"{report['skipped_files']} unreadable file(s) skipped."
    )
    if report["findings"]:
        print(
            "A DRIFT means pending trains for that config would score under a "
            "new protocol and their gate side would be protocol-unequal to the "
            "recorded side — fix the config or re-record before the box trains it."
        )
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
