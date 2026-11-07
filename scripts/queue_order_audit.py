#!/usr/bin/env python3
"""Audit the queue's list order: every consumer must follow its producers.

The runner (`scripts/queue/runner.py`) executes pending tasks strictly in
`tasks.json` list order and stops on the first failure, so list order *is*
dependency order. `tests/test_queue_tasks.py` (steps 16-17) already checks
producer *presence* — a gate's candidate dir must have a producing score
task somewhere in the queue — but presence says nothing about position.
Step 26 had to hand-insert three re-score tasks immediately before the
assemble they feed; had they been appended after it, the assemble would
have copied stale schema-1 scores and every downstream gate would have
decided on them. This audit generalizes that hand fix into a checked
invariant: for every task, each input it reads from the queue must be
written by a task that appears *earlier* in the list.

Dependency model (derived from command text, same token idioms as
`scripts/queue_critical_path.py`):

- train (`train_folds.sh RUN ...`) writes `RUN` (fold predictions).
- score (`score_view.sh RUN SCHEME FMAP GATE`) reads `RUN` (train output)
  and writes `GATE/SCHEME.json` plus `RUN/scores/lofo-SCHEME.json`.
- view (`endpoint_view.sh` / `curved_view.sh BASE OUT SCHEME` chained with
  a score step) reads `BASE` fold predictions and its
  `BASE/scores/lofo-SCHEME.json`, and writes `OUT` folds plus
  `GATE/SCHEME.json`. The chained score reads `OUT` folds produced by the
  view step in the same task, so no external train producer is needed.
- record (`record.sh ID RUN ...`) reads `RUN/scores/*.json` (score output).
- assemble (`mkdir -p DEST && cp SRC/.../VIEW.json DEST/`) reads each
  source gate JSON and writes it into `DEST`.
- gate (`gate.sh NAME INC CAND ...`) reads all three views from both gate
  dirs.
- compare (`compare_schedule.sh` / `compare_seed.sh INC_GATE CAND_GATE
  INC_BASE CAND_BASE|CAND_RUN`) reads both gate dirs and the candidate run
  dir(s). Incumbent LOFO summaries are best-effort by design
  (`compare_schedule.py` skips missing views; step 27), so they are not
  checked here.

Findings:

- ORDER-VIOLATION: a producer exists in the queue but appears at or after
  the consumer. Exit 1. This is the failure the step-26 insert prevented
  by hand.
- NO-PRODUCER (note, exit 0): no queued task writes the input at all.
  Presence is owned by the invariant tests in `tests/test_queue_tasks.py`;
  an incumbent gate dir may legitimately predate the queue, so the audit
  reports these for a human to disposition instead of failing.

Usage:
    uv run python scripts/queue_order_audit.py [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VIEWS = ("spatial", "fault", "trace")

_TRIPLE_RE = re.compile(r"^(s\d+[a-z]?)-(train|score|record)-(.+)$")


def _tokens(cmd: str) -> list[str]:
    # Only the leading tokens matter (script, dirs, config); the trailing
    # quoted hypothesis may contain spaces and escaped quotes, so a plain
    # split is both sufficient and safer than shlex here.
    return cmd.split()


def _find_script_token(tokens: list[str], basename: str) -> int | None:
    for i, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1] == basename:
            return i
    return None


def _parse_task(task: dict) -> dict | None:
    """Parse one queue task into its kind, reads, and writes.

    Returns None for tasks with no queue-internal file dependencies
    (e.g. the s1 wait-for-existing-loop task).
    """
    cmd = task.get("cmd", "")
    tokens = _tokens(cmd)
    task_id = task.get("id", "?")
    for basename in ("endpoint_view.sh", "curved_view.sh"):
        idx = _find_script_token(tokens, basename)
        if idx is not None and len(tokens) >= idx + 4:
            score_idx = _find_script_token(tokens, "score_view.sh")
            gate = tokens[score_idx + 4] if score_idx is not None else None
            return {
                "kind": "view",
                "task_id": task_id,
                "base_run": tokens[idx + 1],
                "run": tokens[idx + 2],
                "scheme": tokens[idx + 3],
                "gate_dir": gate,
            }
    idx = _find_script_token(tokens, "train_folds.sh")
    if idx is not None and len(tokens) >= idx + 2:
        return {"kind": "train", "task_id": task_id, "run": tokens[idx + 1]}
    idx = _find_script_token(tokens, "score_view.sh")
    if idx is not None and len(tokens) >= idx + 5:
        return {
            "kind": "score",
            "task_id": task_id,
            "run": tokens[idx + 1],
            "scheme": tokens[idx + 2],
            "gate_dir": tokens[idx + 4],
        }
    idx = _find_script_token(tokens, "record.sh")
    if idx is not None and len(tokens) >= idx + 3:
        return {"kind": "record", "task_id": task_id, "run": tokens[idx + 2]}
    if "mkdir" in cmd and "cp" in cmd:
        segments = [seg for seg in cmd.replace(";", "&&").split("&&")]
        cp_segment = next((seg for seg in segments if seg.strip().split()[:1] == ["cp"]), None)
        if cp_segment is not None:
            parts = [
                t for t in cp_segment.strip().split()[1:] if not t.startswith("-") and ">" not in t
            ]
            if len(parts) >= 2:
                sources, dest = parts[:-1], parts[-1].rstrip("/")
                return {
                    "kind": "assemble",
                    "task_id": task_id,
                    "dest": dest,
                    "sources": [(str(Path(s).parent), Path(s).stem) for s in sources],
                }
    idx = _find_script_token(tokens, "gate.sh")
    if idx is not None and len(tokens) >= idx + 4:
        return {
            "kind": "gate",
            "task_id": task_id,
            "name": tokens[idx + 1],
            "incumbent_dir": tokens[idx + 2],
            "candidate_dir": tokens[idx + 3],
        }
    for basename in ("compare_seed.sh", "compare_schedule.sh"):
        idx = _find_script_token(tokens, basename)
        if idx is not None and len(tokens) >= idx + 5:
            return {
                "kind": "compare",
                "task_id": task_id,
                "name": basename.replace(".sh", ""),
                "incumbent_dir": tokens[idx + 1],
                "candidate_dir": tokens[idx + 2],
                "incumbent_base": tokens[idx + 3],
                "candidate_base": tokens[idx + 4],
                "seed": basename == "compare_seed.sh",
            }
    return None


class OrderAudit:
    """List-order check over a parsed queue."""

    def __init__(self, tasks: list[dict]):
        self.tasks = tasks
        self.parsed: list[dict] = []
        self.unparsed = 0
        for index, task in enumerate(tasks):
            parsed = _parse_task(task)
            if parsed is None:
                self.unparsed += 1
                continue
            parsed["index"] = index
            self.parsed.append(parsed)
        # Earliest producer per input, by (kind-specific) key.
        self.train_by_run: dict[str, dict] = {}
        self.view_by_run: dict[str, dict] = {}
        self.score_by_run_scheme: dict[tuple[str, str], dict] = {}
        self.score_by_run: dict[str, dict] = {}
        self.gate_by_dir_view: dict[tuple[str, str], dict] = {}
        for parsed in self.parsed:
            kind = parsed["kind"]
            if kind == "train":
                self.train_by_run.setdefault(parsed["run"], parsed)
            elif kind == "view":
                self.view_by_run.setdefault(parsed["run"], parsed)
                self.score_by_run_scheme.setdefault((parsed["run"], parsed["scheme"]), parsed)
                self.score_by_run.setdefault(parsed["run"], parsed)
                if parsed["gate_dir"]:
                    self.gate_by_dir_view.setdefault((parsed["gate_dir"], parsed["scheme"]), parsed)
            elif kind == "score":
                self.score_by_run_scheme.setdefault((parsed["run"], parsed["scheme"]), parsed)
                self.score_by_run.setdefault(parsed["run"], parsed)
                if parsed["gate_dir"]:
                    self.gate_by_dir_view.setdefault((parsed["gate_dir"], parsed["scheme"]), parsed)
            elif kind == "assemble":
                for src_dir, view in parsed["sources"]:
                    # An assemble also *writes* the copied view into dest.
                    self.gate_by_dir_view.setdefault((parsed["dest"], view), parsed)

    def _check_input(
        self,
        consumer: dict,
        label: str,
        producer: dict | None,
        violations: list[dict],
        notes: list[dict],
    ) -> None:
        if producer is None:
            notes.append({"consumer": consumer, "input": label, "kind": "NO-PRODUCER"})
        elif producer["index"] >= consumer["index"]:
            violations.append(
                {
                    "consumer": consumer,
                    "input": label,
                    "producer": producer,
                    "kind": "ORDER-VIOLATION",
                }
            )

    def _run_producer(self, run: str) -> dict | None:
        """Earliest task producing a run's fold predictions (train or view)."""
        candidates = [
            p for p in (self.train_by_run.get(run), self.view_by_run.get(run)) if p is not None
        ]
        return min(candidates, key=lambda p: p["index"]) if candidates else None

    def audit(self) -> dict:
        violations: list[dict] = []
        notes: list[dict] = []

        # 1. Train/score/record triple order within each (step, target) group.
        index_by_id = {t.get("id", "?"): i for i, t in enumerate(self.tasks)}
        groups: dict[tuple[str, str], dict[str, dict]] = {}
        for task in self.tasks:
            match = _TRIPLE_RE.match(task.get("id", ""))
            if not match:
                continue
            step, op, target = match.groups()
            groups.setdefault((step, target), {})[op] = task
        order = ("train", "score", "record")
        for (step, target), ops in sorted(groups.items()):
            positions = {op: index_by_id[task.get("id", "?")] for op, task in ops.items()}
            present = [op for op in order if op in positions]
            for i, earlier in enumerate(present):
                for later in present[i + 1 :]:
                    if positions[earlier] > positions[later]:
                        violations.append(
                            {
                                "consumer": {
                                    "task_id": ops[later]["id"],
                                    "index": positions[later],
                                    "kind": later,
                                },
                                "input": f"train/score/record triple "
                                f"{step}-{target}: {earlier} must precede {later}",
                                "producer": {
                                    "task_id": ops[earlier]["id"],
                                    "index": positions[earlier],
                                    "kind": earlier,
                                },
                                "kind": "ORDER-VIOLATION",
                            }
                        )

        for parsed in self.parsed:
            kind = parsed["kind"]
            if kind == "score":
                self._check_input(
                    parsed,
                    f"run {parsed['run']} fold predictions",
                    self._run_producer(parsed["run"]),
                    violations,
                    notes,
                )
            elif kind == "view":
                self._check_input(
                    parsed,
                    f"run {parsed['base_run']} lofo-{parsed['scheme']}.json",
                    self.score_by_run_scheme.get((parsed["base_run"], parsed["scheme"])),
                    violations,
                    notes,
                )
            elif kind == "record":
                self._check_input(
                    parsed,
                    f"run {parsed['run']} score JSONs",
                    self.score_by_run.get(parsed["run"]),
                    violations,
                    notes,
                )
            elif kind == "assemble":
                for src_dir, view in parsed["sources"]:
                    self._check_input(
                        parsed,
                        f"{src_dir}/{view}.json",
                        self.gate_by_dir_view.get((src_dir, view)),
                        violations,
                        notes,
                    )
            elif kind == "gate":
                for gate_dir in (parsed["incumbent_dir"], parsed["candidate_dir"]):
                    for view in VIEWS:
                        self._check_input(
                            parsed,
                            f"{gate_dir}/{view}.json",
                            self.gate_by_dir_view.get((gate_dir, view)),
                            violations,
                            notes,
                        )
            elif kind == "compare":
                for gate_dir in (parsed["incumbent_dir"], parsed["candidate_dir"]):
                    for view in VIEWS:
                        self._check_input(
                            parsed,
                            f"{gate_dir}/{view}.json",
                            self.gate_by_dir_view.get((gate_dir, view)),
                            violations,
                            notes,
                        )
                if parsed["seed"]:
                    self._check_input(
                        parsed,
                        f"run {parsed['candidate_base']} fold predictions",
                        self._run_producer(parsed["candidate_base"]),
                        violations,
                        notes,
                    )
                else:
                    for view in VIEWS:
                        pattern = re.compile(
                            re.escape(parsed["candidate_base"]) + r"-" + view + r"-b\d+"
                        )
                        producer = None
                        for run, task in self.train_by_run.items():
                            if pattern.fullmatch(run) and (
                                producer is None or task["index"] < producer["index"]
                            ):
                                producer = task
                        self._check_input(
                            parsed,
                            f"run {parsed['candidate_base']}-{view}-bN fold predictions",
                            producer,
                            violations,
                            notes,
                        )
        return {
            "tasks": len(self.tasks),
            "checked": len(self.parsed),
            "unparsed": self.unparsed,
            "violations": violations,
            "notes": notes,
        }


def render(audit_result: dict) -> str:
    lines = [
        (
            f"QUEUE ORDER AUDIT: {audit_result['tasks']} tasks, "
            f"{audit_result['checked']} with queue-internal dependencies checked "
            f"({audit_result['unparsed']} unparsed/ignored)"
        )
    ]
    violations = audit_result["violations"]
    notes = audit_result["notes"]
    lines.append(f"{len(violations)} ORDER-VIOLATION(s), {len(notes)} NO-PRODUCER note(s)")
    for finding in violations:
        consumer = finding["consumer"]
        producer = finding["producer"]
        lines.append(
            f"  VIOLATION {consumer['task_id']} (index {consumer['index']}) "
            f"reads {finding['input']} but producer "
            f"{producer['task_id']} is at index {producer['index']} (after)"
        )
    for note in notes:
        consumer = note["consumer"]
        lines.append(
            f"  NOTE {consumer['task_id']} (index {consumer['index']}) reads "
            f"{note['input']}: no queued producer "
            f"(grandfathered/committed, or genuinely missing — presence is "
            f"covered by tests/test_queue_tasks.py)"
        )
    if not violations:
        lines.append("  OK: every consumer follows its producers in list order.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit queue list order: consumers must follow producers."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    tasks_path = args.root / "scripts" / "queue" / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise TypeError("tasks.json is not a list")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"error: cannot read {tasks_path}: {exc}", file=sys.stderr)
        return 2
    result = OrderAudit(tasks).audit()
    print(render(result), end="")
    return 1 if result["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
