#!/usr/bin/env python3
"""Rank pending queue tasks by how many pending decisions they unblock.

`scripts/queue_status.py` (step 23) answers which queued tasks have
landed; `scripts/gate_status.py` (step 19) answers which decisions are
decided. Neither answers the question the box faces after a long
silence: *what should run first?* A pending train task that feeds a
gate with eight other missing inputs is less urgent than the one
assemble task whose output unblocks seven gates and both compares.

This script builds that ranking from the same committed-tree evidence
the sibling inventories use (``results/``-prefixed paths first, bare
queue paths as fallback). A task counts as already-run only when the
committed tree holds evidence for its own output; box-side-only
progress (run but uncommitted) is invisible here by design, so the box
should commit before trusting the order.

Decision inputs
--------------
- A gate needs one JSON per view (spatial, fault, trace) in both its
  incumbent and candidate gate dirs.
- A compare needs the same gate JSONs on both sides, plus LOFO
  summaries (``<run>/scores/lofo-<view>.json``) from the incumbent and
  candidate run dirs named in its script arguments. The schedule
  compare hard-copies every input; the seed compare skips missing
  views, so its inputs are best-effort (the probe is spatial-only on
  the candidate side — other candidate views are not applicable).
- A missing gate JSON is fed by a pending score/view task writing to
  that gate dir, or by a pending assemble task copying it there. An
  assemble task's own prerequisite is each source file's presence in
  the committed tree; a missing source whose producer score task
  already ran is flagged EVIDENCE-ELSEWHERE (the score landed in the
  run dir but the gate-dir copy was never committed — no queued task
  re-creates it, so the fix is a box-side commit, not queue progress).
  A missing input with no queued producer at all is NO-PRODUCER.
- A pending score/view task's prerequisite is its run trained: a
  pending train task for the same run therefore touches every decision
  its score task touches (transitive edge). Record tasks never feed a
  gate or compare — gates read gate-dir JSONs, not manifests — so they
  rank at zero decisions by construction, not by accident.

Output: per-decision missing inputs with their producer tasks, the
ranked resume order (decisions touched, then queue order), and the
EVIDENCE-ELSEWHERE / NO-PRODUCER flags. Exit 0 whenever the ranking
could be produced; a missing or malformed tasks.json is a hard error
(exit 2) because without the queue the ranking is meaningless.

Usage:
    uv run python scripts/queue_critical_path.py [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VIEWS = ("spatial", "fault", "trace")
KNOWN_VERDICTS = ("ACCEPT", "REJECT", "INCONCLUSIVE")


def _tokens(cmd: str) -> list[str]:
    # Only the leading tokens matter (script, dirs, config); the
    # trailing quoted hypothesis may contain spaces and escaped quotes, so
    # a plain split is both sufficient and safer than shlex here.
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
                # Output run; the prerequisite is the INPUT run (tokens[idx+1]),
                # whose fold predictions the view step transforms.
                "run": tokens[idx + 2],
                "input_run": tokens[idx + 1],
                "scheme": tokens[idx + 3],
                "gate_dir": gate,
            }
    idx = _find_script_token(tokens, "train_folds.sh")
    if idx is not None and len(tokens) >= idx + 6:
        return {
            "kind": "train",
            "task_id": task_id,
            "run": tokens[idx + 1],
            "scheme": tokens[idx + 3],
            "gate_dir": None,
        }
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
        return {
            "kind": "record",
            "task_id": task_id,
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
        "dest": dest,
        # (source dir, view) per copied file, e.g. runs/gate/ref-full/spatial.json.
        "sources": [(str(Path(s).parent), Path(s).stem) for s in sources],
        "files": [Path(s).name for s in sources],
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
        if idx is not None and len(tokens) >= idx + 5:
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
                "incumbent_dir": tokens[idx + 1],
                "candidate_dir": tokens[idx + 2],
                "incumbent_base": tokens[idx + 3],
                "candidate_base": tokens[idx + 4],
                "seed": basename == "compare_seed.sh",
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


def _production_status(root: Path, prod: dict) -> str:
    run = prod["run"]
    if _exists_in(root, run, "manifest.json"):
        return "RECORDED"
    if _score_present(root, prod):
        return "SCORED"
    if _fold_count(root, run):
        return "TRAINED"
    return "PENDING"


def _assemble_status(root: Path, task: dict) -> str:
    present = [f for f in task["files"] if _exists_in(root, task["dest"], f)]
    if len(present) == len(task["files"]):
        return "DONE"
    if present:
        return "PARTIAL"
    return "PENDING"


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


def _compare_ran(root: Path, compare: dict) -> bool:
    if compare["compare_dir"] is None:
        return False
    return _candidate_dirs(root, compare["compare_dir"])[0].joinpath("compare.log").exists()


class CriticalPath:
    """Dependency graph from pending tasks to pending decisions."""

    def __init__(self, root: Path, tasks: list[dict]):
        self.root = root
        self.tasks = tasks
        self.production: list[dict] = []
        self.assembles: list[dict] = []
        self.gates: list[dict] = []
        self.compares: list[dict] = []
        self.unparsed = 0
        queue_dir = root / "scripts" / "queue"
        for index, task in enumerate(tasks):
            parsed = (
                _parse_production_task(task)
                or _parse_assemble_task(task)
                or _parse_gate_task(task)
                or _parse_compare_task(task, queue_dir)
            )
            if parsed is None:
                self.unparsed += 1
                continue
            parsed["index"] = index
            if parsed["kind"] in ("train", "score", "view", "record"):
                parsed["status"] = _production_status(root, parsed)
                self.production.append(parsed)
            elif parsed["kind"] == "assemble":
                parsed["status"] = _assemble_status(root, parsed)
                self.assembles.append(parsed)
            elif parsed["kind"] == "gate":
                self.gates.append(parsed)
            else:
                self.compares.append(parsed)
        # Indexes over pending work only; already-run tasks are evidence,
        # not producers, and are reported as EVIDENCE-ELSEWHERE instead.
        self.score_by_gate: dict[tuple[str, str], list[dict]] = {}
        self.score_by_run: dict[tuple[str, str], list[dict]] = {}
        self.train_by_run: dict[str, list[dict]] = {}
        for prod in self.production:
            if prod["kind"] in ("score", "view") and prod["gate_dir"]:
                self.score_by_gate.setdefault((prod["gate_dir"], prod["scheme"]), []).append(prod)
            if prod["kind"] in ("score", "view"):
                self.score_by_run.setdefault((prod["run"], prod["scheme"]), []).append(prod)
            if prod["kind"] == "train":
                self.train_by_run.setdefault(prod["run"], []).append(prod)
        self.assemble_by_dest: dict[str, list[dict]] = {}
        for task in self.assembles:
            self.assemble_by_dest.setdefault(task["dest"], []).append(task)

    @staticmethod
    def _needs_run(task: dict) -> bool:
        if task["kind"] == "assemble":
            return task["status"] != "DONE"
        return task["status"] == "PENDING"

    def _gate_input_producers(self, gate_dir: str, view: str) -> tuple[list[dict], list[str]]:
        """Pending tasks feeding a missing (gate_dir, view) JSON, plus flags.

        Flags: EVIDENCE-ELSEWHERE (a producer already ran but the gate-dir
        copy was never committed), NO-PRODUCER (no queued task makes it).
        """
        producers: list[dict] = []
        flags: list[str] = []
        scores = self.score_by_gate.get((gate_dir, view), [])
        pending_scores = [t for t in scores if self._needs_run(t)]
        if pending_scores:
            producers.extend(pending_scores)
        elif scores:
            flags.append("EVIDENCE-ELSEWHERE")
        for task in self.assemble_by_dest.get(gate_dir, []):
            if view + ".json" not in task["files"]:
                continue
            if self._needs_run(task):
                producers.append(task)
                # The assemble's own prerequisites: a missing source file
                # whose producer score task is still pending also feeds it.
                for src_dir, src_view in task["sources"]:
                    if src_view != view or _exists_in(self.root, src_dir, view + ".json"):
                        continue
                    src_scores = self.score_by_gate.get((src_dir, src_view), [])
                    pending_src = [t for t in src_scores if self._needs_run(t)]
                    if pending_src:
                        producers.extend(pending_src)
                    elif src_scores:
                        flags.append("EVIDENCE-ELSEWHERE")
                    else:
                        flags.append("NO-PRODUCER")
            elif task["status"] == "PARTIAL":
                pass  # PARTIAL needs a run, handled by _needs_run above.
        if not producers and not flags:
            flags.append("NO-PRODUCER")
        return producers, sorted(set(flags))

    def _lofo_producers(self, run_dir: str, view: str) -> tuple[list[dict], list[str]]:
        """Pending score tasks feeding a missing LOFO summary, plus flags."""
        if _exists_in(self.root, run_dir, f"scores/lofo-{view}.json"):
            return [], []
        scores = self.score_by_run.get((run_dir, view), [])
        pending = [t for t in scores if self._needs_run(t)]
        if pending:
            return pending, []
        if scores:
            return [], ["EVIDENCE-ELSEWHERE"]
        return [], ["NO-PRODUCER"]

    def _expand_producers(self, producers: list[dict]) -> set[str]:
        """Task ids feeding a decision, including transitive train edges."""
        ids = {t["task_id"] for t in producers}
        for task in producers:
            if task["kind"] in ("score", "view"):
                for train in self.train_by_run.get(task["run"], []):
                    if self._needs_run(train):
                        ids.add(train["task_id"])
        return ids

    def analyze(self) -> dict:
        decisions: list[dict] = []
        touches: dict[str, set[str]] = {}
        for gate in self.gates:
            decided, verdict = _gate_decided(self.root, gate)
            if decided or verdict == "BROKEN":
                continue
            missing: list[dict] = []
            feeders: set[str] = set()
            for side, gate_dir in (
                ("incumbent", gate["incumbent_dir"]),
                ("candidate", gate["candidate_dir"]),
            ):
                for view in VIEWS:
                    if _exists_in(self.root, gate_dir, view + ".json"):
                        continue
                    producers, flags = self._gate_input_producers(gate_dir, view)
                    missing.append(
                        {
                            "input": f"{side} {gate_dir}/{view}.json",
                            "producers": [t["task_id"] for t in producers],
                            "flags": flags,
                        }
                    )
                    feeders |= self._expand_producers(producers)
            decisions.append(
                {"kind": "gate", "name": gate["name"], "missing": missing, "feeders": feeders}
            )
            touches.setdefault(gate["name"], set()).update(feeders)
        for compare in self.compares:
            if _compare_ran(self.root, compare):
                continue
            missing: list[dict] = []
            feeders: set[str] = set()
            for side, gate_dir in (
                ("incumbent", compare["incumbent_dir"]),
                ("candidate", compare["candidate_dir"]),
            ):
                for view in VIEWS:
                    if _exists_in(self.root, gate_dir, view + ".json"):
                        continue
                    producers, flags = self._gate_input_producers(gate_dir, view)
                    effort = "best-effort" if compare["seed"] else "required"
                    missing.append(
                        {
                            "input": f"{side} {gate_dir}/{view}.json ({effort})",
                            "producers": [t["task_id"] for t in producers],
                            "flags": flags,
                        }
                    )
                    feeders |= self._expand_producers(producers)
            # LOFO summaries: incumbent base expands per view; the
            # candidate side is a single run for seed compares, a base
            # for schedule compares.
            cand_run = compare["candidate_base"]
            cand_scheme = next(
                (v for v in VIEWS if f"-{v}-" in cand_run or cand_run.endswith(f"-{v}")),
                None,
            )
            for view in VIEWS:
                for side, run_dir in (
                    ("incumbent", f"{compare['incumbent_base']}-{view}-b16"),
                    ("candidate", cand_run if compare["seed"] else f"{cand_run}-{view}-b16"),
                ):
                    if compare["seed"] and side == "candidate" and cand_scheme != view:
                        continue  # Seed probe is spatial-only on this side.
                    producers, flags = self._lofo_producers(run_dir, view)
                    if producers or flags:
                        effort = "best-effort" if compare["seed"] else "required"
                        missing.append(
                            {
                                "input": f"{side} LOFO {run_dir}/scores/lofo-{view}.json ({effort})",
                                "producers": [t["task_id"] for t in producers],
                                "flags": flags,
                            }
                        )
                        feeders |= self._expand_producers(producers)
            decisions.append(
                {
                    "kind": "compare",
                    "name": compare["name"],
                    "missing": missing,
                    "feeders": feeders,
                }
            )
            touches.setdefault(compare["name"], set()).update(feeders)
        # Rank every task that still needs a run by decisions touched.
        ranked: list[dict] = []
        by_id: dict[str, dict] = {}
        for task in self.production + self.assembles:
            by_id[task["task_id"]] = task
        touched: dict[str, set[str]] = {}
        for decision in decisions:
            for task_id in decision["feeders"]:
                touched.setdefault(task_id, set()).add(decision["name"])
        for task in self.production + self.assembles:
            if not self._needs_run(task):
                continue
            names = touched.get(task["task_id"], set())
            # "Runnable now" is judged from committed state only. A
            # pending score task can never run before its train task
            # (any run evidence would move it out of PENDING), so scores
            # name their prerequisite instead of claiming readiness. A
            # view task's prerequisite is its INPUT run's training, not
            # its output run's folds.
            if task["kind"] == "train":
                runnable, note = True, "runnable now"
            elif task["kind"] == "score":
                trains = [t["task_id"] for t in self.train_by_run.get(task["run"], [])]
                note = f"after {'/'.join(trains) if trains else 'training ' + task['run']}"
                runnable = False
            elif task["kind"] == "view":
                input_run = task.get("input_run", task["run"])
                runnable = (
                    _exists_in(self.root, input_run, "manifest.json")
                    or _score_present(
                        self.root, {"run": input_run, "scheme": None, "gate_dir": None}
                    )
                    is not None
                    or _fold_count(self.root, input_run) > 0
                )
                note = "runnable now" if runnable else f"after training {input_run}"
            elif task["kind"] == "assemble":
                missing_sources = [
                    f"{src_dir}/{src_view}.json"
                    for src_dir, src_view in task["sources"]
                    if not _exists_in(self.root, src_dir, src_view + ".json")
                ]
                runnable = not missing_sources
                note = "runnable now" if runnable else "blocked: " + ", ".join(missing_sources)
            else:  # record tasks never feed a decision
                runnable, note = False, "no decision impact"
            ranked.append(
                {
                    "task_id": task["task_id"],
                    "kind": task["kind"],
                    "decisions": sorted(names),
                    "runnable": runnable,
                    "note": note,
                    "index": task["index"],
                }
            )
        ranked.sort(key=lambda r: (-len(r["decisions"]), r["index"]))
        return {
            "decisions": decisions,
            "ranked": ranked,
            "pending_gates": sum(1 for d in decisions if d["kind"] == "gate"),
            "pending_compares": sum(1 for d in decisions if d["kind"] == "compare"),
            "decided_gates": sum(1 for g in self.gates if _gate_decided(self.root, g)[0]),
            "unparsed": self.unparsed,
            "by_id": by_id,
        }


def render(analysis: dict) -> str:
    lines = [
        (
            f"PENDING DECISIONS: {analysis['pending_gates']} gates, "
            f"{analysis['pending_compares']} compares "
            f"({analysis['decided_gates']} gates decided, "
            f"{analysis['unparsed']} unparsed tasks ignored)"
        )
    ]
    for decision in analysis["decisions"]:
        lines.append(
            f"\n{decision['kind'].upper()} {decision['name']}: "
            f"{len(decision['missing'])} missing inputs"
        )
        for item in decision["missing"]:
            producers = ", ".join(item["producers"]) if item["producers"] else "-"
            flags = f" [{'; '.join(item['flags'])}]" if item["flags"] else ""
            lines.append(f"  MISSING {item['input']} <- {producers}{flags}")
    lines.append("\nRESUME ORDER (pending tasks by decisions unblocked):")
    for rank in analysis["ranked"]:
        if not rank["decisions"]:
            continue
        lines.append(
            f"  {rank['task_id']} [{rank['kind']}] "
            f"unblocks {len(rank['decisions'])}: {', '.join(rank['decisions'])} "
            f"({rank['note']})"
        )
    zero = [r for r in analysis["ranked"] if not r["decisions"]]
    if zero:
        lines.append(
            f"\nNO DECISION IMPACT ({len(zero)} pending tasks, e.g. record "
            "tasks that never feed a gate/compare): "
            + ", ".join(r["task_id"] for r in zero[:10])
            + (" ..." if len(zero) > 10 else "")
        )
    duplicates: dict[str, list[str]] = {}
    for decision in analysis["decisions"]:
        for item in decision["missing"]:
            if len(item["producers"]) > 1:
                duplicates.setdefault(item["input"], item["producers"])
    if duplicates:
        lines.append("\nDUPLICATE PRODUCERS (running one suffices):")
        for item_input, producers in duplicates.items():
            lines.append(f"  {item_input} <- {', '.join(producers)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank pending queue tasks by decisions unblocked.")
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
    analysis = CriticalPath(args.root, tasks).analyze()
    print(render(analysis), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
