#!/usr/bin/env python3
"""Audit argument consistency of queued commands: values, not just files.

The refs audit (step 38) checks that every wrapper, config, and script a
queued command *names* still exists in the repo. It cannot catch a command
whose arguments are all real files but wrong for each other: a trace train
scored against the spatial fold map, a buffer-16 run dir fed buffer 32, a
record whose run id does not match its run dir, a gate name whose buffer
suffix disagrees with its candidate dir. Any of those fails on the box —
where the runner stops the whole queue on the first non-zero exit — only
after resumption, burning the drain margin the forecast measures (step 34).

This audit parses each queued command positionally (shlex, per `&&`
segment) and checks every argument against the wrapper's documented
contract (the `# usage:` line in each `scripts/queue/*.sh`) and against
the other arguments of the same command:

- train_folds.sh RUN CONFIG SCHEME FOLD_MAP BUFFER [SEED]: scheme must be
  one of spatial/fault/trace (the argparse choices in train_full_map.py
  and score_cv.py); the fold map must be the *current* map for that
  scheme; the buffer must be a positive integer and must match a `-b<N>`
  suffix on the run dir when one is present; the run dir must carry the
  scheme; an explicit seed must be eight digits; the config must match
  the run dir (reference_unet.yaml for ref-cv-* runs,
  cand_<slug>.yaml for cand-<slug>-* runs).
- score_view.sh RUN SCHEME FOLD_MAP GATE_DIR: same scheme / fold-map /
  scheme-in-run-dir checks; the gate dir must sit under runs/gate/.
- record.sh RUN_ID RUN_DIR CONFIG ...: the run id must equal the run
  dir's basename; the config must match the run dir as above.
- gate.sh NAME INC_GATE CAND_GATE CONFIG ...: the name must match
  CAND-<SLUG>-B16|B32|TRACE or ENDPOINT-EXT-<n>; both gate dirs must sit
  under runs/gate/; the name's slug must appear in the candidate dir and
  a -B16/-B32 suffix must agree with the dir's -b16/-b32 suffix; the
  config must match the candidate dir's slug (CAND-* gates only).
- endpoint_view.sh / curved_view.sh BASE_RUN OUT_RUN SCHEME: scheme
  valid, both run dirs carry the scheme.
- rival_threshold.sh RUN SCHEME FOLD_MAP EXPECTED [FACTOR]: scheme /
  fold-map checks; the expected score must be a float in (0, 1).
- compare_seed.sh / compare_schedule.sh INC_GATE CAND_GATE INC_BASE
  CAND_BASE: gate dirs under runs/gate/, run bases under runs/. A 5th
  note argument is tolerated: the wrappers ignore it (the compare
  scripts take no hypothesis), so it is dropped, not mis-recorded.

Two quoting traps this audit models from how the runner executes
commands (`subprocess.call(["bash", "-c", cmd])`):

- A hypothesis written with `\"` escapes (as the s10/s11 gate tasks
  were) is not one argument under bash: `\"` outside quotes becomes a
  literal `"` character, so the hypothesis splits into separate words
  and the wrapper's $5 is only its first word (`"Reference`). The audit
  reports ESCAPED-QUOTE and checks the remaining arguments against the
  hypothesis as written (with the escapes normalized), so a mangled
  command cannot hide a second defect.
- A `\'` inside a double-quoted note is harmless: bash keeps the
  backslash, the note stays one argument, and the compare wrappers drop
  the 5th argument anyway.

Fold-map currency is the sharp check. The trace map `cv-trace-v1.tif`
is the pre-repair map: it crashed the trace baseline on fold 1
(2026-10-10, withheld label pixel row 3264 col 1516 outside the feature
valid mask, no retained trace body) and was superseded by
`cv-trace-v1b.tif` (that pixel set to -1, fold 0 verified identical).
Every trace task queued since uses v1b — except the s11/s12 trace
trains and scores, which still name v1. A train on v1 would crash at
fold 1 and halt the queue; even a clean run would be protocol-unequal
to ref-b16 (all v1b), so its gate could not be decided.

Deliberately out of scope: whether named files exist (the refs audit
owns that), task order (the order audit, step 33), recorded hashes
(the config-drift audit, step 35), and box-side state — `data/*`
fold maps are never committed, so currency is judged against the
registry below, which cites the supersession in the experiment log.
Double-quoted hypothesis/notes arguments are parsed positionally, never
read as paths, so free text cannot produce false positives.

Findings:

- STALE-FOLD-MAP / BAD-SCHEME / ARG-COUNT / BAD-BUFFER /
  BUFFER-SUFFIX-MISMATCH / SCHEME-DIR-MISMATCH / BAD-SEED /
  CONFIG-DIR-MISMATCH / RECORD-ID-MISMATCH / BAD-GATE-NAME /
  GATE-DIR-MISMATCH / BAD-GATE-DIR / BAD-EXPECTED-SCORE /
  UNKNOWN-WRAPPER / ARG-PARSE-ERROR / ESCAPED-QUOTE: exit 1.
- UNREADABLE-QUEUE: a queue file cannot be read or parsed. Exit 2.
- UNPARSED (note, exit 0): a task whose command names no known wrapper
  at all (e.g. the s1 wait-for-existing-loop task, the mkdir/cp
  assemble tasks).

Usage:
    uv run python scripts/audit_queue_args.py [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

SCHEMES = ("spatial", "fault", "trace")

# Current fold map per scheme. Trace is v1b, not v1: v1 is the pre-repair
# map superseded 2026-10-10 (EXPERIMENT_LOG: fold-1 crash, pixel
# row 3264 col 1516 outside the valid mask). Spatial/fault are still v1.
CURRENT_FOLD_MAP = {
    "spatial": "data/processed/cv-spatial-v1.tif",
    "fault": "data/processed/cv-fault-v1.tif",
    "trace": "data/processed/cv-trace-v1b.tif",
}

_GATE_NAME_RE = re.compile(r"^CAND-[A-Z0-9_]+-(?:B16|B32|TRACE)$|^ENDPOINT-EXT-\d+$")
_BUFFER_SUFFIX_RE = re.compile(r"-b(\d+)(?:-|$)")
_SEED_RE = re.compile(r"^\d{8}$")

_QUEUE_FILES = (("live", "tasks.json"), ("parked", "tasks.parked.json"))

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


def _findings(source: str, findings: list[dict], kind: str, ref: str) -> None:
    findings.append({"source": source, "kind": kind, "ref": ref})


def _check_scheme(source: str, scheme: str, findings: list[dict]) -> bool:
    if scheme not in SCHEMES:
        _findings(source, findings, "BAD-SCHEME", f"scheme {scheme!r} not in {SCHEMES}")
        return False
    return True


def _check_fold_map(source: str, scheme: str, fmap: str, findings: list[dict]) -> None:
    expected = CURRENT_FOLD_MAP[scheme]
    if fmap != expected:
        _findings(
            source,
            findings,
            "STALE-FOLD-MAP",
            f"{fmap} (scheme {scheme} expects {expected})",
        )


def _check_scheme_in_run_dir(source: str, scheme: str, run_dir: str, findings: list[dict]) -> None:
    base = Path(run_dir).name
    if f"-{scheme}-" not in base and not base.endswith(f"-{scheme}"):
        _findings(
            source,
            findings,
            "SCHEME-DIR-MISMATCH",
            f"run dir {run_dir} carries no -{scheme}- segment",
        )


def _check_config_matches_run_dir(
    source: str, run_dir: str, config: str, findings: list[dict]
) -> None:
    """ref-cv-* runs use reference_unet.yaml; cand-<slug>-* runs use cand_<slug>.yaml."""
    base = Path(run_dir).name
    if base.startswith("ref-cv-"):
        expected = "configs/reference_unet.yaml"
        if config != expected:
            _findings(
                source,
                findings,
                "CONFIG-DIR-MISMATCH",
                f"run dir {run_dir} expects {expected}, got {config}",
            )
    elif base.startswith("cand-"):
        slug = base[len("cand-") :].split("-")[0]
        expected = f"configs/candidates/cand_{slug}.yaml"
        if config != expected:
            _findings(
                source,
                findings,
                "CONFIG-DIR-MISMATCH",
                f"run dir {run_dir} expects {expected}, got {config}",
            )


def _check_train(source: str, args: list[str], findings: list[dict]) -> None:
    if len(args) not in (5, 6):
        _findings(source, findings, "ARG-COUNT", f"train_folds.sh takes 5-6 args, got {len(args)}")
        return
    run_dir, config, scheme, fmap, buf = args[:5]
    if not _check_scheme(source, scheme, findings):
        return
    _check_fold_map(source, scheme, fmap, findings)
    _check_scheme_in_run_dir(source, scheme, run_dir, findings)
    _check_config_matches_run_dir(source, run_dir, config, findings)
    if not buf.isdigit() or int(buf) <= 0:
        _findings(source, findings, "BAD-BUFFER", f"buffer {buf!r} is not a positive integer")
    else:
        match = _BUFFER_SUFFIX_RE.search(Path(run_dir).name)
        if match and int(match.group(1)) != int(buf):
            _findings(
                source,
                findings,
                "BUFFER-SUFFIX-MISMATCH",
                f"run dir {run_dir} says b{match.group(1)}, buffer arg is {buf}",
            )
    if len(args) == 6 and not _SEED_RE.match(args[5]):
        _findings(source, findings, "BAD-SEED", f"seed {args[5]!r} is not eight digits")


def _check_score(source: str, args: list[str], findings: list[dict]) -> None:
    if len(args) != 4:
        _findings(source, findings, "ARG-COUNT", f"score_view.sh takes 4 args, got {len(args)}")
        return
    run_dir, scheme, fmap, gate_dir = args
    if not _check_scheme(source, scheme, findings):
        return
    _check_fold_map(source, scheme, fmap, findings)
    _check_scheme_in_run_dir(source, scheme, run_dir, findings)
    if not gate_dir.startswith("runs/gate/"):
        _findings(source, findings, "BAD-GATE-DIR", f"gate dir {gate_dir} not under runs/gate/")


def _check_record(source: str, args: list[str], findings: list[dict]) -> None:
    if len(args) != 5:
        _findings(source, findings, "ARG-COUNT", f"record.sh takes 5 args, got {len(args)}")
        return
    run_id, run_dir, config = args[:3]
    if run_id != Path(run_dir).name:
        _findings(
            source,
            findings,
            "RECORD-ID-MISMATCH",
            f"run id {run_id} != run dir basename {Path(run_dir).name}",
        )
    _check_config_matches_run_dir(source, run_dir, config, findings)


def _check_gate(source: str, args: list[str], findings: list[dict]) -> None:
    if len(args) != 5:
        _findings(source, findings, "ARG-COUNT", f"gate.sh takes 5 args, got {len(args)}")
        return
    name, inc_gate, cand_gate, config = args[:4]
    if not _GATE_NAME_RE.match(name):
        _findings(source, findings, "BAD-GATE-NAME", f"gate name {name!r} matches no known pattern")
        return
    for gate_dir in (inc_gate, cand_gate):
        if not gate_dir.startswith("runs/gate/"):
            _findings(source, findings, "BAD-GATE-DIR", f"gate dir {gate_dir} not under runs/gate/")
    if not name.startswith("CAND-"):
        return
    slug = name[len("CAND-") :].rsplit("-", 1)[0].lower()
    cand_base = Path(cand_gate).name
    if slug not in cand_base and cand_base.removeprefix("cand-").split("-")[0] not in slug:
        _findings(
            source,
            findings,
            "GATE-DIR-MISMATCH",
            f"gate name {name} slug {slug!r} not in candidate dir {cand_gate}",
        )
    if name.endswith(("-B16", "-B32")):
        want = "b16" if name.endswith("-B16") else "b32"
        match = _BUFFER_SUFFIX_RE.search(cand_base)
        if match and f"b{match.group(1)}" != want:
            _findings(
                source,
                findings,
                "GATE-DIR-MISMATCH",
                f"gate name {name} expects -{want} candidate dir, got {cand_gate}",
            )
    if cand_base.startswith("cand-"):
        dir_slug = cand_base[len("cand-") :].split("-")[0]
        expected = f"configs/candidates/cand_{dir_slug}.yaml"
        if config != expected:
            _findings(
                source,
                findings,
                "CONFIG-DIR-MISMATCH",
                f"candidate dir {cand_gate} expects {expected}, got {config}",
            )


def _check_view(source: str, wrapper: str, args: list[str], findings: list[dict]) -> None:
    if len(args) != 3:
        _findings(source, findings, "ARG-COUNT", f"{wrapper} takes 3 args, got {len(args)}")
        return
    base_run, out_run, scheme = args
    if not _check_scheme(source, scheme, findings):
        return
    _check_scheme_in_run_dir(source, scheme, base_run, findings)
    _check_scheme_in_run_dir(source, scheme, out_run, findings)


def _check_rival(source: str, args: list[str], findings: list[dict]) -> None:
    if len(args) not in (4, 5):
        _findings(
            source, findings, "ARG-COUNT", f"rival_threshold.sh takes 4-5 args, got {len(args)}"
        )
        return
    run_dir, scheme, fmap, expected = args[:4]
    if not _check_scheme(source, scheme, findings):
        return
    _check_fold_map(source, scheme, fmap, findings)
    _check_scheme_in_run_dir(source, scheme, run_dir, findings)
    try:
        value = float(expected)
    except ValueError:
        _findings(
            source, findings, "BAD-EXPECTED-SCORE", f"expected score {expected!r} is not a number"
        )
        return
    if not 0.0 < value < 1.0:
        _findings(
            source,
            findings,
            "BAD-EXPECTED-SCORE",
            f"expected score {expected} not in (0, 1)",
        )
    if len(args) == 5:
        try:
            factor = float(args[4])
        except ValueError:
            _findings(source, findings, "BAD-EXPECTED-SCORE", f"factor {args[4]!r} is not a number")
            return
        if factor <= 0:
            _findings(source, findings, "BAD-EXPECTED-SCORE", f"factor {args[4]} not positive")


def _check_compare(source: str, wrapper: str, args: list[str], findings: list[dict]) -> None:
    if len(args) not in (4, 5):
        _findings(source, findings, "ARG-COUNT", f"{wrapper} takes 4 args, got {len(args)}")
        return
    inc_gate, cand_gate, inc_base, cand_base = args[:4]
    for gate_dir in (inc_gate, cand_gate):
        if not gate_dir.startswith("runs/gate/"):
            _findings(source, findings, "BAD-GATE-DIR", f"gate dir {gate_dir} not under runs/gate/")
    for base in (inc_base, cand_base):
        if not base.startswith("runs/"):
            _findings(source, findings, "BAD-GATE-DIR", f"run base {base} not under runs/")


_CHECKERS = {
    "train_folds.sh": _check_train,
    "score_view.sh": _check_score,
    "record.sh": _check_record,
    "gate.sh": _check_gate,
    "endpoint_view.sh": lambda s, a, f: _check_view(s, "endpoint_view.sh", a, f),
    "curved_view.sh": lambda s, a, f: _check_view(s, "curved_view.sh", a, f),
    "rival_threshold.sh": _check_rival,
    "compare_seed.sh": lambda s, a, f: _check_compare(s, "compare_seed.sh", a, f),
    "compare_schedule.sh": lambda s, a, f: _check_compare(s, "compare_schedule.sh", a, f),
}


def check_command(source: str, cmd: str, findings: list[dict]) -> int:
    """Check one queue command; returns the number of wrapper segments checked.

    A command may chain segments with `&&` (e.g. the s3 endpoint tasks run
    a view wrapper then a score). Segments naming no known wrapper (mkdir/cp
    assemble lines, the s1 wait loop) are skipped here and reported as
    UNPARSED by the caller when the whole command has none.
    """
    checked = 0
    for segment in cmd.split("&&"):
        segment = segment.strip()
        if not segment:
            continue
        if '\\"' in segment:
            # The runner executes via `bash -c`: a \" outside quotes is a
            # literal quote character, so a hypothesis written this way
            # splits into separate words and the wrapper's $5 is only its
            # first word. Normalize for the checks below so a mangled
            # command cannot hide a second defect; the finding itself is
            # the defect.
            _findings(
                source,
                findings,
                "ESCAPED-QUOTE",
                f'{segment[:80]}: hypothesis uses \\" escapes — bash splits '
                "it into words, the wrapper's $5 is only the first word; "
                'write plain "..." quotes',
            )
            segment = segment.replace('\\"', '"')
        try:
            tokens = shlex.split(segment)
        except ValueError as exc:
            _findings(source, findings, "ARG-PARSE-ERROR", f"{segment[:80]}: {exc}")
            continue
        if not tokens:
            continue
        wrapper = Path(tokens[0]).name
        if wrapper not in _KNOWN_WRAPPERS:
            continue
        checked += 1
        _CHECKERS[wrapper](source, tokens[1:], findings)
    return checked


def audit(root: Path) -> dict:
    findings: list[dict] = []
    tasks_checked = 0
    segments_checked = 0
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
            per_file[label] = {"tasks": 0, "segments": 0, "unparsed": 0}
            continue
        file_segments = 0
        file_unparsed = 0
        for task in tasks:
            task_id = task.get("id", "?")
            source = f"{label}:{task_id}"
            n = check_command(source, task.get("cmd", ""), findings)
            if n == 0:
                file_unparsed += 1
                unparsed.append(source)
            file_segments += n
        per_file[label] = {
            "tasks": len(tasks),
            "segments": file_segments,
            "unparsed": file_unparsed,
        }
        tasks_checked += len(tasks)
        segments_checked += file_segments

    return {
        "tasks": tasks_checked,
        "segments": segments_checked,
        "per_file": per_file,
        "unparsed": unparsed,
        "unreadable": unreadable,
        "findings": findings,
    }


def render(result: dict) -> str:
    per = result["per_file"]
    lines = [
        (
            f"QUEUE ARG AUDIT: {result['tasks']} tasks "
            f"({per.get('live', {}).get('tasks', 0)} live, "
            f"{per.get('parked', {}).get('tasks', 0)} parked), "
            f"{result['segments']} wrapper segments checked "
            f"({len(result['unparsed'])} unparsed)"
        )
    ]
    findings = result["findings"]
    other = [f for f in findings if f["kind"] != "UNREADABLE-QUEUE"]
    lines.append(f"{len(other)} finding(s), {result['unreadable']} unreadable queue file(s)")
    for finding in findings:
        lines.append(f"  {finding['kind']} {finding['source']}: {finding['ref']}")
    for task in result["unparsed"]:
        lines.append(f"  NOTE {task}: no known wrapper (wait/shell task)")
    if not findings:
        lines.append(
            "  OK: every queued command's arguments are consistent with its "
            "wrapper's contract and with each other."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit argument consistency of queued commands.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = audit(args.root)
    print(render(result), end="")
    if result["unreadable"]:
        return 2
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
