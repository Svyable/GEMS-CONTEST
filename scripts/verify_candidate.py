#!/usr/bin/env python3
"""Run the propose-and-verify acceptance gate for a GEMS experiment candidate.

Compares candidate CV scores against the incumbent on the three validation
views (spatial-block, fault-discovery, trace-completion) using the published
distance-weighted Tversky metric. Both directories must contain the
score_cv.py JSON outputs: spatial.json, fault.json, trace.json.

Acceptance requires a pooled spatial OOF gain above the heuristic margin and
comparable spatial, fault and trace non-regression safeguards. Fault/trace
backgrounds overlap and are never pooled. Re-score old outputs with the current
schema-2 score_cv.py. Every verdict is appended to the honest failure ledger.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from gems.verification import (
    count_trials,
    load_view_results,
    record_trial,
    verify_candidate,
)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Candidate name, e.g. MS-EDGE-01")
    parser.add_argument("--hypothesis", required=True, help="What the candidate was meant to show")
    parser.add_argument("--incumbent-dir", required=True, help="Dir of incumbent score_cv.py JSONs")
    parser.add_argument("--candidate-dir", required=True, help="Dir of candidate score_cv.py JSONs")
    parser.add_argument(
        "--ledger",
        default="docs/candidate-trials.jsonl",
        help="JSONL trial ledger (default: docs/candidate-trials.jsonl)",
    )
    parser.add_argument("--commit", default="", help="Git commit of the candidate run")
    parser.add_argument("--config", default="", help="Candidate config path")
    parser.add_argument("--config-sha256", default="", help="SHA256 of the candidate config")
    parser.add_argument(
        "--trials-before",
        type=int,
        default=None,
        help="Override the auto-counted number of prior trials",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Evaluate and print the report without writing to the ledger",
    )
    args = parser.parse_args()

    incumbent_results = load_view_results(args.incumbent_dir)
    candidate_results = load_view_results(args.candidate_dir)
    if not incumbent_results:
        print(f"error: no view JSONs found in {args.incumbent_dir}", file=sys.stderr)
        return 2
    if not candidate_results:
        print(f"error: no view JSONs found in {args.candidate_dir}", file=sys.stderr)
        return 2

    trials_before = (
        args.trials_before if args.trials_before is not None else count_trials(args.ledger)
    )

    report = verify_candidate(
        args.candidate,
        incumbent_results,
        candidate_results,
        trials_before=trials_before,
        hypothesis=args.hypothesis,
        commit=args.commit or _git_commit(),
        config=args.config,
        config_sha256=args.config_sha256,
    )

    if args.print_only:
        entry = dict(report)
        entry["trials_after"] = trials_before + 1
    else:
        entry = record_trial(args.ledger, report)

    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
