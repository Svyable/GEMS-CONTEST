#!/usr/bin/env python3
"""Evaluate candidate run via automated propose-and-verify experiment gate.

This implements roadmap priority #2: accept a candidate only if it beats the
incumbent on ≥2 of 3 CV views by more than fold-to-fold noise, with a
multiple-comparisons-aware bar that tightens as more candidates are tried.

Usage:
    uv run python scripts/evaluate_candidate.py \\
      --incumbent-id exp-baseline \\
      --candidate-id exp-new-feature \\
      --spatial-incumbent runs/exp-baseline/spatial-scores.json \\
      --spatial-candidate runs/exp-new-feature/spatial-scores.json \\
      --fault-incumbent runs/exp-baseline/fault-scores.json \\
      --fault-candidate runs/exp-new-feature/fault-scores.json \\
      --ledger data/experiment-ledger.json \\
      --output-summary results/exp-new-feature-decision.txt

Every decision (accept or reject) is appended to the ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gems.experiment_gate import (
    append_to_ledger,
    evaluate_candidate,
    format_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated propose-and-verify experiment gate"
    )
    parser.add_argument(
        "--incumbent-id",
        required=True,
        help="Run ID of incumbent (e.g., 'exp-baseline')",
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Run ID of candidate (e.g., 'exp-new-feature')",
    )
    parser.add_argument(
        "--spatial-incumbent",
        type=Path,
        help="Incumbent spatial CV scores JSON (from score_cv.py)",
    )
    parser.add_argument(
        "--spatial-candidate",
        type=Path,
        help="Candidate spatial CV scores JSON",
    )
    parser.add_argument(
        "--fault-incumbent",
        type=Path,
        help="Incumbent fault-discovery CV scores JSON",
    )
    parser.add_argument(
        "--fault-candidate",
        type=Path,
        help="Candidate fault-discovery CV scores JSON",
    )
    parser.add_argument(
        "--trace-incumbent",
        type=Path,
        help="Incumbent trace-completion CV scores JSON",
    )
    parser.add_argument(
        "--trace-candidate",
        type=Path,
        help="Candidate trace-completion CV scores JSON",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="Path to append-only trial ledger JSON",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        help="Optional path to write human-readable decision summary",
    )
    parser.add_argument(
        "--base-alpha",
        type=float,
        default=0.05,
        help="Base significance level before Bonferroni correction (default: 0.05)",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
        help="Minimum views candidate must win (default: 2 of 3)",
    )
    
    args = parser.parse_args()
    
    # Collect provided view scores
    view_scores = {}
    
    if args.spatial_incumbent and args.spatial_candidate:
        if not args.spatial_incumbent.exists():
            print(f"Error: {args.spatial_incumbent} not found", file=sys.stderr)
            return 1
        if not args.spatial_candidate.exists():
            print(f"Error: {args.spatial_candidate} not found", file=sys.stderr)
            return 1
        view_scores["spatial"] = (args.spatial_incumbent, args.spatial_candidate)
    
    if args.fault_incumbent and args.fault_candidate:
        if not args.fault_incumbent.exists():
            print(f"Error: {args.fault_incumbent} not found", file=sys.stderr)
            return 1
        if not args.fault_candidate.exists():
            print(f"Error: {args.fault_candidate} not found", file=sys.stderr)
            return 1
        view_scores["fault-discovery"] = (args.fault_incumbent, args.fault_candidate)
    
    if args.trace_incumbent and args.trace_candidate:
        if not args.trace_incumbent.exists():
            print(f"Error: {args.trace_incumbent} not found", file=sys.stderr)
            return 1
        if not args.trace_candidate.exists():
            print(f"Error: {args.trace_candidate} not found", file=sys.stderr)
            return 1
        view_scores["trace-completion"] = (args.trace_incumbent, args.trace_candidate)
    
    if not view_scores:
        print("Error: No view scores provided", file=sys.stderr)
        print("Provide at least one of: --spatial-*, --fault-*, --trace-*", file=sys.stderr)
        return 1
    
    if len(view_scores) < 2 and args.min_views >= 2:
        print(
            f"Warning: Only {len(view_scores)} view(s) provided, "
            f"but min_views={args.min_views}",
            file=sys.stderr,
        )
    
    print(f"Evaluating candidate: {args.candidate_id}")
    print(f"Against incumbent: {args.incumbent_id}")
    print(f"Views: {', '.join(view_scores.keys())}")
    print()
    
    try:
        decision = evaluate_candidate(
            args.incumbent_id,
            args.candidate_id,
            view_scores,
            args.ledger,
            base_alpha=args.base_alpha,
            min_views_to_win=args.min_views,
        )
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        return 1
    
    # Append to ledger
    try:
        append_to_ledger(decision, args.ledger)
        print(f"Decision appended to ledger: {args.ledger}")
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error writing ledger: {e}", file=sys.stderr)
        return 1
    
    # Print summary
    summary = format_summary(decision)
    print()
    print(summary)
    
    # Optionally save summary
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_summary, "w") as f:
            f.write(summary)
        print()
        print(f"Summary written to: {args.output_summary}")
    
    # Return code: 0 for accept, 1 for reject
    return 0 if decision.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
