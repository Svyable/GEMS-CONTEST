#!/usr/bin/env python3
"""Render a committed gate verdict JSON as a docs/EXPERIMENT_LOG.md entry.

The box commits gate verdicts from scripts/queue/gate.sh to results/gates/
as the verify_candidate JSON schema; interpretation then drafts the log
entry by hand. This script emits that draft so verdict interpretation is
mechanical once box results land.

Usage:
    uv run python scripts/render_gate_verdict.py results/gates/CAND-XXX.json

Prints the markdown block matching the 2026-10-10 ENDPOINT-EXT-02 entry
style. Required keys: candidate, verdict (ACCEPT/REJECT/INCONCLUSIVE),
hypothesis, config, views[] (each with view, delta, required_margin, won,
comparable, reason). Non-comparable views are listed with their reason and
excluded from the win denominator.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REQUIRED_VERDICTS = ("ACCEPT", "REJECT", "INCONCLUSIVE")
EXPECTED_VIEWS = ("spatial", "fault", "trace")


def _fmt_delta(value: float) -> str:
    return f"{value:.2g}"


def _fmt_margin(value: float) -> str:
    return f"{value:.4g}"


def _load_gate(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot parse gate JSON {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit(f"error: gate JSON {path} is not an object")
    missing = [
        key for key in ("candidate", "verdict", "hypothesis", "config", "views")
        if key not in report
    ]
    if missing:
        raise SystemExit(f"error: gate JSON {path} missing keys: {', '.join(missing)}")
    if report["verdict"] not in REQUIRED_VERDICTS:
        raise SystemExit(
            f"error: unknown verdict {report['verdict']!r}; "
            f"expected one of {', '.join(REQUIRED_VERDICTS)}"
        )
    if not isinstance(report["views"], list) or not report["views"]:
        raise SystemExit(f"error: gate JSON {path} has no views")
    return report


def _view_line(view: dict, wins: int, denom: int) -> str:
    name = str(view.get("view", "?")).capitalize()
    if not view.get("comparable", False) or view.get("delta") is None:
        reason = view.get("reason") or "not comparable"
        return f"- **{name}**: not comparable — {reason}"
    delta = float(view["delta"])
    margin = float(view["required_margin"])
    relation = "above" if delta > margin else "below"
    return (
        f"- **{name}**: delta {_fmt_delta(delta)} "
        f"({relation} required margin {_fmt_margin(margin)}), won {wins}/{denom} views"
    )


def _display_path(path: Path) -> str:
    """Repo-relative when the gate JSON sits under the current tree."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def render(report: dict, gate_display_path: str) -> str:
    """Render the markdown log block for a gate report dict."""
    candidate = str(report["candidate"])
    verdict = str(report["verdict"])
    hypothesis = str(report["hypothesis"])
    config = str(report["config"])

    views = report["views"]
    comparable = [v for v in views if v.get("comparable") and v.get("delta") is not None]
    wins = sum(1 for v in comparable if v.get("won"))
    denom = len(comparable)

    timestamp = str(report.get("timestamp_utc") or "")
    try:
        date = datetime.datetime.fromisoformat(timestamp).date().isoformat()
    except ValueError:
        date = datetime.datetime.now(datetime.UTC).date().isoformat()

    view_names = ", ".join(str(v.get("view", "?")) for v in views)

    lines = [
        f"## {date} — Gate decision: {candidate} ({verdict})",
        "",
        f"**Candidate**: {hypothesis}  ",
        f"**Applied to**: gate-view scores ({view_names})  ",
        f"**Config**: `{config}`",
        "",
        f"**Results** ({len(views)} views):",
    ]
    lines.extend(_view_line(v, wins, denom) for v in views)
    lines.append("")

    if verdict == "ACCEPT":
        lines.append(
            f"**Gate verdict**: **ACCEPT**. {candidate} won on {wins} of {denom} views "
            "and cleared the required margin with safeguards intact; promote per "
            "docs/STRATEGY.md."
        )
    elif verdict == "REJECT":
        lines.append(
            f"**Gate verdict**: **REJECT**. {candidate} won on {wins} of {denom} views; "
            "deltas below required margins."
        )
    else:  # INCONCLUSIVE
        lines.append(
            f"**Gate verdict**: **INCONCLUSIVE**. {candidate} could not be judged on "
            f"{len(views) - denom} of {len(views)} views (see reasons above); not a "
            "rejection, not a promotion."
        )

    lines.append("")
    lines.append(
        f"**Results committed**: `{gate_display_path}`, appended to `docs/candidate-trials.jsonl`"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a gate verdict JSON as an EXPERIMENT_LOG.md entry."
    )
    parser.add_argument("gate_json", type=Path, help="results/gates/<NAME>.json")
    args = parser.parse_args(argv)

    report = _load_gate(args.gate_json)
    sys.stdout.write(render(report, _display_path(args.gate_json)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
