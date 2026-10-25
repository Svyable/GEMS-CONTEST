#!/usr/bin/env python3
"""Audit LOFO threshold calibration across committed CV runs.

Every honest CV run in results/runs/ carries a LOFO file
(results/runs/<run>/lofo-<scheme>.json or .../scores/lofo-<scheme>.json)
with one exact-threshold fit per fold: the threshold chosen on the other
folds, the raw score at the default operating point, and the calibrated
score at that threshold. The headline number the repo reports is the
calibrated mean — so the question this audit answers, in one place, is
how much of each headline is threshold fitting and how stable the fitted
thresholds are.

Per run it reports the threshold mean/range across folds, the raw ->
calibrated gain, and the gain share (gain / calibrated mean): the fraction
of the headline score that is calibration, not discrimination. Three
flags, all documented policy tripwires rather than calibrated bounds:

- CEILING: some fold threshold >= 0.999 — the operating point is pinned
  at the probability ceiling, so the binarized map keeps only a handful
  of ultra-confident pixels and the calibrated score is a knife-edge
  artifact, not a robust operating point.
- UNSTABLE: threshold range across folds > 0.05 — the fitted operating
  point does not reproduce across geographic folds.
- CALIBRATION_DOMINATED: gain share > 0.20 — more than a fifth of the
  headline score is threshold fitting.

Usage:
    uv run python scripts/lofo_threshold_audit.py [--root PATH]

Exit code is 0 whenever the audit could be produced, even when every run
is flagged; a missing runs dir, no LOFO files found, a malformed LOFO
file, or a file whose top-level means disagree with its fold means is a
hard error (exit 2), because an unauditable or self-inconsistent committed
record is a reproducibility failure, not a clean bill.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

CEILING_THRESHOLD = 0.999
UNSTABLE_RANGE = 0.05
DOMINATED_SHARE = 0.20
MEAN_TOLERANCE = 1e-6


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_lofo(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot parse LOFO file ({exc})") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path}: LOFO file is not a JSON object")
    folds = data.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{path}: 'folds' must be a non-empty list")
    for fold in folds:
        if not isinstance(fold, dict):
            raise TypeError(f"{path}: each fold must be an object")
        for key in ("threshold", "raw_score", "calibrated_score"):
            if not _is_number(fold.get(key)):
                raise ValueError(f"{path}: fold {fold.get('fold')} lacks numeric {key!r}")
    for key in ("raw_mean", "calibrated_mean"):
        if not _is_number(data.get(key)):
            raise ValueError(f"{path}: missing numeric top-level {key!r}")
    # The committed file must be self-consistent: the top-level means are
    # the fold means (population mean, same convention as score_cv).
    for key in ("raw_mean", "calibrated_mean"):
        fold_key = "raw_score" if key == "raw_mean" else "calibrated_score"
        fold_mean = statistics.fmean(fold[fold_key] for fold in folds)
        if abs(fold_mean - data[key]) > MEAN_TOLERANCE * max(1.0, abs(data[key])):
            raise ValueError(
                f"{path}: top-level {key} ({data[key]}) disagrees with fold mean ({fold_mean})"
            )
    return data


def _audit_run(path: Path, run_id: str) -> dict:
    data = _load_lofo(path)
    folds = data["folds"]
    thresholds = [float(fold["threshold"]) for fold in folds]
    raw_mean = float(data["raw_mean"])
    calibrated_mean = float(data["calibrated_mean"])
    gain = calibrated_mean - raw_mean
    gain_share = gain / calibrated_mean if calibrated_mean else 0.0
    flags = []
    if max(thresholds) >= CEILING_THRESHOLD:
        flags.append("CEILING")
    if max(thresholds) - min(thresholds) > UNSTABLE_RANGE:
        flags.append("UNSTABLE")
    if gain_share > DOMINATED_SHARE:
        flags.append("CALIBRATION_DOMINATED")
    return {
        "run": run_id,
        "scheme": str(data.get("scheme") or path.stem.removeprefix("lofo-")),
        "threshold_mean": statistics.fmean(thresholds),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "threshold_range": max(thresholds) - min(thresholds),
        "threshold_std": statistics.pstdev(thresholds),
        "raw_mean": raw_mean,
        "calibrated_mean": calibrated_mean,
        "gain": gain,
        "gain_share": gain_share,
        "flags": flags,
    }


def _discover_lofo_files(root: Path) -> list[Path]:
    runs_dir = root / "results" / "runs"
    if not runs_dir.is_dir():
        raise ValueError(f"no results/runs/ directory under {root}")
    files = sorted(runs_dir.glob("*/lofo-*.json")) + sorted(runs_dir.glob("*/scores/lofo-*.json"))
    if not files:
        raise ValueError(f"no LOFO files found under {runs_dir}")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit LOFO threshold calibration across committed CV runs"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        files = _discover_lofo_files(args.root)
    except ValueError as exc:
        return _fail(str(exc))
    rows = []
    for path in files:
        # LOFO files live at <run>/lofo-*.json or <run>/scores/lofo-*.json;
        # the run id is the run dir either way.
        run_dir = path.parent.parent if path.parent.name == "scores" else path.parent
        try:
            rows.append(_audit_run(path, run_dir.name))
        except (ValueError, TypeError) as exc:
            return _fail(str(exc))
    rows.sort(key=lambda row: (row["scheme"], row["run"]))
    print("RUN | SCHEME | THRESHOLD mean [min, max] range | RAW -> CAL gain (share) | FLAGS")
    for row in rows:
        flags = ", ".join(row["flags"]) if row["flags"] else "none"
        print(
            f"{row['run']} | {row['scheme']} | "
            f"{row['threshold_mean']:.4g} [{row['threshold_min']:.4g}, "
            f"{row['threshold_max']:.4g}] range {row['threshold_range']:.2g} | "
            f"{row['raw_mean']:.4g} -> {row['calibrated_mean']:.4g} "
            f"gain +{row['gain']:.4g} ({row['gain_share'] * 100:.1f}%) | {flags}"
        )
    flagged = sum(1 for row in rows if row["flags"])
    ceiling = sum(1 for row in rows if "CEILING" in row["flags"])
    unstable = sum(1 for row in rows if "UNSTABLE" in row["flags"])
    dominated = sum(1 for row in rows if "CALIBRATION_DOMINATED" in row["flags"])
    print(
        f"\nSummary: {len(rows)} LOFO runs audited, {flagged} flagged "
        f"({ceiling} ceiling, {unstable} unstable, {dominated} calibration-dominated)."
    )
    print(
        "Flags: CEILING = a fold threshold >= 0.999 (operating point pinned at the "
        "probability ceiling); UNSTABLE = threshold range across folds > 0.05; "
        "CALIBRATION_DOMINATED = threshold-fitting gain is > 20% of the headline score."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
