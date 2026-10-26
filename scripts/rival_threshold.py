#!/usr/bin/env python3
"""Score a CV run's fold predictions at the rival's closed-form threshold rule.

Background (October 2026 briefing letter, "The Threshold Tension"): a rival's
published analysis of the GEMS scoring metric argues the repo's LOFO-calibrated
pixel cutoffs (0.85-0.96, and 1.0 on trace) may be far too strict, and proposes
a closed-form rule instead: mark a pixel when p > 0.2 x expected score. Their
worked example is a 0.08 threshold against a habitual 0.5. The rival's math
has NOT been independently verified, and the experiment that would test it
needs Sven's go-ahead — this script plus the parked s21 queue tasks are the
ready-to-run handoff for that experiment, not its execution.

Operationalization. "Expected score" has no canonical value for a run, so the
experiment pins it to the reference run's LOFO-calibrated mean score for that
view (the best available estimate of what the run achieves), passed as
--expected-score. The parked tasks pass the committed means rounded to 5
decimals (spatial 0.12296, fault 0.05469, trace 0.01856); the rounding moves
the threshold by < 1e-6, below the 0.001 grid resolution used elsewhere in the
repo. Factor defaults to 0.2 (the rival's rule); --factor lets a later
sensitivity check vary it without editing the script.

Method. Mirrors scripts/queue/lofo_calibrate_view.py exactly: same fold
predictions (RUN_DIR/fold-{k}.tif), same truth raster, same per-fold regions
(spatial: fold-k block; fault/trace: the gems.cv fold spec), same
distance-weighted Tversky metric. The only difference is the operating
point: one fixed rival threshold for every fold instead of a per-fold LOFO
fit. The paired baseline comes from the run's own LOFO file (both committed
layouts, scores/lofo-<scheme>.json or lofo-<scheme>.json), so each fold
reports rival_score, lofo_calibrated_score, and delta = rival - lofo_cal.
A missing LOFO file, a fold-set mismatch, or a LOFO file whose top-level
calibrated mean disagrees with its fold mean (> 1e-6, the audit's
convention) is a hard error (exit 2): a comparison against a missing or
self-inconsistent baseline is a reproducibility failure, not a result.

Usage (box; see scripts/queue/rival_threshold.sh):
    uv run python scripts/rival_threshold.py RUN_DIR SCHEME FOLD_MAP \\
        --expected-score 0.12296 [--factor 0.2] [--truth PATH]

Writes RUN_DIR/scores/rival_threshold-<scheme>.json and prints one line per
fold plus the summary. No gate: this is a comparison, not a promotion
candidate — a human decides whether the rival rule is worth adopting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from gems.calibration import binary_threshold
from gems.cv import fault_discovery_fold, trace_completion_fold
from gems.metric import distance_weighted_tversky

MEAN_TOL = 1e-6


def rival_threshold(expected_score: float, factor: float = 0.2) -> float:
    """The rival's closed-form rule: threshold = factor x expected score."""
    return factor * expected_score


def find_lofo_file(run_dir: Path, scheme: str) -> Path:
    """Locate a run's LOFO file across both committed layouts."""
    for candidate in (
        run_dir / "scores" / f"lofo-{scheme}.json",
        run_dir / f"lofo-{scheme}.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no LOFO file for scheme {scheme!r} under {run_dir} "
        f"(looked for scores/lofo-{scheme}.json and lofo-{scheme}.json)"
    )


def load_lofo_calibrated(run_dir: Path, scheme: str) -> dict[int, float]:
    """Fold -> LOFO-calibrated score, with the audit's consistency check."""
    path = find_lofo_file(run_dir, scheme)
    doc = json.loads(path.read_text())
    if doc.get("scheme", scheme) != scheme:
        raise ValueError(f"{path}: scheme {doc.get('scheme')!r} != {scheme!r}")
    folds = doc.get("folds")
    if not folds:
        raise ValueError(f"{path}: no folds")
    by_fold = {int(row["fold"]): float(row["calibrated_score"]) for row in folds}
    mean = float(np.mean(list(by_fold.values())))
    if abs(mean - float(doc["calibrated_mean"])) > MEAN_TOL:
        raise ValueError(
            f"{path}: calibrated_mean {doc['calibrated_mean']} disagrees with "
            f"fold mean {mean} by more than {MEAN_TOL}"
        )
    return by_fold


def fold_regions(
    scheme: str, fold_map_path: Path, truth: np.ndarray, valid: np.ndarray
) -> tuple[list[int], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Per-fold (truth, evaluation mask), exactly as lofo_calibrate_view.py."""
    folds = rasterio.open(fold_map_path).read(1).astype(np.int16)
    if scheme == "spatial":
        ids = sorted(int(v) for v in np.unique(folds[valid]) if v >= 0)
        regions = {k: (truth, valid & (folds == k)) for k in ids}
        return ids, regions
    fold_fn = fault_discovery_fold if scheme == "fault" else trace_completion_fold
    ids = sorted(int(v) for v in np.unique(folds[truth & valid]) if v >= 0)
    regions = {}
    for k in ids:
        spec = fold_fn(truth, folds, fold=k, valid_mask=valid, known_fault_exclusion_pixels=0)
        regions[k] = (spec.validation_truth, spec.evaluation_mask)
    return ids, regions


def score_run_at_threshold(
    run_dir: Path,
    scheme: str,
    fold_map_path: Path,
    threshold: float,
    *,
    expected_score: float,
    factor: float,
    truth_path: Path,
    lofo_file: Path | None = None,
) -> dict:
    """Score every fold at one fixed threshold; pair against LOFO calibration."""
    lofo_by_fold = load_lofo_calibrated(run_dir, scheme)
    if lofo_file is None:
        lofo_file = find_lofo_file(run_dir, scheme)
    with rasterio.open(truth_path) as t:
        truth = t.read(1) > 0
        valid = t.dataset_mask() > 0
    ids, regions = fold_regions(scheme, fold_map_path, truth, valid)
    if sorted(lofo_by_fold) != ids:
        raise ValueError(
            f"LOFO folds {sorted(lofo_by_fold)} != prediction folds {ids}; "
            "cannot pair the rival rule against LOFO calibration"
        )
    rows = []
    for k in ids:
        with rasterio.open(run_dir / f"fold-{k}.tif") as s:
            pred = s.read(1).astype(np.float64)
        truth_k, mask_k = regions[k]
        mask_k = mask_k & np.isfinite(pred)
        rival = distance_weighted_tversky(
            binary_threshold(pred, threshold), truth_k, valid_mask=mask_k
        )
        rows.append(
            {
                "fold": k,
                "rival_score": float(rival),
                "lofo_calibrated_score": float(lofo_by_fold[k]),
                "delta": float(rival - lofo_by_fold[k]),
                "eval_pixels": int(mask_k.sum()),
                "truth_pixels": int((truth_k & mask_k).sum()),
            }
        )
        print(rows[-1], flush=True)
    rival_scores = [r["rival_score"] for r in rows]
    lofo_scores = [r["lofo_calibrated_score"] for r in rows]
    deltas = [r["delta"] for r in rows]
    return {
        "scheme": scheme,
        "method": (
            "Rival closed-form rule: mark pixel when p > factor x expected "
            "score, one fixed threshold for every fold; paired against the "
            "run's own LOFO-calibrated scores. Expected score = reference "
            "run LOFO-calibrated mean for this view (see --expected-score)."
        ),
        "factor": factor,
        "expected_score": expected_score,
        "threshold": threshold,
        "lofo_file": str(lofo_file),
        "folds": rows,
        "rival_mean": float(np.mean(rival_scores)),
        "rival_std": float(np.std(rival_scores)),
        "lofo_calibrated_mean": float(np.mean(lofo_scores)),
        "lofo_calibrated_std": float(np.std(lofo_scores)),
        "mean_delta": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "std_note": "population std (ddof=0), same as score_cv macro_std",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("scheme", choices=["spatial", "fault", "trace"])
    parser.add_argument("fold_map", type=Path)
    parser.add_argument("--expected-score", type=float, required=True)
    parser.add_argument("--factor", type=float, default=0.2)
    parser.add_argument("--truth", type=Path, default=Path("data/raw/existing_faults.tif"))
    args = parser.parse_args(argv)
    if not (args.expected_score > 0):
        parser.error("--expected-score must be positive")
    if not (args.factor > 0):
        parser.error("--factor must be positive")
    threshold = rival_threshold(args.expected_score, args.factor)
    if threshold > 1.0:
        parser.error(
            f"rival threshold {threshold} > 1: factor x expected-score "
            "must be a valid probability threshold"
        )
    try:
        summary = score_run_at_threshold(
            args.run_dir,
            args.scheme,
            args.fold_map,
            threshold,
            expected_score=args.expected_score,
            factor=args.factor,
            truth_path=args.truth,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_dir = args.run_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rival_threshold-{args.scheme}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
