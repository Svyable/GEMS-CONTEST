#!/usr/bin/env python3
"""Optimize probability threshold for the Tversky metric.

DEPRECATION NOTICE: This script is deprecated. Use scripts/calibrate_threshold.py
instead, which implements the exact threshold optimization method from
gems.calibration. The exact method evaluates every distinct threshold state
efficiently and is both faster and more accurate than this grid-search approach.

Migration: replace optimize_threshold.py with calibrate_threshold.py, which
uses --calibration-mask and --evaluation-mask for proper train/test separation.

This script finds the optimal threshold that maximizes the distance-weighted
Tversky index on held-out or full-raster predictions. With beta=0.8 and
alpha=0.2, the optimal threshold is typically much lower than 0.5 to
maximize recall while controlling false positives.

Usage:
    uv run python scripts/optimize_threshold.py \\
      --prediction outputs/model-predictions.tif \\
      --truth data/raw/existing_faults.tif \\
      --method grid \\
      --output results/threshold-optimization.json

    uv run python scripts/optimize_threshold.py \\
      --prediction outputs/model-predictions.tif \\
      --truth data/raw/existing_faults.tif \\
      --method golden \\
      --bracket 0.1 0.5 \\
      --output results/threshold-optimization.json

The output JSON includes optimal_threshold, optimal_score, method, and
optionally the full scores_by_threshold mapping for visualization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from gems.data import load_raster_band
from gems.optimization import (
    ThresholdResult,
    calibrate_predictions,
    optimize_threshold_golden,
    optimize_threshold_grid,
)


def _progress_bar(current: int, total: int, prefix: str = "") -> None:
    """Simple progress bar."""
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = "=" * filled + "-" * (bar_length - filled)
    percent = 100 * current / total
    sys.stdout.write(f"\r{prefix}[{bar}] {percent:5.1f}%")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize threshold for distance-weighted Tversky metric"
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Path to prediction raster (probabilities in [0, 1])",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Path to ground truth binary label raster",
    )
    parser.add_argument(
        "--method",
        choices=["grid", "golden"],
        default="grid",
        help="Optimization method (default: grid)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Tversky alpha (false positive penalty, default: 0.2)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.8,
        help="Tversky beta (false negative penalty, default: 0.8)",
    )
    parser.add_argument(
        "--radius-pixels",
        type=float,
        default=3.0,
        help="Distance tolerance in pixels (default: 3.0 for 300m at 100m resolution)",
    )
    parser.add_argument(
        "--valid-mask",
        type=Path,
        help="Optional path to valid-data mask raster",
    )
    parser.add_argument(
        "--fold-map",
        type=Path,
        help="Optional fold map; only evaluate on specified fold",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold ID to evaluate (requires --fold-map)",
    )
    parser.add_argument(
        "--bracket",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Search bracket for golden method (e.g., 0.1 0.5)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Convergence tolerance for golden method (default: 0.001)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum iterations for golden method (default: 50)",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.0,
        help="Minimum threshold for grid search (default: 0.0)",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=1.0,
        help="Maximum threshold for grid search (default: 1.0)",
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=101,
        help="Number of thresholds for grid search (default: 101)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--save-calibrated",
        type=Path,
        help="Optional: save recalibrated predictions to this GeoTIFF path",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Include full scores_by_threshold mapping in output JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress during optimization",
    )

    args = parser.parse_args()

    if args.fold is not None and args.fold_map is None:
        parser.error("--fold requires --fold-map")

    # Load prediction
    print(f"Loading prediction from {args.prediction}")
    prediction = load_raster_band(args.prediction, band=1)

    # Load truth
    print(f"Loading truth from {args.truth}")
    truth = load_raster_band(args.truth, band=1).astype(bool)

    if prediction.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs truth {truth.shape}"
        )

    # Load optional valid mask
    valid_mask = None
    if args.valid_mask:
        print(f"Loading valid mask from {args.valid_mask}")
        valid_mask = load_raster_band(args.valid_mask, band=1).astype(bool)
        if valid_mask.shape != prediction.shape:
            raise ValueError("Valid mask shape does not match prediction")

    # Load optional fold map
    if args.fold_map:
        print(f"Loading fold map from {args.fold_map}")
        fold_map = load_raster_band(args.fold_map, band=1)
        if fold_map.shape != prediction.shape:
            raise ValueError("Fold map shape does not match prediction")

        if args.fold is None:
            raise ValueError("--fold-map provided but --fold not specified")

        print(f"Restricting evaluation to fold {args.fold}")
        fold_mask = fold_map == args.fold
        if valid_mask is None:
            valid_mask = fold_mask
        else:
            valid_mask = valid_mask & fold_mask

    if valid_mask is not None:
        num_valid = valid_mask.sum()
        num_truth = (truth & valid_mask).sum()
        print(f"Valid pixels: {num_valid:,} ({100 * num_valid / valid_mask.size:.2f}%)")
        print(f"Truth pixels: {num_truth:,} ({100 * num_truth / num_valid:.2f}%)")
    else:
        num_truth = truth.sum()
        print(f"Truth pixels: {num_truth:,} ({100 * num_truth / truth.size:.4f}%)")

    # Progress callback
    callback = None
    if args.verbose:
        evaluations = {"count": 0, "total": args.num_thresholds}

        def callback(threshold: float, score: float):
            evaluations["count"] += 1
            _progress_bar(
                evaluations["count"],
                evaluations["total"],
                prefix="Evaluating thresholds ",
            )

    # Run optimization
    result: ThresholdResult
    if args.method == "grid":
        print(
            f"Running grid search: [{args.min_threshold}, {args.max_threshold}], "
            f"{args.num_thresholds} points"
        )
        thresholds = np.linspace(args.min_threshold, args.max_threshold, args.num_thresholds)

        result = optimize_threshold_grid(
            prediction,
            truth,
            valid_mask=valid_mask,
            alpha=args.alpha,
            beta=args.beta,
            radius_pixels=args.radius_pixels,
            thresholds=thresholds,
            progress_callback=callback,
        )

    elif args.method == "golden":
        bracket = tuple(args.bracket) if args.bracket else (0.0, 1.0)
        print(
            f"Running golden section search: bracket={bracket}, "
            f"tolerance={args.tolerance}, max_iter={args.max_iterations}"
        )

        result = optimize_threshold_golden(
            prediction,
            truth,
            valid_mask=valid_mask,
            alpha=args.alpha,
            beta=args.beta,
            radius_pixels=args.radius_pixels,
            bracket=bracket,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            progress_callback=callback,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    if args.verbose:
        print()  # newline after progress bar

    print(f"\nOptimal threshold: {result.optimal_threshold:.4f}")
    print(f"Optimal score:     {result.optimal_score:.6f}")
    print(f"Method:            {result.method}")

    # Save results
    output_data = {
        "optimal_threshold": result.optimal_threshold,
        "optimal_score": result.optimal_score,
        "method": result.method,
        "parameters": {
            "alpha": args.alpha,
            "beta": args.beta,
            "radius_pixels": args.radius_pixels,
        },
        "inputs": {
            "prediction": str(args.prediction),
            "truth": str(args.truth),
            "valid_mask": str(args.valid_mask) if args.valid_mask else None,
            "fold_map": str(args.fold_map) if args.fold_map else None,
            "fold": args.fold,
        },
    }

    if args.save_scores:
        output_data["scores_by_threshold"] = {
            f"{k:.6f}": v for k, v in result.scores_by_threshold.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {args.output}")

    # Optionally save calibrated predictions
    if args.save_calibrated:
        print(f"Calibrating and saving predictions to {args.save_calibrated}")

        calibrated = calibrate_predictions(
            prediction, optimal_threshold=result.optimal_threshold
        )

        # Copy georeferencing from original prediction
        with rasterio.open(args.prediction) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, count=1, compress="deflate")

            args.save_calibrated.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(args.save_calibrated, "w", **profile) as dst:
                dst.write(calibrated.astype(np.float32), 1)

        print(f"Calibrated predictions saved to {args.save_calibrated}")


if __name__ == "__main__":
    main()
