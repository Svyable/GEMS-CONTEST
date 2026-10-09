#!/usr/bin/env python3
"""Build and score an ensemble of GEMS candidate models.

Loads fold-wise OOF probability rasters from multiple models, combines them
using the specified method (mean, weighted_mean, rank_average), optionally
applies threshold calibration, and writes fold-specific GeoTIFF predictions.
The ensemble can then be scored through scripts/score_cv.py and evaluated
via scripts/verify_candidate.py.

Supports leave-one-fold-out (LOFO) weight fitting for weighted_mean: weights
for fold k are fitted on all other folds to prevent leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from gems.data import sha256_file
from gems.ensemble import (
    EnsembleMember,
    apply_ensemble,
    calibrate_ensemble_threshold,
    fit_lofo_weights,
    load_fold_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config with member predictions and ensemble method",
    )
    parser.add_argument(
        "--truth",
        required=True,
        help="Ground truth raster (for weight fitting and threshold calibration)",
    )
    parser.add_argument(
        "--fold-map",
        required=True,
        help="Fold assignment raster (spatial CV fold map)",
    )
    parser.add_argument(
        "--output-pattern",
        required=True,
        help="Output path pattern with {fold}, e.g. ensemble/fold-{fold}.tif",
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="Apply optimal binary threshold calibration per fold",
    )
    parser.add_argument(
        "--weights-output",
        help="Optional JSON path to save fitted LOFO weights",
    )
    parser.add_argument(
        "--metadata-output",
        help="Optional JSON path to save ensemble metadata (method, members, SHA256s)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    method = config.get("method", "mean")
    if method not in ("mean", "weighted_mean", "rank_average"):
        print(f"error: unknown method '{method}'", file=sys.stderr)
        return 1

    members_config = config.get("members", [])
    if not members_config:
        print("error: config must specify 'members' list", file=sys.stderr)
        return 1

    with rasterio.open(args.truth) as truth_src:
        truth = truth_src.read(1) > 0
        valid = truth_src.dataset_mask() > 0
        reference_path = args.truth

        with rasterio.open(args.fold_map) as fold_src:
            if (fold_src.height, fold_src.width) != (truth_src.height, truth_src.width):
                print("error: fold-map shape mismatch", file=sys.stderr)
                return 1
            fold_map = fold_src.read(1).astype(np.int16)

        fold_ids = sorted(int(v) for v in np.unique(fold_map[valid]) if v >= 0)
        if not fold_ids:
            print("error: fold-map contains no non-negative folds", file=sys.stderr)
            return 1

        members = []
        for member_config in members_config:
            name = member_config.get("name")
            pattern = member_config.get("pattern")
            if not name or not pattern:
                print(f"error: member must have 'name' and 'pattern': {member_config}", file=sys.stderr)
                return 1
            try:
                fold_predictions = load_fold_predictions(
                    pattern, fold_ids, reference_path=reference_path
                )
                members.append(EnsembleMember(name, fold_predictions))
                print(f"Loaded member '{name}' from {pattern}")
            except (ValueError, OSError, RuntimeError) as e:
                print(f"error loading member '{name}': {e}", file=sys.stderr)
                return 1

        weights = None
        if method == "weighted_mean":
            print(f"Fitting LOFO weights for {len(members)} members on {len(fold_ids)} folds...")
            weights = fit_lofo_weights(
                members,
                fold_ids,
                truth,
                fold_map,
                valid_mask=valid,
                alpha=0.2,
                beta=0.8,
                radius_pixels=3.0,
            )
            for fold in fold_ids:
                w = weights.weights[fold]
                print(f"  Fold {fold} weights: {', '.join(f'{x:.3f}' for x in w)}")

            if args.weights_output:
                weights_data = {
                    "method": weights.method,
                    "weights": {str(k): v.tolist() for k, v in weights.weights.items()},
                    "members": [m.name for m in members],
                }
                with open(args.weights_output, "w") as f:
                    json.dump(weights_data, f, indent=2)
                print(f"Saved LOFO weights to {args.weights_output}")

        print(f"Combining predictions using method='{method}'...")
        combined = apply_ensemble(members, fold_ids, method=method, weights=weights)

        if args.calibrate_threshold:
            print("Calibrating optimal thresholds per fold...")
            thresholds = calibrate_ensemble_threshold(
                combined,
                truth,
                fold_map,
                valid_mask=valid,
                alpha=0.2,
                beta=0.8,
                radius_pixels=3.0,
            )
            for fold in fold_ids:
                print(f"  Fold {fold} threshold: {thresholds[fold]:.4f}")
            for fold in fold_ids:
                raw = combined[fold]
                threshold = thresholds[fold]
                calibrated = np.where(
                    raw >= threshold,
                    (raw - threshold) / (1.0 - threshold + 1e-9),
                    raw / (threshold + 1e-9) * 0.5,
                )
                combined[fold] = np.clip(calibrated, 0.0, 1.0).astype(np.float32)

        output_pattern = str(args.output_pattern)
        if "{fold}" not in output_pattern:
            print("error: --output-pattern must contain '{fold}' placeholder", file=sys.stderr)
            return 1

        for fold in fold_ids:
            output_path = output_pattern.format(fold=fold)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=truth_src.height,
                width=truth_src.width,
                count=1,
                dtype="float32",
                crs=truth_src.crs,
                transform=truth_src.transform,
                nodata=truth_src.nodata,
            ) as dst:
                dst.write(combined[fold], 1)
            print(f"Wrote fold {fold} to {output_path}")

        if args.metadata_output:
            metadata = {
                "method": method,
                "members": [
                    {
                        "name": m.name,
                        "pattern": next(
                            (mc["pattern"] for mc in members_config if mc["name"] == m.name), ""
                        ),
                    }
                    for m in members
                ],
                "fold_ids": fold_ids,
                "calibrate_threshold": args.calibrate_threshold,
                "truth_sha256": sha256_file(args.truth),
                "fold_map_sha256": sha256_file(args.fold_map),
            }
            with open(args.metadata_output, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved ensemble metadata to {args.metadata_output}")

    print("Ensemble complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
