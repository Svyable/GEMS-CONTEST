from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from gems.calibration import binary_threshold, exact_threshold_curve
from gems.data import raster_alignment_errors, sha256_file
from gems.metric import distance_weighted_tversky


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit an exact binary threshold and evaluate it on a separate held-out region"
    )
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--calibration-mask", required=True)
    parser.add_argument("--evaluation-mask", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    paths = {
        "prediction": args.prediction,
        "truth": args.truth,
        "calibration_mask": args.calibration_mask,
        "evaluation_mask": args.evaluation_mask,
    }
    arrays, masks = {}, {}
    for name, path in paths.items():
        errors = raster_alignment_errors(args.truth, path)
        if errors:
            raise SystemExit(f"{name}: " + "; ".join(errors))
        with rasterio.open(path) as src:
            if src.count != 1:
                raise SystemExit(f"{name} must be single-band")
            if src.crs != rasterio.crs.CRS.from_epsg(32611) or src.res != (100, 100):
                raise SystemExit(f"{name} must use EPSG:32611 at 100 m resolution")
            arrays[name] = src.read(1)
            masks[name] = src.dataset_mask() > 0
    truth_values = arrays["truth"][masks["truth"]]
    if not np.all(np.isin(truth_values, [0, 1])):
        raise SystemExit("truth must contain binary labels on valid pixels")
    regions = {}
    for name in ("calibration_mask", "evaluation_mask"):
        if not np.all(np.isin(arrays[name][masks[name]], [0, 1])):
            raise SystemExit(f"{name} must contain binary values")
        region = masks[name] & (arrays[name] == 1)
        if not region.any():
            raise SystemExit(f"{name} must contain selected pixels")
        if np.any(region & ~(masks["truth"] & masks["prediction"])):
            raise SystemExit(f"{name} selects pixels outside valid truth/prediction")
        regions[name] = region
    fit = regions["calibration_mask"]
    evaluation = regions["evaluation_mask"]
    if np.any(fit & evaluation):
        raise SystemExit("calibration and evaluation regions must be disjoint")
    prediction = arrays["prediction"]
    truth = arrays["truth"] == 1
    # Validate evaluation values before fitting or writing any report.
    raw_evaluation = distance_weighted_tversky(prediction, truth, valid_mask=evaluation)
    curve = exact_threshold_curve(prediction, truth, valid_mask=fit)
    threshold = curve.best_threshold
    transformed = binary_threshold(prediction, threshold)
    payload = {
        "schema_version": 1,
        "method": "exact_binary_threshold_event_sweep",
        "threshold": threshold,
        "comparison": ">=",
        "metric": {"alpha": 0.2, "beta": 0.8, "radius_pixels": 3.0, "eps": 1e-12},
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "calibration": {
            "pixels": int(fit.sum()),
            "truth_pixels": curve.truth_pixels,
            "threshold_states": len(curve.thresholds),
            "raw_score": distance_weighted_tversky(prediction, truth, valid_mask=fit),
            "binary_fit_score": float(curve.scores.max()),
        },
        "evaluation": {
            "pixels": int(evaluation.sum()),
            "truth_pixels": int((truth & evaluation).sum()),
            "raw_score": raw_evaluation,
            "binary_score": distance_weighted_tversky(transformed, truth, valid_mask=evaluation),
        },
        "limitations": [
            "Both regions must be excluded from training and preprocessing fitting.",
            "Disjoint masks alone do not establish spatial independence; use buffered regions.",
            "Calibration fit scores are selection-biased; evaluation cannot be reused for tuning.",
            "Existing-fault labels are incomplete; this is not a hidden-test or leaderboard score.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"threshold={threshold:.17g} report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
