from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from gems.evaluation import (
    evaluate_fault_discovery_predictions,
    evaluate_spatial_predictions,
)


def _read_aligned(path: str, reference) -> np.ndarray:
    with rasterio.open(path) as src:
        if (src.height, src.width) != (reference.height, reference.width):
            raise SystemExit(f"shape mismatch for {path}")
        if src.crs != reference.crs or src.transform != reference.transform:
            raise SystemExit(f"georeferencing mismatch for {path}")
        return src.read(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score fold-specific GEMS predictions with the published metric"
    )
    parser.add_argument("--truth", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--scheme", choices=("spatial", "fault"), required=True)
    parser.add_argument(
        "--prediction-pattern",
        required=True,
        help="Path pattern containing {fold}, e.g. runs/x/fold-{fold}.tif",
    )
    parser.add_argument("--known-fault-exclusion-pixels", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    with rasterio.open(args.truth) as truth_src:
        truth = truth_src.read(1) > 0
        valid = truth_src.dataset_mask() > 0
        fold_map = _read_aligned(args.fold_map, truth_src).astype(np.int16)
        if args.scheme == "spatial":
            fold_ids = sorted(int(v) for v in np.unique(fold_map[valid]) if v >= 0)
        else:
            fold_ids = sorted(int(v) for v in np.unique(fold_map[truth & valid]) if v >= 0)
        predictions = {
            fold: _read_aligned(args.prediction_pattern.format(fold=fold), truth_src)
            for fold in fold_ids
        }

    if args.scheme == "spatial":
        result = evaluate_spatial_predictions(
            predictions,
            truth,
            fold_map,
            valid_mask=valid,
        )
    else:
        result = evaluate_fault_discovery_predictions(
            predictions,
            truth,
            fold_map,
            valid_mask=valid,
            known_fault_exclusion_pixels=args.known_fault_exclusion_pixels,
        )

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
