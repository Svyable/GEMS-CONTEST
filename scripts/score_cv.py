from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from gems.data import sha256_file
from gems.evaluation import (
    AGGREGATION_METHODS,
    evaluate_fault_discovery_predictions,
    evaluate_spatial_predictions,
    evaluate_trace_completion_predictions,
)

ALPHA = 0.2
BETA = 0.8
RADIUS_PIXELS = 3.0


def _read_aligned(path: str, reference) -> np.ndarray:
    with rasterio.open(path) as src:
        if (src.height, src.width) != (reference.height, reference.width):
            raise SystemExit(f"shape mismatch for {path}")
        if src.crs != reference.crs or src.transform != reference.transform:
            raise SystemExit(f"georeferencing mismatch for {path}")
        return src.read(1)


def _read_training_protocol(pattern: str | None, fold_ids: list[int]) -> dict:
    """An explicitly requested metadata chain must cover every scored fold."""
    if pattern is None:
        return {}
    training_protocol = None
    for fold in fold_ids:
        path = Path(pattern.format(fold=fold))
        try:
            metrics = json.loads(path.read_text())
            epochs = metrics.get("epochs")
            protocol = {
                "buffer_pixels": metrics["buffer_pixels"],
                "seed": metrics["seed"],
                "epochs": len(epochs) if isinstance(epochs, list) else epochs,
                "train_step": metrics["train_step"],
                "config_sha256": metrics.get("config_sha256"),
            }
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            raise SystemExit(f"cannot read training protocol for fold {fold}: {path}: {exc}")
        for key in ("buffer_pixels", "seed", "epochs", "train_step"):
            value = protocol[key]
            minimum = 1 if key in ("epochs", "train_step") else 0
            if type(value) is not int or value < minimum:
                raise SystemExit(f"invalid training protocol {key} for fold {fold}: {path}")
        if training_protocol is None:
            training_protocol = protocol
        elif training_protocol != protocol:
            raise SystemExit(f"training protocol differs for fold {fold}: {path}")
    return {k: v for k, v in (training_protocol or {}).items() if v is not None}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score fold-specific GEMS predictions with the published metric"
    )
    parser.add_argument("--truth", required=True)
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--scheme", choices=("spatial", "fault", "trace"), required=True)
    parser.add_argument(
        "--prediction-pattern",
        required=True,
        help="Path pattern containing {fold}, e.g. runs/x/fold-{fold}.tif",
    )
    parser.add_argument("--known-fault-exclusion-pixels", type=int, default=0)
    parser.add_argument(
        "--metrics-pattern",
        help="Optional path pattern to metrics JSON files containing training protocol, e.g. runs/x/metrics-fold-{fold}.json",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.known_fault_exclusion_pixels < 0:
        parser.error("--known-fault-exclusion-pixels must be non-negative")

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

    training_protocol = _read_training_protocol(args.metrics_pattern, fold_ids)

    if args.scheme == "spatial":
        result = evaluate_spatial_predictions(
            predictions,
            truth,
            fold_map,
            valid_mask=valid,
            alpha=ALPHA,
            beta=BETA,
            radius_pixels=RADIUS_PIXELS,
        )
    else:
        evaluator = (evaluate_fault_discovery_predictions if args.scheme == "fault"
                     else evaluate_trace_completion_predictions)
        result = evaluator(
            predictions,
            truth,
            fold_map,
            valid_mask=valid,
            known_fault_exclusion_pixels=args.known_fault_exclusion_pixels,
            alpha=ALPHA,
            beta=BETA,
            radius_pixels=RADIUS_PIXELS,
        )

    # Bind comparisons to the exact labels, holdout assignment, and metric
    # semantics. Candidate prediction hashes deliberately differ by design.
    result["evaluation_protocol"] = {
        "schema_version": 2,
        "aggregation": AGGREGATION_METHODS[args.scheme],
        "metric": "distance_weighted_tversky",
        "alpha": ALPHA,
        "beta": BETA,
        "radius_pixels": RADIUS_PIXELS,
        "truth_sha256": sha256_file(args.truth),
        "fold_map_sha256": sha256_file(args.fold_map),
        "known_fault_exclusion_pixels": (
            args.known_fault_exclusion_pixels if args.scheme != "spatial" else 0
        ),
    }
    if training_protocol:
        result["evaluation_protocol"].update(training_protocol)

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
