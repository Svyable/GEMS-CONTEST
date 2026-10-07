"""Reproducible spatial-ablation command planning and score summaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def build_spatial_ablation_commands(
    matrix: Mapping[str, Any],
    *,
    features: str,
    labels: str,
    template: str,
    fold_map: str,
    output_root: str,
) -> list[dict[str, Any]]:
    """Build deterministic train_full_map commands from an ablation matrix."""
    experiment_id = str(matrix.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("matrix requires experiment_id")

    raw_folds = matrix.get("folds")
    if not isinstance(raw_folds, Sequence) or isinstance(raw_folds, (str, bytes)):
        raise TypeError("matrix folds must be a non-empty sequence")
    folds = tuple(int(value) for value in raw_folds)
    if not folds or len(set(folds)) != len(folds) or min(folds) < 0:
        raise ValueError("matrix folds must be unique non-negative integers")

    buffer_pixels = int(matrix.get("buffer_pixels", 16))
    overlap = int(matrix.get("overlap", 64))
    if buffer_pixels < 0 or overlap < 0:
        raise ValueError("buffer_pixels and overlap must be non-negative")

    raw_variants = matrix.get("variants")
    if not isinstance(raw_variants, Mapping) or not raw_variants:
        raise ValueError("matrix requires at least one variant")

    root = Path(output_root)
    planned: list[dict[str, Any]] = []
    for variant, raw_spec in raw_variants.items():
        name = str(variant).strip()
        if not name or not isinstance(raw_spec, Mapping):
            raise ValueError("each variant must have a name and mapping config")
        config = str(raw_spec.get("config", "")).strip()
        if not config:
            raise ValueError(f"variant {name!r} requires config")

        registration = bool(raw_spec.get("registration_sensitivity", False))
        for fold in folds:
            run_dir = root / experiment_id / name
            prediction = run_dir / f"fold-{fold}.tif"
            metrics = run_dir / f"fold-{fold}.json"
            command = [
                "uv",
                "run",
                "python",
                "scripts/train_full_map.py",
                "--features",
                features,
                "--labels",
                labels,
                "--template",
                template,
                "--fold-map",
                fold_map,
                "--cv-scheme",
                "spatial",
                "--fold",
                str(fold),
                "--buffer-pixels",
                str(buffer_pixels),
                "--overlap",
                str(overlap),
                "--config",
                config,
                "--output",
                str(prediction),
                "--metrics-json",
                str(metrics),
            ]
            sensitivity = None
            if registration:
                sensitivity = run_dir / f"fold-{fold}-registration.json"
                command.extend(["--registration-sensitivity-json", str(sensitivity)])
            planned.append(
                {
                    "experiment_id": experiment_id,
                    "variant": name,
                    "fold": fold,
                    "config": config,
                    "prediction": prediction,
                    "metrics": metrics,
                    "registration_sensitivity": sensitivity,
                    "command": command,
                }
            )
    return planned


def summarize_spatial_ablation(
    planned: Sequence[Mapping[str, Any]],
    *,
    baseline_variant: str = "control",
) -> dict[str, Any]:
    """Summarize completed fold metrics and paired deltas versus a baseline."""
    by_variant: dict[str, dict[int, float]] = {}
    missing: list[str] = []

    for item in planned:
        variant = str(item["variant"])
        fold = int(item["fold"])
        path = Path(item["metrics"])
        if not path.exists():
            missing.append(path.as_posix())
            continue
        payload = json.loads(path.read_text())
        value = payload.get("holdout_distance_weighted_tversky")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{path} lacks a finite holdout_distance_weighted_tversky")
        by_variant.setdefault(variant, {})[fold] = float(value)

    if missing:
        raise FileNotFoundError("missing metrics: " + ", ".join(missing))
    if baseline_variant not in by_variant:
        raise ValueError(f"baseline variant {baseline_variant!r} is missing")

    baseline = by_variant[baseline_variant]
    expected_folds = set(baseline)
    if not expected_folds:
        raise ValueError("baseline has no folds")

    variants: dict[str, Any] = {}
    for variant, scores in by_variant.items():
        if set(scores) != expected_folds:
            raise ValueError(f"variant {variant!r} does not match baseline fold set")
        ordered = [scores[fold] for fold in sorted(expected_folds)]
        deltas = [scores[fold] - baseline[fold] for fold in sorted(expected_folds)]
        variants[variant] = {
            "fold_scores": {str(fold): scores[fold] for fold in sorted(expected_folds)},
            "mean": mean(ordered),
            "population_std": pstdev(ordered),
            "paired_deltas_vs_baseline": {
                str(fold): scores[fold] - baseline[fold] for fold in sorted(expected_folds)
            },
            "mean_paired_delta_vs_baseline": mean(deltas),
            "wins_vs_baseline": sum(delta > 0 for delta in deltas),
            "losses_vs_baseline": sum(delta < 0 for delta in deltas),
            "ties_vs_baseline": sum(delta == 0 for delta in deltas),
        }

    return {
        "schema_version": 1,
        "metric": "holdout_distance_weighted_tversky",
        "baseline_variant": baseline_variant,
        "folds": sorted(expected_folds),
        "variants": variants,
    }
