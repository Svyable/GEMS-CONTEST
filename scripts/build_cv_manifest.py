from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from gems.cv import (
    assign_fault_components,
    assign_spatial_blocks,
    fault_discovery_fold,
    spatial_fold_masks,
)
from gems.data import fingerprint_file


def _write_fold_raster(path: Path, template, values: np.ndarray) -> None:
    profile = template.profile.copy()
    profile.update(count=1, dtype="int16", nodata=-1, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype("int16"), 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build reproducible spatial or fault-component CV fold artifacts"
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--scheme", choices=("spatial", "fault"), required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--buffer-pixels", type=int, default=16)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    tif_path = prefix.with_suffix(".tif")
    json_path = prefix.with_suffix(".json")

    with rasterio.open(args.features) as feature_src, rasterio.open(args.labels) as label_src:
        if (feature_src.height, feature_src.width) != (label_src.height, label_src.width):
            raise SystemExit("feature and label raster shapes differ")
        if feature_src.crs != label_src.crs or feature_src.transform != label_src.transform:
            raise SystemExit("feature and label raster georeferencing differs")

        shape = (feature_src.height, feature_src.width)
        valid = feature_src.dataset_mask() > 0
        labels = label_src.read(1) > 0

        manifest = {
            "schema_version": 1,
            "scheme": args.scheme,
            "folds": args.folds,
            "seed": args.seed,
            "buffer_pixels": args.buffer_pixels,
            "inputs": {
                "features": fingerprint_file(args.features),
                "labels": fingerprint_file(args.labels),
            },
            "fold_summary": [],
        }

        if args.scheme == "spatial":
            assignments = assign_spatial_blocks(
                shape,
                block_size=args.block_size,
                n_folds=args.folds,
                seed=args.seed,
            )
            rows = np.arange(shape[0]) // args.block_size
            cols = np.arange(shape[1]) // args.block_size
            pixel_folds = assignments[rows[:, None], cols[None, :]].astype("int16")
            pixel_folds[~valid] = -1
            manifest["block_size"] = args.block_size

            for fold_id in range(args.folds):
                fold = spatial_fold_masks(
                    assignments,
                    shape,
                    fold=fold_id,
                    block_size=args.block_size,
                    buffer_pixels=args.buffer_pixels,
                    valid_mask=valid,
                )
                manifest["fold_summary"].append(
                    {
                        "fold": fold_id,
                        "train_pixels": int(fold.train_mask.sum()),
                        "validation_pixels": int(fold.validation_mask.sum()),
                        "excluded_from_training_pixels": int(
                            fold.excluded_from_training.sum()
                        ),
                    }
                )
        else:
            pixel_folds = assign_fault_components(
                labels,
                n_folds=args.folds,
                seed=args.seed,
            ).astype("int16")
            for fold_id in range(args.folds):
                fold = fault_discovery_fold(
                    labels,
                    pixel_folds,
                    fold=fold_id,
                    buffer_pixels=args.buffer_pixels,
                    valid_mask=valid,
                )
                manifest["fold_summary"].append(
                    {
                        "fold": fold_id,
                        "train_fault_pixels": int(fold.train_truth.sum()),
                        "withheld_fault_pixels": int(fold.validation_truth.sum()),
                        "train_valid_pixels": int(fold.train_valid_mask.sum()),
                        "evaluation_pixels": int(fold.evaluation_mask.sum()),
                    }
                )

        _write_fold_raster(tif_path, feature_src, pixel_folds)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {tif_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
