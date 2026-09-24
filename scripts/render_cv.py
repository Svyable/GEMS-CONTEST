from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from gems.visual_qa import fold_preview_rgb


def _read_aligned(path: str, reference) -> np.ndarray:
    with rasterio.open(path) as src:
        if (src.height, src.width) != (reference.height, reference.width):
            raise SystemExit(f"shape mismatch for {path}")
        if src.crs != reference.crs or src.transform != reference.transform:
            raise SystemExit(f"georeferencing mismatch for {path}")
        return src.read(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a compact PNG QA preview of GEMS CV folds"
    )
    parser.add_argument("--fold-map", required=True)
    parser.add_argument("--scheme", choices=("spatial", "fault", "trace"), required=True)
    parser.add_argument("--labels")
    parser.add_argument("--valid-template")
    parser.add_argument("--max-dimension", type=int, default=1600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.scheme in {"fault", "trace"} and not args.valid_template:
        raise SystemExit("--valid-template is required for fault and trace previews")

    with rasterio.open(args.fold_map) as fold_src:
        folds = fold_src.read(1)
        valid = fold_src.dataset_mask() > 0
        labels = _read_aligned(args.labels, fold_src) if args.labels else None
        if args.valid_template:
            with rasterio.open(args.valid_template) as valid_src:
                if (valid_src.height, valid_src.width) != (fold_src.height, fold_src.width):
                    raise SystemExit("valid template shape mismatch")
                if valid_src.crs != fold_src.crs or valid_src.transform != fold_src.transform:
                    raise SystemExit("valid template georeferencing mismatch")
                template_valid = valid_src.dataset_mask() > 0
                valid = (
                    template_valid
                    if args.scheme in {"fault", "trace"}
                    else valid & template_valid
                )

    rgb = fold_preview_rgb(
        folds,
        scheme=args.scheme,
        labels=labels,
        valid_mask=valid,
        max_dimension=args.max_dimension,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "PNG",
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "count": 3,
        "dtype": "uint8",
    }
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))
    print(f"wrote {output} ({rgb.shape[1]}x{rgb.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
