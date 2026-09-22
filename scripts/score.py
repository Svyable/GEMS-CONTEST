from __future__ import annotations

import argparse

import numpy as np
import rasterio

from gems.metric import distance_weighted_tversky


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a prediction against a local held-out label raster"
    )
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--valid-mask", help="Optional raster; nonzero pixels are included")
    args = parser.parse_args()

    with rasterio.open(args.prediction) as src:
        pred = src.read(1).astype(np.float64)
    with rasterio.open(args.truth) as src:
        truth = src.read(1) > 0
    valid = None
    if args.valid_mask:
        with rasterio.open(args.valid_mask) as src:
            valid = src.read(1) > 0

    score = distance_weighted_tversky(pred, truth, valid_mask=valid)
    print(f"distance_weighted_tversky={score:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
