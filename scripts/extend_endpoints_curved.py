#!/usr/bin/env python3
"""Apply P4 orientation-aware curved continuation to a probability raster.

Research: docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md §P4-P5
(Chen et al., curvature-prior geodesic paths, 2023). The straight
endpoint-extension control (ENDPOINT-EXT-02) was measured and REJECTED; this
candidate grows continuations that bend toward probability support instead.

Usage:
    uv run python scripts/extend_endpoints_curved.py \\
        --input predictions/baseline_oof.tif \\
        --output predictions/baseline_curved.tif \\
        --threshold 0.5 \\
        --max-extension-m 300 \\
        --curvature-penalty 0.5

Fixed budgets (per the research doc: fix endpoint/path budgets, penalize
unsupported bridges): --max-paths caps continuations per map,
--min-prob is the per-pixel support floor, --support-threshold rejects
whole paths that cross empty background.

The output will be a float32 GeoTIFF preserving the input's CRS, transform,
and valid-region mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

from gems.curved_continuation import CurvedContinuationParams, apply_curved_continuation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grow curvature-prior fault-trace continuations toward probability support."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input probability raster (float32 GeoTIFF)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output raster with curved continuations painted",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for skeletonization (default: 0.5)",
    )
    parser.add_argument(
        "--max-extension-m",
        type=float,
        default=300.0,
        help="Maximum continuation length in meters (default: 300)",
    )
    parser.add_argument(
        "--curvature-penalty",
        type=float,
        default=0.5,
        help="Cost weight on squared heading change per step (default: 0.5)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="Beam size for the path search (default: 3)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.3,
        help="Per-pixel support floor; paths never grow below this (default: 0.3)",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.4,
        help="Mean path probability required to keep a path (default: 0.4)",
    )
    parser.add_argument(
        "--min-path-length",
        type=int,
        default=3,
        help="Minimum grown path length in pixels (default: 3)",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=200,
        help="Maximum continuations painted per map (default: 200)",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.15,
        help="Exponential decay of painted probability along the path (default: 0.15)",
    )

    args = parser.parse_args()

    print(f"Reading {args.input}")
    with rasterio.open(args.input) as src:
        probabilities = src.read(1)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata

    if probabilities.dtype != np.float32:
        print(f"Warning: input is {probabilities.dtype}, converting to float32")
        probabilities = probabilities.astype(np.float32)

    if nodata is not None:
        valid_mask = probabilities != nodata
    else:
        valid_mask = np.isfinite(probabilities)

    pixel_size_m = abs(transform[0])  # Assuming square pixels
    params = CurvedContinuationParams(
        max_extension_m=args.max_extension_m,
        pixel_size_m=pixel_size_m,
        curvature_penalty=args.curvature_penalty,
        beam_width=args.beam_width,
        min_prob=args.min_prob,
        support_threshold=args.support_threshold,
        min_path_length=args.min_path_length,
        max_paths=args.max_paths,
        decay_rate=args.decay_rate,
    )
    print(f"Pixel size: {pixel_size_m:.1f} m; max steps: {params.max_steps}")

    print("Growing curved continuations...")
    extended = apply_curved_continuation(
        probabilities,
        threshold=args.threshold,
        params=params,
        valid_mask=valid_mask,
    )

    painted_pixels = int(np.sum((extended > probabilities) & valid_mask))
    print(f"Painted {painted_pixels} continuation pixels")

    print(f"Writing {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=nodata if nodata is not None else np.nan,
    )

    with rasterio.open(args.output, "w", **out_profile) as dst:
        dst.write(extended.astype(np.float32), 1)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
