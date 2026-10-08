#!/usr/bin/env python3
"""Apply endpoint-extension post-processing to a probability raster.

This implements roadmap priority #4: skeletonize probability maps, find arc
endpoints, and extend probability along local orientation to recover plausible
continuations of known faults.

Usage:
    uv run python scripts/extend_endpoints.py \\
        --input predictions/baseline_oof.tif \\
        --output predictions/baseline_extended.tif \\
        --threshold 0.5 \\
        --max-distance-m 300 \\
        --decay-rate 0.15

The output will be a float32 GeoTIFF preserving the input's CRS, transform,
and valid-region mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

from gems.endpoint_extension import extend_fault_endpoints


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extend fault trace endpoints to recover continuations and splays."
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
        help="Output extended probability raster",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for skeletonization (default: 0.5)",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=300.0,
        help="Maximum extension distance in meters (default: 300)",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.15,
        help="Exponential decay rate for extended probabilities (default: 0.15)",
    )
    parser.add_argument(
        "--edge-strength",
        type=Path,
        help="Optional edge-strength raster to gate extension",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.3,
        help="Minimum edge strength to allow extension (default: 0.3)",
    )
    
    args = parser.parse_args()
    
    # Read input
    print(f"Reading {args.input}")
    with rasterio.open(args.input) as src:
        probabilities = src.read(1)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata
    
    # Validate
    if probabilities.dtype != np.float32:
        print(f"Warning: input is {probabilities.dtype}, converting to float32")
        probabilities = probabilities.astype(np.float32)
    
    # Create valid mask
    if nodata is not None:
        valid_mask = probabilities != nodata
    else:
        valid_mask = np.isfinite(probabilities)
    
    # Load edge strength if provided
    edge_strength = None
    if args.edge_strength:
        print(f"Reading edge strength from {args.edge_strength}")
        with rasterio.open(args.edge_strength) as edge_src:
            edge_strength = edge_src.read(1).astype(np.float32)
            if edge_strength.shape != probabilities.shape:
                print(
                    f"Error: edge strength shape {edge_strength.shape} "
                    f"does not match input {probabilities.shape}",
                    file=sys.stderr,
                )
                return 1
    
    # Compute pixel size from transform
    pixel_size_m = abs(transform[0])  # Assuming square pixels
    print(f"Pixel size: {pixel_size_m:.1f} m")
    print(f"Extension distance: {args.max_distance_m:.1f} m "
          f"= {args.max_distance_m / pixel_size_m:.1f} pixels")
    
    # Extend endpoints
    print("Extending fault endpoints...")
    extended = extend_fault_endpoints(
        probabilities,
        threshold=args.threshold,
        max_extension_distance_m=args.max_distance_m,
        pixel_size_m=pixel_size_m,
        decay_rate=args.decay_rate,
        edge_strength=edge_strength,
        edge_threshold=args.edge_threshold,
        valid_mask=valid_mask,
    )
    
    # Count extended pixels
    extended_pixels = np.sum((extended > probabilities) & valid_mask)
    print(f"Extended {extended_pixels} pixels")
    
    # Write output
    print(f"Writing {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Update profile for output
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=nodata if nodata is not None else np.nan,
    )
    
    with rasterio.open(args.output, "w", **out_profile) as dst:
        dst.write(extended, 1)
    
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
