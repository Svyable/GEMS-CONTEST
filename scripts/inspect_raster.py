from __future__ import annotations

import argparse

import numpy as np
import rasterio


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a GeoTIFF's geometry and per-band statistics"
    )
    parser.add_argument("path")
    args = parser.parse_args()

    with rasterio.open(args.path) as src:
        print(f"path={args.path}")
        print(f"size={src.width}x{src.height} bands={src.count}")
        print(f"crs={src.crs} epsg={src.crs.to_epsg() if src.crs else None}")
        print(f"transform={src.transform}")
        print(f"resolution={src.res}")
        print(f"bounds={src.bounds}")
        print(f"dtypes={src.dtypes}")
        print(f"nodata={src.nodata}")
        for band in range(1, src.count + 1):
            arr = src.read(band, masked=True)
            vals = np.asarray(arr.compressed(), dtype=np.float64)
            if vals.size:
                q = np.quantile(vals, [0.0, 0.01, 0.5, 0.99, 1.0])
                print(
                    f"band={band:02d} n={vals.size} mean={vals.mean():.5g} "
                    f"std={vals.std():.5g} q0/q1/q50/q99/q100={q.tolist()}"
                )
            else:
                print(f"band={band:02d} no finite values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
