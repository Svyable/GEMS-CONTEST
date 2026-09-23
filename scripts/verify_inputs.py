from __future__ import annotations

import argparse

import rasterio

from gems.data import raster_alignment_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GEMS feature/label raster alignment")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    errors = raster_alignment_errors(args.features, args.labels)
    with rasterio.open(args.features) as features:
        print(
            f"features: {features.width}x{features.height}, bands={features.count}, "
            f"crs={features.crs}, res={features.res}"
        )
        for band in range(1, features.count + 1):
            tags = features.tags(band)
            description = tags.get("description", "")
            category = tags.get("data_category", "")
            print(f"band {band:02d}: {description} [{category}]")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("OK: feature and label rasters are spatially aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
