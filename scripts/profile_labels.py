from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

from gems.data import fingerprint_file
from gems.topology import fault_component_report, spatial_block_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile GEMS label topology before choosing fault/spatial CV details"
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--features",
        help="Optional aligned feature raster whose dataset mask defines the valid study area",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--top-components", type=int, default=20)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    with rasterio.open(args.labels) as label_src:
        labels = label_src.read(1) > 0
        valid = label_src.dataset_mask() > 0
        x_res, y_res = (abs(float(value)) for value in label_src.res)
        raster = {
            "width": label_src.width,
            "height": label_src.height,
            "crs": label_src.crs.to_string() if label_src.crs else None,
            "resolution": [x_res, y_res],
        }

        if args.features:
            with rasterio.open(args.features) as feature_src:
                if (feature_src.height, feature_src.width) != (label_src.height, label_src.width):
                    raise SystemExit("feature and label raster shapes differ")
                if feature_src.crs != label_src.crs or feature_src.transform != label_src.transform:
                    raise SystemExit("feature and label raster georeferencing differs")
                valid &= feature_src.dataset_mask() > 0

    report = {
        "schema_version": 1,
        "labels": fingerprint_file(args.labels),
        "features": fingerprint_file(args.features) if args.features else None,
        "raster": raster,
        "components_8_connected": fault_component_report(
            labels,
            valid_mask=valid,
            connectivity=8,
            pixel_size_x_m=x_res,
            pixel_size_y_m=y_res,
            top_n=args.top_components,
        ),
        "components_4_connected": fault_component_report(
            labels,
            valid_mask=valid,
            connectivity=4,
            pixel_size_x_m=x_res,
            pixel_size_y_m=y_res,
            top_n=args.top_components,
        ),
        "spatial_blocks": spatial_block_report(
            labels,
            block_size=args.block_size,
            valid_mask=valid,
        ),
    }

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
