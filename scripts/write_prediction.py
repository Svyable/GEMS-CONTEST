from __future__ import annotations

import argparse

import numpy as np

from gems.prediction import write_prediction_like_template
from gems.submission import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a .npy probability raster as a template-aligned GEMS GeoTIFF"
    )
    parser.add_argument("--prediction-npy", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prediction = np.load(args.prediction_npy)
    path = write_prediction_like_template(
        prediction,
        template_path=args.template,
        output_path=args.output,
    )
    report = validate_submission(path, args.template)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if not report.ok:
        return 1
    print(f"OK: wrote validated submission-format raster to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
