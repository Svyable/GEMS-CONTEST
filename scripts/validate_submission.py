from __future__ import annotations

import argparse

from gems.submission import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a GEMS GeoTIFF submission")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--template", required=True)
    args = parser.parse_args()

    report = validate_submission(args.submission, args.template)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.ok:
        print("OK: submission matches the template constraints")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
