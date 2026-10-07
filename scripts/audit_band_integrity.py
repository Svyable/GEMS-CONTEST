"""Compare official GEMS rasters with the committed manifest and band physics gate.

Use the metadata-only mode for preparation; it is *never* enough to authorize a
new potential-field transform. The raw scan is read-only and block-streamed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gems.band_audit import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/official.json")
    parser.add_argument("--features")
    parser.add_argument("--labels")
    parser.add_argument("--template")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-hashes", action="store_true", help="For debugging only")
    args = parser.parse_args()
    try:
        report = build_report(args.manifest, features=args.features, labels=args.labels,
                              template=args.template, verify_hashes=not args.skip_hashes)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}; status={report['status']}; "
          f"potential-field={report['physical_transform_gate']}")
    return 2 if report["status"] in {"FAIL", "MANIFEST_FAILURE"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
