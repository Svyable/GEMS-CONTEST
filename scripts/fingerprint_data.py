from __future__ import annotations

import argparse
import json
from pathlib import Path

from gems.data import fingerprint_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a stable SHA256/GeoTIFF metadata manifest for local GEMS data"
    )
    parser.add_argument("paths", nargs="+", help="Files to fingerprint")
    parser.add_argument("--root", help="Make manifest paths relative to this directory")
    parser.add_argument("--output", help="Write JSON here instead of stdout")
    args = parser.parse_args()

    manifest = fingerprint_files(args.paths, root=args.root)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
        print(f"wrote {output} ({manifest['manifest_sha256']})")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
