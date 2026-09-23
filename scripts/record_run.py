from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from gems.run_manifest import build_run_manifest


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record code/data/config/artifact provenance for a GEMS run"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        commit = _git_output("rev-parse", "HEAD")
        dirty = bool(_git_output("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit("record_run.py must be run inside a Git checkout") from exc

    command = args.command
    if not command:
        command = " ".join(shlex.quote(value) for value in sys.argv)

    manifest = build_run_manifest(
        run_id=args.run_id,
        config_path=args.config,
        data_manifest_path=args.data_manifest,
        git_commit=commit,
        git_dirty=dirty,
        artifacts=args.artifact,
        hypothesis=args.hypothesis,
        notes=args.notes,
        command=command,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    if dirty:
        print("WARNING: repository had uncommitted changes when this run was recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
