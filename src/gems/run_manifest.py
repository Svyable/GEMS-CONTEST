from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from gems.data import fingerprint_file, sha256_file


def build_run_manifest(
    *,
    run_id: str,
    config_path: str | Path,
    data_manifest_path: str | Path,
    git_commit: str,
    git_dirty: bool,
    artifacts: Iterable[str | Path] = (),
    hypothesis: str = "",
    notes: str = "",
    command: str = "",
) -> dict:
    """Build a machine-readable record tying a run to code, data, and artifacts."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    config = Path(config_path)
    data_manifest = Path(data_manifest_path)
    if not config.is_file():
        raise FileNotFoundError(config)
    if not data_manifest.is_file():
        raise FileNotFoundError(data_manifest)

    try:
        parsed_data_manifest = json.loads(data_manifest.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON data manifest: {data_manifest}") from exc

    artifact_records = [fingerprint_file(path) for path in artifacts]
    artifact_records.sort(key=lambda item: item["path"])

    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": hypothesis,
        "notes": notes,
        "command": command,
        "code": {
            "git_commit": git_commit,
            "git_dirty": bool(git_dirty),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": {
            "path": config.as_posix(),
            "sha256": sha256_file(config),
        },
        "data_manifest": {
            "path": data_manifest.as_posix(),
            "sha256": sha256_file(data_manifest),
            "manifest_sha256": parsed_data_manifest.get("manifest_sha256"),
        },
        "artifacts": artifact_records,
    }
