from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import rasterio


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, str, bool)):
        return value
    value = float(value)
    return value if math.isfinite(value) else str(value)


def raster_signature(path: str | Path) -> dict[str, Any]:
    """Return stable raster geometry, dtype, nodata, and band metadata."""
    with rasterio.open(path) as src:
        return {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": list(src.dtypes),
            "crs": src.crs.to_string() if src.crs else None,
            "resolution": [_clean_number(v) for v in src.res],
            "transform": [_clean_number(v) for v in tuple(src.transform)],
            "bounds": [_clean_number(v) for v in tuple(src.bounds)],
            "nodata": _clean_number(src.nodata),
            "band_tags": [src.tags(i) for i in range(1, src.count + 1)],
        }


def fingerprint_file(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    """Fingerprint a local competition/external-data file."""
    path = Path(path)
    shown_path = path
    if root is not None:
        try:
            shown_path = path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            shown_path = path

    result: dict[str, Any] = {
        "path": shown_path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() in {".tif", ".tiff"}:
        result["raster"] = raster_signature(path)
    return result


def fingerprint_files(
    paths: Iterable[str | Path], *, root: str | Path | None = None
) -> dict[str, Any]:
    """Create a deterministic manifest for a set of local files."""
    files = [fingerprint_file(path, root=root) for path in paths]
    files.sort(key=lambda item: item["path"])
    manifest = {"schema_version": 1, "files": files}
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest


def raster_alignment_errors(feature_path: str | Path, label_path: str | Path) -> list[str]:
    """Return alignment problems between feature and label rasters."""
    errors: list[str] = []
    with rasterio.open(feature_path) as features, rasterio.open(label_path) as labels:
        if (features.height, features.width) != (labels.height, labels.width):
            errors.append(
                "shape mismatch: "
                f"features={(features.height, features.width)} "
                f"labels={(labels.height, labels.width)}"
            )
        if features.crs != labels.crs:
            errors.append(f"CRS mismatch: features={features.crs} labels={labels.crs}")
        if features.transform != labels.transform:
            errors.append("geotransform mismatch")
        if features.bounds != labels.bounds:
            errors.append("bounds mismatch")
    return errors
