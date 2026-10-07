"""GEMS official-band integrity gate; no model claims or external downloads.

Manifest-only results cannot establish availability, integrity, or physical
units for the organizer's gated rasters. Live reads verify geospatial metadata,
file digests and pixel-level coverage without holding the whole stack in RAM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from gems.data import raster_signature, sha256_file

FEATURE_FILE = "gems-geodawn-numerical-features.tif"
LABEL_FILE = "existing_faults.tif"
TEMPLATE_FILE = "example_submission.tif"
FILES = (FEATURE_FILE, LABEL_FILE, TEMPLATE_FILE)

# Reference identity/order from the supplied 2026-09-23 official-data manifest.
BAND_NAMES = (
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope",
    "tc", "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi",
    "depth_to_base_surf", "ieq_n100a15", "cond_surf",
    "iso_grav_anom_hg", "det_elev_slope",
)
BASE_FIELDS = {"mag_anom", "rtp", "tmi", "iso_grav_anom"}
EXISTING_DERIVATIVES = {
    "tmi_hg", "tmi_vg", "iso_grav_anom_slope", "iso_grav_anom_vg",
    "iso_grav_anom_hg",
}
TOPOGRAPHY = {"det_elev", "det_elev_slope"}
AMBIGUOUS = {"tc", "deq_n100a15", "ieq_n100a15"}


def band_role(name: str) -> str:
    if name in BASE_FIELDS:
        return "potential_field_base_candidate"
    if name in EXISTING_DERIVATIVES:
        return "existing_derivative_control"
    if name in TOPOGRAPHY:
        return "coarse_terrain_control"
    if name in AMBIGUOUS:
        return "definition_needs_review"
    return "other_covariate"


def _geometry(signature: dict[str, Any]) -> dict[str, Any]:
    return {k: signature.get(k) for k in ("crs", "width", "height", "transform", "bounds", "resolution")}


def _manifest_entries(manifest: dict) -> dict[str, dict]:
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if len(entries) != len(manifest["files"]):
        raise ValueError("manifest contains duplicate paths")
    missing = set(FILES) - set(entries)
    if missing:
        raise ValueError(f"manifest missing required organizer rasters: {sorted(missing)}")
    return entries


def audit_manifest(manifest: dict) -> dict:
    entries = _manifest_entries(manifest)
    reference = entries[FEATURE_FILE]["raster"]
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("snapshot_description"):
        warnings.append("manual metadata snapshot; use canonical repository manifest for live checks")
    else:
        recorded = manifest.get("manifest_sha256")
        unsigned = {key: val for key, val in manifest.items() if key != "manifest_sha256"}
        digest = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if not recorded or recorded != digest:
            errors.append("manifest SHA256 self-check failed")
    if any(_geometry(reference) != _geometry(entries[name]["raster"]) for name in FILES[1:]):
        errors.append("feature/label/template grids differ in stored manifest")
    if reference.get("count") != len(BAND_NAMES):
        errors.append(f"expected {len(BAND_NAMES)} feature bands")
    tags = reference.get("band_tags", [])
    if len(tags) != reference.get("count"):
        errors.append("feature band tags are incomplete")
    names = [tags[i].get("band_name") for i in range(min(len(tags), len(BAND_NAMES)))]
    if tuple(names) != BAND_NAMES:
        errors.append("feature band identity or ordering differs from recorded official stack")
    if reference.get("crs") != "EPSG:32611" or reference.get("resolution") != [100.0, 100.0]:
        errors.append("expected EPSG:32611 100-meter feature raster")
    if entries[LABEL_FILE]["raster"].get("count") != 1:
        errors.append("expected one label band")
    if entries[TEMPLATE_FILE]["raster"].get("count") != 1:
        errors.append("expected one submission-template band")

    bands = []
    for index, tag in enumerate(tags, 1):
        name = tag.get("band_name", f"band_{index}")
        units = tag.get("units") or tag.get("unit")
        bands.append({
            "index": index, "name": name, "role": band_role(name),
            "category": tag.get("data_category"),
            "description": tag.get("description"),
            "units_from_tags": units,
            "unit_verification": "NOT_VERIFIED" if units else "NOT_REPORTED",
        })
    warnings.append("stored metadata alone cannot verify real raster contents")
    warnings.append("potential-field transformations blocked until authoritative unit and "
                    "field-definition evidence is recorded")
    return {
        "status": "METADATA_ONLY" if not errors else "MANIFEST_FAILURE",
        "errors": errors,
        "warnings": warnings,
        "source_manifest_sha256": manifest.get("manifest_sha256"),
        "feature_count": len(bands),
        "declared_units": sum(bool(b["units_from_tags"]) for b in bands),
        "grids_match_in_manifest": not any("grids differ" in x for x in errors),
        "bands": bands,
        "physical_transform_gate": "BLOCKED_PENDING_AUTHORITATIVE_UNITS_AND_DEFINITIONS",
    }


def _scan_pixels(feature_path: Path, label_path: Path, template_path: Path) -> dict:
    """Scan official raster values and masks in feature-file block windows."""
    with rasterio.open(feature_path) as feature, rasterio.open(label_path) as label, \
            rasterio.open(template_path) as template:
        bands = [{"index": i, "valid_pixels": 0, "missing_pixels": 0,
                  "valid_inside_feature_support": 0, "minimum": None, "maximum": None}
                 for i in range(1, feature.count + 1)]
        masks = {"feature_valid": 0, "label_valid": 0, "template_valid": 0,
                 "feature_label_disagreement": 0, "label_template_disagreement": 0,
                 "known_positive_outside_feature_support": 0}
        label_values: dict[str, int] = {}
        invalid_label_values = 0
        for _, window in feature.block_windows(1):
            fmask = feature.dataset_mask(window=window) > 0
            ymask = label.dataset_mask(window=window) > 0
            tmask = template.dataset_mask(window=window) > 0
            y = label.read(1, window=window)
            masks["feature_valid"] += int(fmask.sum())
            masks["label_valid"] += int(ymask.sum())
            masks["template_valid"] += int(tmask.sum())
            masks["feature_label_disagreement"] += int(np.count_nonzero(fmask ^ ymask))
            masks["label_template_disagreement"] += int(np.count_nonzero(ymask ^ tmask))
            masks["known_positive_outside_feature_support"] += int(
                np.count_nonzero((y > 0) & ymask & ~fmask)
            )
            vals, counts = np.unique(y[ymask], return_counts=True)
            for value, count in zip(vals, counts, strict=True):
                key = str(value.item())
                label_values[key] = label_values.get(key, 0) + int(count)
                if value not in (0, 1):
                    invalid_label_values += int(count)
            for index, stat in enumerate(bands, 1):
                values = feature.read(index, window=window, masked=True)
                raw = np.asarray(values.data)
                keep = ~np.ma.getmaskarray(values) & np.isfinite(raw)
                n = int(keep.sum())
                stat["valid_pixels"] += n
                stat["missing_pixels"] += raw.size - n
                stat["valid_inside_feature_support"] += int(np.count_nonzero(keep & fmask))
                if n:
                    mn, mx = float(raw[keep].min()), float(raw[keep].max())
                    stat["minimum"] = mn if stat["minimum"] is None else min(stat["minimum"], mn)
                    stat["maximum"] = mx if stat["maximum"] is None else max(stat["maximum"], mx)
        total = feature.width * feature.height
        for stat in bands:
            stat["coverage_fraction"] = stat["valid_pixels"] / total
            stat["coverage_within_feature_support"] = (
                stat["valid_inside_feature_support"] / masks["feature_valid"]
                if masks["feature_valid"] else 0.0
            )
        return {"total_pixels": total, "band_statistics": bands,
                "study_masks": masks, "valid_label_value_counts": label_values,
                "unexpected_valid_label_pixels": invalid_label_values}


def audit_live(manifest: dict, paths: dict[str, Path], *, verify_hashes: bool = True) -> dict:
    """Check file identity against the recorded manifest and scan pixel values.

    Failure of file SHA256 or raster identity is fatal: matching CRS alone is
    insufficient proof that training is using the registered official bands.
    """
    if manifest.get("snapshot_description"):
        raise ValueError("manual snapshot cannot validate live official rasters")
    expected = _manifest_entries(manifest)
    errors: list[str] = []
    warnings: list[str] = []
    signatures = {}
    for name in FILES:
        path = paths[name]
        if not path.is_file():
            raise ValueError(f"required raster not found: {name}")
        entry = expected[name]
        actual_size = path.stat().st_size
        if actual_size != entry["bytes"]:
            errors.append(f"{name}: byte size differs from manifest")
        if verify_hashes:
            if sha256_file(path) != entry["sha256"]:
                errors.append(f"{name}: SHA256 differs from manifest")
        else:
            warnings.append(f"{name}: SHA256 NOT VERIFIED (--skip-hashes used)")
        signature = raster_signature(path)
        signatures[name] = signature
        if signature != entry["raster"]:
            # Record only field names; no gated pixel values or entire tags dumped.
            differences = sorted(k for k in set(signature) | set(entry["raster"])
                                 if signature.get(k) != entry["raster"].get(k))
            errors.append(f"{name}: raster metadata differs in {', '.join(differences)}")

    reference = signatures[FEATURE_FILE]
    for name in FILES[1:]:
        if _geometry(signatures[name]) != _geometry(reference):
            errors.append(f"{name}: grid differs from feature grid")
    if reference.get("count") != len(BAND_NAMES):
        errors.append("live feature stack has wrong band count")
    if any(_geometry(signatures[name]) != _geometry(reference) for name in FILES[1:]):
        return {"status": "FAIL", "errors": errors, "warnings": warnings,
                "sha256_verified": verify_hashes, "scan_skipped": "incompatible grids"}
    if reference.get("count") != len(BAND_NAMES) or any(
        signatures[name].get("count") != 1 for name in FILES[1:]
    ):
        errors.append("unexpected band count; pixel scan is unsafe")
        return {"status": "FAIL", "errors": errors, "warnings": warnings,
                "sha256_verified": verify_hashes, "scan_skipped": "incompatible band counts"}
    scanned = _scan_pixels(paths[FEATURE_FILE], paths[LABEL_FILE], paths[TEMPLATE_FILE])
    if scanned["unexpected_valid_label_pixels"]:
        errors.append("labels contain values other than binary 0/1 in valid training pixels")
    if not scanned["study_masks"]["feature_valid"]:
        errors.append("feature raster contains no valid study area")
    if scanned["study_masks"]["feature_label_disagreement"]:
        warnings.append("feature/label valid-data masks differ; investigate training coverage")
    if scanned["study_masks"]["label_template_disagreement"]:
        warnings.append("label/template valid-data masks differ; investigate submission mask")
    if scanned["study_masks"]["known_positive_outside_feature_support"]:
        warnings.append("some valid known faults fall outside feature coverage")
    for band in scanned["band_statistics"]:
        if band["valid_pixels"] == 0:
            errors.append(f"feature band {band['index']} has no valid pixels")
        elif band["coverage_within_feature_support"] < 1.0:
            warnings.append(f"feature band {band['index']} is missing inside study support")
    return {"status": "FAIL" if errors else ("PASS" if verify_hashes else "UNVERIFIED_HASHES"),
            "errors": errors, "warnings": warnings,
            "sha256_verified": verify_hashes, **scanned}


def build_report(manifest_path: str | Path, *, features: str | Path | None = None,
                 labels: str | Path | None = None, template: str | Path | None = None,
                 verify_hashes: bool = True) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    report = audit_manifest(manifest)
    supplied = (features, labels, template)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError("supply all three official rasters or none of them")
    if all(value is not None for value in supplied):
        paths = dict(zip(FILES, (Path(features), Path(labels), Path(template)), strict=True))
        report["live"] = audit_live(manifest, paths, verify_hashes=verify_hashes)
        if report["errors"] or report["live"]["errors"]:
            report["status"] = "FAIL"
        else:
            report["status"] = report["live"]["status"]
    return report
