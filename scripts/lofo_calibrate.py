#!/usr/bin/env python3
"""Leave-one-fold-out exact threshold calibration on spatial-CV OOF predictions.

For fold k: stitch the OOF raster (each pixel predicted by the model whose spatial
fold held it out), fit the exact threshold via scripts/calibrate_threshold.py on the
union of the *other* folds' validation regions, and evaluate once on fold k.
Fold k's labels never influence its threshold.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

run = Path(sys.argv[1])
pattern = sys.argv[2]  # e.g. fold-{fold}.tif relative to run
fold_map_path = "data/processed/cv-spatial-v1.tif"
truth_path = "data/raw/existing_faults.tif"
work = run / "lofo"
work.mkdir(parents=True, exist_ok=True)
with rasterio.open(truth_path) as t:
    profile = t.profile.copy()
    tvalid = t.dataset_mask() > 0
folds = rasterio.open(fold_map_path).read(1)
ids = sorted(int(v) for v in np.unique(folds[tvalid]) if v >= 0)
oof = np.full(folds.shape, np.nan, dtype=np.float32)
for k in ids:
    with rasterio.open(run / pattern.format(fold=k)) as src:
        p = src.read(1).astype(np.float32)
        pvalid = src.dataset_mask() > 0
    sel = (folds == k) & tvalid & pvalid & np.isfinite(p)
    oof[sel] = p[sel]
pp = profile.copy()
pp.update(dtype="float32", nodata=np.nan, count=1, compress="deflate")
oof_path = work / "oof-stitched.tif"
with rasterio.open(oof_path, "w", **pp) as d:
    d.write(oof, 1)
region_ok = tvalid & np.isfinite(oof)
mp = profile.copy()
mp.update(dtype="uint8", nodata=None, count=1, compress="deflate")
results = []
for k in ids:
    paths = {}
    for name, m in (
        ("cal", region_ok & (folds >= 0) & (folds != k)),
        ("eval", region_ok & (folds == k)),
    ):
        paths[name] = work / f"{name}-mask-fold{k}.tif"
        with rasterio.open(paths[name], "w", **mp) as d:
            d.write(m.astype(np.uint8), 1)
    out = work / f"calibration-fold{k}.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_threshold.py",
            "--prediction",
            str(oof_path),
            "--truth",
            truth_path,
            "--calibration-mask",
            str(paths["cal"]),
            "--evaluation-mask",
            str(paths["eval"]),
            "--output-json",
            str(out),
        ],
        check=True,
    )
    r = json.loads(out.read_text())
    results.append(
        {
            "fold": k,
            "threshold": r["threshold"],
            "raw_score": r["evaluation"]["raw_score"],
            "calibrated_score": r["evaluation"]["binary_score"],
            "calibration_fit_score": r["calibration"]["binary_fit_score"],
            "calibration_raw_score": r["calibration"]["raw_score"],
        }
    )
raw = [r["raw_score"] for r in results]
cal = [r["calibrated_score"] for r in results]
summary = {
    "method": "leave-one-fold-out exact threshold (scripts/calibrate_threshold.py)",
    "folds": results,
    "raw_mean": float(np.mean(raw)),
    "raw_std": float(np.std(raw)),
    "calibrated_mean": float(np.mean(cal)),
    "calibrated_std": float(np.std(cal)),
    "std_note": "population std (ddof=0), same convention as score_cv macro_std",
}
(run / "scores").mkdir(exist_ok=True)
(run / "scores" / "lofo-calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
