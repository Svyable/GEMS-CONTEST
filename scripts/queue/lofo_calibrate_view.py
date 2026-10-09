"""Leave-one-fold-out exact-threshold calibration for any CV view.

usage: lofo_calibrate_view.py RUN_DIR SCHEME FOLD_MAP [--buffer-note ...]
Reads RUN_DIR/fold-{k}.tif, writes RUN_DIR/scores/lofo-calibration.json.

For fold k the threshold maximizes the MEAN fit score over the other folds j,
each scored on exactly the evaluation region/truth that score_cv.py uses for
fold j (spatial: fold-j block; fault/trace: gems.cv fold spec). Fold k's labels
never influence its threshold. Candidate thresholds = union of each other fold's
exact best threshold and a 0.001 grid; each fold's curve is the canonical
gems.calibration.exact_threshold_curve (exact step function), so scores at the
candidates are exact. Evaluation on fold k uses binary_threshold (>=) + the
published metric on fold k's region.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from gems.calibration import binary_threshold, exact_threshold_curve
from gems.cv import fault_discovery_fold, trace_completion_fold
from gems.metric import distance_weighted_tversky

run, scheme, fold_map_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
truth_path = "data/raw/existing_faults.tif"
with rasterio.open(truth_path) as t:
    gt = t.read(1) > 0; valid = t.dataset_mask() > 0
folds = rasterio.open(fold_map_path).read(1).astype(np.int16)
if scheme == "spatial":
    ids = sorted(int(v) for v in np.unique(folds[valid]) if v >= 0)
else:
    ids = sorted(int(v) for v in np.unique(folds[gt & valid]) if v >= 0)

def region(k):
    if scheme == "spatial":
        return gt, valid & (folds == k)
    f = fault_discovery_fold if scheme == "fault" else trace_completion_fold
    spec = f(gt, folds, fold=k, valid_mask=valid, known_fault_exclusion_pixels=0)
    return spec.validation_truth, spec.evaluation_mask

preds, regions, curves = {}, {}, {}
for k in ids:
    with rasterio.open(run / f"fold-{k}.tif") as s:
        preds[k] = s.read(1).astype(np.float64)
    truth_k, mask_k = region(k)
    mask_k = mask_k & np.isfinite(preds[k])
    regions[k] = (truth_k, mask_k)
    curves[k] = exact_threshold_curve(preds[k], truth_k, valid_mask=mask_k)

def score_at(curve, t):
    th = curve.thresholds  # descending; state i predicts values >= th[i]
    idx = np.searchsorted(-th, -t, side="right") - 1  # last i with th[i] >= t
    if idx < 0:
        return float(curve.scores[0])
    return float(curve.scores[idx])

grid = np.round(np.arange(0.0, 1.0005, 0.001), 6)
out = []
for k in ids:
    others = [j for j in ids if j != k]
    cands = np.unique(np.concatenate([grid, [curves[j].best_threshold for j in others]]))
    cands = cands[(cands >= 0) & np.isfinite(cands)]
    means = np.array([np.mean([score_at(curves[j], t) for j in others]) for t in cands])
    t_best = float(cands[int(np.argmax(means))])
    truth_k, mask_k = regions[k]
    raw = distance_weighted_tversky(preds[k], truth_k, valid_mask=mask_k)
    cal = distance_weighted_tversky(binary_threshold(preds[k], t_best), truth_k, valid_mask=mask_k)
    out.append({"fold": k, "threshold": t_best, "fit_mean_other_folds": float(means.max()),
                "raw_score": float(raw), "calibrated_score": float(cal),
                "eval_pixels": int(mask_k.sum()), "truth_pixels": int((truth_k & mask_k).sum())})
    print(out[-1], flush=True)
raw = [r["raw_score"] for r in out]; cal = [r["calibrated_score"] for r in out]
summary = {"scheme": scheme, "method": "LOFO exact threshold: maximize mean exact_threshold_curve fit over other folds; evaluate once on held-out fold",
           "fold_map": fold_map_path, "folds": out,
           "raw_mean": float(np.mean(raw)), "raw_std": float(np.std(raw)),
           "calibrated_mean": float(np.mean(cal)), "calibrated_std": float(np.std(cal)),
           "std_note": "population std (ddof=0), same as score_cv macro_std"}
(run / "scores").mkdir(parents=True, exist_ok=True)
(run / "scores" / f"lofo-{scheme}.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "folds"}))
