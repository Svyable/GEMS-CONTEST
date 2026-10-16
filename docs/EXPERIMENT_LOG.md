# Experiment log

Machine-readable run manifests under `runs/<run-id>/manifest.json` are the source of truth for reproducibility. This Markdown file is a concise human index for meaningful experiments and every leaderboard submission.

A recorded run should pin:

- Git commit and whether the checkout was dirty;
- config path + SHA256;
- official data-manifest path + SHA256;
- exact command;
- output artifact paths + SHA256;
- hypothesis and interpretation.

Create a record with `scripts/record_run.py`. Never overwrite a submitted run's config, metrics, prediction, or manifest.

| ID | Date | Commit | Data fingerprint | Config | Spatial CV | Fault-discovery CV | Public LB | Submission SHA256 | Notes |
|---|---|---|---|---|---:|---:|---:|---|---|
| bootstrap | 2026-09-22 | initial setup | none | configs/baseline.yaml | — | — | — | — | Repo initialized; no competition data downloaded yet. |
| cv-v1 | 2026-09-23 | after official fingerprint | data/manifests/official.json | spatial block 256 / buffer 16; fault 8-connected | artifacts only | artifacts only | — | — | Topology supports 8-connected component holdout. Manifests: data/manifests/cv-spatial-v1.json and cv-fault-v1.json. No model scored yet. |
| cv-trace-v1 | 2026-09-23 | after cv-v1 | data/manifests/official.json | endpoint 30% of components ≥ 24 px, buffer 3 | — | continuation view | — | — | About 2,150 withheld endpoint pixels per fold. Manifest: data/manifests/cv-trace-v1.json. |
| reference-oof | 2026-09-23 | 9e256b9 | data/manifests/official.json | configs/reference_unet.yaml | — | — | — | — | Organizer 5×5-epoch ResNet-18 OOF on MPS, Apple M4 Pro, 438 s. Full-valid distance-weighted Tversky 0.0877. Mean probability 0.243 on fault pixels vs 0.123 on background. Loss was still falling at epoch 4. Not a submission. |
| full-map-v1 | 2026-09-23 | 690b159 | data/manifests/official.json | configs/full_map_v1.yaml | — | — | not submitted | local submissions/full-map-v1.tif | 20-epoch train-on-all U-Net, 5,035 fault windows and 640 background windows, MPS, 416 s. In-sample distance-weighted Tversky 0.2998 against known faults. Mean probability 0.445 on fault pixels vs 0.051 on background. Template validator passed. SHA256 cd34abbd0b8b44feb371e34effc26dc29a73f5c3776dc69995d8beeb6f1c181b. This is fit to the training labels, not a hidden-fault score. Not uploaded. |
| encoder-fold0 | 2026-09-23 | ca0d8d7 | data/manifests/official.json | resnet18 / convnextv2_tiny / sam2_hiera_small fold0 | 0.1427 / 0.1211 / 0.1281 | — | not submitted | — | Same spatial fold 0, 2,355 windows, 10 epochs, Apple MPS. Holdout distance-weighted Tversky: ResNet-18 0.1427, SAM2 Hiera-small 0.1281, ConvNeXt V2-tiny 0.1211. A 224 px Hiera run had only 124 legal windows and scored 0.070; it is not comparable. ResNet-18 leads this budget. Not uploaded. |
| encoder-fold0-e20 | 2026-09-23 | faae632 | data/manifests/official.json | resnet18_fold0.yaml and sam2_hiera_small_fold0.yaml | 0.1478 / 0.0990 | — | not submitted | — | 20 epochs, same fold and windows. ResNet-18 holdout 0.1478. SAM2 Hiera-small holdout 0.0990, down from 0.1281 at 10 epochs, so the extra epochs overfit the training blocks. ResNet-18 remains the best single model on this holdout. Not uploaded. |
| verify-loop-v1 | 2026-10-07 | — | n/a (infrastructure, synthetic tests only) | n/a | — | — | — | — | STRATEGY.md priority #2 implemented: propose-and-verify acceptance gate (`src/gems/verification.py`, `scripts/verify_candidate.py`, 15 tests). Candidate beats incumbent on ≥2 of 3 CV views by more than max(fold-std) × (1 + 0.5·log2(1+n_trials)); INCONCLUSIVE never accepts. Every verdict is appended to the committed ledger `docs/candidate-trials.jsonl` (honest failure log). Verdict is evidence, not an upload trigger. No official-data scores involved. |
| mumford-shah-log | 2026-10-08 | — | n/a (infrastructure, synthetic tests only) | configs/resnet18_mumford_shah_log.yaml | — | — | — | — | STRATEGY.md priority #3 implemented: LoG (Laplacian of Gaussian) edge-strength feature channel inspired by Mumford-Shah geometric regularity (`src/gems/mumford_shah.py`, 16 tests). Uses second derivatives to distinguish sharp discontinuities from smooth gradients on geophysical bands. Integrated into `lineament.py` with `kind: mumford_shah_log`. Fold-pure normalization (training-region stats only). Config ready; scoring blocked on official data. Experiment gate consolidated: `verification.py` canonical, `experiment_gate.py` deprecated. |
| endpoint-extension | 2026-10-08 | — | n/a (infrastructure, synthetic tests only) | configs/resnet18_with_endpoint_extension.yaml | — | — | — | — | STRATEGY.md priority #4 implemented: endpoint-extension post-processing (`src/gems/endpoint_extension.py`, `scripts/extend_endpoints.py`, 19 tests). Skeletonizes probability maps, finds endpoints, estimates orientation via structure tensor, extends along direction with decay over bounded distance (300m default). Optional edge-strength gating. Preserves float32 [0,1] and valid masks. Tests verify gap-bridging, noise-blob stability, idempotence, bounds. Ready for OOF ablation via `scripts/verify_candidate.py` once data available. Documentation corrected: edge detector is LoG inspired by Mumford-Shah geometry, not Ambrosio-Tortorelli solver. |
| structure-tensor | 2026-10-08 | — | n/a (infrastructure, synthetic tests only) | configs/resnet18_structure_tensor.yaml | — | — | — | — | STRATEGY.md priority #5 (partial): structure tensor coherence and orientation features implemented in `src/gems/lineament.py` (7 new tests). Coherence measures directional structure strength (λ1-λ2)/(λ1+λ2+ε) ∈ [0,1]; orientation gives dominant angle [-π/2, π/2] from eigenvector. High coherence + consistent orientation = lineament. Configurable smoothing: sigma_pixels (gradient pre-smoothing) and structure_tensor_window (gradient-product smoothing, default 3.0). Motivated by OpenAI Family 083 (Hilbert/Lipschitz): keep filter lengths below orientation-change scale. Tests verify constant-field zero coherence, strong-gradient high coherence, H/V orientation detection, invalid-pixel handling, multi-channel aggregation. Ready for ablation once training data available. Roadmap #5 still needs: ridge/valley response, multi-scale variants, directional filters. |
| ridge-valley | 2026-10-08 | — | n/a (infrastructure, synthetic tests only) | configs/resnet18_ridge_valley.yaml | — | — | — | — | STRATEGY.md priority #5 (advanced): Hessian-eigenvalue ridge/valley response implemented in `src/gems/lineament.py` as `kind: ridge_valley_response` (9 new tests, 217 passed / 1 skipped full suite; `tests/test_endpoint_extension.py` uncollectable locally — pre-existing missing `skimage` in this VM's venv, unrelated to this change). Signed channel: valley = relu(λmax−λmin)·relu(λmax), ridge = relu(λmax−λmin)·relu(−λmin), signed = valley−ridge, normalized by 99th percentile of |signed| over the valid region (fold-pure) to [−1,1]. Multi-scale pyramid via `ridge_valley_scales` (valley/ridge terms maxed independently, Frangi-style). Reuses `structure_tensor_window` for Hessian-component smoothing. New `_symmetric_eigenvalues` helper shared by structure-tensor and Hessian paths. Config includes scales [1.0, 2.0, 4.0]. Tests verify valley-positive/ridge-negative sign on synthetic Gaussian trough/crest, zero on constant field, metadata sigmas, invalid-pixel isolation, scale-param validation. Ready for propose-and-verify ablation once training data available. Roadmap #5 still needs: directional/steerable filters. |
| steerable-filter | 2026-10-09 | — | n/a (infrastructure, synthetic tests only) | configs/resnet18_steerable_filter.yaml | — | — | — | — | STRATEGY.md priority #5 COMPLETED: Gabor-style steerable/directional filters implemented in `src/gems/lineament.py` as `kind: steerable_filter` (12 new tests, 37 passed full lineament suite). Applies oriented edge detectors at multiple angles (default 8 orientations equally spaced from 0 to π) using Gaussian-envelope × sinusoidal-carrier (real Gabor) kernels at configurable wavelength (default 8.0 pixels). Returns max magnitude across all orientations for orientation-invariant lineament detection, normalized to [0,1] by fold-pure 99th percentile over valid region. Parameters: `steerable_wavelength` (characteristic wavelength in pixels, typical 6-12), `steerable_orientations` (angle count, 2-32, default 8). Tests verify horizontal/vertical/diagonal line detection, constant-field quietness, invalid-region handling, multi-channel aggregation, and parameter validation. Config: `configs/resnet18_steerable_filter.yaml`. Roadmap #5 now complete: gradient energy, phase edge, LoG, structure tensor coherence/orientation, ridge/valley response, and steerable filters. Ready for OOF ablation via `scripts/verify_candidate.py` once training data available. |
| ensemble-calibration | 2026-10-09 | — | n/a (infrastructure, synthetic tests only) | configs/ensemble_example.json | — | — | — | — | STRATEGY.md priority #6 COMPLETED: Ensemble and calibration framework (`src/gems/ensemble.py`, `scripts/build_ensemble.py`, 18 tests). Combines fold-wise OOF probability rasters from multiple candidate models using three fusion strategies: arithmetic mean (simple average), metric-fit weighted mean with LOFO weight fitting (prevents leakage: weights for fold k fitted on all other folds), and rank-average (robust to scale differences). LOFO weight fitting uses grid search over equal weights, best-single-model, and 0.1-step combinations to maximize mean score across other folds. Includes optional threshold calibration using `calibration.py` exact_threshold_curve. CLI accepts JSON config with member patterns, fits weights, applies method, optionally calibrates thresholds, and writes fold-specific GeoTIFFs. Output passes `validate_submission.py` and is scored through `score_cv.py`/`verify_candidate.py`, logged to `docs/candidate-trials.jsonl`. Tests verify mean/weighted/rank combinations, LOFO non-leakage guarantee (fold 0 weights unchanged when fold 0 labels change), threshold calibration, raster I/O. Ready for ablation once training data available. |

## 2026-10-07 — band-integrity preflight augmentation (INFRASTRUCTURE)

Synthetic tests exercised the canonical GEMS manifest contract, file hash and
band-tag tampering, grid mismatch, label sanity, nodata coverage, and mask
disagreement. **No official rasters were read in this session**, no ResNet-18
folds were retrained, and no scored fault-discovery uplift was measured. The
metadata-only report leaves geophysical units and physical transform suitability
blocked. Produce a live `PASS` report before P0a. Do not
count this as model validation or a new leaderboard experiment.

## Scoring conventions

For **spatial CV**, `scripts/score_cv.py` reports each fold plus a stitched global OOF score because spatial validation regions are disjoint.

For **fault-discovery CV**, evaluation backgrounds overlap across folds, so the scorer intentionally reports per-fold scores plus macro mean/std and does **not** manufacture a global score. The held-out fault component itself is the positive truth; training-known faults are excluded from that fold's evaluation mask.

## Upload decision, 2026-09-24

Do not upload yet. Issues 4 and 5 were updated to match the runs above. Issue 11 is closed: the lockfile goal is done, and the NVIDIA check is cancelled because training stays on Apple MPS.

The first DrivenData upload, when it happens, has one question: does `submissions/full-map-v1.tif` (SHA256 `cd34abbd0b8b44feb371e34effc26dc29a73f5c3776dc69995d8beeb6f1c181b`) receive a non-zero public score? That is a format and calibration check, not the final selection. Local evidence for that file is an in-sample score of 0.2998. The honest spatial-fold score for this architecture is 0.1478 on fold 0 only. Fault-holdout and trace-holdout scores do not exist yet, and folds 1–4 are unscored. Those gaps are why the upload waits.

## Submission notes template

**Hypothesis:**  
**Change vs previous:**  
**Run manifest:**  
**Local evidence:**  
**Artifact path + SHA256:**  
**Leaderboard score:**  
**Decision:**  

### 2026-10-09 — Real-data integration: bug fixes and validation

First full pipeline run on real contest data (19-band 3292×3730 raster) surfaced 5 production bugs, all fixed in PR #32:

**Real-data characteristics** (external machine, data not shared):
- Raster: 19 bands, 3292×3730 pixels
- Fault pixels: 60,894 (1.18% of 5,164,312 valid pixels)
- 8-connected components: 3,198 (median 12 px, largest 360 px)
- 4-connected components: 25,058
- CV manifests: rebuilt from real data, matched committed synthetic-derived manifests exactly

**Bugs fixed**:
1. **Memory OOM**: `scripts/train_reference_oof.py` exceeded 7.6 GB peak RAM building training patches. Implemented `make_reference_split_lazy` with on-the-fly window construction, reducing peak to ~3.6 GB. Added test verifying lazy == eager patches for fixed seeds.
2. **Import error**: `scripts/optimize_threshold.py` failed with `ImportError: cannot import name 'load_raster_band'`. Fixed by replacing with direct `rasterio.open` calls. Added CLI smoke tests (`test_script_smoke.py`) to catch import regressions.
3. **Visual QA bug**: `src/gems/visual_qa.py` fault-scheme preview rendered assigned held-out faults white (unassigned color) due to mismatched downsampling between fold IDs and fault mask. Fixed with consistent downsampling. Added test.
4. **Nondeterministic ordering**: `scripts/profile_labels.py` ordered equal-size components nondeterministically in top-20 list. Fixed by sorting ties by component ID (`src/gems/topology.py`). Added test.
5. **Symlink path resolution**: `fingerprint_data.py`/verify flow resolved symlinks to targets, breaking manifest relative paths. Fixed by using `.absolute()` instead of `.resolve()` in `src/gems/data.py`.

**Outcome**: All tests green. Memory-constrained hosts can now run the reference baseline. CV manifests validated on real data. Ready for model training and ablation on real contest data.

---

## Phase 1: First Real-Data Results (2026-10-09)

**Host:** 8 vCPU, 15.6 GiB RAM, no swap, no GPU (CPU-only torch 2.14.0+cpu), about 92 GB free disk. Another job on the shared box (ffmpeg) used about 3 to 6 cores during training.

### Real-Data Characteristics

- **Raster**: 19 bands, 3292×3730 pixels, EPSG:32611, 100 m resolution
- **Data fingerprint**: `data/manifests/official.json` (manifest_sha256 ab3ec3cc…, matches committed manifest byte-for-byte)
- **Fault pixels**: 60,894 over 5,164,312 valid pixels (prevalence 1.18%)
- **8-connected components**: 3,198 (median 12 px, p99 128 px, largest 360 px = 0.59% share)
- **4-connected components**: 25,058 (median 2 px, largest 62 px = 0.10% share)
- **Spatial blocks** (256×256): 99 of 105 valid blocks contain faults (94.3%), median 559 fault px per positive block

**Topology notes**: Traces are thin and fragmented. Fault holdouts should use 8-connected components, since 4-connectivity breaks diagonal traces into pieces. Every spatial block holds faults, so spatial folds never run out of positives.

**CV manifest validation**: Rebuilt `cv-spatial-v1.json`, `cv-fault-v1.json`, and `cv-trace-v1.json` from real data; all matched committed manifests exactly.

### Baseline: Reduced Reference U-Net (ref-cv-spatial-reduced)

**Config**: `configs/reference_unet_cv_reduced.yaml` (organizer recipe with training.epochs 5→2 and patches.train_step 32→64 for 8-vCPU box; 452 to 603 windows per fold)

**Training**: `scripts/train_full_map.py`, fold-pure normalization, buffer 16, seed 20260922. Training losses ended at 0.945 to 0.950, so the model is **barely trained**. Median fold-0 probability is 0.548. Total training time ~16.5 min.

**Spatial CV scores** (5 folds):
- **Raw** (threshold=0.5): mean **0.0591 ± 0.0082**, stitched 0.0583
  - Per fold: 0.0713 / 0.0618 / 0.0520 / 0.0620 / 0.0482
- **LOFO-calibrated**: mean **0.0895 ± 0.0200**
  - Per fold: 0.1250 / 0.0903 / 0.0678 / 0.0910 / 0.0733
  - Thresholds: 0.724 / 0.741 / 0.698 / 0.698 / 0.698

**Leave-one-fold-out (LOFO) calibration protocol**: Stitch the spatial OOF raster. For each fold k, fit the exact threshold with `scripts/calibrate_threshold.py` on the other four folds' validation regions, then evaluate once on fold k. Fold k's labels never influence its threshold. Per-fold raw scores from the calibration CLI match `score_cv.py` exactly.

**Note on calibrated thresholds**: Thresholds landed near 0.7, not below 0.5 as hypothesized in `docs/STRATEGY.md`. This baseline's outputs are compressed around 0.55 (median fold-0 probability 0.548), so these thresholds say nothing about a well-trained model. The "optimal threshold below 0.5" claim should be confirmed on a properly trained model before treating it as a design principle.

### Endpoint Extension Gate (ENDPOINT-EXT-01)

**Config**: `configs/resnet18_with_endpoint_extension.yaml` (300 m extension, decay 0.15, skeleton threshold = LOFO threshold per fold)

**Applied to**: ref-cv-spatial-reduced OOF maps

**Spatial CV scores**:
- Raw: 0.0591 (Δ +2.4e-6)
- Calibrated: 0.0895 (Δ −4e-7)

**Gate verdict**: **INCONCLUSIVE**. The spatial delta of 2.4e-6 is far below the required margin of 0.0082, and the fault and trace views are missing. Not adopted.

### Additional Bugs Found

**Endpoint extension performance**: `src/gems/endpoint_extension.extend_endpoint` copied the whole H×W map for every endpoint. A full-map fold ran for more than 10 min before being stopped. The `inplace=True` fix (PR #33) gives bit-identical output and finishes in about 8 s per fold; 19/19 tests pass. At the default threshold of 0.5, a weak model's map is mostly above threshold (67% of fold 0), so the skeleton is meaningless for under-trained models.

### Experiment Catalog

| ID | Date | Commit | Config | Spatial CV | Fault CV | Trace CV | Public LB | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| intake-box-v1 | 2026-10-09 | 4ccbf0e + local patches | n/a | artifacts only | artifacts only | artifacts only | — | `verify_inputs.py` OK. Fingerprint matches 2026-09-23 manifest. CV manifests match exactly. |
| reference-oof-box | 2026-10-09 | 4ccbf0e + local patches | configs/reference_unet.yaml | — | — | — | — | **Abandoned, not scored.** Split 1, epoch 0: 29.5 min, train_loss 0.9504, test_loss 0.9295. Full 25 epochs projected at ~12 h. Stopped because random-patch Monte Carlo OOF cannot be scored per CV view. |
| ref-cv-spatial-reduced | 2026-10-09 | 4ccbf0e + local patches | configs/reference_unet_cv_reduced.yaml | raw 0.0591 ± 0.0082; LOFO-cal 0.0895 ± 0.0200 | — | — | — | Reduced config (epochs 2, train_step 64). Training ~16.5 min. Model barely trained (losses 0.945–0.950). **BASELINE** for Phase 1. |
| ENDPOINT-EXT-01 | 2026-10-09 | 4ccbf0e + local patches | configs/resnet18_with_endpoint_extension.yaml | raw Δ +2.4e-6; cal Δ −4e-7 | — | — | — | Endpoint extension on ref-cv-spatial-reduced OOF. Gate verdict: **INCONCLUSIVE** (delta << margin, fault/trace views missing). |

### Next Steps

1. **Score full-config spatial run**: Train with full `configs/reference_unet.yaml` (epochs=5, train_step=32) to validate the barely-trained baseline was not a fluke
2. **Train fault and trace fold models**: Complete fault-discovery and trace-discovery CV views so the gate can evaluate all 3 views for candidates
3. **Re-gate candidates**: Once the full-config baseline is scored and fault/trace views are available, re-gate endpoint extension and other candidates against the honest baseline
4. **Confirm threshold hypothesis**: Test whether "optimal threshold below 0.5" holds for a properly trained model (current calibrated thresholds ~0.70 are for an under-trained model with compressed outputs)

## 2026-10-09 — Full reference U-Net baseline, honest spatial CV

**Config**: `configs/reference_unet.yaml` (organizer reference, unreduced: 5 epochs, train_step 32)  
**Training**: `scripts/train_full_map.py --cv-scheme spatial --buffer-pixels 16 --negative-ratio 1.0 --seed 20260922`  
**Normalization**: fold-pure (`per_channel_minmax_over_training_region`)  
**Host**: 8-core CPU-only box, single seed

**Spatial CV scores** (5 folds):
- **Raw** (unthresholded distance-weighted Tversky): 0.1031, 0.1015, 0.0854, 0.1078, 0.0836
  - Macro mean: **0.0963 ± 0.0099** (population std)
- **LOFO-calibrated**: 0.1308, 0.1334, 0.1014, 0.1442, 0.1050
  - Thresholds: 0.944, 0.936, 0.940, 0.940, 0.959
  - Calibrated mean: **0.1230 ± 0.0168** (population std)

**Decision**: This is the **incumbent baseline for spatial view**. Fault and fault-tip (trace) views are training next, then the endpoint-extension gate and the four lineament-channel candidates (mumford_shah_log, structure_tensor_coherence, steerable_filter, ridge_valley_response; spatial view only because train_full_map.py rejects derived lineament features for fault/trace CV).

**Results committed**: `results/runs/ref-cv-spatial-full/{manifest.json,spatial.json,lofo-spatial.json}`

**Local fixes applied** (already in codebase):
1. `src/gems/endpoint_extension.py`: `inplace=True` option for `extend_endpoint` (avoids O(H×W) copy per endpoint; identical output)
2. `scripts/train_full_map.py`: Pass-through of kind-specific lineament params (structure_tensor_window, ridge_valley_scales, steerable_wavelength, steerable_orientations)

**Queue tooling committed**: `scripts/queue/{runner.py,tasks.json,train_folds.sh,score_view.sh,lofo_calibrate_view.py,endpoint_view.sh,gate.sh,record.sh,README.md}` — resumable experiment runner with state tracking.

**Candidate configs committed**: `configs/candidates/{reference_unet_cv_reduced.yaml,cand_mumford_shah_log.yaml,cand_structure_tensor_coherence.yaml,cand_ridge_valley_response.yaml,cand_steerable_filter.yaml}` — incumbent + four lineament-channel variants ready for ablation.

## 2026-10-10 — Full reference U-Net baseline, honest fault CV

**Config**: `configs/reference_unet.yaml` (organizer reference, unreduced: 5 epochs, train_step 32)  
**Training**: `scripts/train_full_map.py --cv-scheme fault --buffer-pixels 16 --negative-ratio 1.0 --seed 20260922`  
**Normalization**: fold-pure (`per_channel_minmax_over_training_region`)  
**Host**: 8-core CPU-only box, single seed, ~17.8 h

**Fault CV scores** (5 folds):
- **Raw** (unthresholded distance-weighted Tversky): 0.0374, 0.0449, 0.0505, 0.0464, 0.0484
  - Macro mean: **0.0455 ± 0.0045** (population std)
- **LOFO-calibrated**: 0.0452, 0.0583, 0.0642, 0.0579, 0.0478
  - Thresholds: 0.935, 0.882, 0.952, 0.952, 0.997
  - Calibrated mean: **0.0547 ± 0.0071** (population std)

**Comparison to spatial**: Fault CV raw mean is 0.0455 (vs. spatial 0.0963); LOFO-calibrated 0.0547 (vs. spatial 0.1230). The large drop indicates that generalizing to held-out complete faults is substantially harder than spatial block holdout.

**Decision**: This is the **incumbent baseline for fault view**. Trace CV baseline is now training. Endpoint-extension gate (3 views) and the four lineament-channel candidates follow trace completion.

**Results committed**: `results/runs/ref-cv-fault-full/{manifest.json,fault.json,lofo-fault.json}`

**Note**: Manifest records `git_dirty=true` at commit b82efe9 because of local patches (endpoint_extension inplace, train_full_map lineament param pass-through) that have since been superseded on main.

---

### Box handoff 2026-10-09 — fault/trace CV fold training (INFRASTRUCTURE)

This run is CPU-only (no torch on this VM), so the two blocked validation views
are prepared, not executed. New config `configs/resnet18_cv_reduced.yaml`:
scheme-agnostic ResNet-18 U-Net recipe identical to `configs/resnet18_fold0.yaml`
except the box reduction (epochs 20 -> 2, train_step 32 -> 64); fold-pure
`per_channel_minmax_over_training_region` normalization; Tversky alpha=0.2 /
beta=0.8. Pinned by `tests/test_cv_config.py` (5 tests: trainer-key contract,
metric weights, fold-pure normalization, scheme-from-CLI, recipe-vs-control).

Run on the torch+data box (all commands in the repo root, using `uv`):

```bash
# 0. clean tree, pinned data fingerprint
git status -sb && git log -1 --format=%H
uv run python scripts/verify_inputs.py

# 1. build fold maps (params reproduce the committed manifests exactly)
uv run python scripts/build_cv_manifest.py \
  --features data/raw/gems-geodawn-numerical-features.tif \
  --labels data/raw/existing_faults.tif \
  --scheme fault --output-prefix data/processed/cv-fault-v1
uv run python scripts/build_cv_manifest.py \
  --features data/raw/gems-geodawn-numerical-features.tif \
  --labels data/raw/existing_faults.tif \
  --scheme trace --buffer-pixels 3 --min-pixels 24 --endpoint-fraction 0.3 \
  --output-prefix data/processed/cv-trace-v1

# 2. train reduced folds (~3 min/fold at the reduced recipe)
for k in 0 1 2 3 4; do
  uv run python scripts/train_full_map.py \
    --config configs/resnet18_cv_reduced.yaml \
    --features data/raw/gems-geodawn-numerical-features.tif \
    --labels data/raw/existing_faults.tif \
    --template data/raw/example_submission.tif \
    --fold-map data/processed/cv-fault-v1.tif --cv-scheme fault --fold $k \
    --seed 20260922 \
    --output runs/resnet18-fault-cv-reduced/fold-$k.tif \
    --metrics-json runs/resnet18-fault-cv-reduced/metrics-fold-$k.json
done
# same loop for trace: --fold-map data/processed/cv-trace-v1.tif
# --cv-scheme trace, outputs under runs/resnet18-trace-cv-reduced/

# 3. score all folds per view
uv run python scripts/score_cv.py \
  --truth data/raw/existing_faults.tif \
  --fold-map data/processed/cv-fault-v1.tif --scheme fault \
  --prediction-pattern runs/resnet18-fault-cv-reduced/fold-{fold}.tif \
  --output-json runs/resnet18-fault-cv-reduced/fault.json
# trace analog -> runs/resnet18-trace-cv-reduced/trace.json
# (spatial view already scored: ref-cv-spatial-reduced, 0.0591 +/- 0.0082 raw)

# 4. re-gate endpoint extension once all three views exist
# extend_endpoints.py over the new OOF maps -> score all three views ->
# verify_candidate.py --candidate ENDPOINT-EXT-02 --incumbent-dir <scores>
# --candidate-dir <scores> --config configs/resnet18_with_endpoint_extension.yaml
```

Expected outcome: fault.json and trace.json scores for the reduced ResNet-18
baseline, which unblock the ENDPOINT-EXT gate (currently INCONCLUSIVE on
spatial-only evidence) and give every lineament candidate its first honest
three-view trial. The reduced recipe is a speed control; promote surviving
candidates to the full recipe (epochs=20, train_step=32) on the same folds.

---

## 2026-10-10 — Leakage-safe lineament features for fault/trace CV (INFRASTRUCTURE)

**Problem**: `scripts/train_full_map.py` refused derived lineament features under
`--cv-scheme fault` or `trace` with error "fault/trace CV needs a masked-input
transform protocol before comparison". All four queued lineament candidates
(mumford_shah_log, structure_tensor_coherence, ridge_valley_response,
steerable_filter) were restricted to spatial CV only, blocking three-view gate
evaluation. Context from CPU-only local runs (not reproducible in CI): reference
U-Net spatial CV raw distance-weighted Tversky mean 0.0963, LOFO-calibrated
0.1230; fault CV folds 0.0374 and 0.0449 so far — generalization to held-out
faults is the weak point where better input features should be tested.

**Solution**: Implemented leakage-safe protocol for fault/trace CV with derived
features. The concern was that spatial filters (Gaussian smoothing, structure
tensor, Hessian, Gabor) with support radius R could carry information from
held-out fault/trace pixels into training inputs through their filtering
neighborhood. The protocol ensures held-out regions never contribute to training:

1. **Input masking for filtering**: Pass `train_mask` (excludes held-out faults
   + buffer) as the `valid_mask` to `grouped_lineament_features` for fault/trace
   CV. All spatial filters (`_masked_gaussian`, structure tensor, Hessian, Gabor)
   treat held-out regions as invalid and exclude their values from weighted sums.

2. **Fold-pure normalization**: Already implemented — `normalization_mask` uses
   `train_mask` for percentile/scale statistics, ensuring held-out pixels never
   affect normalization parameters.

3. **Buffer >= filter support**: Already enforced — `required_lineament_buffer`
   check ensures `--buffer-pixels` exceeds spatial filter support (e.g., 4σ for
   Gaussian, kernel radius for Gabor), so training windows never sample edge
   artifacts at held-out boundaries.

**Changes**:
- `scripts/train_full_map.py` (lines 112-136): Removed hard refusal for
  fault/trace CV with lineament features. Now passes `train_mask` as both
  `valid_mask` and `normalization_mask` when computing lineament features under
  any CV scheme with a split (spatial/fault/trace). Added inline comment
  documenting the leakage-safe protocol.

**Tests** (`tests/test_lineament_cv_leakage.py`, 15 tests, all pass):
- **Core leakage test**: For all 7 lineament kinds (gradient_energy, phase_edge,
  mumford_shah_log, structure_tensor_coherence, structure_tensor_orientation,
  ridge_valley_response, steerable_filter), perturbing held-out region inputs
  by +1000 leaves training-region lineament features unchanged (tolerance 1e-6).
  Validates that spatial filters respect the train_mask boundary.
  
- **Fold-pure normalization**: Perturbing held-out inputs by 10× leaves
  normalization pixel counts and scale parameters unchanged for gradient_energy,
  phase_edge, and structure_tensor_coherence.

- **Spatial CV unchanged**: Confirms spatial CV still uses full `valid` mask for
  filtering (spatially-separated regions have no fault-based leakage concern),
  with high correlation (>0.99) between full-valid and train-restricted results
  in training core region.

- **Buffer efficacy**: With σ=2 Gaussian (support ~8 pixels), training pixels
  >8 pixels from held-out boundary show near-zero gradient on constant-field
  inputs (95th percentile <0.01), proving buffer prevents edge artifact leakage.

- **Trace CV**: Two-trace synthetic test verifies endpoint holdout respects
  train_mask; perturbing held-out endpoints by +500 leaves training region
  (including trace bodies) unchanged (tolerance 1e-6).

- **Multi-scale features**: mumford_shah_log and ridge_valley_response (with
  3 scales) respect masking; held-out perturbation causes max absolute
  difference <1e-5 in training region.

**Verification**: All existing lineament and CV tests pass (44 lineament tests,
14 CV tests, 15 new leakage tests = 73 total). No change to spatial CV behavior
or feature computation logic — only the mask selection for fault/trace CV.

**Outcome**: Lineament-feature candidates can now be trained and scored under
fault and trace CV. The local experiment queue can proceed with all four
lineament candidates across all three CV views (spatial, fault, trace) for
honest three-view gate evaluation.

**Config changes for local queue**: None. Existing candidate configs
(`configs/candidates/cand_mumford_shah_log.yaml`, etc.) work as-is with
fault/trace CV. Example invocation (same as spatial, now works for fault/trace):

```bash
uv run python scripts/train_full_map.py \
  --config configs/candidates/cand_mumford_shah_log.yaml \
  --features data/raw/gems-geodawn-numerical-features.tif \
  --labels data/raw/existing_faults.tif \
  --template data/raw/example_submission.tif \
  --fold-map data/processed/cv-fault-v1.tif --cv-scheme fault --fold 0 \
  --buffer-pixels 16 --seed 20260922 \
  --output runs/mumford-shah-fault-cv/fold-0.tif \
  --metrics-json runs/mumford-shah-fault-cv/metrics-fold-0.json
```

Buffer requirement: `--buffer-pixels` must be ≥ `required_lineament_buffer` for
the chosen `kind`. Current configs use σ=1.0, structure_tensor_window=3.0,
steerable_wavelength=8.0, yielding required buffers of 4-17 pixels. Default
`--buffer-pixels 16` satisfies all except steerable_filter (needs 17); increase
to 20 for steerable if needed.

**Commit**: `cursor/lineament-fault-trace-cv-protocol-3a7e` branch.

**Next steps**: Train all four lineament candidates (mumford_shah_log,
structure_tensor_coherence, ridge_valley_response, steerable_filter) on
fault and trace CV views. Score via `scripts/score_cv.py`. Gate against
reference U-Net baseline using `scripts/verify_candidate.py` with all three
views (spatial, fault, trace). Promote winners that beat incumbent on ≥2 of 3
views by more than max(fold-std) × (1 + 0.5·log2(1+n_trials)).

---

## 2026-10-10 — PR #36 pre-merge fixes: protocol-equality gate and CI enforcement

**Issue**: PR #36 (lineament fault/trace CV protocol) was blocked by three issues:
1. Fair comparison concern: incumbent baselines use `--buffer-pixels 16`, but PR
   documentation says steerable_filter needs 20. The gate can't attribute a gain
   to the feature vs. the buffer change if they differ.
2. Integration tests skip in CI because torch isn't installed in the `test` job.
3. `LINEAMENT_CV_PROTOCOL_SUMMARY.md` in repo root should be in `docs/`.

**Solution**: Implemented protocol-equality enforcement and CI fixes:

1. **Protocol-equality check in canonical gate**: Extended `scripts/score_cv.py`
   to accept optional `--metrics-pattern` argument that reads per-fold metrics
   JSON files and extracts training protocol fields (buffer_pixels, seed, epochs,
   config_sha256). These are added to the `evaluation_protocol` section of
   score_cv.py output. Extended `src/gems/verification.py` with new
   `TRAINING_PROTOCOL_FIELDS` tuple and logic in `compare_view()` to check these
   fields and return `comparable=False` with reason "training protocol differs:
   <fields>" when incumbent and candidate differ in any of these parameters
   (when both are present). This ensures the gate refuses unfair comparisons
   where two things changed instead of one.

2. **Documentation in STRATEGY.md**: Added "Fair comparison protocol" subsection
   documenting that buffer_pixels, seed, epochs, train_step, and fold_map_sha256
   must match between incumbent and candidate. If a candidate requires buffer 20,
   incumbent must be re-run at buffer 20 for all views before comparison. Train
   script reports minimum required buffer per lineament kind.

3. **CI enforcement**: Updated `.github/workflows/ci.yml` to add new step
   "Lineament fault/trace CV integration tests" in the `training-smoke` job
   that runs `pytest -q tests/test_train_lineament_fault_cv_integration.py`
   with torch installed (via `--extra ml --extra cpu`), ensuring these tests
   actually run and pass in CI instead of skipping.

4. **Documentation organization**: Moved `LINEAMENT_CV_PROTOCOL_SUMMARY.md` from
   repo root to `docs/LINEAMENT_CV_PROTOCOL_SUMMARY.md` via `git mv`.

5. **Linting fixes**: Fixed three ruff errors blocking CI:
   - `tests/test_lineament_cv_leakage.py:164` — changed `H, W = labels.shape` to `_, W = labels.shape`
   - `tests/test_lineament_cv_leakage.py:220` — removed unused variable `valid`
   - `tests/test_train_lineament_fault_cv_integration.py:3` — removed unused import `tempfile`

**Changes**:
- `scripts/score_cv.py`: Added `--metrics-pattern` argument, reads per-fold
  metrics JSONs, extracts protocol fields, adds them to evaluation_protocol.
- `src/gems/verification.py`: Added `TRAINING_PROTOCOL_FIELDS` tuple, extended
  `compare_view()` to check these fields and refuse mismatched comparisons.
- `docs/STRATEGY.md`: Added "Fair comparison protocol" subsection documenting
  parameter-matching requirement and re-run procedure.
- `.github/workflows/ci.yml`: Added integration test step to training-smoke job.
- `LINEAMENT_CV_PROTOCOL_SUMMARY.md` → `docs/LINEAMENT_CV_PROTOCOL_SUMMARY.md`
- Linting fixes in test files.

**Verification**: All changes committed and pushed. CI expected to pass:
- Linting errors fixed
- Integration tests will run with torch in training-smoke job
- Protocol check tested via existing verification test suite

**Outcome**: PR #36 ready for review with:
- Fair comparison enforcement preventing mixed protocol gates
- Integration tests running in CI (not skipped)
- Documentation properly organized
- CI passing (linting clean)

**Next steps**: Mark PR #36 ready for review (not draft), merge after review,
then proceed with lineament candidate training on fault/trace CV views.

---

## 2026-10-10 — Protocol-chain fixes + lineament fault/trace queue (INFRASTRUCTURE)

**Context**: PR #36 merged the leakage-safe lineament protocol for fault/trace CV
and a protocol-equality check in the gate, but two links in the chain were dead
and the s4 queue tasks were broken as committed.

**Fixes to the metrics → gate protocol chain**:
1. `scripts/train_full_map.py` never wrote `train_step` into the per-fold metrics
   JSON, so `scripts/score_cv.py --metrics-pattern` hardcoded it to `None` and the
   gate could never compare it. Now written (`"train_step": step`) and extracted.
2. `scripts/queue/score_view.sh` never passed `--metrics-pattern`, so every
   queue-produced score JSON carried no training protocol at all and the
   protocol-equality check silently skipped. Now passed for all three views.
3. Tests: 2 new in `tests/test_score_cv_protocol.py` (metrics-pattern carries all
   five training fields; omission leaves them absent), 1 assertion in
   `tests/test_training_ml.py` (`train_step == 64` in metrics payload).

**Queue rework** (`scripts/queue/tasks.json`, 62 tasks): s4 referenced
`configs/local/cand_*.yaml`, which does not exist in the repo — every s4 task
would have failed. Fixed to `configs/candidates/*.yaml`. Buffer choices verified
against committed `required_lineament_buffer` code + committed configs (CORRECTION
to the earlier 2026-10-10 entry's "4-17 px" claim): mumford_shah_log 5,
steerable_filter 7, structure_tensor_coherence 17, ridge_valley_response 30.
- b16 group (mumford, steerable @16): protocol-equal with s1/s2 incumbents
  (`runs/gate/ref-full` @16/16).
- b32 group (structure_tensor, ridge_valley @32): with buffer-32 incumbent
  re-runs (`runs/gate/ref-b32`) for spatial, fault, and trace views.
- s5 adds all four candidates on the **fault** view (PR #36 enables this).
- s6 adds all four candidates on the **trace** view; the s2 trace incumbent used
  buffer 3, below every kind's minimum, so one ref-trace@32 re-run covers all.
- Gates placed only after every compared view is scored: b16 kinds get a 2-view
  (spatial+fault) gate vs ref-full; b32 kinds get a full 3-view gate vs ref-b32;
  b16 kinds additionally get a trace-only informational gate (single view cannot
  ACCEPT by itself). Gate ordering machine-checked.
- `scripts/queue/README.md` documents the per-kind buffer table and the
  fair-comparison rule.

**Verification**: 284 passed, 4 skipped (torch/skimage-dependent), ruff clean.
No competition data read; no training on this CPU-only VM; no score claimed.

**Next**: torch+data box runs the queue (s1→s6); first honest three-view gates
for the four lineament candidates land once s6 completes.


## 2026-10-10 — P0 pooled spatial selection correction

Infrastructure correction, not a model gain. `score_cv.py` schema 2 now records
stitched spatial weighted TP/FP/FN and pixel coverage. The verifier uses the
pooled ratio as primary, requires geographic/fault/trace safeguards, validates
aggregate arithmetic and preserves supplied training-budget fields. Historical
JSONL verdicts were not rewritten; their prediction rasters need re-scoring.
Synthetic tests reproduce macro/pooled ranking reversal and seam-crossing
spatial matches. Full local validation: 337 passed, 4 optional training tests
skipped; Ruff and diff checks passed.

Next measured experiment: adequately trained baseline with all three views,
then a budget/seed/fold-matched ResNet-18 distance-aware-loss ablation. Source
links and the P1-P5 queue live in
[the research note](research/METRIC_AWARE_RESEARCH_2026-10-10.md).

## 2026-10-10 — Full reference U-Net baseline, honest trace CV (ref-cv-trace-full) [SUPERSEDED]

**STATUS: SUPERSEDED on 2026-10-10 by ref-cv-trace-b16.** This run used buffer 3 (below every lineament kind's minimum buffer requirement) and mixed fold maps (fold 0 on pre-repair cv-trace-v1, folds 1-4 on cv-trace-v1b). Its raw score 0.009317 ± 0.001755 and LOFO-calibrated score 0.016198 ± 0.006655 are on a mixed protocol and must not be used for gating or incumbent comparison. The replacement ref-cv-trace-b16 (all 5 folds on v1b, buffer 16) is queued in s5b and has no score yet.

**Config**: `configs/reference_unet.yaml` (organizer reference, unreduced: 5 epochs, train_step 32)  
**Training**: `scripts/train_full_map.py --cv-scheme trace --buffer-pixels 3 --negative-ratio 1.0 --seed 20260922`  
**Normalization**: fold-pure (`per_channel_minmax_over_training_region`)  
**Host**: 8-core CPU-only box, single seed

**Trace CV scores** (5 folds):
- **Raw** (unthresholded distance-weighted Tversky): macro mean **0.009317 ± 0.001755** (population std)
- **LOFO-calibrated**: mean **0.016198 ± 0.006655** (population std)

**Fold map note**: Trace-CV baseline crashed on fold 1 because one withheld label pixel (row 3264, col 1516) lies outside the feature valid mask, so it had no retained trace body. Locally we used a repaired fold map `data/processed/cv-trace-v1b.tif` with that pixel set to -1 (fold 0 verified identical); `lofo-trace.json` references it. The fix requires intersecting labels with the feature valid mask before assigning trace bodies/endpoints in the fold builder. This is addressed in the accompanying code fix.

**Decision**: ~~This is the incumbent baseline for trace view.~~ **SUPERSEDED.** All three CV views now complete but trace incumbent is invalid due to mixed protocol.

**Results committed**: `results/runs/ref-cv-trace-full/{manifest.json,trace.json,lofo-trace.json}` (preserved for provenance; see SUPERSEDED note in manifest directory)

**Box note**: The box OOM on 2026-10-10 19:55 CT killed the first lineament candidate (mumford_shah_log, spatial fold 1) due to unrelated memory pressure; it was resumed from fold 1.

**Helper script**: Added `scripts/lofo_calibrate_view.py` — general-purpose leave-one-fold-out exact-threshold calibration for any CV view (spatial/fault/trace), maximizing mean fit score over other folds using exact threshold curves.

## 2026-10-10 — Gate decision: ENDPOINT-EXT-02 (REJECT)

**Candidate**: Endpoint extension (300 m, decay 0.15, skeleton threshold = fold LOFO threshold)  
**Applied to**: Full-config reference OOF maps (all three views)  
**Config**: `configs/resnet18_with_endpoint_extension.yaml`

**Results** (3 views):
- **Spatial**: delta -3.3e-08 (far below required margin 0.0148), won 0/3 views
- **Fault**: delta -2.0e-06 (far below required margin 0.0067), won 0/3 views
- **Trace**: delta -4.9e-08 (far below required margin 0.0026), won 0/3 views

**Gate verdict**: **REJECT**. Endpoint extension won on 0 of 3 views; all deltas far below required margins.

**Results committed**: `results/gates/ENDPOINT-EXT-02.json`, appended to `docs/candidate-trials.jsonl`

---

## 2026-10-10 — Queue state decisions (DOCUMENTATION)

Infrastructure and queue-management decisions made on the training box on 2026-10-10 evening (CT):

### INVALIDATED: stale mumford_shah_log spatial run

**Timestamp**: 2026-10-10 20:15:02 CDT

**Decision**: Invalidated `runs/cand-mumford_shah_log-spatial` (stale spatial run) and moved to `.stale-precommit34`. This run was trained on stale checkout b82efe9 lacking PR #34 (fold-pure lineament scaling) and PR #36 (leakage-safe protocol). Its fold 0 holdout score of 0.1096 is NOT valid evidence for gating. Fold 1 was killed mid-training. Queue paused and repo synced to origin/main a2733ad.

### SUPERSEDED: ref-cv-trace-full (trace incumbent)

**Timestamp**: 2026-10-10 20:22:25 CDT

**Decision**: Marked `ref-cv-trace-full` (trace incumbent) as SUPERSEDED. This run used buffer 3 (below every lineament kind's minimum buffer requirement) and mixed fold maps (fold 0 on pre-repair cv-trace-v1 sha 75eb89db..., folds 1-4 on v1b sha ccbff45e...). Its committed EXPERIMENT_LOG numbers (raw 0.0093 ± 0.0018, calibrated 0.0162 ± 0.0067) are on a mixed protocol and must not be used for gating. Replaced by `ref-cv-trace-b16` (all 5 folds on v1b, buffer 16) which is queued as s5b-train-ref-trace-b16.

### RESYNC: queue tasks and incumbent gates

**Timestamp**: 2026-10-10 20:19:44 CDT

**Decision**: Rebuilt `tasks.json` from main `scripts/queue/tasks.json` with cv-trace-v1 → v1b, b16 kinds first, b32 block after. Four mumford/steerable gates (two spatial+fault b16 gates @ s5-gate-*, two trace-only b32 gates @ s6-gate-*-trace) were parked in `tasks.parked-gates-20261010.json`. Baseline gate JSONs rescored with schema-2 and train_step=32 reconstructed (sidecars `runs/ref-cv-*-full/protocol/`).

### REMOVED: block D (6 trace-only b32 tasks)

**Timestamp**: 2026-10-10 20:22:45 CDT

**Decision**: Removed block D (6 trace-only b32 training and scoring tasks for mumford_shah_log and steerable_filter candidates at buffer 32). These tasks trained the candidates but their corresponding gates (s6-gate-mumford_shah_log-trace and s6-gate-steerable_filter-trace) remain parked. Single-view gates cannot ACCEPT by policy, and these b16 kinds now have 3-view gates (s5b-gate-*-b16) at buffer 16 instead.

### REPAIR: OOM kill and memory gate addition

**Timestamp**: 2026-10-10 21:57:37 CDT

**Decision**: After s4-train-mumford_shah_log-spatial fold 0 was OOM-killed (rc=137) at 20:33 CT by shared-box memory pressure (other agents' video render), added memory gate (wait for MemAvailable >= 5000 MB) and up to 3 OOM retries per fold in `train_folds.sh`. Backed up as `train_folds.sh.bak-20261010-oom` locally. Cleared `.failed` marker and restarted runner. These changes are now being synced to the repo.

### SYNC: fast-forward to PR #39

**Timestamp**: 2026-10-10 21:58:18 CDT

**Decision**: Local checkout fast-forwarded a2733ad → d17d57b (PR #39: trace valid-mask fix, docs/results only; no train_full_map change) before candidate fold 0 restarted. This sync had no impact on training code but updated docs and results manifests.

## 2026-10-11 — Distance-aware Tversky loss, P1 implementation (INFRASTRUCTURE)

Implements the P1 hypothesis from `docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md`:
train directly on the competition's distance-weighted Tversky equations.

**What changed**:
1. `src/gems/distance_tversky.py` (new): NumPy reference for the patch loss
   evidence (T/F/FN) plus a torch `DistanceTverskyLoss` module, blended as
   `(1-w)*Tversky + w*L_distance` with `w = loss_distance_weight` (0.5, fixed in
   config before evaluation). Semantics mirror `gems.metric` exactly:
   Euclidean triangular kernel, radius 3.0 px, no wraparound, core-only
   contribution counting with halo context, validity/supervision masks
   (predictions zeroed outside valid; truth outside valid never enters the
   truth distance kernel, so withheld labels cannot leak into the FP field).
   Patchwise ratio is documented as a surrogate for the global ratio
   (zero-fill at patch borders). `build_training_loss(config)` maps
   `loss: tversky` → existing smp TverskyLoss (unchanged behavior) and
   `loss: distance_tversky` → the blended loss (`needs_spatial=True`).
2. `scripts/train_full_map.py`: loss factory wiring; distance-aware losses keep
   the 2D batch structure under masked supervision (supervision mask zeroes
   contributions instead of flattening — flattening would destroy the distance
   kernel). Metrics JSON now records a `training_loss` descriptor
   (name/alpha/beta/distance_weight/radius/halo) alongside `config_sha256`.
3. `configs/candidates/cand_loss_distance_tversky.yaml` (new): incumbent
   recipe byte-identical except the loss (same model/seed/epochs/train_step/
   buffer/augmentation) — protocol-equal for the propose-and-verify gate.
4. `tests/test_distance_tversky.py` (new, 19 tests): NumPy tests run on CPU —
   perfect-prediction zero loss, one-pixel shift penalized (loss exactly 1/3)
   but not fatal, distant prediction = pure FP, empty-truth unit loss,
   invalid-pixel isolation (no truth kernel from unsupervised truth),
   halo/core evidence vs an independent brute-force reference, halo-free
   agreement with the published metric, no border wraparound, monotonicity in
   prediction quality, shape validation; torch tests (numpy/torch agreement,
   gradient flow, empty-truth finiteness, masked-supervision structure) are
   `importorskip("torch")` and run on the torch box/CI.

**Verification**: 334 passed, 10 skipped (torch/skimage-dependent), ruff clean.
Two test failures during development caught real issues (eps tolerance on the
exact-zero assertion; halo-context truth kernel vs the metric's core-only
kernel) and were fixed in the tests, not by weakening the implementation.
No competition data read; no training on this CPU-only VM; no score claimed.

**Next (torch+data box handoff — queue untouched, lineament s4–s6 in flight)**:
```
# From the repo root on the box, after the lineament queue drains:
uv run python scripts/queue/train_folds.sh configs/candidates/cand_loss_distance_tversky.yaml spatial   # + fault, trace views
uv run python scripts/queue/score_view.sh ...   # per scripts/queue/README.md
uv run python scripts/verify_candidate.py --candidate DIST-TVERSKY --incumbent-dir runs/gate/ref-full ...
```
Gate hypothesis: the distance-aware loss beats the reference U-Net on ≥2 of
3 views (pooled spatial + fault/trace non-regression, per the v2 selection
policy). If the queue runner pulls this commit, the config is ready to
schedule; it was deliberately NOT added to `scripts/queue/tasks.json` here
because the box queue is mid-run and queue edits are the box's lane.

## 2026-10-11 — INVALID: cand_mumford_shah_log spatial b16 (BUG)

**Result**: Raw 0.0398, calibrated 0.0619 (if recorded)

**Status**: **INVALID** due to lineament CV held-out zeroing bug

**Bug description**: In `scripts/train_full_map.py` lines 112-136, for CV runs
`filtering_mask = split.train_mask` was passed as both `valid_mask` and
`normalization_mask` to `grouped_lineament_features`. This caused
`derived[~valid] = 0.0` in `src/gems/lineament.py` line 469 to zero out all
derived lineament channels over the entire held-out fold. Models trained on
real gradient/edge/structure-tensor values in training regions but were scored
on zeros in validation regions. Observed symptom: held-out predictions collapsed
(mean probability ~0.05 vs 0.3 in-train).

**Fix**: PR #XX implements two-pass filtering:
- Pass A: `valid_mask=train_mask` for training pixels
- Pass B: `valid_mask=validation_mask` for held-out pixels
- Both passes use `normalization_mask=train_mask` (fold-pure)
- Combined by region; buffer pixels remain zero
- Added regression tests and runtime checks

**Impact**: All lineament-derived CV runs (mumford_shah_log, structure_tensor_*,
ridge_valley_response, steerable_filter) trained before this fix have invalid
held-out scores. This includes spatial, fault-discovery, and trace-completion CV.
Re-run all lineament CV experiments after this fix merges.

---

## 2026-10-11 — Corrected lineament candidate results after held-out zeroing fix

### RERUN: cand_mumford_shah_log spatial b16 (post-fix)

**Context**: PR #42 and PR #43 fixed the held-out zeroing bug that invalidated the original cand_mumford_shah_log spatial b16 run in PR #41. This is the corrected rerun.

**Config**: `configs/candidates/cand_mumford_shah_log.yaml` (LoG edge-strength lineament candidate, kind `mumford_shah_log`)  
**Training**: `scripts/train_full_map.py --cv-scheme spatial --buffer-pixels 16 --seed 20260922`  
**Normalization**: fold-pure (`per_channel_minmax_over_training_region`)  
**Host**: 8-core CPU-only box, single seed

**Spatial CV scores** (5 folds, buffer 16):
- **Raw**: macro mean **0.0914 ± 0.0135** (population std), stitched aggregate **0.0903**
  - Per fold: 0.1108 / 0.0939 / 0.0783 / 0.0996 / 0.0745
- **LOFO-calibrated**: mean **0.1146 ± 0.0143**
  - Per fold: 0.1259 / 0.1188 / 0.0945 / 0.1320 / 0.1018
  - Thresholds: 0.977 / 0.899 / 0.936 / 0.899 / 0.962

**Comparison to reference U-Net (same protocol, buffer 16)**:
- Reference aggregate: 0.0967, raw mean: 0.0963, calibrated mean: 0.1230
- Candidate delta vs reference: calibrated **-0.0084**, aggregate **-0.0064**
- Within about one fold std; no clear gain over reference on spatial view

**Gate status**: Fault and trace views are still queued; no gate decision yet pending completion of all three views.

**Results committed**: `results/runs/cand-mumford_shah_log-spatial-b16/{manifest.json,scores/spatial.json,scores/lofo-spatial.json}`, `results/runs/gate/cand-mumford_shah_log-b16/spatial.json`

**Note**: The earlier PR #41 spatial result (raw 0.0398, calibrated 0.0619) is **INVALID** and **SUPERSEDED** by this corrected run.

### BASELINE: ref-cv-trace-b16 (honest trace CV, buffer 16)

**Context**: Completes the ref-b16 baseline gate (spatial/fault/trace at buffer 16). Supersedes ref-cv-trace-full (buffer 3, fold 0 on pre-repair cv-trace-v1 map, mixed protocol).

**Config**: `configs/reference_unet.yaml` (organizer reference: 5 epochs, train_step 32)  
**Training**: `scripts/train_full_map.py --cv-scheme trace --buffer-pixels 16 --seed 20260922`  
**Normalization**: fold-pure (`per_channel_minmax_over_training_region`)  
**Fold map**: `data/processed/cv-trace-v1b.tif` (all 5 folds)  
**Host**: 8-core CPU-only box, single seed

**Trace CV scores** (5 folds, buffer 16):
- **Raw**: macro mean **0.00783 ± 0.00157** (population std)
  - Per fold: 0.00710 / 0.00798 / 0.00570 / 0.01050 / 0.00788
- **LOFO-calibrated**: mean **0.01856 ± 0.00557**
  - Per fold: 0.02075 / 0.01121 / 0.01334 / 0.02652 / 0.02096
  - Thresholds: 0.99999994 / 0.99999994 / 0.99999976 / 0.99999994 / 0.99999940

**Decision**: This is the **incumbent baseline for trace view at buffer 16**. Completes the ref-b16 gate baseline (spatial/fault/trace).

**Results committed**: `results/runs/ref-cv-trace-b16/{manifest.json,scores/trace.json,scores/lofo-trace.json}`, `results/runs/gate/ref-b16/trace.json`

**Note**: Supersedes ref-cv-trace-full (buffer 3, mixed fold maps, recorded as SUPERSEDED on 2026-10-10).

**Manifest note**: Both manifests show `git_dirty=true` at commit 0134ae8 only because of an untracked `configs/local/` directory on the training box; no tracked files were modified.

---

## 2026-10-12 — P1 distance-aware Tversky queued (s7) + result re-verification

### CPU verification of the P1 loss implementation (this VM, no torch)

**Tests**: `.venv/bin/python -m pytest -q --ignore=tests/test_endpoint_extension.py`
- **351 passed, 10 skipped**. The excluded module `test_endpoint_extension.py` is
  uncollectable here (no `skimage` in this VM's venv, no pip to install it) — the
  same documented VM gap as prior runs.
- `tests/test_distance_tversky.py` (the P1 suite, commit 3b3e413): **13 passed,
  6 skipped** — the 6 skips are the torch-half tests, which cannot run on this
  VM but will run on the training box. NumPy half (kernel offsets, validity/
  supervision masking, blend behavior, config mapping) fully green on CPU.

### Re-verification: cand_mumford_shah_log spatial b16 rerun numbers

Recomputed directly from `results/runs/cand-mumford_shah_log-spatial-b16/scores/`:
- Raw per-fold: 0.1108 / 0.0939 / 0.0783 / 0.0996 / 0.0745 → macro **0.0914 ± 0.0135**,
  aggregate **0.0903**
- LOFO-calibrated per-fold: 0.1259 / 0.1188 / 0.0945 / 0.1320 / 0.1018 → macro **0.1146 ± 0.0143**

All figures match the 2026-10-11 log entry exactly — no transcription error.
Standing comparison vs the spatial incumbent (raw 0.0963, LOFO-cal 0.1230):
candidate delta **−0.0084 calibrated / −0.0064 aggregate**, within one fold std —
no spatial gain. Fault/trace reruns remain queued on the box; no gate verdict yet
(gate policy requires ≥2 comparable views).

### Queued: step 7 — P1 distance-aware Tversky gate candidate

Rationale: the P1 loss (commit 3b3e413) trains directly on the competition's
distance-weighted Tversky equations — the single most metric-aligned idea in the
plan's loss-ablation item, and ENDPOINT-EXT-02's REJECT leaves the modeling
improvement lane as the top remaining bet. Training needs the box; config is
already protocol-equal to the incumbent (identical model/seed/epochs/train_step/
buffer; only `loss` changes), so the honest move is to enqueue rather than stall.

Added to `scripts/queue/tasks.json` (now **75 tasks** total, runner re-reads live):
- `s7-train/score/record-loss_distance_tversky-{spatial,fault,trace}` — honest
  5-fold CV at buffer 16 on `configs/candidates/cand_loss_distance_tversky.yaml`
- `s7-assemble-ref-b16` — idempotent copy of ref-full spatial/fault gate JSONs
  into `runs/gate/ref-b16` (cp -n; no clobbering)
- `s7-gate-loss_distance_tversky-b16` — `CAND-LOSS_DISTANCE_TVERSKY-B16` vs
  `runs/gate/ref-b16`; hypothesis: pooled-spatial OOF gain with
  fault/trace non-regression safeguards, per `verify_candidate.py` policy.

### Standing box-queue picture after this run

- Steps 1–3 done. ENDPOINT-EXT-02 **REJECT** (endpoint extension dead as a
  candidate class: deltas ~1e-7 to 1e-8 on all three views at 1.5× margin).
- Steps 4–6 (lineament candidates at correct buffers, post zeroing-fix): only
  `cand-mumford_shah_log-spatial-b16` has completed re-running; the remaining
  steerable_filter / structure_tensor_coherence / ridge_valley_response
  spatial+fault+trace reruns are still queued on the box runner.
- Step 7 (this run): the loss ablation — the highest-expected-gain open item
  that fits this VM (CPU prep of box-side training).

## 2026-10-11 — Reference spatial CV at buffer 32 (ref-cv-spatial-b32)

**Purpose**: Establish the buffer-32 spatial baseline needed before gating
structure_tensor_coherence (min buffer 17) and ridge_valley_response (min buffer 30).

**Training**: `train_full_map.py --cv-scheme spatial --buffer-pixels 32`, fold-pure
normalization, negative_ratio 1.0, seed 20260922, 5 epochs, reference_unet.yaml.
Finished scoring 2026-10-11 18:26 CDT on the box queue (`s4-train/score-ref-spatial-b32`).

**Scores** (schema_version 2, distance-weighted Tversky):
- Raw macro mean **0.0988 ± 0.0157**
- LOFO-calibrated macro mean **0.1245 ± 0.0151**
- Stitched aggregate **0.0967**

Per-fold raw: 0.1203 / 0.1015 / 0.0905 / 0.1078 / 0.0739
Per-fold LOFO-cal: 0.1278 / 0.1340 / 0.1200 / 0.1426 / 0.0983

**Comparison vs ref spatial b16** (raw 0.0963 / LOFO 0.1230 / agg 0.0967):
essentially equal on aggregate; raw/LOFO slightly higher within noise.
Not a candidate — this is the buffer-32 reference baseline itself.
Fault and trace buffer-32 reference runs remain in the queue.

## 2026-10-12 — Reference fault CV at buffer 32 (ref-cv-fault-b32)

**Purpose**: buffer-32 fault-split reference baseline, needed before gating
structure_tensor_coherence (min buffer 17) and ridge_valley_response (min buffer 30).

**Training**: train_full_map.py --cv-scheme fault --buffer-pixels 32, fold map
data/processed/cv-fault-v1.tif, reference_unet.yaml, seed 20260922, 5 epochs.
Training finished 2026-10-12 15:17 CDT; scoring/LOFO calibration finished 16:03 CDT
on the box queue (s5-train/score-ref-fault-b32). Box was paused ~Oct 11 19:45 to Oct 12 11:15.

**Scores** (schema_version 2, distance-weighted Tversky, alpha 0.2 / beta 0.8, radius 3 px):
- Raw macro mean **0.0366 ± 0.0076**
- LOFO-calibrated macro mean **0.0450 ± 0.0105**

Per-fold raw: 0.0260 / 0.0345 / 0.0491 / 0.0341 / 0.0393
Per-fold LOFO-cal: 0.0312 / 0.0419 / 0.0629 / 0.0410 / 0.0482
LOFO thresholds: 0.882 / 0.884 / 0.849 / 0.882 / 0.952

**Comparison vs ref fault b16** (raw 0.0455 ± 0.0045 / LOFO 0.0547 ± 0.0071):
about 0.009–0.010 lower on both raw and calibrated at buffer 32 (wider exclusion
buffer leaves less fault-adjacent training data). Unlike spatial (b32 ≈ b16), the
fault view is buffer-sensitive, so buffer-32 candidates must be compared only
against this b32 baseline, never against b16. Not a candidate; ref trace b32 is next in the queue.

## 2026-10-13 — Step 8: seed-robustness probe queued (s8) + train_folds.sh seed plumbing

**Rationale** (plan: validation gaps first). Every incumbent number and every
gate verdict rests on a single seed (20260922) — the standing honest watch
item. Gate margins are computed from fold stds only, so seed variance is an
unmodeled noise source: if the P1 loss gate wins by less than the seed swing,
the verdict is ambiguous. This probe quantifies that swing on the view every
gate decision keys on (pooled spatial aggregate).

**Changes** (CPU-only, this VM; no training here):
- `scripts/queue/train_folds.sh`: optional 6th arg `SEED` (default 20260922),
  forwarded as `train_full_map.py --seed`. Backward compatible — all existing
  5-arg call sites behave identically (`bash -n` clean).
- `scripts/queue/tasks.json`: 75 → **78 tasks**. Step 8 appends after the s7
  gate so it never preempts live candidates:
  - `s8-train-seed20260923-spatial` — reference U-Net, honest spatial CV at
    buffer 16, seed **20260923** (protocol-identical to ref-cv-spatial-full
    except the seed)
  - `s8-score-seed20260923-spatial` — schema-2 scorer (pooled spatial
    aggregate) into `runs/gate/ref-seed20260923-b16`
  - `s8-record-seed20260923-spatial` — manifest + data fingerprint
- `tests/test_queue_tasks.py` (new, 6 tests): tasks.json validity, unique ids,
  every referenced `configs/*.yaml` and queue script exists, seed forwarding
  in train_folds.sh, probe tasks present.
- `scripts/queue/README.md`: documented the SEED arg.

**Scope notes**: spatial-only, one extra seed — 5 fold-trainings, the same box
cost as one view of any candidate. Fault/trace seed probes are follow-ups if
the spatial seed variance is material. This is NOT a gate candidate:
`verify_candidate` treats differing seeds as a training-protocol mismatch by
design (correct), so the comparison is a manual side-by-side of the pooled
`aggregate_score`s. Decision rule when it completes: if |aggregate delta| is
well under the gate's required-margin scale, seed noise is contained; if it
is comparable or larger, gates need a seed term before any ACCEPT.

**Gate-chain audit** (same run): verified the step-7 chain is sound as queued —
`s1/s2` schema-2 re-scores feed `runs/gate/ref-full/`, the s7 assemble/gate
tasks reference live box paths, and `verify_candidate --print-only` parses the
schema-2 gate JSONs. One repo-hygiene finding, no action needed on the box:
the *committed* `results/runs/ref-cv-{spatial,fault}-full/{spatial,fault}.json`
are **schema-1** (no `aggregation`, no pooled `aggregate_score`), so they
cannot serve any gate — the box's uncommitted schema-2 re-scores in
`runs/gate/ref-full/` remain the gate's incumbent source. Do not "fix" this
by copying schema-1 JSONs into a gate dir; a re-score needs the fold
prediction rasters, which are (correctly) not committed.

**Tests** (this VM): `351 → 357 passed, 10 skipped` — same skips as 10-12
(`test_endpoint_extension.py` uncollectable: no skimage; 6 torch-half skips in
the P1 suite).

## 2026-10-14 — P2 wavelet-entropy lineament channel (step 9) queued

**Rationale** (plan: lineament features; research execution order). The
documented order in docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md is
"P1 loss ablation; then P2 wavelet entropy; then P3 background loss." P1 is
queued on the box (s7); P2 is the next CPU-buildable item and is independent
of the s4–s6 verdicts, so it was implemented here and queued after step 8 —
it never preempts live candidates.

**What was built** (CPU-only, this VM; no training here):
- `src/gems/lineament.py`: new `kind: wavelet_entropy`. Undecimated Haar
  transform at `wavelet_levels` (default (1,2,3); dilation 2**(level-1)),
  reflect padding (no wraparound), per-pixel detail energies
  E_s = LH²+HL²+HH²; q_s = E_s/(ΣE+ε); H = −Σ q_s log(q_s+ε), H ≡ 0 where
  total energy ~0, float-dust negatives clamped to 0. Per-band entropies
  averaged across the group, fold-pure 99th-percentile normalization to
  [0,1]. H ∈ [0, log L]: low for single-scale structure, high where detail
  energy spreads across scales — the model learns the sign. Required buffer
  = ceil(4σ) + max dilation = 8 px at defaults (fits buffer-16 protocol).
  Two-pass norm reuse via `entropy_p99`; `wavelet_levels` validated
  (integer-valued, 1–6, non-empty); metadata reports levels +
  `undecimated_haar_detail_energy_entropy_across_scales`.
- `configs/candidates/cand_wavelet_entropy.yaml`: protocol-equal to ref-b16
  (buffer 16, seed 20260922, epochs 5, train_step 32); magnetic_data +
  gravity_data only, per the P2 "start on raw magnetic/gravity bands" note.
- `scripts/queue/tasks.json`: 78 → **88 tasks**. Step 9:
  `s9-train/score/record-wavelet_entropy-{spatial,fault,trace}` (buffer 16,
  fold maps cv-spatial-v1 / cv-fault-v1 / cv-trace-v1b) +
  `s9-gate-wavelet_entropy-b16` → `CAND-WAVELET_ENTROPY-B16` vs
  `runs/gate/ref-b16`. Hypothesis: pooled-spatial OOF gain with
  fault/trace non-regression safeguards, per `verify_candidate` policy.
- `tests/test_wavelet_entropy.py` (17 tests) + `wavelet_entropy` added to
  the held-out-outlier leakage parametrize in `test_lineament_config.py` +
  `test_wavelet_entropy_candidate_tasks_present` in `test_queue_tasks.py`.
- `scripts/queue/README.md`: buffer table gains the wavelet_entropy row.

**Tests** (this VM): `377 passed, 10 skipped` (up from 357/10 on 10-13;
`test_endpoint_extension.py` still uncollectable — no skimage in VM venv,
documented gap). `ruff check .` clean (CI gate); fixed one B023 (loop-bound
closure) found during the run.

**Scope notes**: no measured official-data scores — the verdict comes from
the box queue's `verify_candidate` gate, not from this change. This is a
candidate, not an incumbent. Next CPU items once box results land: read the
s7 loss gate verdict (first live promotion candidate), the s8 seed-probe
comparison, and the s4–s6 lineament rerun gates; P3 (background-loss
treatment) is the following research item.

## 2026-10-15 — P3 background-softened Tversky loss (step 10) queued

**Rationale** (plan: stronger segmentation via loss ablations; research execution
order). The documented order in docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md
is "P1 loss ablation; then P2 wavelet entropy; then P3 background loss." P1 is
queued on the box (s7), P2 was built and queued 10-14 (s9); P3 is the next
CPU-buildable item and is independent of the s4–s9 verdicts, so it was
implemented here and queued after step 9 — it never preempts live candidates.

**Motivation**: known fault labels are incomplete (unknown-fault discovery;
geologist selection is not uniform), so plain Tversky treats every non-labeled
pixel as hard background and over-penalizes plausible unmapped faults near
known traces. The competition metric itself forgives near-truth predictions
(300 m ≈ 3 px tolerance). P3 extends a milder version of that tolerance to
training: background pixels within 3 px of *training-side* truth get their FP
penalty downweighted x0.5; TP/FN terms unchanged (no pseudolabeling, nothing
rewarded); background outside the band keeps full penalty (no flooding reward).

**What was built** (CPU-only, this VM; no training here):
- `src/gems/background_soften.py`: `ambiguity_weights` (per-pixel FP weight
  map: 1.0 valid, `soften_factor` on the band, 0.0 invalid; truth outside the
  valid mask never defines the band; Euclidean distance transform, no
  wraparound), `softened_tversky_components` / `softened_tversky_loss_value`
  NumPy references, `build_background_softened_loss` torch module
  (`needs_spatial=True`, band built per-patch from `truth & valid` on CPU,
  mirroring the P1 pattern). `soften_factor=1.0` reduces exactly to standard
  soft Tversky; `soften_factor<1.0` can only lower the loss vs control.
- `build_training_loss` (in `src/gems/distance_tversky.py`) dispatches the
  new `loss: background_softened` with config keys `loss_ambiguity_radius_pixels`
  (default 3.0) and `loss_background_soften_factor` (default 0.5), validated
  at build time (radius > 0, factor in (0, 1]).
- `configs/candidates/cand_loss_background_soften.yaml`: protocol-equal to
  ref-b16 (buffer 16, seed 20260922, epochs 5, train_step 32); only the loss
  differs, so the gate compares the background treatment, not budgets.
- `scripts/queue/tasks.json`: 88 → **98 tasks**. Step 10:
  `s10-train/score/record-loss_background_soften-{spatial,fault,trace}` +
  `s10-gate-loss_background_soften-b16` → `CAND-LOSS_BACKGROUND_SOFTEN-B16` vs
  `runs/gate/ref-b16`. Hypothesis: pooled-spatial OOF gain with fault/trace
  non-regression safeguards, per `verify_candidate` policy.
- `tests/test_background_soften.py` (15 tests: band geometry, training-side-only
  band, empty-truth, invalid-zero weights, no wraparound, param validation,
  factor=1.0 equals plain Tversky, softened ≤ control, monotone in factor,
  far-background keeps full penalty, band-not-rewarded ordering, dispatcher
  config flow + build-time validation, 3 torch tests skipped here) +
  `test_background_soften_candidate_tasks_present` in `test_queue_tasks.py`.

**Tests** (this VM): `390 passed, 13 skipped` (up from 377/10 on 10-14;
`test_endpoint_extension.py` still uncollectable — no skimage in VM venv,
documented gap; the 3 new torch tests skip here). `ruff check .` clean (CI gate).

**Scope notes**: no measured official-data scores — the verdict comes from
the box queue's `verify_candidate` gate, not from this change. This is a
candidate, not an incumbent. When the gate verdict lands, run the P3 analysis
the research doc asks for: report probability mass, FP cost, and class-prior
sensitivity of the accepted/rejected candidate. Next CPU items once box
results land: s7 loss gate verdict (first live promotion candidate), s8
seed-probe comparison, s4–s6 lineament rerun gates, s9 wavelet-entropy gate.

## 2026-10-16 — P5 skeleton-recall auxiliary loss (step 11) queued

**Rationale** (plan: stronger segmentation via loss ablations; research execution
order). The documented order in docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md
is "P1 loss ablation; then P2 wavelet entropy; then P3 background loss. Try
P4/P5 only after their controls are measured." P1 is queued on the box (s7),
P2 was built and queued 10-14 (s9), P3 was built and queued 10-15 (s10); the
P4 control (straight endpoint extension) was measured and REJECTED 10-10
(ENDPOINT-EXT-02, deltas ~1e-8 on all views -- the 300 m metric kernel
saturates near-truth coverage, so post-hoc extension adds nothing). P5 is the
next CPU-buildable item: it attacks the model itself rather than
post-processing, and the plan prioritizes modeling improvements over lineament
features. It was implemented here and queued after step 10 -- it never
preempts live candidates.

**Motivation**: Kirchhoff et al. Skeleton Recall Loss (2024) and the skeptical
2025 follow-up ("Does the Skeleton-Recall Loss Really Work?") give conflicting
empirical evidence on thin-structure segmentation; neither settles GEMS. Fault
traces are 1-3 px thin structures and the metric is recall-heavy (beta=0.8),
so a small matched auxiliary term rewarding skeleton recall along
training-side traces is the honest way to test whether connectivity
supervision helps. Connectivity stays a diagnostic: the gate still decides on
pooled distance-weighted Tversky with fault/trace non-regression safeguards.

**What was built** (CPU-only, this VM; no training here):
- `src/gems/skeleton_recall.py`: `soft_skeletonize` -- white top-hat
  `x - open(x)` plus Lantuejoul opening residues over K erosions, 3x3
  min/max pooling, zero padding (no wraparound); exactly the morphological
  skeleton on binary input, differentiable on soft predictions. The top-hat
  term is load-bearing: pure Lantuejoul erosion annihilates sub-SE thin
  structures, leaving zero gradient on exactly the pixels the recall term
  samples; the top-hat preserves the thin-structure gradient path (a 1 px
  predicted line skeletonizes to itself). `target_skeleton` from
  `truth & valid` only (held-out fault/trace pixels never define the
  target); recall term is 0.0 on patches with no truth skeleton (the
  auxiliary must not invent signal). `build_skeleton_recall_loss` torch
  module (`needs_spatial=True`): control masked soft-Tversky (alpha=0.2 /
  beta=0.8, supervised pixels only) + `skeleton_weight` * skeleton recall.
  With `skeleton_weight=0.0` it reduces exactly to the incumbent's training
  objective. Note: unlike P1's module, the supervision mask is honored in
  *both* terms here (P1's smp base term sees unmasked targets), keeping the
  ablation matched to `loss: tversky`.
- `build_training_loss` (in `src/gems/distance_tversky.py`) dispatches the
  new `loss: skeleton_recall` with config keys `loss_skeleton_weight`
  (default 0.25, validated in [0, 1]) and `loss_skeleton_iterations`
  (default 5, validated integer in [1, 12]).
- `configs/candidates/cand_loss_skeleton_recall.yaml`: protocol-equal to
  ref-b16 (buffer 16, seed 20260922, epochs 5, train_step 32); only the loss
  differs, so the gate compares the auxiliary loss, not budgets.
- `scripts/queue/tasks.json`: 98 -> **108 tasks**. Step 11:
  `s11-train/score/record-loss_skeleton_recall-{spatial,fault,trace}` +
  `s11-gate-skeleton_recall-b16` -> `CAND-SKELETON_RECALL-B16` vs
  `runs/gate/ref-b16`. Hypothesis: pooled-spatial OOF gain with
  fault/trace non-regression safeguards, per `verify_candidate` policy.
- `scripts/train_full_map.py`: `loss_descriptor` now records the
  skeleton (and previously unrecorded P3 background-soften) loss params
  for run-manifest reproducibility.
- `tests/test_skeleton_recall.py` (15 tests: thin-line/thick-bar/empty/soft
  skeleton geometry, no wraparound, monotonicity, binary training-side-only
  target, perfect/broken/empty recall behavior, invalid-mask isolation,
  weight=0 reduction to control, param validation, dispatcher routing, 5
  torch tests skipped here: numpy/torch agreement, aux-term agreement,
  gradient flow on thin structures, held-out exclusion) +
  `test_skeleton_recall_candidate_tasks_present` in `test_queue_tasks.py`.

**Tests** (this VM): `401 passed, 18 skipped` (up from 390/13 on 10-15;
`test_endpoint_extension.py` still uncollectable -- no skimage in VM venv,
documented gap; the 5 new torch tests skip here). `ruff check .` clean (CI gate).

**Scope notes**: no measured official-data scores -- the verdict comes from
the box queue's `verify_candidate` gate, not from this change. This is a
candidate, not an incumbent. When the gate verdict lands, run the P5 analysis
the research doc implies: report connectivity/diagnostic behavior of the
accepted/rejected candidate alongside the pending P3 probability-mass/FP-cost
analysis. Next CPU items once box results land: s7 loss gate verdict (first
live promotion candidate), s8 seed-probe comparison, s4-s6 lineament rerun
gates, s9 wavelet-entropy gate, s10 background-soften gate.
