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
