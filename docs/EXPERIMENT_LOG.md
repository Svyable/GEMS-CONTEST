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
