# Competition strategy

The hidden target is not “reproduce the known USGS/INGENIOUS fault map.” It is **discover fault pixels missing from it**. Known faults are useful supervision and geological context, but organizer clarification says those known pixels are masked in scoring.

A September 23 organizer clarification makes the target broader than “geologically separate new structures”: a new-fault label can include previously unmapped continuation, splay, parallel strand, or other newly mapped geometry of an existing fault system. Proximity to a known fault is therefore **not** a reason to suppress a candidate by itself.

## Phase 0 — reproduce and measure

First reproduce the organizer U-Net reference solution and establish a local metric implementation, run manifest, submission validator, and spatial CV. Do not spend leaderboard submissions until local scoring is trustworthy.

Use at least three validation views:

1. **Spatial-block holdout:** withhold contiguous geographic blocks and buffer their boundaries so nearby pixels from the same structure cannot leak across train/validation.
2. **Fault-component holdout:** withhold complete fault traces/components where the real label topology makes that interpretation meaningful.
3. **Trace-completion holdout:** withhold long sections, endpoints, branches, or vector features from otherwise known fault systems, then test whether the model extends/reconstructs plausible geometry from the feature evidence.

Random pixel CV will materially overstate performance because fault traces are spatially connected and nearby geophysical pixels are autocorrelated. Component-only CV can also miss an important part of the competition if hidden labels include extensions of already known systems.

Do not implement trace segmentation from synthetic assumptions. Inspect the official raster/vector labels first; use source-vector feature IDs when meaningful, or define reproducible graph/segment rules from the observed topology.

### Fair comparison protocol

The three-view gate (`scripts/verify_candidate.py`) compares immutable evaluation
protocols. Supply `score_cv.py --metrics-pattern` for training metadata; when
supplied, its presence and values must agree between runs. Config hashes can
differ for the declared experiment. The checked training/evaluation fields are:

- `buffer_pixels` (spatial buffer around held-out regions)
- `seed` (random seed for reproducibility)
- `epochs` (training duration)
- `train_step` (patch sampling stride)
- `fold_map_sha256` (CV fold assignment)

If a candidate feature (e.g. `steerable_filter`) requires `--buffer-pixels 20` but the incumbent used 16, **the incumbent must be re-run at buffer 20 for all three views** before comparison. Otherwise the gate cannot attribute a difference to the feature change vs. the buffer change. Alternatively, if the candidate's filter support permits buffer 16, use that; `scripts/train_full_map.py` reports the minimum required buffer per lineament kind.

## Phase 1 — strong segmentation baseline

Reproduce the official U-Net, then standardize the pipeline around patch inference and full-raster overlap blending. Track per-band normalization computed only from training portions of each fold.

Loss candidates should reflect sparsity and recall pressure: BCE + soft Dice/Tversky, focal/Tversky variants, and distance-aware targets. Always choose by the **published competition metric** on honest held-out data, not by training loss.

## Phase 2 — features that expose lineaments

Add derived features where physically justified:

- multi-scale gradients, Laplacian and edge magnitude;
- structure-tensor coherence/orientation;
- ridge/valley and curvature measures;
- local contrast at multiple radii;
- directional / steerable filters;
- Hough/line-segment response maps;
- multi-scale hillshade/slope/curvature from the permitted high-resolution DEM.

Treat each derived channel as an ablation. A channel only stays if it clears the pooled spatial improvement gate and the
geographic, complete-fault and trace-completion non-regression safeguards.

## Phase 3 — geologic priors and external data

External data is encouraged if we have the necessary rights/license for competition use and sponsor evaluation. Prioritize sources with clear provenance and relevant physics:

- permitted 1 m USGS 3DEP DEM tiles provided/linked by the competition;
- GeoDAWN source products and metadata;
- broader INGENIOUS/Great Basin context layers if license terms are compatible;
- USGS geology/fault products useful as contextual inputs, with leakage checks.

Maintain a license ledger before training on any new source.

## Phase 4 — ensembles and candidate discovery

High-value diversity is likely to come from architecture, feature family, spatial scale, fold/seed, and post-processing. Blend probabilities, then tune calibration against honest CV.

Because `beta=0.8` and `alpha=0.2`, the metric rewards recall more heavily than precision, but probability spread still incurs FP cost. Calibrate rather than hard-threshold too early.

For final-round upside, inspect high-confidence predictions in **both** classes of opportunity:

- isolated or previously unmapped structures away from known traces;
- plausible continuations, endpoint extensions, splays, and parallel strands near known fault systems.

Produce a geologist-facing candidate layer with supporting feature evidence. Expert verification can add genuine previously unmapped pixels to the final label set.

## Leaderboard policy

The scored-submission allowance is three submissions in a rolling window, not a calendar-week reset. Each upload must answer a hypothesis. Log commit SHA/config, model/folds, local scores, artifact SHA256, leaderboard score, interpretation, and next experiment. Do not repeatedly tune to tiny public-LB changes.

---

## Next steps (2026-10-09)

### Immediate priorities (ranked)

1. **Score full-config spatial baseline** – First real-data run used a reduced config (epochs 2, train_step 64) on an 8-vCPU CPU-only box. The model was barely trained (losses 0.945–0.950) but produced the first honest spatial CV baseline: raw 0.0591 ± 0.0082, LOFO-calibrated 0.0895 ± 0.0200. Train with full `configs/reference_unet.yaml` (epochs=5, train_step=32) to validate this baseline and confirm it was not a fluke of the reduced configuration. This establishes the proper baseline for all future comparisons.

2. **Train fault and trace fold models** – Complete fault-discovery and trace-discovery CV views. The gate currently only has spatial CV scores; fault and trace views are required to evaluate candidates properly under the pooled-spatial-primary gate policy (a spatial win plus all three comparable non-regression safeguards).

3. **Re-gate candidates against full baseline** – Once the full-config spatial baseline is scored and fault/trace views are available, re-gate endpoint extension (ENDPOINT-EXT-01: currently INCONCLUSIVE) and other candidates against the honest full baseline.

2. **Automated propose-and-verify experiment loop** – ✅ COMPLETED 2026-10-07/08
   
   **Canonical implementation:** `src/gems/verification.py`, `scripts/verify_candidate.py`.
   Updated 2026-10-10: accepts only a stitched pooled spatial gain above the
   heuristic max fold-std × (1 + 0.5·log2(1 + n)), with all three comparable
   non-regression safeguards. Spatial per-fold results remain geographic
   safeguards; fault/trace macro means remain separate because background overlaps.
   Schema-2 reports validate weighted aggregate evidence and pixel coverage.
   Uses the JSONL ledger at `docs/candidate-trials.jsonl`; old reports need re-scoring.
   Full policy: [CANDIDATE_PIPELINE.md](CANDIDATE_PIPELINE.md#propose-and-verify-acceptance-gate-2026-10-07).
   
   *Note:* An alternate statistical gate (`experiment_gate.py`, paired t-tests with Bonferroni) 
   was implemented 2026-10-08 but deprecated in favor of the simpler, more transparent 
   verification.py approach. The canonical escalation is a search-pressure heuristic, not a statistical
   multiple-comparisons guarantee. Verdicts inform upload decisions, don't trigger uploads.

3. **LoG edge-strength feature channel (inspired by Mumford-Shah geometry)** – ✅ COMPLETED 2026-10-08
   
   Implemented Laplacian of Gaussian (LoG) edge detection for geophysical bands 
   (`src/gems/mumford_shah.py`, 16 tests). This is a classical edge detector using second 
   derivatives, not an Ambrosio-Tortorelli solver. Integrated into derived feature pipeline 
   (`src/gems/lineament.py`) with new `kind: mumford_shah_log` option. Distinguishes sharp 
   discontinuities from smooth gradients, capturing where gravity, magnetic, conductivity, and 
   strain-rate fields jump. Uses fold-pure normalization (training-region-only stats) to prevent 
   validation leakage. Candidate config: `configs/resnet18_mumford_shah_log.yaml`. The feature 
   design is motivated by OpenAI Family 366 (Mumford-Shah regularity), which confirms the 
   geometric prior: fault networks are smooth arcs with bounded curvature. Ready for ablation 
   through the propose-and-verify gate once training data is available.

4. **Endpoint-extension post-processing for fault tips and splays** – ✅ COMPLETED 2026-10-08
   
   Implemented skeletonization-based endpoint extension (`src/gems/endpoint_extension.py`, 
   `scripts/extend_endpoints.py`, 19 tests). Finds skeleton endpoints, estimates local orientation 
   via structure tensor/PCA, extends probability along that direction with exponential decay over 
   bounded distance (300m default, configurable). Optional edge-strength gating prevents extension 
   into unsupported regions. Preserves float32 [0,1] range and valid-region masks. Config: 
   `configs/resnet18_with_endpoint_extension.yaml`. Tests verify gap-bridging (broken lines with 
   gaps < extension distance), noise-blob stability (isolated blobs don't sprout long tails), mask 
   preservation, idempotence/bounded growth. With β=0.8, this targets cheap recall on organizer-confirmed 
   "newly mapped continuations" class. Motivated by Mumford-Shah arc-termination geometry (Family 366). 
   Ready for OOF ablation via canonical gate once training data is available.

5. **Multi-scale gradient and structure-tensor features** – ✅ COMPLETED 2026-10-09
   
   **Implemented:** Structure tensor coherence and orientation channels (`src/gems/lineament.py`, 
   7 tests). Coherence (λ1-λ2)/(λ1+λ2+ε) measures directional structure strength [0,1]; orientation 
   gives dominant angle [-π/2, π/2]. High coherence + consistent orientation = lineament. Configurable 
   smoothing windows (sigma_pixels for gradient pre-smoothing, structure_tensor_window for gradient-product 
   smoothing). Config: `configs/resnet18_structure_tensor.yaml`. Motivated by OpenAI Family 083 
   (Hilbert/Lipschitz): keep filter lengths below orientation-change scale (~1/|∇v|).

   **Implemented 2026-10-08:** Hessian-eigenvalue ridge/valley response + multi-scale
   pyramid (`src/gems/lineament.py` `kind: ridge_valley_response`, 9 new tests;
   config `configs/resnet18_ridge_valley.yaml`). signed = valley−ridge where
   valley = relu(λmax−λmin)·relu(λmax) and ridge = relu(λmax−λmin)·relu(−λmin),
   normalized by the fold-pure 99th percentile of |signed| to [−1,1]. The
   `ridge_valley_scales` pyramid maxes valley/ridge terms independently
   (Frangi-style); `structure_tensor_window` smooths the Hessian components.
   Distinguishes fault-parallel troughs/crests from isotropic blobs and step
   edges (already covered by gradient/structure-tensor channels).
   
   **Implemented 2026-10-09:** Gabor-style steerable/directional filters
   (`src/gems/lineament.py` `kind: steerable_filter`, 12 new tests;
   config `configs/resnet18_steerable_filter.yaml`). Applies oriented edge
   detectors at `steerable_orientations` equally-spaced angles (0 to π), using
   Gaussian-envelope × sinusoidal-carrier kernels at `steerable_wavelength`.
   Returns max magnitude across orientations for orientation-invariant lineament
   detection, normalized to [0,1] by fold-pure 99th percentile. Completes the
   full suite: gradient energy, phase edge, LoG, structure tensor coherence/
   orientation, ridge/valley response, and steerable filters. Each channel must
   pass the propose-and-verify gate on all three CV views once training data
   is available.

6. **Ensemble and calibration framework** – ✅ COMPLETED 2026-10-09
   
   Implemented infrastructure to combine fold-wise OOF probability rasters from multiple
   candidate models (`src/gems/ensemble.py`, `scripts/build_ensemble.py`, 18 unit tests).
   Supports three fusion strategies: arithmetic mean, metric-fit weighted mean with
   leave-one-fold-out (LOFO) weight fitting, and rank-average (robust to scale differences).
   LOFO weight fitting prevents leakage: weights for fold k are fitted on all other folds,
   so fold k's score does not use weights fit on fold k's labels. Includes optional
   threshold calibration using the canonical exact_threshold_curve from `calibration.py`.
   Output GeoTIFFs pass `validate_submission.py` and are scored through `score_cv.py` and
   `verify_candidate.py`, logged to `docs/candidate-trials.jsonl`. Example config:
   `configs/ensemble_example.json`. Tests verify mean/weighted/rank-average combinations,
   LOFO non-leakage guarantee, threshold calibration, and raster I/O. Ready for ablation
   on candidate models once training data is available.

7. **Trace-continuation validation and synthetic tests** – Expand the endpoint-holdout CV with synthetic gap-insertion tests to measure whether models can plausibly extend fault geometry from partial context, as the competition requires for newly mapped continuations.

### Recent additions

- **Threshold tooling consolidation (2026-10-08)**: `src/gems/optimization.py` and `scripts/optimize_threshold.py` deprecated in favor of `src/gems/calibration.py` (PR #17). The exact threshold method evaluates every distinct threshold state efficiently via event-based cumulative sums, making it both faster and more accurate than grid search. `optimization.py` now wraps `calibration.py` with deprecation warnings for backward compatibility. Use `scripts/calibrate_threshold.py` going forward.

- **Threshold optimization (2026-10-07)**: Added exact threshold optimization via `src/gems/calibration.py`. Given beta=0.8 (recall) vs alpha=0.2 (precision), the hypothesis was that optimal thresholds would be much lower than 0.5. **Update 2026-10-09**: First real-data baseline (reduced epochs, barely trained) produced LOFO-calibrated thresholds near 0.70, not below 0.5. However, this model's outputs cluster near 0.55 (median fold-0 probability 0.548), so these thresholds reflect an under-trained model with compressed outputs. The "optimal threshold below 0.5" hypothesis should be confirmed on a properly trained model before treating it as a design principle. The exact method finds the global optimum efficiently without search. Ready for use as soon as fold predictions are available.

- **OpenAI math research integration (2026-10-07)**: Added `docs/OPENAI_MATH_LEADS.md`, which evaluates the 722-paper release for GEMS applicability. The top actionable lead is the *method* (massive generate-and-verify search with strict acceptance criteria), not the mathematics. Two geometry papers suggest concrete features and post-processing: Mumford-Shah regularity for edge-strength channels and endpoint extension, and Hilbert-transform stability for adaptive directional-filter design. See the document for full analysis and sources.


## Metric-aware research queue (2026-10-10)

P0 pooled-score selection is implemented and tested. Before P1 distance-aware
training, re-score retained predictions with schema 2, finish the adequately
trained baseline, and obtain complete fault/trace holdouts. Then run a matched
ResNet-18 conventional-loss versus conventional-plus-distance-aware-loss ablation.
P2 is wavelet energy entropy on physically appropriate potential-field bands;
P3 is a small background-penalty ablation for incomplete labels. Geodesic
continuation and skeleton loss remain later controlled experiments.

Source checks, equations and acceptance conditions are recorded in
[METRIC_AWARE_RESEARCH_2026-10-10.md](research/METRIC_AWARE_RESEARCH_2026-10-10.md).
No new candidate improvement is established by this queue or the P0 correction.
