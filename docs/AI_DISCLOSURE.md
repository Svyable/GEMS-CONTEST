# Generative AI disclosure log

The GEMS official rules allow generative AI in solution development but require competitors to disclose the extent and manner of its use in the submission narrative.

Keep this file factual and current.

## 2026-09-22 — repository bootstrap

OpenAI ChatGPT (GPT-5.6 Sol) was used to:

- research and summarize the public competition description, metric, submission format, official rules, and public organizer clarifications;
- create the initial repository structure and documentation;
- draft reusable Python utilities for the published distance-weighted Tversky metric and GeoTIFF submission validation;
- propose an experiment and modeling strategy.

Competition data was not supplied to the model during this bootstrap. All generated code is subject to local tests, human review, empirical validation, and the competitor's responsibility for accuracy/authorship representations.

## 2026-09-22 — provenance, CV, and reference-baseline tooling

OpenAI ChatGPT (GPT-5.6 Sol) was used to:

- inspect the public organizer reference-solution repository and transcribe its published U-Net parameters;
- identify and document the filename discrepancy between the public problem page and reference notebook;
- implement deterministic SHA256/raster-metadata manifests for local data provenance;
- implement buffered spatial-block folds and complete connected-fault holdouts;
- add tests and a CLI that produces fold rasters and JSON summaries for reproducible validation.

No gated competition data was provided to the model during this work. The generated validation design is our experimental methodology, not an organizer-provided scoring implementation, and must be checked empirically once the real rasters are available.

## 2026-09-22 — executable organizer-baseline reproduction

OpenAI ChatGPT (GPT-5.6 Sol) was used to:

- inspect the public organizer notebook's full patch-splitting and Monte Carlo training cells;
- implement a NumPy-tested reproduction of its patch selection, normalization, padding, and OOF assembly;
- implement a PyTorch/segmentation-models-pytorch CLI for reproducing the published U-Net baseline once gated data is available;
- document deliberate reproducibility/output-format fixes that differ from the notebook's incidental behavior.

No gated competition data was supplied to the model, and no reference training run has yet been claimed as reproduced.

## 2026-09-22 — full-raster inference plumbing

OpenAI ChatGPT (GPT-5.6 Sol) was used to:

- implement deterministic overlapping raster windows and weighted probability blending;
- implement a template-aligned float32 GeoTIFF prediction writer with range/finite checks;
- add unit tests for full coverage, edge padding, exact reconstruction under overlap, valid masks, and submission-format output;
- document the train-on-all candidate inference contract.

No gated competition data was supplied to the model and no model-performance claim was made from this infrastructure.

## 2026-09-23 — fold-aware evaluation and run provenance

OpenAI ChatGPT (GPT-5.6 Sol) was used to:

- update the local metric so NaN/nodata is permitted outside an explicit valid mask while remaining invalid inside it;
- implement fold-aware spatial and held-out-fault scoring using the published distance-weighted Tversky metric;
- define spatial OOF aggregation and intentionally avoid a synthetic global score for overlapping fault-discovery evaluation regions;
- implement machine-readable run manifests tying experiments to Git state, config/data hashes, commands, and artifact hashes;
- add unit tests and workflow documentation.

No gated competition data was supplied to the model and no empirical model-performance claim was made.

## 2026-09-23 — label-topology intake diagnostics

OpenAI ChatGPT (GPT-5.6 Sol) was used to:

- implement descriptive 4-connected and 8-connected fault-component topology reporting;
- report component-size distributions, largest-component share, bounding-box scales, and spatial-block label sparsity;
- add an intake CLI and tests so the real label topology can determine the appropriate fault-holdout unit rather than assuming connected components are always suitable;
- document that no arbitrary topology pass/fail threshold should be applied before inspecting the official labels.

No gated competition data was supplied to the model and no conclusion about the actual GEMS label topology was made.

## 2026-09-23 — CV visual QA

OpenAI ChatGPT (GPT-5.6 Sol) was used to:

- implement a dependency-light RGB/PNG fold preview renderer;
- preserve thin labeled traces during preview downsampling using block-wise positive-pixel aggregation;
- distinguish spatial-fold QA from fault-component QA;
- require a separate valid-study template for fault previews so fold-raster nodata/background is not misinterpreted;
- add unit tests and document the visual review gate before model training.

No gated competition data was supplied to the model and no visual conclusion about the actual GEMS folds was made.

## 2026-09-23 — organizer/forum clarification sweep

OpenAI ChatGPT (GPT-5.6 Sol) was used to review current public DrivenData competition/forum material and update the strategy based on organizer statements, including:

- known USGS/INGENIOUS pixels are masked from scoring;
- competition “new fault” pixels may include newly mapped continuations, splays, parallel strands, or other geometry of an existing fault system;
- the three-scored-submission allowance operates on a rolling window rather than a calendar-week reset;
- public questions about test-fault identification methods and a reported label-band discrepancy remain tracked for follow-up rather than assumed resolved.

These were public organizer/community materials; no gated competition data was supplied to the model.

## 2026-09-23 — official data intake

Grok (xAI Grok 4.7, in Grok Build) was used to:

- place the four organizer files already downloaded by the competitor into `data/raw/` without renaming them;
- record the DrivenData dataset list and the actual filenames in the repository docs and config;
- run the repository's input verification, raster inspection, and SHA256/GeoTIFF fingerprint scripts.

The model saw filenames, file sizes, the DEM-link JSON shape (a list of USGS TNM URLs), and raster metadata produced by the local inspection scripts. Raw raster pixels were not pasted into the conversation. Hashes and metadata in `data/manifests/official.json` were produced by local code, not invented by the model.

## 2026-09-23 — Apple vision-encoder candidates

Grok (xAI Grok 4.7, in Grok Build) was used to:

- choose open-weight image encoders that run as a U-Net on Apple MPS, after checking that a plain DINOv2 ViT cannot supply this decoder's feature pyramid;
- add a spatial-fold training mask so those models are scored on held-out ground rather than on the faults they were trained on;
- wire ConvNeXt V2-Tiny and the Segment Anything 2 Hiera-Small encoder into that loop.

The model saw raster metadata, training logs, and scores. It did not receive raw raster pixels. Measured scores come from the local metric code.

## Add future entries

For each material use, record date, tool/model, what information was provided, what was generated, and what verification was performed before relying on the output.

## 2026-10-07 — candidate validation preprocessing repair

OpenAI ChatGPT/Codex inspected repository code, issue status, and a user-provided
research shortlist. It implemented training-region normalization, candidate raster
alignment/fold checks, preprocessing and input-hash reporting, regression tests,
and documentation. No raw competition rasters were available in this workspace;
verification uses synthetic arrays and the repository test suite. No model training,
leaderboard submission, or improvement in competition score is claimed.

## 2026-10-07 — OpenAI math research screening and exact threshold optimization

OpenAI ChatGPT/Codex was used to inspect the October 6 OpenAI mathematics release,
its public catalogue, the official GEMS metric, repository code, and current issue
status. It screened possible geophysical connections, recorded Svyable's prize
objective, derived an exact binary-threshold event sweep from the official metric,
and implemented a separate-calibration/evaluation CLI with SHA256 input provenance.
Verification compares all threshold states on synthetic fixtures against both an
independent coordinate-distance oracle and the existing metric, with geospatial
CLI rejection tests. No OpenAI proof artifacts, competition rasters, or external
datasets were imported; no paid API/compute, model training, submission, or measured
competition-performance gain is claimed. The release's individual proofs were not
independently verified in this work, and its unreleased model was not accessed.

## 2026-10-07 — metric-aware threshold optimization (PR #16)

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) implemented threshold optimization
tools that directly maximize the distance-weighted Tversky metric. Given the competition's
beta=0.8 (recall weighting) vs alpha=0.2 (precision weighting) and 300m tolerance,
the optimal probability threshold is typically much lower than 0.5. The implementation
includes:

- Grid search across threshold candidates with emphasis on [0, 0.5] range
- Golden section search for efficient continuous optimization  
- Prediction calibration utilities that remap optimal thresholds to 0.5
- Comprehensive unit tests with synthetic linear fault patterns
- CLI tool (`scripts/optimize_threshold.py`) with fold-aware evaluation support
- Source code: `src/gems/optimization.py`, `scripts/optimize_threshold.py`, `tests/test_optimization.py`

No competition data was available; testing uses synthetic arrays and the repository
metric implementation. No model predictions, leaderboard submission, or score claim
is made. All code in this work was AI-generated by the cloud agent. This infrastructure
is ready for use once fold-specific predictions exist.

## 2026-10-07 — OpenAI math release research document and roadmap integration (PR #16)

An AI assistant (using web search over github.com/openai/math) researched and authored
the document `docs/OPENAI_MATH_LEADS.md` analyzing the 722-paper OpenAI math release
for GEMS applicability. Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) then:

- Committed the AI-authored research document with minor formatting edits
- Integrated the top actionable lead (automated propose-and-verify experiment loops with
  statistical acceptance criteria) into the ranked roadmap in `docs/STRATEGY.md`
- Ranked Mumford-Shah-inspired features (edge-strength channel, endpoint-extension
  post-processing) among the next priorities based on their geometric relevance to
  fault networks and the competition's recall-weighted metric
- Updated this disclosure file

Both the research document and the integration work were AI-generated. No new executable
code, model training, or competition submission was produced in this step; the contribution
was research synthesis, prioritization, and documentation.

## 2026-10-07 — runnable discovery and endpoint validation

OpenAI ChatGPT/Codex inspected candidate training, sampling, fold generation and
evaluation code, plus installed locked library source. It implemented explicit
spatial/fault/trace training splits, sparse masked supervision, paired target/mask
augmentation, trace-aware scoring, input-protocol metadata, and CPU training-smoke
CI. Tests poison withheld feature values, check that excluded logits receive no
loss gradient, reject malformed fold maps, and train real randomly initialized
U-Nets on synthetic GeoTIFFs through validated inference and scoring. The work
used synthetic data and local CPU execution only; it did not use raw competition
data, download pretrained weights, run paid compute/API calls, or submit predictions.
No official-data model improvement is claimed. Masked-input training is documented
as an ablation with potential artificial-edge and distribution-shift effects.

## 2026-10-07 — MS-EDGE-01 lineament feature implementation

OpenAI ChatGPT (GPT-5.6 Sol) used the public OpenAI mathematics research scan, the
repository's official feature-band metadata, and existing candidate-training code
to implement grouped lineament feature transforms and wire them into the candidate
U-Net experiment path. The implementation includes a plain grouped-gradient
control and a bounded local Ambrosio--Tortorelli-inspired phase-edge transform,
with synthetic unit tests, provenance metadata, spatial-buffer checks, and an
explicit block on fault/trace CV until masked-input transform semantics are
specified. The model did not receive raw competition raster pixels in this chat,
no paid compute or model training was initiated, and no competition-performance
improvement is claimed.

## 2026-10-07 — REG-SENS-01 registration robustness diagnostic

OpenAI ChatGPT (GPT-5.6 Sol) implemented a spatial-registration stress test for the
candidate training path. It shifts one physical feature family at a time by each
neighboring one-pixel offset, reuses the same trained fold model, recomputes only
held-out inference windows, and records distance-weighted Tversky deltas with
input/config hashes. Synthetic tests cover the perturbation helpers and an
end-to-end CPU U-Net train/infer/report path. No raw competition pixels were
provided in this chat, no paid compute was initiated, and no claim is made that
the official feature stack is actually misregistered.

## 2026-10-07 — propose-and-verify acceptance gate

Muse (Anthropic) implemented the automated propose-and-verify experiment loop,
the top infrastructure priority from the OpenAI math-research scan
(`docs/OPENAI_MATH_LEADS.md` #1). The deliverable is new executable code:
`src/gems/verification.py` (acceptance rule with search-pressure escalation:
win ≥2 of 3 CV views by more than max fold-to-fold std × (1 + 0.5·log2(1+n
trials))), `scripts/verify_candidate.py` (CLI over score_cv.py JSONs with a
committed JSONL trial ledger), and 15 unit/CLI tests in
`tests/test_verification.py` — all passing on synthetic scores, plus ruff-clean.
Documentation was updated in `docs/STRATEGY.md`, `docs/CANDIDATE_PIPELINE.md`,
and `docs/EXPERIMENT_LOG.md`.

No competition data was used (this VM has none); no model was trained, no
official-data score is claimed, and no DrivenData upload was made. A verdict is
documented as evidence for the upload decision, not an upload trigger.

## 2026-10-07 — spatial ablation matrix automation

OpenAI ChatGPT (GPT-5.6 Sol) implemented a dry-run-by-default experiment matrix for
the unchanged ResNet-18, grouped-gradient, and MS-EDGE-01 recipes across all five
spatial folds. It also implemented paired fold-score aggregation and attaches the
registration-sensitivity diagnostic only to the unchanged control. Unit tests use
synthetic JSON metrics and command plans; no official raster training, paid
compute, or leaderboard submission was launched by this automation work.

## 2026-10-07 — official band-integrity audit augmentation

OpenAI ChatGPT (GPT-6) drafted a read-only, block-streamed preflight tool using
the repository's existing NumPy/Rasterio dependencies and manifest fingerprint
helpers. It checks SHA256, file sizes, band identities/order, raster metadata,
label values, nodata coverage, and mask disagreements. Synthetic pytest fixtures
cover both pass and fail cases. No gated raster pixel data, model checkpoints,
GPU training, external datasets, paid compute, submission uploads, or
competition score improvements were involved. The tool was added to the
repository via a CI-gated pull request; actual-data preflight is still pending.


## 2026-10-07 — score provenance and acceptance-gate hardening

OpenAI ChatGPT (GPT-6) inspected the candidate verifier, scoring CLI, synthetic
regression tests, and repository CI. It added exact truth/fold-map fingerprints
and explicit metric semantics to score reports, made the propose-and-verify
gate fail closed on incomparable protocols or inconsistent fold aggregates,
and added synthetic regression tests. The work used source code and synthetic
fixtures only, not gated competition rasters or external datasets. No paid
compute, model retraining, leaderboard upload, or measured GEMS improvement
is claimed. CI verification is required before merge.

## 2026-10-08 — threshold tooling consolidation and statistical experiment gate

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) performed two tasks:

1. **Threshold consolidation:** Deprecated `src/gems/optimization.py` and
   `scripts/optimize_threshold.py` in favor of the exact threshold method in
   `src/gems/calibration.py` (from PR #17). The exact method evaluates every
   distinct threshold state via event-based cumulative sums, making it both
   faster and more accurate than grid search. The optimization module now wraps
   calibration with deprecation warnings. All existing tests updated to expect
   "exact" method and deprecation warnings. No functionality removed; backward
   compatibility maintained.

2. **Statistical experiment gate:** Implemented automated propose-and-verify gate
   using paired t-tests with Bonferroni correction (`src/gems/experiment_gate.py`,
   `scripts/evaluate_candidate.py`, 13 unit tests). Accepts candidates only if they
   beat incumbent on ≥2 of 3 CV views with statistical significance. Significance
   threshold tightens with trial count (α' = α / (trials × views)) to control
   family-wise error rate. Appends all decisions to immutable JSON ledger. Testing
   uses synthetic fold scores only. No competition data, model training, or
   leaderboard submission involved.

All code in this work is AI-generated by the cloud agent. Tests pass, linter
clean, CI-verifiable. Implements roadmap priorities from OPENAI_MATH_LEADS.md.

## 2026-10-08 — LoG edge-strength features (Mumford-Shah-inspired) and canonical experiment gate

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) performed two tasks:

1. **LoG edge-strength feature channel inspired by Mumford-Shah geometry (roadmap #3):**
   Implemented Laplacian of Gaussian (LoG) edge detection for geophysical bands
   (`src/gems/mumford_shah.py`, 16 unit tests). This is a classical edge detector using second
   derivatives, not an Ambrosio-Tortorelli variational solver. The design is motivated by
   OpenAI Family 366 (Mumford-Shah regularity), which confirms faults follow smooth-arc geometry.
   Integrated into derived feature pipeline (`src/gems/lineament.py`) with new `kind: mumford_shah_log`
   option. Includes fold-pure normalization using training-region-only statistics
   to prevent validation leakage. Candidate config created (`configs/resnet18_mumford_shah_log.yaml`).
   All tests use synthetic step/gradient patterns; no competition data or training involved.

2. **Canonical experiment gate consolidation:** Unified the two existing experiment gates
   (verification.py with escalating fold-std thresholds, and experiment_gate.py with
   paired t-tests) by marking experiment_gate.py as deprecated in favor of verification.py
   as the canonical implementation. Added deprecation warnings to `src/gems/experiment_gate.py`
   and `scripts/evaluate_candidate.py`. The canonical gate (`scripts/verify_candidate.py`)
   uses committable JSONL ledger at `docs/candidate-trials.jsonl`, better protocol validation
   (truth/fold SHA256 checks), and simpler paired-fold-differences with escalating thresholds.

All code AI-generated by cloud agent. Tests pass (16 new Mumford-Shah tests, all lineament
tests pass). Linter clean, CI-verifiable. Implements OPENAI_MATH_LEADS.md Family 366
(Mumford-Shah regularity of fault traces).

## 2026-10-08 — Endpoint-extension post-processing and documentation corrections

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) performed two tasks:

1. **Endpoint-extension post-processing (roadmap #4):** Implemented skeletonization-based
   endpoint extension to recover plausible fault continuations and splays
   (`src/gems/endpoint_extension.py`, `scripts/extend_endpoints.py`, 19 unit tests).
   Finds skeleton endpoints, estimates local orientation via structure tensor/PCA, extends
   probability along that direction with exponential decay over bounded distance (300m default,
   tied to metric tolerance). Optional edge-strength gating prevents extension into unsupported
   regions. Preserves float32 [0,1] range and valid-region masks. Config:
   `configs/resnet18_with_endpoint_extension.yaml`. Tests verify gap-bridging, noise-blob
   stability, mask preservation, idempotence, and bounds. No competition data; synthetic tests only.

2. **Documentation corrections:** Corrected prior overstatements in `docs/AI_DISCLOSURE.md`,
   `docs/STRATEGY.md`, `docs/OPENAI_MATH_LEADS.md`, and `docs/EXPERIMENT_LOG.md` to accurately
   describe the edge detector as Laplacian of Gaussian (LoG) inspired by Mumford-Shah geometric
   regularity, not an Ambrosio-Tortorelli variational solver. The implementation is a classical
   edge detector using second derivatives; the Mumford-Shah result motivates the geometric prior
   (smooth arcs, tips, Y-junctions) but does not prescribe the algorithm.

All code AI-generated by cloud agent. Tests pass (19 new endpoint-extension tests, 219 total).
Linter clean, CI-verifiable. Implements roadmap #4 motivated by Mumford-Shah arc-termination
geometry (Family 366) and organizer-confirmed "newly mapped continuations" target class.

## 2026-10-08 — Structure tensor coherence and orientation features (roadmap #5)

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) implemented structure tensor analysis
for lineament detection (`src/gems/lineament.py` extended with two new feature kinds,
7 unit tests in `tests/test_lineament.py`). Structure tensor decomposes smoothed gradient
outer products into eigenvalues and eigenvectors, revealing:

- **Coherence:** (λ1 - λ2) / (λ1 + λ2 + ε) ∈ [0,1], where 0 = isotropic (no preferred
  direction) and 1 = perfectly linear (strong directional structure). High coherence
  indicates lineaments.

- **Orientation:** dominant direction angle in radians [-π/2, π/2] from the structure
  tensor eigenvector, indicating the direction of maximum change perpendicular to the
  lineament.

Implementation includes `structure_tensor_coherence` and `structure_tensor_orientation`
as new `kind` options in `grouped_lineament_features()`, with configurable smoothing
windows (defaults: sigma_pixels=1.0 for gradient pre-smoothing, structure_tensor_window=3.0
for gradient-product smoothing). Motivated by OpenAI Family 083 (Hilbert transform stability
along Lipschitz directions): keep filter lengths below the orientation-change scale (~1/|∇v|).
Tests verify zero coherence on constant fields, high coherence on strong linear gradients,
correct orientation detection for horizontal and vertical gradients, invalid-pixel handling,
and multi-channel aggregation. Candidate config: `configs/resnet18_structure_tensor.yaml`.

All code AI-generated by cloud agent. Tests pass (7 new structure tensor tests, 227 total,
1 skipped). Linter clean, CI-verifiable. Implements part of STRATEGY.md roadmap priority #5
(multi-scale gradient and structure-tensor features). No competition data, model training,
or leaderboard submission involved; testing uses synthetic gradient patterns only.

## 2026-10-08 — Hessian ridge/valley response and multi-scale pyramid (roadmap #5)

Anthropic Claude Sonnet 4.5 (Muse cron worker) extended `src/gems/lineament.py`
with a new `kind: ridge_valley_response` derived-feature channel plus a `_symmetric_eigenvalues`
helper (9 new unit tests in `tests/test_lineament.py`), and added candidate config
`configs/resnet18_ridge_valley.yaml`. The channel computes the Hessian of each smoothed
band, averages the Hessian components across channels, and forms a signed response:
valley = relu(λmax−λmin)·relu(λmax), ridge = relu(λmax−λmin)·relu(−λmin),
signed = valley−ridge — positive on concave-up troughs, negative on concave-down crests.
`ridge_valley_scales` computes the response at each pyramid scale and maxes the valley/ridge
terms independently (Frangi-style multi-scale detection). Normalization divides by the 99th
percentile of |signed| over the valid region only (fold-pure) and clips to [−1,1].
`structure_tensor_window` is reused for Hessian-component smoothing. Strategy and experiment-log
docs were updated to record the new state of roadmap priority #5 (directional/steerable filters
remain).

All code AI-generated by cloud agent. Tests pass (217 passed, 1 skipped, full suite excluding
`tests/test_endpoint_extension.py`, which is uncollectable on this VM due to a pre-existing
missing `skimage` dependency unrelated to this change). Ruff check clean; new code is
ruff-format clean (pre-existing format drift elsewhere in the touched files left untouched
to keep the diff minimal). CI-gated.
Implements most of STRATEGY.md roadmap priority #5 (ridge/valley + multi-scale). No competition
data, model training, or leaderboard submission involved; testing uses synthetic Gaussian
trough/crest patterns only.

## 2026-10-09 — Gabor-style steerable/directional filters (roadmap #5, complete)

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) completed STRATEGY.md roadmap priority #5
by implementing Gabor-style steerable/directional filters (`src/gems/lineament.py`
`kind: steerable_filter`, 12 new unit tests in `tests/test_lineament.py`). The implementation
applies oriented edge detectors at multiple angles (configurable `steerable_orientations`,
default 8, equally spaced from 0 to π) using Gaussian-envelope × sinusoidal-carrier (real
Gabor) kernels at configurable `steerable_wavelength` (default 8.0 pixels, typical 6-12).
Returns max magnitude across all orientations for orientation-invariant lineament detection,
normalized to [0,1] by fold-pure 99th percentile over the valid region. Tests verify
horizontal/vertical/diagonal line detection, constant-field quietness, invalid-region
handling, multi-channel aggregation, and parameter validation. Candidate config:
`configs/resnet18_steerable_filter.yaml`.

Roadmap priority #5 is now complete: gradient energy, phase edge, Laplacian of Gaussian,
structure tensor coherence/orientation, Hessian ridge/valley response, and steerable filters.
Each channel is ready for OOF ablation via the canonical propose-and-verify gate once
training data is available.

All code AI-generated by cloud agent. Tests pass (37 passed, full lineament suite). Ruff
check clean. CI-gated. No competition data, model training, or leaderboard submission
involved; testing uses synthetic line patterns only.

## 2026-10-09 — Ensemble and calibration framework (roadmap #6, complete)

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) completed STRATEGY.md roadmap priority #6
by implementing the ensemble and calibration framework (`src/gems/ensemble.py`,
`scripts/build_ensemble.py`, 18 unit tests in `tests/test_ensemble.py`). The framework
combines fold-wise out-of-fold (OOF) probability rasters from multiple candidate models
using three fusion strategies:

1. **Arithmetic mean**: simple average of all member predictions per fold
2. **Metric-fit weighted mean with LOFO**: fits weights to maximize the published
   distance-weighted Tversky metric using leave-one-fold-out (LOFO) to prevent leakage.
   Weights for fold k are fitted on all other folds (j != k), ensuring fold k's score
   does not use weights fit on fold k's labels. Grid search over equal weights,
   best-single-model, and all 0.1-step combinations (for 2-3 members).
3. **Rank-average**: converts predictions to normalized ranks before averaging, robust
   to scale differences between models

The framework includes optional threshold calibration using the canonical
`exact_threshold_curve` from `calibration.py`. CLI tool (`scripts/build_ensemble.py`)
accepts JSON config with member patterns, fits LOFO weights, applies the specified
method, optionally calibrates thresholds, and writes fold-specific GeoTIFF predictions.
Output passes `validate_submission.py` and is scored through `score_cv.py` and
`verify_candidate.py`, logged to `docs/candidate-trials.jsonl`.

Tests verify mean/weighted/rank-average combinations, the LOFO non-leakage guarantee
(test proves fold 0 weights unchanged when fold 0 labels change, since they're fitted
on fold 1 only), threshold calibration, and raster I/O with georeferencing checks.
Example config: `configs/ensemble_example.json`.

All code AI-generated by cloud agent. Tests pass (18 passed, full ensemble suite).
Ruff check clean. CI-gated. No competition data, model training, or leaderboard
submission involved; testing uses synthetic fold predictions and linear patterns only.
Ready for ablation on candidate models once training data is available.

---

## 2026-10-09 (PR #32): real-data pipeline bug fixes

### Changes
Real contest data testing (19-band 3292×3730 raster, 60,894 fault pixels = 1.18% prevalence) surfaced production issues fixed in this PR:
1. Memory optimization: `make_reference_split_lazy` with on-the-fly window construction reduces peak memory from >7.6 GB to ~3.6 GB (`src/gems/reference_baseline.py`, `scripts/train_reference_oof.py`)
2. Import error fix: `scripts/optimize_threshold.py` replaced missing `load_raster_band` with direct `rasterio.open` calls
3. Visual QA bug: fault-scheme preview used mismatched downsampling between fold IDs and fault mask, rendering assigned held-out faults white (fixed in `src/gems/visual_qa.py`)
4. Nondeterministic ordering: `scripts/profile_labels.py` top-20 components now sorted deterministically by `(size desc, component_id asc)` (`src/gems/topology.py`)
5. Symlink handling: `fingerprint_file` no longer resolves symlinks to targets, preserving manifest relative paths (`src/gems/data.py`)

### Real-Data Findings
- Raster: 19 bands, 3292×3730 pixels
- Fault pixels: 60,894 (1.18% of 5,164,312 valid pixels)
- 8-connected components: 3,198 (median 12 px, largest 360 px)
- 4-connected components: 25,058
- CV manifests rebuilt from real data matched committed synthetic-derived manifests exactly

### Rationale
Bug discovery via real-data integration required systematic debugging and memory profiling by generative AI. Fixes preserve reproducibility and CI health.

### Verification
- `pytest tests/test_reference_baseline.py::test_lazy_split_matches_eager_patches` (lazy == eager patches for fixed seeds)
- `pytest tests/test_visual_qa.py::test_fault_preview_consistent_downsampling`
- `pytest tests/test_topology.py::test_component_report_deterministic_ties`
- `pytest tests/test_script_smoke.py` (all scripts --help smoke tests green)
- Full test suite: `pytest` (all tests green)
- Linted with `ruff check` (clean)

## 2026-10-09 — lineament feature integrity repair

OpenAI ChatGPT (GPT-6) reviewed the public repository code, candidate configs,
and current experiment status. It repaired incompatibilities between three
derived-feature YAML configs and the actual candidate trainer; passed
tensor, multiscale Hessian, and steerable parameters through the trainer;
separated training-only feature-scale fitting from full-region transforms;
tightened spatial buffer checks for filter support; and added synthetic
regression tests and documentation. No raw gated competition rasters were
supplied to this assistant and no new model training, held-out score, or prize
improvement is claimed from this code work.
