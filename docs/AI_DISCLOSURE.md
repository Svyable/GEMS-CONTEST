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

## 2026-10-09 — fault/trace CV training handoff config

This worker (scheduled cron, Muse) added `configs/resnet18_cv_reduced.yaml`
(scheme-agnostic reduced ResNet-18 CV config for fault-discovery and
trace-completion fold training), `tests/test_cv_config.py` (5 schema tests),
and the box handoff commands in `docs/EXPERIMENT_LOG.md`. No raw gated
competition data was read or committed; no model training ran on this CPU-only
VM and no new held-out score is claimed. The handoff targets the
torch+data box for the fault/trace CV views blocked here.

## 2026-10-10 — PR #36 pre-merge fixes (protocol-equality gate and CI)

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) prepared PR #36 for merge by
implementing three fixes per user request:

1. **Protocol-equality check**: Extended `scripts/score_cv.py` to optionally read
   training protocol from per-fold metrics JSON files (buffer_pixels, seed, epochs,
   config_sha256) and include them in the evaluation_protocol output. Extended
   `src/gems/verification.py` to check these fields (TRAINING_PROTOCOL_FIELDS) and
   refuse comparisons where they differ, preventing unfair gates where baseline and
   candidate differ in two ways (e.g., buffer 16 vs 20 plus a feature change).
   Documented in `docs/STRATEGY.md` that if candidates require different buffer_pixels,
   the incumbent must be re-run at the same buffer for fair comparison.

2. **CI enforcement of integration tests**: Updated `.github/workflows/ci.yml` to add
   a new step in the `training-smoke` job that runs
   `tests/test_train_lineament_fault_cv_integration.py` with torch installed, ensuring
   these tests actually run in CI instead of skipping.

3. **Documentation organization**: Moved `LINEAMENT_CV_PROTOCOL_SUMMARY.md` from
   repository root to `docs/` for better organization.

4. **Linting fixes**: Fixed three ruff linting errors (unused imports/variables) in
   test files that were blocking CI.

No competition data was provided; no model training, leaderboard submission, or
score improvement is claimed. All changes verified via existing test suite and CI.

## 2026-10-10 — protocol-chain fixes and lineament fault/trace queue (gems-daily-advance)

This worker (scheduled cron, Muse) made two groups of changes:

1. **Metrics → gate protocol chain (correctness)**: `scripts/train_full_map.py` now
   writes `train_step` into the per-fold metrics JSON; `scripts/score_cv.py`
   extracts it from `--metrics-pattern` instead of hardcoding `None`; and
   `scripts/queue/score_view.sh` passes `--metrics-pattern` so every queue-produced
   score JSON carries the full training protocol (buffer_pixels, seed, epochs,
   train_step, config_sha256) that `src/gems/verification.py` compares. Added 2
   tests in `tests/test_score_cv_protocol.py` and 1 assertion in
   `tests/test_training_ml.py`.

2. **Experiment queue (validation gaps)**: Fixed the s4 lineament tasks to reference
   the committed `configs/candidates/*.yaml` (`configs/local/` does not exist in
   the repo) and to use protocol-equal buffers per kind — buffer 16 for
   mumford_shah_log / steerable_filter (required 5/7 px), buffer 32 for
   structure_tensor_coherence / ridge_valley_response (required 17/30 px) with a
   matching buffer-32 incumbent re-run. Added s5 (fault view) and s6 (trace view)
   lineament tasks for all four candidates, enabled by PR #36, with gates placed
   only after all compared views are scored. Documented the buffer table in
   `scripts/queue/README.md`. (Correction to the 2026-10-10 log entry: the
   verified required buffers are 5/7/17/30 px, not "4-17".)

No competition data was read; no model training ran on this CPU-only VM and no
new held-out score is claimed. The queue targets the torch+data box.


## 2026-10-10 — Pooled spatial candidate selection and research queue

Codex checked the supplied research against official metric/organizer pages and
primary literature, implemented schema-2 stitched weighted-evidence reporting,
and changed candidate selection to a mandatory pooled spatial improvement plus
comparable geographic/fault/trace non-regression safeguards. It also repaired
training-protocol metadata preservation (config hashes remain provenance rather
than equality keys), added synthetic ranking/seam/protocol regression tests, and
updated strategy/research docs. Existing ledger entries were preserved.

Validation covers scoring and selection behavior, not improved official-data
predictions. Distance-aware loss, wavelet entropy and incomplete-label training
remain proposed ablations. No external dataset, checkpoint, paid compute, model
API call or competition submission was introduced.

## 2026-10-11 — Distance-aware Tversky loss (P1)

Muse implemented the P1 distance-aware differentiable Tversky loss from
`docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md`:
`src/gems/distance_tversky.py` (NumPy reference + torch module + loss factory),
`scripts/train_full_map.py` wiring (spatial-loss branch under masked
supervision, `training_loss` descriptor in metrics JSON),
`configs/candidates/cand_loss_distance_tversky.yaml`, and
`tests/test_distance_tversky.py` (19 tests, NumPy half runs on CPU, torch half
gated on torch availability).

No competition data was read; no model training ran on this CPU-only VM and no
new held-out score is claimed. The candidate config targets the torch+data box.

## 2026-10-11 — Lineament CV held-out zeroing bug fix (PR #XX)

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) fixed a critical data-leakage bug in
`scripts/train_full_map.py` where, during CV runs, `filtering_mask = split.train_mask`
was passed as both `valid_mask` and `normalization_mask` to `grouped_lineament_features`.
This caused `derived[~valid] = 0.0` in `src/gems/lineament.py` to zero all derived
lineament channels over the entire held-out fold, so models trained on real gradient/edge
values but were scored on zeros (observed symptom: held-out predictions collapsed,
mean prob ~0.05 vs 0.3 in-train).

**Fix implemented (two-pass filter)**:
- Pass A: `valid_mask=train_mask` for training pixels
- Pass B: `valid_mask=validation_mask` for held-out pixels  
- Both passes use `normalization_mask=train_mask` (fold-pure: percentiles/max fit on
  training pixels only, never held-out)
- Combine by region: training where train_mask, validation where validation_mask,
  zero in buffer
- No held-out values influence training-pixel features (PR #36's leakage tests still pass)

**Regression tests added** (`tests/test_lineament_cv_leakage.py`, 3 new parametrized tests):
1. `test_two_pass_produces_varying_held_out_features`: verifies derived channels are
   non-constant-zero in held-out region (catches the zeroing bug)
2. `test_two_pass_fold_pure_normalization`: verifies held-out perturbations don't affect
   training-region features (normalization is fold-pure)
3. Runtime check in `train_full_map.py`: fails the run if any derived channel is constant
   zero over validation pixels

**Documentation**:
- Marked the `cand_mumford_shah_log` spatial b16 result (raw 0.0398, calibrated 0.0619,
  if recorded) as INVALID in `docs/EXPERIMENT_LOG.md` due to this bug
- Updated this disclosure

All code AI-generated by cloud agent. Tests pass (21 new assertions, full lineament-leakage
suite green). Ruff check clean. CI-verifiable. No competition data, model training, or
leaderboard submission involved; testing uses synthetic fault patterns only.

## 2026-10-11 — Lineament CV scale-reuse fix (PR #XX)

Anthropic Claude Sonnet 4.5 (Cursor Cloud Agent) fixed an incomplete implementation in
PR #42 where Pass B attempted to refit normalization scale constants on disjoint masks,
causing held-out derived channels to be 10-300x smaller than training channels (e.g.,
magnetic training median/p99 of 0.358/0.999 vs held-out 0.011/0.122).

**Root cause**: In Pass B, `fit & valid` was empty (disjoint train/eval masks), so
normalization was skipped (denom = 0.0), leaving raw unnormalized features in held-out
regions.

**Fix (scale parameter reuse)**:
1. Pass A computes and stores all normalization constants in metadata['norm_params']:
   - mumford_shah_log: band percentiles (2/98), grad_norm_max, edge_max
   - ridge_valley_response: signed_p99
   - steerable_filter: response_p99
2. Pass B receives norm_params and applies precomputed constants unchanged (no refitting)
3. Updated `compute_edge_strength` to accept and return edge_max for reuse
4. Tightened runtime check to fail when held-out vs train median or p99 differ by >5x
   (not just when channel is all zeros)

**Regression test** (`test_two_pass_distribution_match_with_norm_params`):
- Creates synthetic pattern appearing in both training and held-out regions
- Verifies Pass B distribution matches Pass A within 5x on median and p99
- Catches the scale-mismatch bug (all 7 lineament kinds)

**Documentation**:
- Corrected `docs/EXPERIMENT_LOG.md` line ~666: spatial CV *was* affected (has split),
  not "unaffected"
- Updated this disclosure

All code AI-generated by cloud agent. Tests include new distribution-matching regression
test (7 parametrized cases). Ruff check clean. CI-verifiable. No competition data, model
training, or leaderboard submission involved; testing uses synthetic gradient patterns.

## 2026-10-12 — P1 queue step (s7), result re-verification, docs

- CPU test verification (this VM): 351 passed / 10 skipped suite-wide
  (`test_endpoint_extension.py` uncollectable — no skimage in this venv);
  `tests/test_distance_tversky.py` 13 passed / 6 skipped (torch-half tests skip
  without torch; they run on the training box).
- Independently recomputed the 2026-10-11 cand_mumford_shah_log spatial b16
  rerun statistics from committed score JSONs; all published figures match
  exactly. No changes needed.
- Added 11 tasks to `scripts/queue/tasks.json` (step 7): train/score/record the
  P1 distance-aware Tversky candidate on spatial/fault/trace at buffer 16,
  an idempotent ref-b16 assembly step, and the `CAND-LOSS_DISTANCE_TVERSKY-B16`
  gate task — all following the existing s5b task conventions. Task ids
  `s7-{train,score,record,gate}-loss_distance_tversky-*`.
- Appended this experiment-log entry and this disclosure.

All work AI-generated by cloud agent. No model training, raw data, checkpoints,
submission artifacts, or leaderboard interaction involved; no competition data
was read — analysis used only committed score JSONs and configs.

## 2026-10-12 — Commit ref spatial CV buffer-32 baseline scores

Cursor Grok 4.6 (cloud agent) copied commit-safe JSON scores for
`ref-cv-spatial-b32` (reference U-Net, spatial CV, buffer 32) from the
provided result artifacts into `results/runs/` and appended experiment-log
and reference-baseline notes. Numbers were not invented or altered.

No model training, raw rasters, checkpoints, prediction GeoTIFFs, or
DrivenData submission in this change.

## 2026-10-13 — Seed-robustness probe (step 8) + train_folds.sh seed plumbing

Cloud agent: added an optional 6th `SEED` arg to `scripts/queue/train_folds.sh`
(default 20260922, backward compatible, forwarded as `train_full_map.py --seed`);
appended 3 step-8 queue tasks to `scripts/queue/tasks.json` (train/score/record
the spatial incumbent at seed 20260923 — the single-seed validation gap);
added `tests/test_queue_tasks.py` (6 tests: queue integrity + seed plumbing);
documented the SEED arg in `scripts/queue/README.md`; appended the experiment-log
entry and this disclosure. Also audited the step-7 gate chain and confirmed it
is sound as queued (schema-2 gate JSONs on the box; committed ref-full scores
are schema-1 and correctly not used for gating).

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — analysis used only
committed score JSONs and configs.

## 2026-10-14 — P2 wavelet-entropy lineament channel (step 9) + queue tasks

Cloud agent: implemented the P2 hypothesis from
docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md as `kind: wavelet_entropy`
in `src/gems/lineament.py` — an adaptation (not a reproduction) of Dwivedi
(2026) wavelet-space entropy: undecimated Haar detail energies at
wavelet_levels (default (1,2,3), dilation 2**(level-1)), per-pixel
energy-share entropy H = -sum_s q_s log(q_s + eps) with H = 0 defined at
zero total energy, fold-pure 99th-percentile normalization to [0,1], and a
conservative required buffer of smoothing + max dilation (8 px at defaults,
fits the buffer-16 protocol). Added `configs/candidates/cand_wavelet_entropy.yaml`
(magnetic_data + gravity_data only, per the P2 "start on raw magnetic/gravity
bands" instruction), `tests/test_wavelet_entropy.py` (17 tests: Haar energy
properties, entropy bounds, step-edge localization, two-pass fold-pure reuse,
invalid-pixel isolation, level/buffer/kwarg validation), extended the
held-out-outlier leakage test to the new kind, appended 10 step-9 queue
tasks (train/score/record × 3 views at buffer 16 + CAND-WAVELET_ENTROPY-B16
gate vs runs/gate/ref-b16, queued after step 8 so live candidates are never
preempted), added the step-9 integrity test, and updated the README buffer
table. Full suite: 377 passed, 10 skipped (same documented skips as
2026-10-13); `ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic rasters in the VM venv. Training/scoring/gating run on the box
queue; the verdict will come from verify_candidate, not from this change.

## 2026-10-15 — P3 background-softened Tversky loss (step 10) + queue tasks

Cloud agent: implemented the P3 hypothesis from
docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md as `loss: background_softened`
in `src/gems/background_soften.py` — an adaptation (not a reproduction) of the
incomplete-labels motivation in Needles in the Landscape (2025) and Zhao et al.
ICCV 2023: an ambiguity band (Euclidean distance < `loss_ambiguity_radius_pixels`,
default 3.0, matching the metric's ~3 px tolerance) around training-side truth
only, where the false-positive penalty is downweighted by the fixed
`loss_background_soften_factor` (default 0.5). TP/FN terms unchanged (no
pseudolabeling), far background keeps full penalty (no flooding reward), truth
outside the valid mask never defines the band, no wraparound at borders.
Extended `build_training_loss` with the new dispatch, added
`configs/candidates/cand_loss_background_soften.yaml` (protocol-equal to
ref-b16), `tests/test_background_soften.py` (15 tests: band geometry,
training-side-only band, param validation, factor=1.0 equals plain Tversky,
softened-never-exceeds-control, torch/numpy agreement paths), the step-10
queue-task integrity test, and appended 10 step-9-following queue tasks
(train/score/record × 3 views at buffer 16 + CAND-LOSS_BACKGROUND_SOFTEN-B16
gate vs runs/gate/ref-b16). Full suite: 390 passed, 13 skipped (same documented
skips as 2026-10-14 plus 3 new torch tests skipped in the VM venv);
`ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic rasters in the VM venv. Training/scoring/gating run on the box
queue; the verdict will come from verify_candidate, not from this change.

## 2026-10-16 — P5 skeleton-recall auxiliary loss (step 11) + queue tasks

Cloud agent: implemented the P5 hypothesis from
docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md as `loss: skeleton_recall`
in `src/gems/skeleton_recall.py` — an adaptation (not a reproduction) of the
conflicting 2024 Skeleton Recall Loss / skeptical 2025 follow-up literature:
a soft morphological skeleton (white top-hat plus Lantuejoul opening residues,
3x3 min/max pooling, zero padding, no wraparound; the top-hat term preserves
the thin-structure gradient path that pure erosion would annihilate) with a
fixed-weight recall term against the training-side truth skeleton, added to
the control masked soft-Tversky (alpha=0.2/beta=0.8, supervised pixels only;
weight=0.0 reduces exactly to the incumbent objective). Truth outside the
valid mask never defines the target; the auxiliary contributes nothing on
patches with no truth skeleton. Extended `build_training_loss` with the new
dispatch, added `configs/candidates/cand_loss_skeleton_recall.yaml`
(protocol-equal to ref-b16), `tests/test_skeleton_recall.py` (15 tests: thin
skeleton geometry, training-side-only target, recall behavior, invalid-mask
isolation, weight-0 reduction, torch/numpy agreement paths), the step-11
queue-task integrity test, appended 10 step-11 queue tasks
(train/score/record × 3 views at buffer 16 + CAND-SKELETON_RECALL-B16 gate vs
runs/gate/ref-b16), and recorded the skeleton/P3 loss params in the
train_full_map run manifest. Full suite: 401 passed, 18 skipped (same
documented skips as 2026-10-15 plus 5 new torch tests skipped in the VM venv);
`ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic rasters in the VM venv. Training/scoring/gating run on the box
queue; the verdict will come from verify_candidate, not from this change.

## 2026-10-17 — 10-epoch longer-schedule reference (step 12) + manual compare script

Cloud agent: built the "longer schedules" item from docs/STRATEGY.md Phase 1
(research doc P0 "finish the adequately trained baseline"): added
`configs/candidates/cand_longer_schedule.yaml` — the incumbent reference
recipe with training epochs doubled 5 -> 10, everything else
(data/patches/model/augmentation/seed) identical so the comparison isolates
schedule length. The acceptance gate refuses epoch mismatches by design, so
also wrote `scripts/compare_schedule.py` (CPU-only paired comparison of two
gate dirs: requires identical evaluation protocol and identical
seed/buffer/train_step/fold map with exactly the epochs field differing;
reports paired fold-by-fold raw deltas plus LOFO-calibrated deltas from
lofo-*.json summaries; writes nothing, no ledger, no verdict) and
`scripts/queue/compare_schedule.sh` (assembles raw + LOFO dirs from the
per-view run dirs on the box). Added `tests/test_longer_schedule.py` (8
tests: config epochs/loss, protocol-equality modulo epochs, compare script
delta reporting, refusal of equal epochs / seed mismatch / fold mismatch,
missing-view tolerance), the step-12 queue-task integrity test, and appended
10 step-12 queue tasks (train/score/record x 3 views at buffer 16 +
s12-compare-longer_schedule-b16 vs runs/gate/ref-b16). Full suite: 409
passed, 18 skipped (same documented skips as 2026-10-16; the
test_endpoint_extension.py skimage import remains a VM-venv gap); `ruff
check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic JSONs/rasters in the VM venv. Training/scoring run on the box
queue; the schedule verdict comes from the manual compare task, not from
this change.

## 2026-10-18 — P4 curved continuation candidate (step 13)

Authored `src/gems/curved_continuation.py` (numpy-only: Zhang–Suen thinning,
skeleton endpoint finding, PCA endpoint orientation, curvature-prior beam-search
path growth with a per-pixel support floor, and decaying painting anchored at
the skeleton threshold), `scripts/extend_endpoints_curved.py` (rasterio CLI),
`scripts/queue/curved_view.sh` (per-fold application with LOFO thresholds,
mirroring endpoint_view.sh), and `configs/candidates/cand_curved_continuation.yaml`
(incumbent recipe + fixed post-processing budgets: 300 m, curvature penalty
0.5, beam 3, min_prob 0.3, support_threshold 0.4, max 200 paths). Added
`tests/test_curved_continuation.py` (37 tests: thinning/endpoint/orientation
behavior, path growth along corridors, support-gap stopping, curvature-penalty
effect, valid-mask respect, unsupported-bridge rejection, paint decay bounds,
param validation), the --help smoke test, the step-13 queue-task integrity
test, and appended 4 step-13 queue tasks (curved post-process + score x 3 views
at buffer 16 + s13-gate-curved_continuation-b16 vs runs/gate/ref-b16). Full
suite: 448 passed, 18 skipped (same documented skips; test_endpoint_extension.py
skimage import remains a VM-venv gap); `ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic rasters in the VM venv. Training/scoring run on the box queue; the
P4 verdict comes from the gate task, not from this change.

## 2026-10-19 — Seed-probe comparison tooling (step 14) + paired manual compare script

Authored `scripts/compare_seed.py` and `scripts/queue/compare_seed.sh`
(specifying the comparison the gate refuses by design: seed is the ONLY
tolerated training-protocol difference). Added `tests/test_compare_seed.py`
(8 tests: paired delta reporting + seed-effect std, spatial-only probe with
missing-view skip, refusal of equal seeds / a second differing field / fold
mismatch / invalid scores / missing protocol fields, no-LOFO skip, shell
executable check) and the s14 queue-task presence test in
`tests/test_queue_tasks.py`. Appended step-14 queue task
`s14-compare-seed20260923-b16` (script compares runs/gate/ref-b16 against
runs/gate/ref-seed20260923-b16 with both LOFO summaries; writes nothing, no
ledger, no verdict -- a human decides whether the single-seed watch item can
be retired). Queue 122 -> 123 tasks.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read -- all verification used
synthetic JSON fixtures in the VM venv. Training/scoring run on the box
queue; the seed-variance answer comes from the step-14 task once the step-8
probe trains, not from this change.

## 2026-10-20 — Queue provenance fix: s6-record-ref-trace-b32 + integrity test

Authored the missing `s6-record-ref-trace-b32` queue task in
`scripts/queue/tasks.json` (buffer-32 trace reference now has the full
train/score/record triple like every other queued reference run) and
`test_s6_ref_trace_b32_record_task_present` in `tests/test_queue_tasks.py`
(regression test pinning the triple: incumbent config, buffer 32, incumbent
seed, gate-dir score target). Queue 123 -> 124 tasks. Runner keys on task ids
and `record.sh` only reads the run dir, so the task is safe to execute
regardless of box progress.

## 2026-10-21 — Queue provenance fix, generalized (step 16): s4/s5 record tasks + triple-invariant test

Authored the missing `s4-record-ref-spatial-b32` and `s5-record-ref-fault-b32`
queue tasks in `scripts/queue/tasks.json` (both buffer-32 incumbent reference
runs shipped train+score with no record, the same failure mode the 10-20 fix
closed for s6), plus `test_train_score_record_triple_invariant` in
`tests/test_queue_tasks.py` (queue-wide regression: every `<step>-train-<target>`
must have matching score and record tasks) and
`test_s4_s5_ref_b32_record_tasks_present`. Queue 124 -> 126 tasks. Runner
keys on task ids and `record.sh` only reads the run dir, so both tasks are
safe to execute regardless of box progress. Full suite: 461 passed, 18
skipped (same documented skips; test_endpoint_extension.py skimage import
remains a VM-venv gap); `ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic fixtures and the queue JSON in the VM venv. Training/scoring run on
the box queue; the s7/s8/s12/s14 verdicts and the s4–s6/s9–s11/s13 gates still
await box results.

## 2026-10-22 — Queue integrity, generalized (step 17): gate/compare candidate-prerequisite invariant tests

Authored `test_gate_candidate_dirs_produced_in_queue` and
`test_compare_candidate_inputs_produced_in_queue` in
`tests/test_queue_tasks.py` (queue-wide regression: every gate.sh/compare
task's candidate-side inputs — gate dirs, run dirs/prefixes — must be
produced by in-queue score/train tasks; incumbent inputs grandfathered and
verified manually), plus the manual 126-task wiring audit and the s7
training-protocol comparability re-verification that motivated them. Full
suite: 463 passed, 18 skipped (same documented skips; test_endpoint_extension.py
skimage import remains a VM-venv gap); `ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic fixtures and the queue JSON in the VM venv. Training/scoring run on
the box queue; the s7/s8/s12/s14 verdicts and the s4–s6/s9–s11/s13 gates still
await box results.

## 2026-10-23 — Verdict-interpretation tooling (step 18): render_gate_verdict.py

Authored `scripts/render_gate_verdict.py` (reads a committed
`results/gates/<NAME>.json` in the verify_candidate schema and prints the
docs/EXPERIMENT_LOG.md verdict block in the 2026-10-10 ENDPOINT-EXT-02 entry
style: date from `timestamp_utc`, hypothesis/config lines, per-view delta vs
required margin with consistent %.2g/%.4g formatting, ACCEPT/REJECT/
INCONCLUSIVE verdict lines, non-comparable views listed with their reason and
excluded from the win denominator, plus the committed-results checklist line)
and `tests/test_gate_verdict.py` (6 tests, including a golden test against
the real `results/gates/ENDPOINT-EXT-02.json` asserting the renderer
reproduces the hand-written 10-10 entry's content). Motivation: the step-17
note pins verdict interpretation (s7 loss gate, s8/s14 seed probe, s4–s6
lineament rerun gates, s9–s11/s13 candidate gates, s12 schedule compare) as
the next CPU item once the box produces results; this makes that step
mechanical instead of hand-written. Full suite: 469 passed, 18 skipped (same
documented skips; test_endpoint_extension.py skimage import remains a VM-venv
gap); `ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic fixtures and the committed ENDPOINT-EXT-02 gate JSON. Training/scoring
run on the box queue; the s7/s8/s12/s14 verdicts and the s4–s6/s9–s11/s13 gates
still await box results (no new box commits since the 10-12 b32 baselines).

## 2026-10-24 — Gate-status inventory (step 19): scripts/gate_status.py

Authored `scripts/gate_status.py` (parses scripts/queue/tasks.json for
gate.sh and compare_*.sh tasks and prints one row per queued decision:
DECIDED with verdict when results/gates/<NAME>.json parses, BROKEN when it
exists but is unparseable or carries an unknown verdict, PENDING otherwise
with per-view candidate-input coverage; RAN/PENDING for manual compares via
their compare.log; gate JSONs with no queued task listed as orphans; the
box-relative runs/gate/... and runs/compare/... dirs resolve to their
committed results/... locations first) and `tests/test_gate_status.py`
(4 tests, including a real-repo golden test: 10 gates queued, 1 decided,
9 pending; 2 compares, 0 ran; 0 orphans). Motivation: step 18 made verdict
interpretation mechanical, but no single command showed which queued
decisions had landed — the daily run hand-grepped tasks.json against
results/gates/. Full suite: 473 passed, 18 skipped (same documented skips;
test_endpoint_extension.py skimage import remains a VM-venv gap);
`ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — all verification used
synthetic fixtures and the committed queue/gate state. Training/scoring run
on the box queue; the s7/s8/s12/s14 verdicts and the s4–s6/s9–s11/s13 gates
still await box results (no new box commits since the 10-12 b32 baselines).

## 2026-10-25 — LOFO threshold audit (step 20): scripts/lofo_threshold_audit.py

Authored `scripts/lofo_threshold_audit.py` (discovers every committed
LOFO file in both committed layouts, validates self-consistency of the
top-level means against fold means, and reports per-run threshold
mean/range, raw -> calibrated gain, and gain share, with CEILING /
UNSTABLE / CALIBRATION_DOMINATED tripwire flags) and
`tests/test_lofo_threshold_audit.py` (6 tests, including a real-repo
golden test over all 7 committed LOFO runs). Motivation: the standing
watch item that LOFO thresholds are unusually high had no quantification
from committed numbers alone. Findings: trace thresholds pin at ~1.0
with 42–58% of the headline trace score from fitting; spatial
thresholds are stable but ~21% of the headline is fitting; fault
thresholds do not reproduce across folds. Full suite: 479 passed, 18
skipped (same documented skips; test_endpoint_extension.py skimage
import remains a VM-venv gap); `ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or leaderboard
interaction involved; no competition data was read — the audit reads only
committed LOFO summary JSONs (thresholds and per-fold scores), never rasters
or predictions. All verification used synthetic fixtures and the committed
LOFO state. Training/scoring run on the box queue; the s7/s8/s12/s14 verdicts
and the s4–s6/s9–s11/s13 gates still await box results (no new box commits
since the 10-12 b32 baselines).

## 2026-10-26 — Threshold experiment handoff (step 21): scripts/rival_threshold.py + parked s21 tasks

Authored `scripts/rival_threshold.py` (scores a completed CV run's fold
predictions at the rival's closed-form threshold rule, p > 0.2 x expected
score, mirroring `scripts/queue/lofo_calibrate_view.py` exactly — same
predictions, truth, per-fold regions, and metric — and pairs each fold
against the run's own LOFO-calibrated score with hard errors on a missing
or self-inconsistent LOFO file), `scripts/queue/rival_threshold.sh` (box
wrapper), and three parked queue tasks (`s21-rival-threshold-*-b16`,
spatial/fault/trace vs the ref-b16 incumbent; expected scores pinned to
the committed LOFO-calibrated means: 0.12296 / 0.05469 / 0.01856), plus
`tests/test_rival_threshold.py` (8 tests, including a synthetic end-to-end
run and a real-repo golden test tying each parked task to its committed
LOFO mean). Motivation: step 20 showed 21–58% of each headline score is
threshold fitting, and the October 20 briefing's open question — whether
the rival's rule beats the calibrated cutoffs — had no runnable form.
The experiment is still unapproved: the tasks are parked, not live, and
this step is the handoff (unpark on Sven's go-ahead), not execution. Full
suite: 487 passed, 18 skipped (same documented skips;
test_endpoint_extension.py skimage import remains a VM-venv gap); `ruff
check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or
leaderboard interaction involved; no competition data was read — the
parked tasks reference box-side rasters and fold maps that are never
committed, and all verification used synthetic fixtures and the committed
LOFO state. Training/scoring run on the box queue; the s7/s8/s12/s14
verdicts and the s4–s6/s9–s11/s13 gates still await box results (no new
box commits since the 10-12 b32 baselines).

## 2026-10-27 — Committed-results integrity audit (step 22): scripts/audit_results.py + tests

Authored `scripts/audit_results.py` (one audit over the committed
results record: score-file shape and protocol constants, LOFO mean and
raw-score cross-checks, run-manifest artifact existence with bytes and
SHA256 verification including relocated-content detection, gate-report
arithmetic against the required_margin formula in
src/gems/verification.py, and ledger-vs-gate-report consistency) and
`tests/test_audit_results.py` (11 tests, including a real-repo golden
test asserting the exact finding counts and synthetic clean / hard-error
/ finding-only fixtures). Motivation: no tool had ever verified the
committed record itself, and with the box silent since 10-12 the
committed numbers are the only evidence available. Findings on the real
repo: 33 findings, 0 hard errors — all scored numbers verify, but 25
per-fold detail files recorded in manifests were never committed, 6
score files were committed at a different path than recorded (content
verified by SHA256), ref-cv-fault-b32 still has no manifest, and the
ENDPOINT-EXT-01 ledger entry has no gate report. Full suite: 519
passed, 18 skipped (measured 508/18 baseline without this step);
`ruff check .` clean per CI.

No model training, raw data, checkpoints, submission artifacts, or
leaderboard interaction involved; no competition data was read — the
audit reads only committed JSONs, never rasters or predictions. The
fold files it flags as missing live only on the box (if the box run
dirs still hold them); recovering them needs a box-side commit. The
s7/s8/s12/s14 verdicts and the s4–s6/s9–s11/s13 gates still await box
results (no new box commits since the 10-12 b32 baselines); the
threshold experiment is still unapproved.
