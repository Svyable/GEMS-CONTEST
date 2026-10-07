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
