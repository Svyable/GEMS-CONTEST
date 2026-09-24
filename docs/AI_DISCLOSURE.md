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

## Add future entries

For each material use, record date, tool/model, what information was provided, what was generated, and what verification was performed before relying on the output.
