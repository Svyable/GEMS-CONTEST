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

Treat each derived channel as an ablation. A channel only stays if it improves multiple validation views rather than one convenient split.

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
