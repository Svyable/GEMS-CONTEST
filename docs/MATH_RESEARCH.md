# Mathematical research for Svyable's GEMS entry

Checked 2026-10-07. Goal: win GEMS through reproducible hidden-fault discovery.

## Recent OpenAI release and what transfers

OpenAI's [October 6 release](https://openai.com/index/sharing-ai-progress-in-mathematics/)
links to [openai/math](https://github.com/openai/math), whose README describes 722
manuscripts in 372 families, at different verification stages. Some have Lean
formalizations; unformalized results may have issues. The producing model is
described as unreleased. We have not imported it, reproduced its results, or
checked these proof artifacts ourselves.

Catalogue inspection was pinned to commit
`adc7f1241b42e322a6451854ab7e4b4c146bf78a`; see the
[overview source](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/overview.tex).
The following are screening decisions based on its theorem summaries, not proof
reviews or established GEMS improvements.

| Family | Catalogue subject | GEMS connection / next evidence needed | Decision |
| --- | --- | --- | --- |
| 365 | Joint metric/connection recovery; contrasting conductivity nonuniqueness | Conductivity is an input layer, but GEMS has raster features, not the theorem's boundary measurement setup. Do not infer unique fault geometry from one layer. Test independent feature-family ablations. | WATCH; assumptions do not match our observations |
| 372 | Uniqueness in smooth isotropic elasticity | Potential conceptual relevance to deformation, but static boundary displacement/traction and smooth Lamé moduli are not supplied GEMS data. Need an actual forward model and matching observations. | WATCH; no executable transfer established |
| 374 | Sharp stability exponent for optimal transport maps | Could inform a future alignment sensitivity experiment, but transport maps between probability measures are not fault probabilities. Need a licensed alignment method and held-out ablation. | WATCH; no direct metric guarantee |
| 212–214 | Percolation/geodesic and critical-cluster results | Candidate traces are spatial networks, but iid or quasi-transitive assumptions do not describe geologic rasters. Do not impose a theoretical critical threshold on model confidence. | DEFER |

The [unit-distance result](https://openai.com/index/model-disproves-discrete-geometry-conjecture/)
uses algebraic number theory to construct extremal point sets. Its exact unit-distance
problem does not match GEMS's 300 m triangular kernel. No score increase follows
from using that theorem. The [First Proof update](https://openai.com/index/first-proof-submissions/)
also records a proof attempt later judged incorrect. Both support treating AI
reasoning as a proposal that must survive independent checks.

**Adopt now:** derive a concrete optimization from the organizer's published
objective, produce an executable implementation, and compare it against an
independently written oracle. This is our application of verification discipline;
the derivation below is not a theorem copied from the OpenAI release.

## Exact threshold optimization

For threshold `t`, let `b_t(x) = 1[p(x) >= t]`. The finite raster can change binary
state only at a distinct valid prediction value. Include one threshold strictly
above the maximum for the empty-prediction state.

For each truth pixel `g`, its contribution is the largest triangular weight among
active neighbors: `m_g(t) = max_{x: p(x)>=t} k(d(x,g))`, with zero for an empty set.
Activating predictions in descending probability order therefore creates only
nonnegative increments to `m_g`. Tied probabilities activate together.

The false-positive contribution of each activated pixel is fixed:
`1 - max_g k(d(x,g))`. Sum these FP events and the TP increments cumulatively,
then calculate `FN = number_of_truth_pixels - TP` and the published DTI ratio.
This enumerates every achievable binary threshold state exactly (up to floating
point arithmetic), without rescoring the whole raster at every threshold.

Implementation: `src/gems/calibration.py`. Cost is sorting valid probabilities and
local truth-neighbor events; memory is `O(N + K*G)` for valid pixels `N`, truth
pixels `G`, and positive-kernel offsets `K`. At the official radius, `K=25`.
It still allocates arrays proportional to the full valid raster and is not a
streaming implementation. Test realistic memory usage on the training host.

Tests compare **every** threshold against a coordinate-distance oracle and the
existing NumPy metric for multiple radii, seeded rasters, masks, edges and ties.
CLI tests check geometry, binary masks, nodata and calibration/evaluation overlap.
These are executable numerical checks, not a Lean formalization.

Binary thresholds are candidates, not a mandate to discard confidence. The binary
family need not beat the unchanged soft raster, and its best fit score is biased
by selection. Evaluate both unchanged and thresholded predictions on a separate
held-out region. No real-data score improvement is claimed for this implementation.

## Experiment order

1. Rerun the ResNet-18 control with training-only normalization on all five spatial
   folds. Existing fold-0 scores used older preprocessing and are historical.
2. Produce buffered calibration and evaluation subregions both excluded from
   training/preprocessing. Fit one threshold on calibration, then measure it once
   on evaluation; preserve immutable masks and input hashes. Ordinary OOF folds
   used as mutual calibration/evaluation are not automatically independent: a
   calibration fold's model may have trained on the evaluation fold's labels.
3. Implement and score fault-discovery and trace-completion supervision, which the
   current spatial-only trainer does not support. Verify calibration across these
   tasks before promoting it to final-map inference.
4. Compare distance-aware training, directional features and external DEM channels
   one at a time under the same validation/provenance contract.

The blocker here is access to official rasters and held-out predictions on the
training host, not absence of another speculative architecture. This change runs
without paid model calls or paid compute and does not upload a submission.
