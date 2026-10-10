# Metric-aware GEMS selection and experiments

Checked 2026-10-10 against the supplied October 9 research, current repository
code, organizer statements and primary literature. P0 is implemented; P1-P5
are hypotheses, with no new official-data training results.

## P0 — correct what selects the incumbent

The [October 1 organizer answer](https://community.drivendata.org/t/leaderboard-aggregation-pooled-over-public-test-pixels-or-mean-of-per-chunk-scores/11550)
confirms that public and private scoring pool pixel contributions before forming
one Tversky ratio. Final re-evaluation covers the whole GeoDAWN area.

The local spatial evaluator already stitches each fold's validation region into
one OOF prediction. The previous verifier instead selected on macro fold means,
with a two-of-three-view vote. Those objectives can choose different candidates.
Schema-2 scoring now records weighted TP/FP/FN from the **stitched** raster, plus
truth/valid coverage. The verifier validates that evidence and uses its pooled
score as primary. Per-fold geographic scores and fault/trace macro means remain
non-regression safeguards. All three views must be comparable for acceptance.

Do not reconstruct pooled evidence by summing independently masked fold TP/FP/FN:
the three-pixel kernel can match across a spatial seam. Fault and trace masks
reuse background and cannot be pooled as disjoint evaluated pixels either.
Synthetic tests cover both seam matching and a macro/pooled ranking reversal.

The spatial improvement hurdle retains the existing search-pressure heuristic:
maximum fold standard deviation times `1 + 0.5 log2(1 + trials)`. Safeguard
tolerance is the un-escalated maximum fold std; a geographic fold or discovery
macro mean may not regress beyond it. These are policy choices, not calibrated
confidence bounds or proof of hidden-label generalization. Reports identify the
selection policy as `pooled_spatial_with_nonregression_v2`.

Supplied seed, epoch count, training stride and buffer metadata now survive
parsing and are compared. Config hashes may differ across candidate/control
recipes. With `--metrics-pattern`, the scorer requires consistent metadata for
every scored fold rather than emitting a usable report after warnings. Without
that option, reports certify evaluation comparability only, not equal training
budgets. Re-score retained prediction rasters; preserve historical ledger entries.

## P1 — distance-aware differentiable Tversky

The [official metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
uses a triangular kernel with radius three pixels. For evaluated label set G:

\[
m_g(p)=\max_{x:\|x-g\|\le3}p_x(1-\|x-g\|/3),\quad
T=\sum_{g\in G}m_g(p),\quad FN=|G|-T.
\]

FP mass also matters; proximity-weighted matching alone is insufficient:

\[
F=\sum_{x\in V}p_x\left[1-\max_{g\in G}(1-\|x-g\|/3)_+\right].
\]

For alpha 0.2 and beta 0.8, the proposed loss is

\[
L_{distance}=1-\frac{T}{0.2T+0.2F+0.8|G|+\epsilon}.
\]

The finite local max admits subgradients almost everywhere. This is a direct
adaptation of the competition equations, not an existing measured GEMS advance.
Keep a conventional BCE/Tversky component: when a patch has no positives, this
ratio alone gives no useful background gradient. Avoid replacing the current
loss or encoder before a controlled ablation.

Implementation requirements: three-pixel context halos, core-only contribution
counting, no wraparound, validity and supervision masks, and no withheld labels
in target kernels or FP distance fields. Both prediction and truth context must
come from eligible training pixels; uncertain boundary cores must be excluded.
A patchwise ratio is a surrogate for the global ratio, so patch sampling and
reduction must be documented rather than described as exact global training.

Acceptance experiment: conventional loss versus conventional-plus-distance loss
with the same ResNet-18, data fingerprints, spatial folds, seeds, optimizer steps
and training budget. Fix blend weight before evaluation; tune it inside training
folds if necessary. Test NumPy metric agreement, one-pixel shifts, empty targets,
invalid pixels, halos, and gradients before training. Promote through P0 only
after independently scored fault and trace safeguards are available.

## P2 — potential-field wavelet entropy

[Dwivedi (23 February 2026)](https://doi.org/10.3389/feart.2026.1771807)
compares potential-field edge operators on synthetic magnetic prisms and a
Laxmi Basin gravity example. Wavelet space entropy performs well in its noisy
synthetic boundary tests and marine case. The method uses Haar decomposition
and windowed entropy; source boundaries need not be fault traces.

A small **adaptation**, not reproduction of that estimator, is

\[
q_s(x)=\frac{E_s(x)}{\sum_jE_j(x)+\epsilon},\qquad
H(x)=-\sum_s q_s(x)\log(q_s(x)+\epsilon).
\]

Define zero-energy behavior explicitly; fix scales/window support; preserve
fold-pure normalization and avoid leakage through filter support. Start on
physically suitable raw magnetic/gravity bands, not every channel or already
derived slopes. Compare against grouped gradients with the same model/budget.

[Abbassi and Cheng, curvilinear lineament extraction](https://doi.org/10.1016/j.cageo.2024.105768)
combines principal-component wavelet analysis, hysteresis and Bayesian tuning.
Its reported improvement concerns its own F-beta evaluation; it establishes no
GEMS gain. No MATLAB code, new dependency or source dataset is imported here.

## P3 — incomplete labels

[Needles in the Landscape (2025)](https://arxiv.org/abs/2510.16814)
studies geospatial archaeological discovery with pseudolabeling under sparse
labels. It motivates a test of background treatment, not a transferable fault
detector. [Zhao et al., ICCV 2023](https://arxiv.org/abs/2308.15081)
develop class-prior-free PU learning for hyperspectral imagery and analyze
unlabeled samples dominating optimization. This is a 2023 result, not a newly
released 2026 method.

First compare a fixed, modest reduction of ambiguous-background penalty against
the control on complete-fault holdouts. Fit ambiguity only from training-side
evidence. Report probability mass, FP cost and class-prior sensitivity. Geologist
selection is not uniform; classical PU assumptions need auditing. Do not
pseudolabel the entire unmarked raster or reward flooded predictions.

## P4-P5 — geometry and topology after the baseline

[Chen et al., curvature-prior geodesic paths (2023)](https://arxiv.org/abs/2308.15729)
supports orientation-aware curved continuation as a later candidate. First
measure the implemented straight endpoint extension against a full baseline.
Fix endpoint/path budgets and penalize unsupported bridges.

[Skeleton Recall Loss (2024)](https://arxiv.org/abs/2404.03010)
and [Does the Skeleton-Recall Loss Really Work? (2025)](https://arxiv.org/abs/2508.11374)
provide conflicting empirical evidence on thin-structure segmentation. Neither
settles GEMS performance. A small matched auxiliary-loss ablation is appropriate;
connectivity is a diagnostic, not a replacement for distance-weighted scoring.

## Marginal geometry gain — derived, not yet an estimator

With fixed evaluated truth and mask, write
`D = 0.2T + 0.2F + 0.8|G| + epsilon` and `S = T/D`.
For any change whose new denominator stays positive,

\[
\frac{T+\Delta T}{D+0.2\Delta T+0.2\Delta F}>\frac{T}{D}
\iff (1-0.2S)\Delta T>0.2S\Delta F.
\]

The strict inequality restores the missing comparison sign in the supplied
research text. Coverage saturates through the max kernel: adjacent additions
can add FP mass without increasing T. Measure these increments directly for
thin/widened/extended predictions on development folds. Hidden-test increments
are unavailable; a learned estimator needs a separate holdout and nested
selection. This algebra is not permission to tune on hidden labels.

## Execution order and uncertainty

1. Re-score retained OOF predictions under P0; finish the adequately trained
   baseline and obtain complete fault/trace reports. Historical reduced-training
   means near 0.0591 raw and 0.0895 calibrated remain historical experiment
   results; they do not establish a new pooled improvement.
2. Run the P1 loss ablation; then P2 wavelet entropy; then P3 background loss.
3. Try P4/P5 only after their controls are measured. Every trial remains logged.

[Stock (2025), spatial-block choices](https://doi.org/10.3389/frsen.2025.1531097)
studies how block design affects evaluation in synthetic marine remote sensing.
Pre-register a block-size sensitivity check without choosing the easiest split.
The [October 7 organizer clarification](https://community.drivendata.org/t/questions-about-the-final-round-rules/11556)
allows final faults to be added, moved/changed or removed; known-fault exclusions
stay fixed and no extra fault-specific weights are used. This reinforces the
need for geographically separate discovery safeguards and label uncertainty.

No new architecture, external training data, weights, paid compute or submission
was introduced by this research/selection update. Official rasters and retained
OOF artifacts are not present in this checkout, so new measured training results
remain the next evidence gate on the training host.
