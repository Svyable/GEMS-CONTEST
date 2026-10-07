# OpenAI math and reasoning releases — GEMS applicability scan

_Checked 2026-10-07. This is a research note, not a performance claim._

## Bottom line

The October 6 OpenAI mathematics release is important for GEMS mostly because it
demonstrates a high-throughput **propose -> formalize/check -> revise** research
workflow. Most individual theorems do not match the GEMS observation model.

One result is materially closer to the contest than the inverse-problem results
already screened: **family 366, planar Mumford--Shah regularity**. Mumford--Shah
models are built around piecewise-smooth fields separated by discontinuity sets,
and the released result characterizes local minimizer geometry using smooth arcs,
regular crack tips, and three-way junctions. That does not make faults Mumford--Shah
minimizers, but it gives us a concrete, cheap-to-test hypothesis: derive
Mumford--Shah/Ambrosio--Tortorelli-style lineament channels from the supplied
continuous geophysical bands and test them as additional inputs under honest GEMS
cross-validation.

A second useful transfer is methodological rather than theorem-level: use the
release's emphasis on independent verification to keep exact metric oracles,
property tests, immutable run manifests, and one-hypothesis experiments. A third
candidate is a **registration-sensitivity stress test** motivated by the optimal
transport stability results, without claiming those theorems bound our model.

## Sources checked

- OpenAI, “Sharing AI progress in mathematics,” 2026-10-06:
  https://openai.com/index/sharing-ai-progress-in-mathematics/
- OpenAI math repository:
  https://github.com/openai/math
- Pinned catalogue used for this screen:
  https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/overview.tex
- Family 366 preprint, “Interior regularity of planar Mumford–Shah minimizers”:
  https://github.com/openai/math/tree/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Interior-regularity-of-planar-Mumford-Shah-minimizers-September-24-2026
- Ambrosio & Tortorelli (1990), classical elliptic approximation of the
  Mumford--Shah functional:
  https://doi.org/10.1002/cpa.3160430805
- OpenAI, “On the Navier–Stokes Millennium Prize Problem,” 2026-09-08:
  https://openai.com/index/navier-stokes-solution/
- OpenAI, “Research acceleration: The view inside OpenAI,” 2026-09-06:
  https://openai.com/index/research-acceleration-view-inside-openai/
- OpenAI, “Ten advances in mathematics and theoretical computer science,”
  2026-08-01:
  https://openai.com/index/ten-advances-in-mathematics/
- OpenAI, “An OpenAI model has disproved a central conjecture in discrete
  geometry,” 2026-05-20:
  https://openai.com/index/model-disproves-discrete-geometry-conjecture/
- OpenAI, “Our First Proof submissions,” 2026-02-20:
  https://openai.com/index/first-proof-submissions/
- OpenAI, “Introducing GPT-6.1 Sol,” 2026-09-29:
  https://openai.com/index/introducing-gpt-6-1-sol/

The October 6 repository describes 722 manuscripts in 372 families, many but not
all with Lean formalizations. OpenAI reports roughly 4,000 posed problems and an
average per-result compute equivalent to about three hours of ChatGPT Pro thinking
with the unreleased internal model. We treat every unverified result as a research
lead, not an authority.

## Release-to-GEMS screen

| Release / result | What is actually new | GEMS relevance | Decision |
| --- | --- | --- | --- |
| OpenAI math release (Oct 6) | Large catalogue, revision protocol, many Lean checks, reasoning summaries and compute accounting | Strong template for research hygiene and machine-checkable subproblems | ADOPT workflow |
| Family 366 — planar Mumford--Shah regularity | Local discontinuity sets of minimizers are regular arcs, crack tips, or three-way junctions under the theorem's assumptions | Closest mathematical analogue to line-like discontinuities in geophysical rasters; suggests phase-field/edge channels | TEST |
| Families 360 / 374 — optimal transport regularity/stability | Regularity and sharp stability statements for transport maps in specific geometric settings | No direct relation to fault probabilities, but motivates a measured georegistration sensitivity experiment | TEST robustness only |
| Family 365 — metric/connection recovery and conductivity nonuniqueness | Boundary inverse-problem results under specific PDE observation models | Supplied conductivity is a raster feature, not a Dirichlet-to-Neumann operator; no fault-recovery guarantee | WATCH / do not transfer theorem |
| Family 372 — isotropic elasticity inverse uniqueness | Static boundary data determine smooth Lamé moduli under the paper's assumptions | GEMS has strain-rate layers, not the required boundary displacement/traction data | WATCH / no executable transfer |
| Navier--Stokes + Lean (Sep 8) | AI-generated research proof with formal verification | Verification process is relevant; the fluid PDE result itself is not a fault-mapping method | ADOPT verification lesson only |
| Ten advances / unit-distance result | Novel mathematics across unrelated domains | No defensible mapping from those theorems to the GEMS metric or data | DEFER |
| First Proof | Research-level proof attempts plus external checking; at least one attempt was later judged incorrect | Reinforces independent verification and not trusting fluent proofs | ADOPT verification lesson |
| GPT-6.1 Sol / research acceleration | Stronger scientific/coding-agent workflows and lower-cost long-horizon execution | Useful for code/eval automation if budget is authorized; benchmark strength does not imply GEMS score | OPTIONAL process accelerator |

## Candidate experiment: MS-EDGE-01

### Hypothesis

Fault-relevant evidence is often expressed as coherent discontinuities or narrow
transition zones across multiple geophysical fields. A numerical approximation to
Mumford--Shah can produce an edge/phase field that emphasizes such discontinuities.
Adding those maps as derived channels may help the segmentation model generalize
from mapped faults to unmapped continuations, splays, and parallel strands.

### Important non-claim

The OpenAI family 366 theorem is **not** a geological law. Do not impose 120-degree
junctions, crack-tip counts, or any theorem-specific topology on predicted faults.
At the pinned release commit, no `lean/docs/366.md` formalization entry was found;
treat the theorem summary as a research lead even before considering its practical
transfer.

### Minimal implementation

1. Start only after the corrected ResNet-18 control has been rerun on all five
   buffered spatial folds.
2. Use only continuous supplied feature bands for the first test. Compute all
   normalization/scale parameters from the fold training region.
3. Build one or more Ambrosio--Tortorelli-style phase/edge maps at a small number
   of registered spatial scales. Prefer family-level composites (for example,
   magnetic, gravity, topographic, deformation) over dozens of highly correlated
   per-band maps in the first experiment.
4. Add the maps as channels to the same ResNet-18 U-Net, keeping optimizer,
   sampling, augmentation, epochs, folds, and metric fixed.
5. Evaluate on all buffered spatial folds and, once implemented, fault-discovery
   and trace-completion validation. Keep the original feature stack as the control.
6. Record the transform parameters and source-band list in the immutable run
   config/manifest.

### Promotion gate

Do not keep the feature because it “looks geological.” Promote it only if the same
frozen recipe improves multiple honest validation views and the gain is not driven
by a single convenient fold. Report fold-by-fold deltas, not only a pooled number.

### Failure modes to watch

- lithologic or acquisition boundaries may look like faults;
- variational smoothing may erase weak lineaments;
- too many derived channels may simply increase capacity and overfit;
- a phase-field solver may be expensive relative to simple gradients/structure
  tensors;
- edge maps can duplicate existing directional features without adding information.

A cheap comparison against gradients, structure-tensor coherence, and steerable
filters should be part of the same ablation family.

## Candidate experiment: REG-SENS-01

The optimal-transport results do **not** provide a stability theorem for this
competition model. Their useful prompt is narrower: quantify how sensitive our
predictions are to small spatial misregistration between feature families.

1. Group input bands by physical/source family.
2. At evaluation time, shift one family at a time by each neighboring one-pixel
   offset (100 m) while holding the trained model fixed.
3. Measure the held-out distance-weighted Tversky delta, prediction disagreement,
   and whether errors concentrate near candidate traces.
4. If a useful model is brittle, test small group-wise translation augmentation or
   an inference ensemble over plausible alignments as a separate experiment.
5. Do not introduce any shift correction without evidence from held-out folds.

This is unusually relevant because one pixel is 100 m and the official metric has
300 m spatial support; an apparent score tolerance does not mean the input
representation is robust to misregistration.

## Agentic research loop

OpenAI's September research-acceleration report is consistent with using coding
agents to increase experiment throughput, but it also reports substantial human
steering on longer tasks. For this repository:

- humans set the prize-relevant hypothesis and decide what evidence is sufficient;
- agents may implement transforms, tests, configs, run summaries, and comparisons;
- each experiment gets a commit, immutable config, data fingerprint, and objective
  metric output;
- merge when CI/objective gates pass; do not add review rituals as blockers;
- do not initiate paid API or compute usage without an authorized budget;
- do not confuse more experiments with more independent evidence.

GPT-6.1 Sol is a possible implementation/research agent because OpenAI reports
strong scientific workflow and coding performance, but no model benchmark is
evidence that it will improve GEMS predictions.

## Priority relative to current blockers

1. **Rerun the corrected control first.** The repository already says historical
   fold-0 scores used contaminated full-raster normalization.
2. **Implement fault-discovery / trace-completion supervision.** Without that,
   feature claims can still optimize the wrong notion of success.
3. **Run MS-EDGE-01.** It is the strongest theorem-adjacent idea from the October 6
   release because it connects to line/discontinuity geometry and needs no external
   data.
4. **Run REG-SENS-01.** This is a robustness test that can expose a hidden
   preprocessing weakness before more architecture work.
5. Only then use stronger agentic parallelism to fan out ablations, and only within
   the repository's compute/budget policy.

The October 6 release should therefore change our **research queue**, not our
beliefs about the current best GEMS model. No competition gain is claimed until
these experiments run on the official data under the existing validation contract.
