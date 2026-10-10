# Primary sources and technical references

_Last checked: 2026-09-23._

## Competition

- DrivenData overview: https://www.drivendata.org/competitions/306/competition-doe-gems/page/966/
- DrivenData problem description / metric / submission format: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- DrivenData about page: https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/
- Official rules (September 2026 PDF): https://docs.nlr.gov/docs/fy26osti/96647.pdf
- Organizer reference solution: https://github.com/drivendataorg/gems-prize-reference-solution
- Public scoring clarification — known USGS/INGENIOUS pixels are masked in both rounds: https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516
- Public target clarification — “new fault” can include newly mapped continuation/geometry of an existing fault system: https://community.drivendata.org/t/where-do-you-draw-the-line/11536
- Public submission-quota clarification — three-submission allowance resets on a rolling window: https://community.drivendata.org/t/weekly-submissions/11524
- Organizer/community thread to monitor — how private test faults were identified and what data/fault types informed them: https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527
- Community question to monitor — training-label/reference-solution band discrepancy: https://community.drivendata.org/t/why-does-the-training-fault-labels-file-in-the-data-tab-have-a-single-band-while-the-labels-in-the-reference-solution-repo-have-19-bands/11529

## Source datasets / geology

- GeoDAWN: Glen & Earney (2024), USGS; DOI https://doi.org/10.5066/P93LGLVQ
- INGENIOUS Great Basin Regional Dataset Compilation (2022): https://doi.org/10.15121/1881483
- USGS / EERE GeoDAWN background: https://www.usgs.gov/news/featured-story/usgs-and-eere-collaborating-strengthen-americas-energy-and-resource

## Methods highlighted by the organizers

- Mattéo et al. (2021), automatic fault mapping in optical/topographic data with deep learning: https://doi.org/10.1029/2020JB021269
- Hermant, Kiersnowski & Bellanger (2025), deep learning to map Quaternary faults in the Western USA: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf

Do not assume a dataset is competition-legal merely because it is public. Record its license and confirm the competition permits the intended use before incorporating it.


## Candidate technical sources for unknown-fault discovery (reviewed 2026-10-07)

These sources are **methods and hypotheses**, not demonstrated GEMS score gains. The official
competition references above remain the authority on data, evaluation and admissibility.
Keep code and raw data out of this source register unless they are independently checked.
Priority means *next experiment value*, not evidence of improved holdout scores.

| Priority | Source | What it could add | First falsifiable GEMS check | Important qualification |
|---|---|---|---|---|
| P0 | [scarplet — fault-scarp template matching](https://github.com/stgl/scarplet) | DEM-derived scarp likelihood independent of another image encoder | On permitted high-resolution DEM, compare a scale-aware scarp-response channel against unchanged ResNet-18 controls on the **same** spatial, complete-fault and trace-completion folds | Faults do not always produce surface scarps; test scale/DEM quality and preserve transform when downsampling to the 100 m output grid |
| P0 | [Harmonica — potential-field transforms](https://www.fatiando.org/harmonica/latest/user_guide/transformations.html), [API](https://www.fatiando.org/harmonica/latest/api/index.html) | Magnetic/gravity horizontal and upward derivatives, total gradient amplitude, tilt angle and scale-separated maps | Add one physically interpretable transform at a time; compare to direct grouped gradients using identical folds and budgets | Apply only to identified, appropriate physical bands on evenly spaced grids; pad for FFT edges; check sign, units, nodata and registration |
| P0 | [xDEM terrain attributes](https://xdem.readthedocs.io/en/stable/terrain.html) | Multi-scale slope, profile/planform curvature, topographic position, roughness and hillshade | Test a small grouped terrain-feature channel set against base bands, first on spatial folds | Derived 1 m DEM features can disappear when naively reduced to 100 m; document aggregation, resolution, coverage and allowed source terms |
| P1 | [scikit-image structure tensor](https://scikit-image.org/docs/stable/api/skimage.feature.html) | Local orientation, coherence and anisotropy maps from physical raster groups | Compare tensor eigenvalue/coherence channel against the already implemented grouped-gradient/MS-edge controls | Already a dependency; avoid extra packages. Guard smoothing support against holdout buffers and compute fold-pure transforms |
| P1 | [clDice — topology-preserving thin-structure segmentation](https://github.com/jocpae/clDice) ([CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_CVPR_2021_paper.html)) | Soft-skeleton continuity loss for broken or thin predicted traces | BCE/Tversky control versus small auxiliary soft-clDice loss; score with the **official distance-weighted metric** across all three holdouts | A connected prediction is not necessarily a valid fault; excess bridging creates false positives. Do not swap official metric for clDice |
| P1 / research | [Kiryo et al. (2017), non-negative positive-unlabeled learning](https://arxiv.org/abs/1703.00593); [pulearn implementation/docs](https://github.com/pulearn/pulearn) | Formal tools for cases where mapped-fault positives are incomplete and unlabelled pixels are not guaranteed negatives | Only after auditing label-selection mechanism and estimating/sweeping class-prior sensitivity; run matched controls without treating all unlabeled pixels as true negatives | Classical PU identification assumptions (e.g., selected-completely-at-random positives) may fail badly for geologists' mapped traces; do not assume applicability from the name |
| P2 | [fractopo — fracture/lineament network analysis](https://github.com/nialov/fractopo) ([JOSS 2023](https://doi.org/10.21105/joss.05300)) | Branch, endpoint, orientation and trace-connectivity diagnostics after vectorization | Compare candidate-geometry QA on known withheld segments and false-bridge rates; only later consider postprocessing | Network descriptors are not geological truth, and vectorization/snap thresholds can invent junctions |
| P2 | [scikit-image probabilistic Hough transform](https://scikit-image.org/docs/stable/api/skimage.transform.html) | Fast, reproducible line-segment proposal baseline for directional structure | Compare oriented line-response postprocessing to unmodified probability maps on held-out traces | Straight-line prior may miss curved or en-echelon faults; record line length/gap thresholds and false connectors |
| Validation anchor | [Roberts et al. (2017), cross-validation with spatial structure](https://doi.org/10.1111/ecog.02881) | Literature basis for buffered geographic holdouts instead of random pixel/patch splits | Keep fixed independent spatial, complete-fault and trace-completion evaluation views; use stitched pooled spatial evidence and separate overlapping-background discovery safeguards before declaring a win | Spatial CV is necessary but not proof of generalization to truly unknown geology |

### Experiment order and gates

1. **Do not multiply baselines:** finish a full fold-specific, training-only-normalized ResNet-18 control and record score variance first; historical fold-0 scores computed with full-raster normalization are not valid comparisons.
2. **Prioritize new geological evidence:** test Harmonica potential-field responses, xDEM morphometry, and scarplet-style scarp likelihood before another pretrained encoder. First make a simple feature map with source-band identity, physical units, nodata behavior, raster transform, and resolution checks.
3. **Then test network-aware objectives:** structure-tensor channels, auxiliary soft-clDice, and line/graph postprocessing must each enter a separate, budget-matched ablation. Check FP bridging and genuinely withheld continuations, not just apparent connectedness.
4. **Treat PU as conditional research:** map the positive-label selection mechanism and sensitivity to unknown class priors before any method changes; do not use external known-fault products as a shortcut to hidden labels.
5. **Acceptance:** require improvement under the repository's independent proposal/verification gate, against spatial, complete-fault, and trace-completion controls, with reproducible hashes, fixed folds, and no leakage. An attractive map or paper result is not a GEMS gain.

### Data, licenses, and traceability

These are **reference links only**; no new source datasets, dependencies, weights, or downloaded
code have been imported. Review each code license, maintenance status, transitive dependency
and any external data rights before adoption. Log any data product in
[EXTERNAL_DATA_LEDGER.md](EXTERNAL_DATA_LEDGER.md). Verify eligibility against official rules
and organizer clarification; use the published metric and existing submission-format checks.


## Metric-aware research source update (2026-10-10)

- Organizer pooled-scoring clarification (2026-10-01): https://community.drivendata.org/t/leaderboard-aggregation-pooled-over-public-test-pixels-or-mean-of-per-chunk-scores/11550
- Organizer final-label/exclusion/weight clarification (2026-10-07): https://community.drivendata.org/t/questions-about-the-final-round-rules/11556
- Checked wavelet, PU, topology and spatial-block literature with experiment boundaries:
  [research/METRIC_AWARE_RESEARCH_2026-10-10.md](research/METRIC_AWARE_RESEARCH_2026-10-10.md).

These references add no external training data, code or weights.
