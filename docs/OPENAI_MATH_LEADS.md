# OpenAI math release: what's useful for GEMS

_Researched 2026-10-07. Intended repo path: `docs/OPENAI_MATH_LEADS.md`._

**Short version.** OpenAI's October 2026 math release is real and big: 722 manuscripts in 372 families, including a claimed bound of 9/4 on the matrix-multiplication exponent. Almost none of the *mathematics* will move our fault-detection score. The 9/4 result is an asymptotic statement about arithmetic complexity and gives us no faster matrix multiply at our raster sizes. What carries over is the *method*: a model generates a large number of candidates, and an automated verifier filters them. In the release the verifier is Lean/Comparator. Ours is the published Tversky metric on honest spatial CV. One paper, the planar Mumford–Shah regularity result, suggests a useful geometric prior and feature channel. A second suggests a directional-filter design rule. Everything else is not applicable.

Verification status: the primary facts below come from the `openai/math` GitHub repository (cloned 2026-10-07, single commit `adc7f12`, dated 2026-10-06 14:58 -07:00). The OpenAI blog post ("Sharing AI progress in mathematics") returned 403/JS-wall to our fetchers, so we only quote it from search snippets and secondary coverage. Neither we nor anyone else has peer-reviewed or Lean-checked these claims. The repo README itself says unformalized results "could have issues."

---

## 1. The "9/4 matrix multiplier" paper

- **Title:** *An Upper Bound of 9/4 for the Matrix Multiplication Exponent*
- **Author / date:** "OpenAI", dated **October 2, 2026** (PDF built Oct 5). "9/4" is the exponent 9/4 = 2.25. It is *not* a September 4 date.
- **Link:** https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/paper.pdf (13 pages)
- **Catalog entry:** Family 107, "Matrix multiplication with exponent at most 9/4." Companion writeups in the same family:
  - *Complex Matrix Multiplication Below 2.258 and Rectangular Bounds* (Sept 24, 2026; ω < 2.258, dual exponent α > 0.465)
  - *Staggered extraction for exact matrix multiplication over every field* (Sept 24, 2026; ω < 2.371054886006746 over every field)
- **Lean status:** the repo's scope note `lean/docs/107.md` says the formalized results include ω(ℂ) ≤ 9/4, α > 0.465, and ω < 2.371054886006746 over every field, in the division-free arithmetic model. It adds explicitly that this concerns "arithmetic complexity, rather than bit complexity or practical crossover sizes." We have not built or checked the Lean ourselves.

**Plain summary.** ω is the smallest number such that two n×n matrices can be multiplied with about n^ω arithmetic operations as n → ∞. The schoolbook method gives 3. Strassen (1969) got below 3. Decades of "laser method" refinements (Coppersmith–Winograd → … → Alman–Vassilevska Williams et al.) brought it to ≈2.3713. The latest prior bound, ω < 2.371177, is cited in the paper as Dupont et al. and was announced by Google DeepMind and collaborators with AlphaEvolve in Aug 2026. The OpenAI paper claims **ω ≤ 2.25**, a far larger jump than recent records. Instead of tuning the laser method, it works in Strassen's *asymptotic spectrum*: "tensor characters" are numerical invariants that add over direct sums, multiply over tensor products, and are monotone under restriction. Bounding every character on the matrix-multiplication tensor bounds ω. The key new construction separates tensor blocks that share variables on one leg. It uses a finite Fourier projection to guess block labels and a degeneration that penalizes label mismatch quadratically. The paper applies this to *polynomial multiplication* and derives a growth profile P(a,b) with P(a, 3h+a−1) ≥ 3·P(a,h). That forces P(a,a) ≥ a^{4/3}, which gives t ≤ 3/4 and hence ω ≤ 3t ≤ 9/4. One step, the existence of a detecting character, uses compactness and a fixed-point theorem. The paper states the proof "does not specify a competitive finite matrix size," and commentators describe it as non-constructive in practice.

**Relevance to GEMS: none in practice.** Our stack is about 10^2–10^4 pixels per side and runs on GPU BLAS/cuDNN and FFTs. No implementable algorithm comes out of this paper, and even if one did, the crossover size would be astronomically larger than anything we multiply. It is a landmark for theoretical CS but not a lever for our score.

---

## 2. The "~700 papers" release

- **What:** 722 mathematical manuscripts in 372 "result families," produced by an **unreleased internal OpenAI model** as part of model-development evaluations on open research problems. Most manuscripts are dated Sept 23 – Oct 5, 2026.
- **Where:** https://github.com/openai/math (Apache-2.0). It was announced Oct 6, 2026 in the OpenAI post https://openai.com/index/sharing-ai-progress-in-mathematics/ and covered by New Scientist (Oct 7), Unite.AI, The Information, Techmeme, and others.
- **Organization (from the README):**
  - `overview.pdf`: family descriptions grouped by 17 disciplines (number theory, algebraic/complex geometry, real/complex analysis, convex & metric geometry, theoretical CS, dynamics, combinatorics, algebra, probability/stat-mech, logic, group theory, mathematical physics, operator algebras, topology, functional analysis, differential geometry, PDE).
  - `CONTENTS.md`: the manuscript map, with each family's result statement and its papers and abstracts.
  - `preprints/`: one directory per manuscript (PDF, LaTeX source, BibTeX, build notes).
  - `lean/`: one Lean library plus `formalization.yaml`, which lists 162 papers with a formalized main result. 235 of the 372 families link a Lean scope note. Verification uses the Lean FRO's `comparator` (`lean/ComparatorChallenges/`).
  - `reasoning_traces/`: 10 abridged "summarized chain of thought" PDFs (e.g., irrationality exponent of π, Mahler conjectures, Mézard–Parisi formula).
- **How the results were produced (README):** the model was posed about 4,000 problems. On average each result used about three hours of ChatGPT Pro-equivalent thinking compute. Outputs were then aggregated into families and filtered for significance. Exceptions: the Riemann-zeta zero-free-region work, whose Re(s) > 11/12 writeup was human-edited, and the Hodge conjecture for CM abelian varieties.
- **Context, from secondary sources only and not verified against openai.com:** OpenAI consulted the IAS Advisory Group on Mathematics and AI. That group's Sept 29, 2026 recommendations ask labs to disclose the model, prompts, compute, and how many comparable problems were attempted and failed. Coverage also says OpenAI announced on Sept 8, 2026 a Navier–Stokes finite-time-singularity result from the same internal model. We did not verify that claim from a primary source.

---

## 3. Ranked shortlist for GEMS

Ranked by realistic expected impact on the hidden-set distance-weighted Tversky score (300 m triangular kernel, α=0.2, β=0.8, known faults masked).

### #1. The method: massive generate-and-verify search. Relevance: **high (as process, not math)**
- **Source:** README "How the results were produced"; Lean/Comparator verification (`lean/ComparatorChallenges/README.md`); `reasoning_traces/*.pdf`, which show long chains of failed approaches before a successful one.
- **Mapping:** OpenAI's leverage came from scale (about 4,000 problems) times a strict, automatic verifier times a significance filter. Our analogue:
  - **Generator:** agent-proposed candidates for derived feature channels, losses, architectures, scales, ensembling weights, and post-processing.
  - **Verifier:** `scripts/score_cv.py` with the published metric on *all three* CV views (spatial-block, fault-component, trace-completion), plus `validate_submission.py`.
  - **Significance filter:** accept a change only if it beats the incumbent on ≥2 of 3 views by more than fold-to-fold standard deviation. Otherwise discard it.
  - **Honest failure log:** record every attempt, not only the winners, in `docs/EXPERIMENT_LOG.md`, mirroring the IAS "report attempts and failures" norm. This is also our main defense against overfitting CV through many comparisons. The more candidates we try, the larger the improvement margin we should demand.
- **Blunt caveat:** a Lean proof is a *perfect* verifier. Our CV is a *noisy proxy* for a hidden label set of newly mapped faults with a different distribution. Search pressure will exploit CV noise unless the acceptance bar rises with the number of trials. Spend the three-per-rolling-window leaderboard submissions only on pre-registered hypotheses.

### #2. Family 366: planar Mumford–Shah regularity. Relevance: **moderate (geometric prior / feature idea)**
- **Paper:** *Interior regularity of planar Mumford–Shah minimizers* (Sept 24, 2026), https://github.com/openai/math/blob/main/preprints/Interior-regularity-of-planar-Mumford-Shah-minimizers-September-24-2026/Interior-regularity-of-planar-Mumford-Shah-minimizers-September-24-2026.pdf. Not Lean-formalized according to the catalog.
- **Claim:** for planar Mumford–Shah minimizers (fit a piecewise-smooth image u to data g, paying for the length of the jump set K), K is locally only (a) a C^{1,α} arc, (b) an arc ending at a "crack tip," or (c) three arcs meeting at 120°. Only finitely many components meet any compact region.
- **Mapping:** Mumford–Shah is *the* classical variational model for "where does a 2-D field jump," and faults are exactly lines across which the gravity, magnetic, conductivity, and strain-rate fields jump. The new theorem says that, in this model, the edge set is guaranteed to be a clean network of smooth arcs, terminations, and Y-junctions. That geometry matches fault traces: segments, tips, and splays. Concrete uses:
  1. **Unsupervised feature channel.** Use edge detection (e.g., Laplacian of Gaussian) to identify where geophysical fields jump. The Mumford-Shah regularity result motivates the geometric prior (smooth arcs, tips, Y-junctions) but does not prescribe the implementation. For this competition, LoG is practical and scalable. Feed the edge indicator field to the U-Net as a candidate channel and ablate it like any Phase-2 feature.
  2. **Post-processing prior.** Skeletonize the probability map into arcs, tips, and junctions. Prune isolated specks. **Extend arc endpoints along local orientation.** With β=0.8 and a 300 m tolerance, recovering continuations of mapped faults is cheap recall, which matches the Sept 23 organizer clarification that continuations count.
- **Blunt caveat:** the *functional* dates from 1989 and Ambrosio–Tortorelli from 1990. The new result is a regularity guarantee and does not introduce a new algorithm. It justifies the prior but does not hand us code.

### #3. Family 083: Hilbert transforms along Lipschitz directions. Relevance: **low–moderate (filter-design heuristic)**
- **Paper:** *A uniform Hilbert transform estimate for Lipschitz directions* (Sept 25, 2026), https://github.com/openai/math/blob/main/preprints/A-uniform-Hilbert-transform-estimate-for-Lipschitz-directions-September-25-2026/main.pdf (Lean scope note available).
- **Claim:** a planar Hilbert transform taken along a Lipschitz unit vector field v(x) is L²-bounded, i.e. stable, when the integration length is at most an absolute constant divided by v's Lipschitz constant. This settles Stein's weak-(2,2) conjecture at that scale.
- **Mapping:** potential-field geophysics already uses Hilbert/derivative-based edge detectors (analytic signal, tilt angle). Our Phase-2 plan includes structure-tensor orientation and steerable filters. "Filter along the local strike direction" is exactly a directional transform along a vector field. The theorem's practical message: **keep the along-strike filter length below the scale over which strike orientation changes (~1/|∇v|).** Longer filters on a curving orientation field are where instability lives. Use this to set per-pixel adaptive filter lengths from structure-tensor coherence and curvature.
- **Blunt caveat:** the constant is not explicit for engineering purposes. The rule of thumb is a sensible heuristic we could reach without the paper.

### #4. Family 081: Riesz transforms and rectifiability. Relevance: **low**
- **Paper:** *Riesz transforms and uniform rectifiability in higher codimension* (Sept 24, 2026), https://github.com/openai/math/blob/main/preprints/Riesz-transforms-and-uniform-rectifiability-in-higher-codimension-September-24-2026/paper.pdf
- **Mapping:** this links "a singular integral is bounded on a set" to "the set is made of curve- or surface-like pieces," which is conceptually the fault-detection question. Riesz transforms also underlie the monogenic signal (phase/orientation edge features).
- **Blunt caveat:** the theorem covers dimension d ≥ 4 and 2 ≤ n ≤ d−2. The planar curve case we care about was settled decades ago. Monogenic and phase-congruency features are classical and can be tried without this paper.

### #5. Family 107: matrix multiplication exponent ≤ 9/4. Relevance: **none in practice**
- See §1. It is asymptotic and gives no practical algorithm or crossover at our scale. It belongs in conversations, not in the pipeline.

### #6. Family 130: exact Fourier transforms below n log n. Relevance: **none in practice**
- **Paper:** *An explicit power saving for the exact discrete Fourier transform* (Sept 25, 2026), https://github.com/openai/math/blob/main/preprints/An-explicit-power-saving-for-the-exact-discrete-Fourier-transform-September-25-2026/main.pdf
- **Claim:** O(n (log n)^{1−δ}) operations with δ = 10^{−13}. That is a theoretical milestone, but the speedup for any real raster is effectively zero. FFT-based multi-scale filtering stays on cuFFT/numpy.

### #7. Families 365 and 372: Calderón-type inverse problems (conductivity, isotropic elasticity). Relevance: **none for scoring**
- **Papers:** *Smooth Anisotropic Uniqueness in the Calderón Problem from One Boundary Patch* (Sept 24), *Nonuniqueness for Bounded Measurable Scalar Conductivities in Three Dimensions* (Sept 23), *Global Uniqueness for the Smooth Isotropic Elasticity Inverse Problem* (Sept 24). All are in `preprints/`.
- **Mapping:** these are thematically the closest to geophysics, covering when boundary measurements determine conductivity or elastic moduli. Our inputs are already-inverted products such as conductivity and depth-to-source rasters, so the theorems do not change modeling. At most they are a reminder that a non-smooth (rough) conductivity can be *non-unique*. Interpret the conductivity bands as smoothed and ambiguous near sharp contrasts, which is exactly where faults are.

### #8. Family 139: subpolynomial log-concave sampling. Relevance: **none**
- **Paper:** *Subpolynomial query complexity for well-conditioned log-concave sampling* (Sept 26). Uncertainty quantification is in our plan (the MC U-Net baseline and calibration), but this result assumes a well-conditioned log-concave potential (I ⪯ ∇²V ⪯ 2I). Neural-network posteriors are nothing like that.

**Considered and rejected:** Poisson–Voronoi percolation and Cardy's formula (224), Brenier-map stability (374), planar halving lines (183), trace reconstruction (122), and Kakeya/Falconer (073/074). None maps to a measurable change in our pipeline.

---

## What actually moves the score (own suggestions, not from the release)

Nothing in the release addresses optimizing a recall-weighted, distance-tolerant Tversky metric. The practical levers remain:
1. **Train on the metric.** The 300 m triangular kernel is a linear convolution, so a soft distance-weighted Tversky loss using the same kernel and α/β is differentiable. Compare it against BCE+Tversky under all three CV views.
2. **Calibrate for the metric, not for log-loss.** Choose temperature, threshold, and dilation by maximizing CV Tversky. β=0.8 favors spreading mass within about 300 m of likely traces but punishes diffuse background.
3. **Recover continuations.** Endpoint extension and splay completion near known faults (see #2) target exactly the "new pixels" class confirmed by the organizer.
4. **Diversity ensembles** across features, scales, and seeds, accepted through the #1 verify-and-filter gate.

---

## Sources

Primary (GitHub `openai/math`, commit `adc7f12`, 2026-10-06):
- Repository and README: https://github.com/openai/math
- Manuscript map: https://github.com/openai/math/blob/main/CONTENTS.md
- Overview PDF: https://github.com/openai/math/blob/main/overview.pdf
- 9/4 paper: https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/paper.pdf
- Family 107 Lean scope: https://github.com/openai/math/blob/main/lean/docs/107.md
- Formalization catalogue: https://github.com/openai/math/blob/main/lean/formalization.yaml
- Comparator instructions: https://github.com/openai/math/blob/main/lean/ComparatorChallenges/README.md
- Mumford–Shah (366), Hilbert/Lipschitz (083), Riesz (081), DFT (130), Calderón/elasticity (365/372), and log-concave sampling (139): `preprints/` links above

OpenAI announcement (not directly fetchable from our environment, 403):
- https://openai.com/index/sharing-ai-progress-in-mathematics/

Secondary coverage (used only for context):
- New Scientist, "OpenAI announces 722 mathematical discoveries in one go" (2026-10-07): https://www.newscientist.com/article/2592421-openai-announces-722-mathematical-discoveries-in-one-go/
- Unite.AI (2026-10-06; the article states it is AI-generated): https://www.unite.ai/openai-releases-722-math-manuscripts-from-an-unreleased-ai-model/
- The Information briefing: https://www.theinformation.com/briefings/openai-publishes-700-new-math-papers-ai-made-solutions
- Techmeme (Strogatz comment on ω ≤ 2.25): https://www.techmeme.com/261006/p42
- Prior record ω < 2.371177 (Google DeepMind / AlphaEvolve, Aug 2026), cited in the 9/4 paper as Dupont et al.: https://www.linkedin.com/posts/pushmeet-kohli-4838994_improving-the-matrix-multiplication-exponent-activity-7495483046163472385-cw9b
