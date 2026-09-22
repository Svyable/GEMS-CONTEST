# GEMS-CONTEST

Competition workspace for the **DOE / NLR Geologic Enhanced Mapping System (GEMS) Prize Challenge** on DrivenData.

The goal is to identify previously unmapped geologic faults in the GeoDAWN study area from geophysical and topographic data. The competition closes **2026-12-03 23:59 UTC** and uses a **distance-weighted Tversky index** with a 300 m tolerance (`alpha=0.2`, `beta=0.8`).

## Start here

1. Join the competition and accept the rules on DrivenData.
2. Download the competition files into `data/raw/` (competition data is intentionally gitignored).
3. Create an environment:

   ```bash
   uv sync --extra cpu --extra dev
   # or, on a compatible NVIDIA system:
   uv sync --extra cu126 --extra dev
   ```

4. Validate your local files:

   ```bash
   uv run python scripts/inspect_raster.py data/raw/training_features.tif
   uv run python scripts/validate_submission.py \
     --submission data/raw/sample_submission.tif \
     --template data/raw/sample_submission.tif
   ```

5. Read [`docs/STRATEGY.md`](docs/STRATEGY.md) before modeling and log every experiment in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).
6. Benchmark against the organizer reference solution linked in [`docs/SOURCES.md`](docs/SOURCES.md).

## Repository layout

```text
configs/                 experiment configuration
data/                    local-only competition/external data (ignored)
docs/                    rules, strategy, sources, experiment log, AI disclosure
scripts/                 CLI utilities for inspection, scoring and submission checks
src/gems/                reusable metric and geospatial helpers
tests/                   unit tests
.github/workflows/       CI
```

## Non-negotiable submission checks

A valid submission must be a single-band `float32` GeoTIFF in **EPSG:32611**, at **100 m** resolution, with the same shape/bounds/transform as the organizer template and finite prediction values in `[0, 1]` inside the valid study region. Use the template-based validator before every upload.

The organizer has clarified that pixels corresponding to the existing USGS/INGENIOUS faults are masked out of evaluation, including the final re-scoring round. That makes this an **unknown-fault discovery** problem, not merely a trace-reproduction problem.

## Competition hygiene

- Do not commit competition data, downloaded DEM tiles, model checkpoints, or generated GeoTIFF submissions.
- Track provenance and license terms for every external dataset before using it.
- Keep local validation spatial: random pixel splits are leakage-prone for long connected fault traces.
- The rules allow generative AI in solution development, but require disclosure. Keep [`docs/AI_DISCLOSURE.md`](docs/AI_DISCLOSURE.md) current.
- Only one final submission is selected for both prize rounds; leaderboard feedback is limited, so optimize for generalization rather than public-LB chasing.

See [`docs/COMPETITION.md`](docs/COMPETITION.md) and [`docs/RULES_CHECKLIST.md`](docs/RULES_CHECKLIST.md) for the working brief.
