# GEMS-CONTEST

Competition workspace for the **DOE / NLR Geologic Enhanced Mapping System (GEMS) Prize Challenge** on DrivenData.

The goal is to identify previously unmapped geologic faults in the GeoDAWN study area from geophysical and topographic data. The competition closes **2026-12-03 23:59 UTC** and uses a **distance-weighted Tversky index** with a 300 m tolerance (`alpha=0.2`, `beta=0.8`).

## Start here

1. Join the competition and accept the rules on DrivenData.
2. Download the competition files into `data/raw/` (competition data is intentionally gitignored). Use the exact downloaded filenames; the public problem page currently says `training_features.tif`, while the organizer reference notebook currently uses `numeric_features.tif`.
3. Create an environment:

   ```bash
   uv sync --extra ml --extra cpu --extra dev
   # or, on a compatible NVIDIA system:
   uv sync --extra ml --extra cu126 --extra dev
   ```

4. Verify and fingerprint the downloaded inputs:

   ```bash
   uv run python scripts/verify_inputs.py \
     --features data/raw/training_features.tif \
     --labels data/raw/labels.tif

   uv run python scripts/fingerprint_data.py \
     data/raw/training_features.tif \
     data/raw/labels.tif \
     data/raw/sample_submission.tif \
     data/raw/1m_DEM_links.csv \
     --root data/raw \
     --output data/manifests/official.json
   ```

   Adjust names to match the actual download. Commit the manifest, not the raw data.

5. Build immutable validation fold artifacts:

   ```bash
   uv run python scripts/build_cv_manifest.py \
     --features data/raw/training_features.tif \
     --labels data/raw/labels.tif \
     --scheme spatial \
     --output-prefix data/processed/cv-spatial-v1

   uv run python scripts/build_cv_manifest.py \
     --features data/raw/training_features.tif \
     --labels data/raw/labels.tif \
     --scheme fault \
     --output-prefix data/processed/cv-fault-v1
   ```

6. Reproduce the organizer's Monte Carlo U-Net as an out-of-fold diagnostic:

   ```bash
   uv run python scripts/train_reference_oof.py \
     --features data/raw/training_features.tif \
     --labels data/raw/labels.tif \
     --output outputs/reference-oof.tif \
     --metrics-json runs/reference-oof.json
   ```

   This is a reference **OOF benchmark**, not a competition submission model.

7. Validate every submission candidate against the organizer template:

   ```bash
   uv run python scripts/validate_submission.py \
     --submission submissions/candidate.tif \
     --template data/raw/sample_submission.tif
   ```

8. Read [`docs/REFERENCE_BASELINE.md`](docs/REFERENCE_BASELINE.md), [`docs/CANDIDATE_PIPELINE.md`](docs/CANDIDATE_PIPELINE.md), and [`docs/STRATEGY.md`](docs/STRATEGY.md) before modeling. Log every experiment in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

## Repository layout

```text
configs/                 experiment configuration
data/manifests/          commit-safe hashes/metadata for gated data
data/raw|external/       local-only competition/external data (ignored)
docs/                    rules, strategy, baseline notes, experiment log, AI disclosure
scripts/                 inspection, fingerprinting, CV, scoring and submission checks
src/gems/                metric, data and validation helpers
tests/                   unit tests
.github/workflows/       CI
```

## Non-negotiable submission checks

A valid submission must be a single-band `float32` GeoTIFF in **EPSG:32611**, at **100 m** resolution, with the same shape/bounds/transform as the organizer template and finite prediction values in `[0, 1]` inside the valid study region. Use the template-based validator before every upload.

The organizer has clarified that pixels corresponding to the existing USGS/INGENIOUS faults are masked out of evaluation, including the final re-scoring round. That makes this an **unknown-fault discovery** problem, not merely a trace-reproduction problem.

## Competition hygiene

- Do not commit competition data, downloaded DEM tiles, model checkpoints, or generated GeoTIFF submissions.
- Track provenance and license terms for every external dataset before using it.
- Keep local validation spatial and fault-aware: random pixel/patch splits are leakage-prone for long connected traces.
- Reproduce the organizer U-Net first, but use buffered spatial CV and complete-fault holdouts for model selection.
- The rules allow generative AI in solution development, but require disclosure. Keep [`docs/AI_DISCLOSURE.md`](docs/AI_DISCLOSURE.md) current.
- Only one final submission is selected for both prize rounds; leaderboard feedback is limited, so optimize for generalization rather than public-LB chasing.

See [`docs/COMPETITION.md`](docs/COMPETITION.md) and [`docs/RULES_CHECKLIST.md`](docs/RULES_CHECKLIST.md) for the working brief.
