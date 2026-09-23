# Organizer reference U-Net reproduction

_Last checked against the public organizer repository: 2026-09-22._

The organizer's notebook is a useful floor, but it is not our validation design.

## Exact public notebook recipe

The notebook:

- loads `numeric_features.tif` and `labels.tif`;
- replaces feature values below -1e38 with NaN;
- min/max normalizes each feature channel over the full raster;
- uses all feature channels;
- makes 128×128 patches;
- selects random non-overlapping positive patches for test data, with `test_proportion=0.5`;
- removes those test patches before generating overlapping training patches with stride 32;
- trains five Monte Carlo splits with seeds 0, 10, 20, 30, and 40;
- uses `segmentation_models_pytorch.Unet` with a ResNet-18 ImageNet encoder;
- uses AdamW at 1e-4 and binary Tversky loss (`alpha=0.2`, `beta=0.8`);
- trains five epochs per split;
- applies random resized crop, horizontal/vertical flips, and ±30° rotation;
- selects the lowest test-loss model in each split and adds each held-out prediction divided by five to the output raster.

The reproducible parameter transcription is in `configs/reference_unet.yaml`.

## Our executable reproduction

`scripts/train_reference_oof.py` turns that notebook into a CLI. The patch construction and organizer padding behavior are isolated in `src/gems/reference_baseline.py` and covered by unit tests.

Run:

```bash
uv sync --extra ml --extra cpu --extra dev --frozen
uv run python scripts/train_reference_oof.py \
  --features data/raw/training_features.tif \
  --labels data/raw/labels.tif \
  --output outputs/reference-oof.tif \
  --metrics-json runs/reference-oof.json
```

Use `--extra cu126` instead of `--extra cpu` on a compatible NVIDIA system.

We make three deliberate engineering fixes around the notebook while preserving its modeling recipe:

1. model/augmentation RNGs are seeded per Monte Carlo split so a run can be reproduced;
2. the output GeoTIFF is written as `float32`;
3. pixels outside the label raster's valid-data mask are written as NaN.

The notebook also pads by a full extra patch when an image dimension is exactly divisible by 128. Our reproduction keeps that quirk because changing it would stop being a faithful baseline comparison.

## Important filename discrepancy

The current DrivenData problem-description page says the provided feature file is `training_features.tif`, while the public reference notebook currently hard-codes `numeric_features.tif`. Do not silently rename data. After downloading, record the actual organizer filenames in the data manifest and point commands/configs at them.

## Why this output is not a competition candidate

The reference notebook predicts only randomly held-out positive tiles across its Monte Carlo splits. It is therefore an out-of-fold diagnostic of the known-label task, not a train-on-all model designed to infer unknown faults over the complete study region.

That distinction matters: we will reproduce this result to establish the software/data baseline, then train candidate models under validation that better approximates unknown-fault discovery.

## Why we will not use its CV for model selection

The reference notebook intentionally demonstrates a simple benchmark. Its random patch holdout can place nearby portions of the same connected geological structure in training and validation, and its full-raster normalization uses validation-region feature statistics. Both are acceptable for reproducing the baseline; neither is a strong estimate of unknown-fault generalization.

Our comparison sequence is therefore:

1. reproduce the organizer notebook faithfully;
2. score its OOF raster using the published distance-weighted Tversky implementation;
3. rerun the same architecture under buffered spatial-block CV;
4. rerun it under complete connected-fault holdout;
5. only then change architecture/features/losses.

This isolates gains from better validation, better modeling, and better data rather than mixing them together.
