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
  --features data/raw/gems-geodawn-numerical-features.tif \
  --labels data/raw/existing_faults.tif \
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

The DrivenData problem page says `training_features.tif` and the public reference notebook hard-codes `numeric_features.tif`. The 2026-09-23 download is `gems-geodawn-numerical-features.tif`, with labels in `existing_faults.tif`. Those downloaded names are kept as-is and recorded in `data/manifests/official.json` and `configs/reference_unet.yaml`.

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

## Full-config spatial CV results (2026-10-09)

**Status**: Complete. This is the incumbent baseline for spatial view.

**Config**: `configs/reference_unet.yaml` (unreduced: 5 epochs, train_step 32)  
**Training**: Honest spatial CV with 16-pixel buffer, fold-pure normalization, negative_ratio 1.0, seed 20260922  
**Raw spatial scores**: 0.1031, 0.1015, 0.0854, 0.1078, 0.0836 (mean 0.0963 ± 0.0099)  
**LOFO-calibrated**: 0.1308, 0.1334, 0.1014, 0.1442, 0.1050 (mean 0.1230 ± 0.0168)

Results and provenance: `results/runs/ref-cv-spatial-full/`

## Spatial CV results at buffer 32 (2026-10-11)

**Status**: Complete. Buffer-32 spatial reference baseline (`run_id` `ref-cv-spatial-b32`). Not a candidate and not a replacement for the buffer-16 spatial incumbent.

**Config**: `configs/reference_unet.yaml` (unreduced: 5 epochs, train_step 32)  
**Training**: Honest spatial CV with 32-pixel buffer, fold-pure normalization, negative_ratio 1.0, seed 20260922  
**Raw spatial scores**: macro mean **0.0988 ± 0.0157**, stitched aggregate **0.0967**  
**LOFO-calibrated**: macro mean **0.1245 ± 0.0151**

Comparable to ref spatial b16 (raw 0.0963 / LOFO 0.1230 / agg 0.0967): essentially equal. This baseline is required before gating `structure_tensor_coherence` (min buffer 17) and `ridge_valley_response` (min buffer 30). Fault-b32 is now scored (`ref-cv-fault-b32`); trace-b32 remains in the queue.

Results and provenance: `results/runs/ref-cv-spatial-b32/`

## Full-config fault CV results (2026-10-10)

**Status**: Complete. This is the incumbent baseline for fault view.

**Config**: `configs/reference_unet.yaml` (unreduced: 5 epochs, train_step 32)  
**Training**: `train_full_map.py --cv-scheme fault --buffer-pixels 16`, fold-pure normalization, negative_ratio 1.0, seed 20260922. CPU training ~17.8 h.  
**Raw fault scores**: 0.0374, 0.0449, 0.0505, 0.0464, 0.0484 (mean 0.0455 ± 0.0045)  
**LOFO-calibrated**: 0.0452, 0.0583, 0.0642, 0.0579, 0.0478 (mean 0.0547 ± 0.0071)  
**Thresholds**: 0.935, 0.882, 0.952, 0.952, 0.997

**Comparison to spatial**: Fault CV raw mean is 0.0455 (vs. spatial 0.0963); LOFO-calibrated 0.0547 (vs. spatial 0.1230). The large drop indicates that generalizing to held-out complete faults is substantially harder than spatial block holdout.

**Note**: Manifest records `git_dirty=true` at commit b82efe9 because of local patches (endpoint_extension inplace, train_full_map lineament param pass-through) that have since been superseded on main.

Results and provenance: `results/runs/ref-cv-fault-full/`

## Fault CV results at buffer 32 (2026-10-12)

**Status**: Complete. Buffer-32 fault-split reference baseline (`run_id` `ref-cv-fault-b32`). Not a candidate and not a replacement for the buffer-16 fault incumbent.

**Config**: `configs/reference_unet.yaml` (unreduced: 5 epochs, train_step 32)  
**Training**: Honest fault CV with 32-pixel buffer, fold-pure normalization, negative_ratio 1.0, seed 20260922  
**Raw fault scores**: macro mean **0.0366 ± 0.0076**  
**LOFO-calibrated**: macro mean **0.0450 ± 0.0105**

Lower than ref fault b16 (raw 0.0455 ± 0.0045 / LOFO 0.0547 ± 0.0071) by about 0.009–0.010. Buffer-32 candidates must be compared only against this b32 baseline, never against b16. Required before gating `structure_tensor_coherence` (min buffer 17) and `ridge_valley_response` (min buffer 30). Trace-b32 reference is next in the queue.

Results and provenance: `results/runs/ref-cv-fault-b32/`

## Full-config trace CV results (2026-10-10) [SUPERSEDED]

**Status**: SUPERSEDED on 2026-10-10 by ref-cv-trace-b16. This run used buffer 3 (below every lineament kind's minimum buffer requirement) and mixed fold maps (fold 0 on pre-repair cv-trace-v1, folds 1-4 on cv-trace-v1b). The raw score 0.009317 ± 0.001755 and LOFO-calibrated score 0.016198 ± 0.006655 are on a mixed protocol and must not be used for gating or incumbent comparison.

**Config**: `configs/reference_unet.yaml` (unreduced: 5 epochs, train_step 32)  
**Training**: `train_full_map.py --cv-scheme trace --buffer-pixels 3`, fold-pure normalization, negative_ratio 1.0, seed 20260922  
**Raw trace scores**: macro mean **0.009317 ± 0.001755** (MIXED PROTOCOL, NOT VALID FOR GATING)  
**LOFO-calibrated**: mean **0.016198 ± 0.006655** (MIXED PROTOCOL, NOT VALID FOR GATING)

**Note**: Fold 0 trained on cv-trace-v1 (pre-repair), folds 1-4 on cv-trace-v1b (post-repair). Buffer 3 is below the minimum buffer requirement for all lineament kinds (mumford_shah_log needs 5, steerable_filter needs 7, structure_tensor_coherence needs 17, ridge_valley_response needs 30). The replacement ref-cv-trace-b16 (all 5 folds on v1b, buffer 16) is queued and has no score yet.

Results and provenance: `results/runs/ref-cv-trace-full/` (preserved with SUPERSEDED note)
