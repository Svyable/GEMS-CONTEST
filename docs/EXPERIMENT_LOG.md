# Experiment log

Machine-readable run manifests under `runs/<run-id>/manifest.json` are the source of truth for reproducibility. This Markdown file is a concise human index for meaningful experiments and every leaderboard submission.

A recorded run should pin:

- Git commit and whether the checkout was dirty;
- config path + SHA256;
- official data-manifest path + SHA256;
- exact command;
- output artifact paths + SHA256;
- hypothesis and interpretation.

Create a record with `scripts/record_run.py`. Never overwrite a submitted run's config, metrics, prediction, or manifest.

| ID | Date | Commit | Data fingerprint | Config | Spatial CV | Fault-discovery CV | Public LB | Submission SHA256 | Notes |
|---|---|---|---|---|---:|---:|---:|---|---|
| bootstrap | 2026-09-22 | initial setup | none | configs/baseline.yaml | — | — | — | — | Repo initialized; no competition data downloaded yet. |
| cv-v1 | 2026-09-23 | after official fingerprint | data/manifests/official.json | spatial block 256 / buffer 16; fault 8-connected | artifacts only | artifacts only | — | — | Topology supports 8-connected component holdout. Manifests: data/manifests/cv-spatial-v1.json and cv-fault-v1.json. No model scored yet. |
| cv-trace-v1 | 2026-09-23 | after cv-v1 | data/manifests/official.json | endpoint 30% of components ≥ 24 px, buffer 3 | — | continuation view | — | — | About 2,150 withheld endpoint pixels per fold. Manifest: data/manifests/cv-trace-v1.json. |
| reference-oof | 2026-09-23 | 9e256b9 | data/manifests/official.json | configs/reference_unet.yaml | — | — | — | — | Organizer 5×5-epoch ResNet-18 OOF on MPS, Apple M4 Pro, 438 s. Full-valid distance-weighted Tversky 0.0877. Mean probability 0.243 on fault pixels vs 0.123 on background. Loss was still falling at epoch 4. Not a submission. |
| full-map-v1 | 2026-09-23 | 690b159 | data/manifests/official.json | configs/full_map_v1.yaml | — | — | not submitted | local submissions/full-map-v1.tif | 20-epoch train-on-all U-Net, 5,035 fault windows and 640 background windows, MPS, 416 s. In-sample distance-weighted Tversky 0.2998 against known faults. Mean probability 0.445 on fault pixels vs 0.051 on background. Template validator passed. SHA256 cd34abbd0b8b44feb371e34effc26dc29a73f5c3776dc69995d8beeb6f1c181b. This is fit to the training labels, not a hidden-fault score. Not uploaded. |
| encoder-fold0 | 2026-09-23 | ca0d8d7 | data/manifests/official.json | resnet18 / convnextv2_tiny / sam2_hiera_small fold0 | 0.1427 / 0.1211 / 0.1281 | — | not submitted | — | Same spatial fold 0, 2,355 windows, 10 epochs, Apple MPS. Holdout distance-weighted Tversky: ResNet-18 0.1427, SAM2 Hiera-small 0.1281, ConvNeXt V2-tiny 0.1211. A 224 px Hiera run had only 124 legal windows and scored 0.070; it is not comparable. ResNet-18 leads this budget. Not uploaded. |
| encoder-fold0-e20 | 2026-09-23 | faae632 | data/manifests/official.json | resnet18_fold0.yaml and sam2_hiera_small_fold0.yaml | 0.1478 / 0.0990 | — | not submitted | — | 20 epochs, same fold and windows. ResNet-18 holdout 0.1478. SAM2 Hiera-small holdout 0.0990, down from 0.1281 at 10 epochs, so the extra epochs overfit the training blocks. ResNet-18 remains the best single model on this holdout. Not uploaded. |

## Scoring conventions

For **spatial CV**, `scripts/score_cv.py` reports each fold plus a stitched global OOF score because spatial validation regions are disjoint.

For **fault-discovery CV**, evaluation backgrounds overlap across folds, so the scorer intentionally reports per-fold scores plus macro mean/std and does **not** manufacture a global score. The held-out fault component itself is the positive truth; training-known faults are excluded from that fold's evaluation mask.

## Submission notes template

**Hypothesis:**  
**Change vs previous:**  
**Run manifest:**  
**Local evidence:**  
**Artifact path + SHA256:**  
**Leaderboard score:**  
**Decision:**  
