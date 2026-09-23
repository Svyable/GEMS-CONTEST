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
