# Data intake and label-topology gate

The official rasters are the next empirical decision point. Do not move directly from download to model training.

## Intake sequence

1. Verify feature/label geometry with `scripts/verify_inputs.py`.
2. Fingerprint every organizer file into `data/manifests/official.json`.
3. Run `scripts/profile_labels.py` and save `data/manifests/label-profile.json`.
4. Inspect the profile before generating fault-discovery folds.
5. Only then build spatial/fault fold artifacts and run the organizer baseline.

## What the topology profile answers

The fault-discovery validation currently has a simple default: hold out complete 8-connected positive components. That is only useful if the label raster's connected components correspond reasonably well to separable geological structures.

The profile therefore reports both 4-connected and 8-connected topology, including:

- number of connected components;
- total positive/fault pixels and prevalence;
- component-size quantiles;
- largest component size and share of all positive pixels;
- bounding-box dimensions of the largest components;
- fraction of candidate spatial blocks that contain any labeled fault;
- positive-pixel distribution across those blocks.

## What not to infer automatically

There is no universal threshold such as “largest component share > X means failure.” A very large component could represent a genuinely continuous mapped system, an artifact of rasterized intersections, or many traces joined at crossings. The report is a diagnostic, not a geology classifier.

If the real labels collapse into one or a few giant connected networks, the next step is to inspect those networks and define a more meaningful grouping, such as separated trace segments, buffered graph branches, or source-vector feature IDs if the organizer data includes them. We should make that change from evidence in the official labels, not from a synthetic assumption.

## Commit policy

`data/manifests/official.json` and `data/manifests/label-profile.json` may be committed because they contain hashes and descriptive metadata rather than the gated rasters themselves. Confirm that any additional derived artifact does not reconstruct or redistribute restricted competition data before committing it.
