# Data intake and label-topology gate

The official rasters are the next empirical decision point. Do not move directly from download to model training.

## Intake sequence

1. Verify feature/label geometry with `scripts/verify_inputs.py`.
2. Fingerprint every organizer file into `data/manifests/official.json`.
3. Run `scripts/profile_labels.py` and save `data/manifests/label-profile.json`.
4. Inspect the profile before generating fault-discovery folds.
5. Build the spatial/fault fold artifacts.
6. Render both with `scripts/render_cv.py` and visually inspect them before training.
7. Only then run the organizer baseline.

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

## Fold-image QA

After building fold rasters, render compact PNG previews:

```bash
uv run python scripts/render_cv.py \
  --fold-map data/processed/cv-spatial-v1.tif \
  --labels data/raw/existing_faults.tif \
  --scheme spatial \
  --output data/processed/cv-spatial-v1.png

uv run python scripts/render_cv.py \
  --fold-map data/processed/cv-fault-v1.tif \
  --labels data/raw/existing_faults.tif \
  --valid-template data/raw/gems-geodawn-numerical-features.tif \
  --scheme fault \
  --output data/processed/cv-fault-v1.png
```

The renderer block-reduces fault presence so thin traces are not simply dropped when a large raster is shrunk for review. Fault previews require the valid-study template because `-1` is both the fault-fold background value and GeoTIFF nodata.

Inspect for unexpectedly fragmented traces, giant single-fold networks, obvious fold imbalance, strange study-boundary behavior, and spatial folds that divide a structure in ways likely to inflate or destabilize validation.

## Official topology, 2026-09-23

`data/manifests/label-profile.json` was produced from `existing_faults.tif` and the feature-raster valid mask.

8-connected components are the fault-holdout unit. There are 3,198 of them, 60,894 fault pixels (1.18% of 5,164,312 valid pixels), a median size of 12 pixels (1.2 km), and a largest component of 360 pixels (0.59% of fault pixels). That is a set of separable traces, not one giant network. 4-connected labeling splits the same rasters into 25,058 pieces with a median size of 2 pixels, because the traces are mostly diagonal, so it is not the holdout unit.

Spatial blocks of 256 pixels leave 105 valid blocks, and 99 of those contain at least one fault pixel. Five folds therefore all have supervision. Fold maps are `data/processed/cv-spatial-v1.tif` and `data/processed/cv-fault-v1.tif`; their JSON summaries are committed under `data/manifests/`. Preview PNGs were inspected: spatial folds are mixed across the study area, with white traces crossing block boundaries as this scheme intends, and fault-fold colors are interleaved along short traces rather than painting one region a single color. Fault-fold positive counts are balanced at about 12,180 withheld pixels per fold. Spatial validation area is less even (about 0.90M to 1.24M valid pixels) because blocks are balanced by count, not by valid area inside the irregular study mask. That imbalance is recorded and is not large enough to reject the first artifact.

No organizer vector file was downloaded, so there are no source-feature IDs to group by. A trace-completion holdout is a separate view, not a replacement. `cv-trace-v1` withholds one endpoint of each 8-connected component of at least 24 pixels (30% of that component, along its long axis) and keeps the rest of the trace as training truth. The buffer is 3 pixels, the metric radius, so the withheld endpoint does not leak into training inputs while the trace body stays visible. The five folds each withhold about 2,150 endpoint pixels and still train on about 58,700 fault pixels. The summary is `data/manifests/cv-trace-v1.json`.

## Commit policy

`data/manifests/official.json`, `data/manifests/label-profile.json`, and the CV JSON summaries may be committed because they contain hashes and descriptive metadata rather than the gated rasters themselves. Fold GeoTIFFs and preview PNGs stay under gitignored `data/processed/`: the previews depict the label traces. Confirm that any additional derived artifact does not reconstruct or redistribute restricted competition data before committing it.
