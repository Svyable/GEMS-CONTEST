# GEMS working brief

_Last checked: 2026-09-23._

## Objective

Predict geologic fault presence across the GeoDAWN study area. The supplied features are a multiband 100 m GeoTIFF covering geophysical and related measurements, while the training labels are existing USGS/INGENIOUS fault data. The hidden test labels are newly identified fault pixels not captured by those existing labels.

Organizer clarification on 2026-09-23: “new fault” is pixel-based for competition purposes. It can include newly mapped geometry of an existing fault system, such as an unmapped continuation, splay, or parallel strand; it does not need to be a geologically separate structure.

## Evaluation

The metric is a distance-weighted Tversky index with:

- `alpha = 0.2` for false-positive penalty;
- `beta = 0.8` for false-negative penalty;
- a triangular spatial kernel with 300 m support (3 pixels at 100 m resolution).

False negatives therefore matter substantially more than false positives, but diffuse low-confidence probability everywhere is still penalized. The key modeling problem is calibrated **recall of plausible fault geometry** without flooding large areas with probability mass.

Organizer forum clarification (2026-09-16): known USGS/INGENIOUS fault pixels are masked/excluded from evaluation in both the initial and final rounds. The scoring mask means predicting known pixels does not hurt, but proximity to a known trace is not itself excluded: newly mapped continuations/branches can still be new-fault labels.

## Provided features

The official problem description currently lists the 100 m `training_features.tif` stack as including:

- surface conductivity and conductive-base depth;
- detrended elevation and detrended-elevation slope;
- dilatation rate, shear strain rate, and second invariant of strain rate;
- isostatic gravity anomaly and its slope;
- magnetic anomaly/intensity derivatives and top-of-crustal magnetic source-depth estimate;
- earthquake density.

The competition also provides `1m_DEM_links.csv` for high-resolution DEM retrieval.

## Submission contract

One GeoTIFF:

- EPSG:32611 (UTM zone 11N)
- 100 m resolution
- same bounds/shape/transform as training/template raster
- one band
- `float32`
- confidence/probability in `[0, 1]`
- null/NaN outside valid bounds as required by the template

Run `scripts/validate_submission.py` against the official sample submission before uploading.

## Competition mechanics

- Deadline: **2026-12-03 23:59 UTC**.
- Prize pool: $300,000.
- Initial round: top five private-test submissions receive $10,000 each.
- Final round: same selected final submission is rescored after expert review expands the new-fault label set; prizes are $100k / $70k / $40k / $25k / $15k.
- Up to three scored submissions are allowed in a **rolling** window; the organizer clarified there is no fixed weekly reset date/time.
- Only one final submission is selected per participating entity/team for both prize rounds.
- Finalists must provide complete code assets and documentation sufficient to reproduce results and run predictions on new data.

## What this means for us

1. Treat public leaderboard score as noisy evidence, not the optimization target.
2. Use spatial holdouts, whole-fault holdouts, and trace-completion/endpoint withholding rather than random pixels.
3. Do not automatically downweight predictions merely because they are close to known faults.
4. Preserve reproducibility from the first experiment because finalist verification requires it.
5. Favor candidate geometry that a geologist could plausibly verify in the final expert-review phase.
6. Keep an external-data license ledger and an AI-use disclosure as we work.
