# GEMS working brief

_Last checked: 2026-09-22._

## Objective

Predict geologic fault presence across the GeoDAWN study area. The supplied features are a multiband 100 m GeoTIFF covering geophysical and related measurements, while the training labels are existing fault data. The hidden test labels are newly identified faults not contained in the existing public fault database.

## Evaluation

The metric is a distance-weighted Tversky index with:

- `alpha = 0.2` for false-positive penalty;
- `beta = 0.8` for false-negative penalty;
- a triangular spatial kernel with 300 m support (3 pixels at 100 m resolution).

False negatives therefore matter substantially more than false positives, but diffuse low-confidence probability everywhere is still penalized. The key modeling problem is calibrated **recall of plausible lineaments** without flooding large areas with probability mass.

Organizer forum clarification (2026-09-16): known USGS/INGENIOUS fault pixels are masked/excluded from evaluation in both the initial and final rounds. The final-round label set remains focused on newly identified/verified faults.

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
- The official rules state up to three scored submissions per week and only one selected final submission per participating entity/team for both prize rounds.
- Finalists must provide complete code assets and documentation sufficient to reproduce results and run predictions on new data.

## What this means for us

1. Treat public leaderboard score as noisy evidence, not the optimization target.
2. Use spatial holdouts and fault-level/region-level withholding rather than random pixels.
3. Preserve reproducibility from the first experiment because finalist verification requires it.
4. Favor candidate faults that a geologist could plausibly verify in the final expert-review phase.
5. Keep an external-data license ledger and an AI-use disclosure as we work.
