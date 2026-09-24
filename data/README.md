# Local data layout

Competition data is intentionally not stored in Git.

After joining the DrivenData competition and accepting the terms, put the organizer-provided files under `data/raw/`. Keep external sources under `data/external/<source>/` with a short provenance/license note.

Official download, placed under `data/raw/` on 2026-09-23. Filenames are the ones DrivenData served; they are not renamed.

| DrivenData dataset | Local file | What it is |
|---|---|---|
| Submission format | `example_submission.tif` | Template GeoTIFF of known INGENIOUS quaternary faults. Submitting this file scores zero because known faults are excluded from evaluation. |
| Numerical features | `gems-geodawn-numerical-features.tif` | Multiband numerical feature raster. |
| Training fault labels | `existing_faults.tif` | GDR / INGENIOUS quaternary faults v2. |
| Digital elevation model links JSON | `tnm_items.json` | 894 URLs for 1 m DEM tiles from the USGS National Map (TNM). |

No separate fault-label vector file was in this download. The public problem page calls the feature file `training_features.tif`, and the organizer reference notebook hard-codes `numeric_features.tif`. The downloaded name `gems-geodawn-numerical-features.tif` is the one commands use. The DEM manifest is `tnm_items.json`, not `1m_DEM_links.csv`.

`scripts/verify_inputs.py` and `scripts/inspect_raster.py` on 2026-09-23: all three rasters match. They are 3292×3730, EPSG:32611, 100 m, with bounds (243350, 4135550)–(572550, 4508550). Features are 19 float32 bands with nodata about -3.4e38. Labels are one int8 band with nodata -1. The submission template is one float32 band with NaN nodata. `tnm_items.json` has 894 unique URLs. Hashes are in `data/manifests/official.json`.

```text
data/
  raw/          gitignored organizer files listed above
  manifests/
    official.json
  external/
    usgs-geodawn/
    usgs-3dep/
    ingenious/
  interim/
  processed/
```

Before modeling:

```bash
uv run python scripts/verify_inputs.py \
  --features data/raw/gems-geodawn-numerical-features.tif \
  --labels data/raw/existing_faults.tif
uv run python scripts/fingerprint_data.py \
  data/raw/gems-geodawn-numerical-features.tif \
  data/raw/existing_faults.tif \
  data/raw/example_submission.tif \
  data/raw/tnm_items.json \
  --root data/raw --output data/manifests/official.json

uv run python scripts/profile_labels.py \
  --labels data/raw/existing_faults.tif \
  --features data/raw/gems-geodawn-numerical-features.tif \
  --output-json data/manifests/label-profile.json
```

Do not rename organizer files unless a config records the mapping. Do not commit downloaded competition data unless the competition's data terms explicitly permit redistribution.
