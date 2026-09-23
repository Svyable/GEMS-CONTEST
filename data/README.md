# Local data layout

Competition data is intentionally not stored in Git.

After joining the DrivenData competition and accepting the terms, put the organizer-provided files under `data/raw/`. Keep external sources under `data/external/<source>/` with a short provenance/license note.

Suggested layout:

```text
data/
  raw/
    training_features.tif
    labels.tif
    <organizer fault-label vector file(s)>
    sample_submission.tif
    1m_DEM_links.csv
  manifests/
    official.json
  external/
    usgs-geodawn/
    usgs-3dep/
    ingenious/
  interim/
  processed/
```

The public problem-description page currently calls the feature raster `training_features.tif`; the organizer's public reference notebook currently hard-codes `numeric_features.tif`. Use the actual filename you receive and record it in `data/manifests/official.json`.

Before modeling:

```bash
uv run python scripts/verify_inputs.py --features <features.tif> --labels <labels.tif>
uv run python scripts/fingerprint_data.py <downloaded files...> \
  --root data/raw --output data/manifests/official.json

uv run python scripts/profile_labels.py \
  --labels <labels.tif> \
  --features <features.tif> \
  --output-json data/manifests/label-profile.json
```

Do not rename organizer files unless a config records the mapping. Do not commit downloaded competition data unless the competition's data terms explicitly permit redistribution.
