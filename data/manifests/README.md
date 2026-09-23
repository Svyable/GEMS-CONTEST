# Data manifests

Generated manifests belong here and may be committed because they contain hashes and metadata, not the underlying gated data.

After downloading the official files, run for example:

```bash
uv run python scripts/fingerprint_data.py \
  data/raw/training_features.tif \
  data/raw/labels.tif \
  data/raw/sample_submission.tif \
  data/raw/1m_DEM_links.csv \
  --root data/raw \
  --output data/manifests/official.json
```

Use the actual filenames from the download page. The public competition page currently calls the feature raster `training_features.tif`; the organizer reference notebook currently uses `numeric_features.tif`, so the manifest is the source of truth for our local copy.
