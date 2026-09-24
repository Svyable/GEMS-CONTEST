# Data manifests

Generated manifests belong here and may be committed because they contain hashes and metadata, not the underlying gated data.

After downloading the official files, run for example:

```bash
uv run python scripts/fingerprint_data.py \
  data/raw/gems-geodawn-numerical-features.tif \
  data/raw/existing_faults.tif \
  data/raw/example_submission.tif \
  data/raw/tnm_items.json \
  --root data/raw \
  --output data/manifests/official.json
```

`data/manifests/official.json` records the 2026-09-23 DrivenData download. Public pages still say `training_features.tif` / `1m_DEM_links.csv`, and the reference notebook still says `numeric_features.tif`. The manifest filenames are the source of truth.
