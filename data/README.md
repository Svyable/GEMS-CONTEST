# Local data layout

Competition data is intentionally not stored in Git.

After joining the DrivenData competition and accepting the terms, put the organizer-provided files under `data/raw/`. Keep external sources under `data/external/<source>/` with a short provenance/license note.

Suggested layout:

```text
data/
  raw/
    training_features.tif
    <organizer fault-label raster>
    <organizer fault-label vector file(s)>
    sample_submission.tif
    1m_DEM_links.csv
  external/
    usgs-geodawn/
    usgs-3dep/
    ingenious/
  interim/
  processed/
```

Do not rename organizer files unless a script/config records the mapping. Do not commit downloaded competition data unless the competition's data terms explicitly permit redistribution.
