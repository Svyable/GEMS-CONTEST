#!/bin/bash
# usage: train_folds.sh RUN_DIR CONFIG SCHEME FOLD_MAP BUFFER_PIXELS
# Resumable: folds whose fold-k.json exists are skipped.
set -o pipefail
WORKSPACE="${GEMS_WORKSPACE:-/workspace/gems/GEMS-CONTEST}"
cd "$WORKSPACE"
RUN=$1; CFG=$2; SCHEME=$3; FMAP=$4; BUF=$5
mkdir -p $RUN
for k in 0 1 2 3 4; do
  if [ -f $RUN/fold-$k.json ] && [ -f $RUN/fold-$k.tif ]; then echo "skip fold $k"; continue; fi
  s=$(date +%s); echo "$(date +%T) START $RUN fold $k"
  uv run python -u scripts/train_full_map.py \
    --features data/raw/gems-geodawn-numerical-features.tif \
    --labels data/raw/existing_faults.tif --template data/raw/example_submission.tif \
    --config $CFG --fold-map $FMAP --cv-scheme $SCHEME --fold $k --buffer-pixels $BUF \
    --output $RUN/fold-$k.tif --metrics-json $RUN/fold-$k.json.tmp 2>&1 \
    | while IFS= read -r l; do echo "$(date +%T) [f$k] $l"; done
  rc=${PIPESTATUS[0]}
  echo "$(date +%T) END fold $k exit=$rc elapsed_s=$(( $(date +%s)-s ))"
  [ $rc -eq 0 ] || exit $rc
  mv $RUN/fold-$k.json.tmp $RUN/fold-$k.json
done
