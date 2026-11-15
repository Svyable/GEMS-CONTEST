#!/bin/bash
# usage: train_folds.sh RUN_DIR CONFIG SCHEME FOLD_MAP BUFFER_PIXELS [SEED]
# Resumable: folds whose fold-k.json exists are skipped.
# SEED defaults to 20260922 (the incumbent seed); a seed-robustness probe
# passes a second seed explicitly.
set -o pipefail
WORKSPACE="${GEMS_WORKSPACE:-/workspace/gems/GEMS-CONTEST}"
cd "$WORKSPACE"
RUN=$1; CFG=$2; SCHEME=$3; FMAP=$4; BUF=$5; SEED=${6:-20260922}
mkdir -p $RUN
for k in 0 1 2 3 4; do
  if [ -f $RUN/fold-$k.json ] && [ -f $RUN/fold-$k.tif ]; then echo "skip fold $k"; continue; fi
  attempt=0
  while :; do
  # Memory gate (added 2026-10-10 21:58 CT after two rc=137 OOM kills caused by other agents' jobs on the shared box):
  # wait until MemAvailable >= GEMS_MIN_AVAIL_MB before starting a fold.
  # Bounded (step 41): the wait used to be unbounded, so a memory-hogging
  # job elsewhere on the shared box stalled the queue silently and
  # indefinitely — the doctor could only diagnose RUNNING-WAIT-MEM. After
  # GEMS_WAIT_MEM_TIMEOUT_S (default 21600s = 6h; a transient hog under 6h
  # still waits out) the fold gives up with exit 3, so the runner records
  # a .failed marker and stops for inspection instead of stalling forever.
  # GEMS_MEMINFO / GEMS_WAIT_MEM_SLEEP_S are test hooks (defaults
  # /proc/meminfo and 60s); box behavior is unchanged except for the bound.
  MIN_MB=${GEMS_MIN_AVAIL_MB:-5000}
  MEMINFO=${GEMS_MEMINFO:-/proc/meminfo}
  WAIT_TIMEOUT_S=${GEMS_WAIT_MEM_TIMEOUT_S:-21600}
  WAIT_SLEEP_S=${GEMS_WAIT_MEM_SLEEP_S:-60}
  waited=0
  while :; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' "$MEMINFO")
    avail=${avail:-0}
    [ "$avail" -ge "$MIN_MB" ] && break
    if [ "$waited" -ge "$WAIT_TIMEOUT_S" ]; then
      echo "$(date +%T) WAIT-MEM-TIMEOUT fold $k: MemAvailable=${avail}MB < ${MIN_MB}MB after ${waited}s (limit ${WAIT_TIMEOUT_S}s); giving up so the runner records a failure instead of stalling forever"
      exit 3
    fi
    [ $waited -eq 0 ] && echo "$(date +%T) WAIT-MEM fold $k: MemAvailable=${avail}MB < ${MIN_MB}MB"
    waited=$((waited+WAIT_SLEEP_S)); sleep $WAIT_SLEEP_S
  done
  [ $waited -gt 0 ] && echo "$(date +%T) MEM-OK fold $k after ${waited}s: MemAvailable=${avail}MB"
  s=$(date +%s); echo "$(date +%T) START $RUN fold $k seed=$SEED"
  uv run python -u scripts/train_full_map.py \
    --features data/raw/gems-geodawn-numerical-features.tif \
    --labels data/raw/existing_faults.tif --template data/raw/example_submission.tif \
    --config $CFG --fold-map $FMAP --cv-scheme $SCHEME --fold $k --buffer-pixels $BUF \
    --seed "$SEED" \
    --output $RUN/fold-$k.tif --metrics-json $RUN/fold-$k.json.tmp 2>&1 \
    | while IFS= read -r l; do echo "$(date +%T) [f$k] $l"; done
  rc=${PIPESTATUS[0]}
  echo "$(date +%T) END fold $k exit=$rc elapsed_s=$(( $(date +%s)-s ))"
  if [ $rc -eq 137 ] && [ $attempt -lt 3 ]; then attempt=$((attempt+1)); echo "$(date +%T) OOM-RETRY fold $k attempt $attempt"; rm -f $RUN/fold-$k.json.tmp; sleep 120; continue; fi
  break
  done
  [ $rc -eq 0 ] || exit $rc
  mv $RUN/fold-$k.json.tmp $RUN/fold-$k.json
done
