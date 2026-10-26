#!/bin/bash
# usage: rival_threshold.sh RUN_DIR SCHEME FOLD_MAP EXPECTED_SCORE [FACTOR]
# Score a completed CV run's fold predictions at the rival's closed-form
# threshold rule (mark pixel when p > FACTOR x EXPECTED_SCORE; default
# FACTOR 0.2), paired per fold against the run's own LOFO-calibrated scores.
# Writes RUN_DIR/scores/rival_threshold-SCHEME.json.
#   EXPECTED_SCORE: the reference run's LOFO-calibrated mean for this view
#   (parked s21 tasks pin the committed means, 5 decimals).
# Part of the threshold experiment (step 21): PARKED until Sven green-lights
# it. Runs nothing on its own; see scripts/rival_threshold.py.
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
RUN=$1; SCHEME=$2; FMAP=$3; EXPECTED=$4; FACTOR=${5:-0.2}
uv run python scripts/rival_threshold.py $RUN $SCHEME $FMAP \
  --expected-score $EXPECTED --factor $FACTOR
