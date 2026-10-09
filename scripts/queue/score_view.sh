#!/bin/bash
# usage: score_view.sh RUN_DIR SCHEME FOLD_MAP GATE_DIR  -> GATE_DIR/SCHEME.json + RUN_DIR/scores/lofo-SCHEME.json
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
RUN=$1; SCHEME=$2; FMAP=$3; GATE=$4
mkdir -p $GATE $RUN/scores
uv run python scripts/score_cv.py --truth data/raw/existing_faults.tif --fold-map $FMAP --scheme $SCHEME \
  --prediction-pattern "$RUN/fold-{fold}.tif" --output-json $GATE/$SCHEME.json > $RUN/scores/score_cv-$SCHEME.log
cp $GATE/$SCHEME.json $RUN/scores/$SCHEME.json
uv run python /workspace/gems/queue/lofo_calibrate_view.py $RUN $SCHEME $FMAP
