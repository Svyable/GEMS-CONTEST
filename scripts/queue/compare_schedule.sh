#!/bin/bash
# usage: compare_schedule.sh INC_GATE_DIR CAND_GATE_DIR INC_RUN_BASE CAND_RUN_BASE
# Assembles raw gate JSONs + LOFO summaries from per-view run dirs, then runs
# scripts/compare_schedule.py. RUN_BASE is the common run-dir prefix, e.g.
#   runs/ref-cv        -> runs/ref-cv-spatial-b16/scores/lofo-spatial.json
#   runs/cand-longer_schedule -> runs/cand-longer_schedule-fault-b16/scores/lofo-fault.json
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
INC_GATE=$1; CAND_GATE=$2; INC_BASE=$3; CAND_BASE=$4
CMP=runs/compare/longer_schedule-b16
rm -rf $CMP; mkdir -p $CMP/inc-raw $CMP/cand-raw $CMP/inc-lofo $CMP/cand-lofo
for view in spatial fault trace; do
  cp $INC_GATE/$view.json $CMP/inc-raw/$view.json
  cp $CAND_GATE/$view.json $CMP/cand-raw/$view.json
  cp ${INC_BASE}-${view}-b16/scores/lofo-$view.json $CMP/inc-lofo/lofo-$view.json
  cp ${CAND_BASE}-${view}-b16/scores/lofo-$view.json $CMP/cand-lofo/lofo-$view.json
done
uv run python scripts/compare_schedule.py $CMP/inc-raw $CMP/cand-raw \
  --incumbent-lofo $CMP/inc-lofo --candidate-lofo $CMP/cand-lofo \
  | tee $CMP/compare.log
