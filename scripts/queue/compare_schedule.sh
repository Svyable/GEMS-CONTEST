#!/bin/bash
# usage: compare_schedule.sh INC_GATE_DIR CAND_GATE_DIR INC_RUN_BASE CAND_RUN_BASE
# Assembles raw gate JSONs + LOFO summaries from per-view run dirs, then runs
# scripts/compare_schedule.py. RUN_BASE is the common run-dir prefix, e.g.
#   runs/ref-cv        -> runs/ref-cv-spatial-b16/scores/lofo-spatial.json
#   runs/cand-longer_schedule -> runs/cand-longer_schedule-fault-b16/scores/lofo-fault.json
# Every copy is guarded: compare_schedule.py skips a view whose gate JSON is
# missing on either side and skips LOFO-calibrated deltas when a LOFO summary
# is missing, but a bare cp under set -e would abort the whole compare
# before the tool ever runs. That bit the s12 compare specifically: the
# incumbent b16 spatial/fault LOFO summaries have no queued producer
# (step-24 finding; only the s5b trace LOFO exists), so an unguarded cp
# made the compare unrunnable as scripted even after every producer task
# completed. Missing inputs surface as SKIPPED lines in compare.log.
# (Same guarded idiom as compare_seed.sh.)
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
INC_GATE=$1; CAND_GATE=$2; INC_BASE=$3; CAND_BASE=$4
CMP=runs/compare/longer_schedule-b16
rm -rf $CMP; mkdir -p $CMP/inc-raw $CMP/cand-raw $CMP/inc-lofo $CMP/cand-lofo
for view in spatial fault trace; do
  [ -f $INC_GATE/$view.json ] && cp $INC_GATE/$view.json $CMP/inc-raw/$view.json || true
  [ -f $CAND_GATE/$view.json ] && cp $CAND_GATE/$view.json $CMP/cand-raw/$view.json || true
  [ -f ${INC_BASE}-${view}-b16/scores/lofo-$view.json ] \
    && cp ${INC_BASE}-${view}-b16/scores/lofo-$view.json $CMP/inc-lofo/lofo-$view.json || true
  [ -f ${CAND_BASE}-${view}-b16/scores/lofo-$view.json ] \
    && cp ${CAND_BASE}-${view}-b16/scores/lofo-$view.json $CMP/cand-lofo/lofo-$view.json || true
done
uv run python scripts/compare_schedule.py $CMP/inc-raw $CMP/cand-raw \
  --incumbent-lofo $CMP/inc-lofo --candidate-lofo $CMP/cand-lofo \
  | tee $CMP/compare.log
