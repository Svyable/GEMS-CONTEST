#!/bin/bash
# usage: compare_seed.sh INC_GATE_DIR CAND_GATE_DIR INC_RUN_BASE CAND_RUN_DIR
# Assembles raw gate JSONs + LOFO summaries from per-view run dirs, then runs
# scripts/compare_seed.py. The step-8 seed probe is spatial-only; views missing
# on either side are skipped by the comparison script.
#   INC_RUN_BASE: runs/ref-cv        -> runs/ref-cv-spatial-b16/scores/lofo-spatial.json
#   CAND_RUN_DIR: runs/ref-cv-spatial-b16-seed20260923 -> CAND_RUN_DIR/scores/lofo-spatial.json
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
INC_GATE=$1; CAND_GATE=$2; INC_BASE=$3; CAND_RUN=$4
CMP=runs/compare/seed20260923-b16
rm -rf $CMP; mkdir -p $CMP/inc-raw $CMP/cand-raw $CMP/inc-lofo $CMP/cand-lofo
for view in spatial fault trace; do
  [ -f $INC_GATE/$view.json ] && cp $INC_GATE/$view.json $CMP/inc-raw/$view.json || true
  [ -f $CAND_GATE/$view.json ] && cp $CAND_GATE/$view.json $CMP/cand-raw/$view.json || true
  [ -f ${INC_BASE}-${view}-b16/scores/lofo-$view.json ] \
    && cp ${INC_BASE}-${view}-b16/scores/lofo-$view.json $CMP/inc-lofo/lofo-$view.json || true
  [ -f $CAND_RUN/scores/lofo-$view.json ] \
    && cp $CAND_RUN/scores/lofo-$view.json $CMP/cand-lofo/lofo-$view.json || true
done
uv run python scripts/compare_seed.py $CMP/inc-raw $CMP/cand-raw \
  --incumbent-lofo $CMP/inc-lofo --candidate-lofo $CMP/cand-lofo \
  | tee $CMP/compare.log
