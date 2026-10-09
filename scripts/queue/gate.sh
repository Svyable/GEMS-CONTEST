#!/bin/bash
# usage: gate.sh NAME INCUMBENT_GATE_DIR CANDIDATE_GATE_DIR CONFIG "HYPOTHESIS"
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
mkdir -p /workspace/gems/phase2-outputs/gates
uv run python scripts/verify_candidate.py --candidate "$1" --hypothesis "$5" \
  --incumbent-dir $2 --candidate-dir $3 --config $4 \
  --ledger /workspace/gems/phase1-outputs/candidate-trials.jsonl | tee /workspace/gems/phase2-outputs/gates/$1.json
