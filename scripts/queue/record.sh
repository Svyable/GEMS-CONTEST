#!/bin/bash
# usage: record.sh RUN_ID RUN_DIR CONFIG "HYPOTHESIS" "NOTES"  (records + exports commit-safe files)
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
ID=$1; RUN=$2; CFG=$3
ARTS=$(ls $RUN/scores/*.json $RUN/fold-?.json 2>/dev/null | sed 's/^/--artifact /' | tr '\n' ' ')
uv run python scripts/record_run.py --run-id $ID --config $CFG --data-manifest data/manifests/official.json \
  $ARTS --hypothesis "$4" --notes "$5" --command "queue: /workspace/gems/queue/tasks.json" --output $RUN/manifest.json
O=/workspace/gems/phase2-outputs/runs/$ID; mkdir -p $O
cp $RUN/manifest.json $RUN/scores/*.json $O/ 2>/dev/null || true
cp $RUN/fold-?.json $O/ 2>/dev/null || true
cp $CFG $O/config.yaml
