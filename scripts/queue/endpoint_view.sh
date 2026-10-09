#!/bin/bash
# usage: endpoint_view.sh BASE_RUN OUT_RUN SCHEME
# Extend each fold map with skeleton threshold = that fold's LOFO threshold (fit without its labels).
set -eo pipefail
cd /workspace/gems/GEMS-CONTEST
BASE=$1; OUT=$2; SCHEME=$3; mkdir -p $OUT
for k in 0 1 2 3 4; do
  [ -f $OUT/fold-$k.tif ] && continue
  t=$(python3 -c "import json;print([f for f in json.load(open('$BASE/scores/lofo-$SCHEME.json'))['folds'] if f['fold']==$k][0]['threshold'])")
  uv run python scripts/extend_endpoints.py --input $BASE/fold-$k.tif --output $OUT/fold-$k.tif.tmp.tif --threshold $t
  mv $OUT/fold-$k.tif.tmp.tif $OUT/fold-$k.tif
  echo "fold $k threshold=$t"
done
