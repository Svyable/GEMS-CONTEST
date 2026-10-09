# GEMS Experiment Queue

Resumable CPU-based experiment queue runner and helper scripts for GEMS competition workflows.

## Components

- **runner.py**: Main queue runner. Reads tasks from `tasks.json` and executes them sequentially, tracking state in `state/*.{done,failed,started}`. Stops on first failure for inspection.
- **tasks.json**: Queue task definitions (one per experiment). Re-read before each task execution, allowing live task additions.
- **train_folds.sh**: Resumable 5-fold training script. Skips folds that already have outputs.
- **score_view.sh**: Score a completed fold-map experiment for one CV view.
- **lofo_calibrate_view.py**: Leave-one-fold-out exact-threshold calibration. For fold k, finds the threshold that maximizes mean score over the other 4 folds, then evaluates once on fold k.
- **endpoint_view.sh**: Apply endpoint extension to fold predictions.
- **gate.sh**: Apply the propose-and-verify gate to a candidate.
- **record.sh**: Create a run manifest with provenance metadata.

## Environment Variables

All scripts support environment variable overrides with sensible defaults:

- `GEMS_WORKSPACE`: Repository root (default: `/workspace/gems/GEMS-CONTEST`)
- `QUEUE_DIR`: Queue state/log directory (default: `/workspace/gems/queue`)

Example:
```bash
export GEMS_WORKSPACE=/workspace
export QUEUE_DIR=/workspace/scripts/queue
```

## Usage

### Start the queue

```bash
cd /workspace
export GEMS_WORKSPACE=/workspace
nohup python3 scripts/queue/runner.py >> scripts/queue/queue.log 2>&1 &
```

The runner enforces single-instance execution with flock on `runner.lock`.

### Monitor progress

```bash
cat scripts/queue/STATUS.txt
cat scripts/queue/status.json
tail -f scripts/queue/queue.log
```

### Resume after failure

1. Inspect the failure: `cat scripts/queue/state/<task-id>.failed`
2. Fix the issue (code, data, config)
3. Delete the failed marker: `rm scripts/queue/state/<task-id>.failed`
4. Restart the queue (it will resume from the failed task)

### Manual task execution

Individual helper scripts can be run directly for testing or one-off runs:

```bash
# Train all folds for one view
./scripts/queue/train_folds.sh runs/my-exp configs/reference_unet.yaml spatial data/processed/cv-spatial-v1.tif 16

# LOFO calibration
python3 scripts/queue/lofo_calibrate_view.py runs/my-exp spatial data/processed/cv-spatial-v1.tif

# Score a view
./scripts/queue/score_view.sh runs/my-exp spatial data/processed/cv-spatial-v1.tif
```

## State Management

- `state/<task-id>.started`: Task start timestamp
- `state/<task-id>.done`: Task completion timestamp and exit code (rc=0)
- `state/<task-id>.failed`: Task failure timestamp and non-zero exit code
- `STATUS.txt`: Human-readable queue status snapshot
- `status.json`: Machine-readable queue status with full task metadata
- `queue.log`: Complete log of all queue output (append-only)

Tasks are idempotent: helper scripts check for existing outputs and skip completed work.
