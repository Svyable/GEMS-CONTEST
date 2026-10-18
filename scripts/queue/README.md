# GEMS Experiment Queue

Resumable CPU-based experiment queue runner and helper scripts for GEMS competition workflows.

## Components

- **runner.py**: Main queue runner. Reads tasks from `tasks.json` and executes them sequentially, tracking state in `state/*.{done,failed,started}`. Stops on first failure for inspection.
- **tasks.json**: Queue task definitions (one per experiment). Re-read before each task execution, allowing live task additions.
- **tasks.parked.json**: Parked tasks (removed from active queue but kept for reference).
- **train_folds.sh**: Resumable 5-fold training script. Skips folds that already have outputs. Includes memory gate (waits for available memory before starting each fold) and OOM retry logic (retries folds killed with exit code 137 up to 3 times). Optional 6th arg `SEED` (default `20260922`, the incumbent seed) forwarded as `train_full_map.py --seed`; used by the step-8 seed-robustness probe.
- **score_view.sh**: Score a completed fold-map experiment for one CV view.
- **lofo_calibrate_view.py**: Leave-one-fold-out exact-threshold calibration. For fold k, finds the threshold that maximizes mean score over the other 4 folds, then evaluates once on fold k.
- **endpoint_view.sh**: Apply endpoint extension to fold predictions.
- **curved_view.sh**: Apply P4 curved continuation to fold predictions (same
  per-fold LOFO skeleton thresholds as endpoint_view.sh).
- **gate.sh**: Apply the propose-and-verify gate to a candidate.
- **record.sh**: Create a run manifest with provenance metadata.

## Environment Variables

All scripts support environment variable overrides with sensible defaults:

- `GEMS_WORKSPACE`: Repository root (default: `/workspace/gems/GEMS-CONTEST`)
- `QUEUE_DIR`: Queue state/log directory (default: `/workspace/gems/queue`)
- `GEMS_MIN_AVAIL_MB`: Minimum available memory in MB before starting a fold (default: 5000). Used by `train_folds.sh` memory gate.

Example:
```bash
export GEMS_WORKSPACE=/workspace
export QUEUE_DIR=/workspace/scripts/queue
export GEMS_MIN_AVAIL_MB=8000
```

## Memory Gate and OOM Retry (added 2026-10-10)

`train_folds.sh` includes two resilience mechanisms for shared-box training:

1. **Memory gate**: Before starting each fold, waits until `/proc/meminfo` reports `MemAvailable >= GEMS_MIN_AVAIL_MB` (default 5000 MB). Logs wait time when memory is insufficient. Prevents OOM kills from concurrent jobs.

2. **OOM retry**: If a fold exits with code 137 (OOM kill), retries up to 3 times with 120-second delays between attempts. Logs each retry attempt. This handles transient memory pressure from other agents on the shared box.

These features were added after two OOM kills (exit code 137) interrupted training due to unrelated memory pressure on 2026-10-10 evening CT.

## Task Parking

Tasks removed from the active queue but kept for reference are stored in `tasks.parked.json`. Current parked tasks:
- **s6-gate-mumford_shah_log-trace** and **s6-gate-steerable_filter-trace**: Trace-only b32 gates for b16 kinds. Parked by user decision 2026-10-10 because single-view gates cannot ACCEPT by policy. The corresponding b16 kinds now have 3-view gates (s5b-gate-*-b16) at buffer 16.

## Fair-comparison buffer protocol

Lineament filter support differs per feature kind. `scripts/train_full_map.py`
refuses `--buffer-pixels` below `required_lineament_buffer(kind)` (verified
against `src/gems/lineament.py` on 2026-10-10):

| kind | required buffer (px) | queue buffer | incumbent |
|---|---|---|---|
| `mumford_shah_log` | 5 | 16 | `ref-full` (spatial/fault @16) |
| `steerable_filter` | 7 | 16 | `ref-full` (spatial/fault @16) |
| `wavelet_entropy` | 8 | 16 | `ref-b16` (3 views @16) |
| `structure_tensor_coherence` | 17 | 32 | `ref-b32` re-run @32 |
| `ridge_valley_response` | 30 | 32 | `ref-b32` re-run @32 |

A candidate must share its incumbent's buffer (and seed/epochs/train_step/fold
map) or the propose-and-verify gate refuses the comparison as protocol-unequal
(see `docs/STRATEGY.md` "Fair comparison protocol"). Larger buffers exclude more
training pixels near held-out regions, so a buffer-32 candidate is only ever
compared to the buffer-32 incumbent — never to the buffer-16 run.

`score_view.sh` passes `--metrics-pattern` so every score JSON carries the
training protocol (`buffer_pixels`, `seed`, `epochs`, `train_step`,
`config_sha256`) the gate compares.

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
