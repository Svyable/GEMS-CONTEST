# GEMS Experiment Queue

Resumable CPU-based experiment queue runner and helper scripts for GEMS competition workflows.

## Components

- **runner.py**: Main queue runner. Reads tasks from `tasks.json` and executes them sequentially, tracking state in `state/*.{done,failed,started}`. Stops on first failure for inspection.
- **tasks.json**: Queue task definitions (one per experiment). Re-read before each task execution, allowing live task additions.
- **queue_insert.py** (in `scripts/`, step 26): Insert new tasks before a named anchor task. The runner executes pending tasks in list order, so a task that must precede an already-queued consumer (e.g. a re-score that must run before the assemble that copies its output) cannot be appended — it has to be inserted. The tool validates anchor uniqueness, id uniqueness, and non-empty cmds before writing anything, and refuses to write on any failure. Use `--dry-run` first.
- **queue_order_audit.py** (in `scripts/`, step 33): Audit the queue's list order — every consumer must follow its producers (train before score before record; score after its run's train; view after its base run's LOFO score; assemble after the scores that wrote its source gate JSONs; gate/compare after the tasks that wrote both gate dirs and the candidate run dirs). Presence is checked by the invariant tests; this checks position. Exit 0 when clean, 1 on any ORDER-VIOLATION, 2 on an unreadable `tasks.json`. Inputs with no queued producer at all are NO-PRODUCER notes, not violations — an incumbent gate dir may predate the queue, and the step-8 seed probe is spatial-only by design (its compare skips missing views).
- **queue_forecast.py** (in `scripts/`, step 34): Forecast whether the queue can drain before the safe cutoff (2026-12-03 22:00 UTC). Observed throughput is recorded runs/day from committed manifest `created_utc` (a lower bound — scored-but-unrecorded runs are reported alongside); drain days = pending train tasks / rate, since 5-fold trains dominate wall-clock and the runner is sequential. Per pending gate/compare it reports the train tasks on its critical path (from `queue_critical_path.py` feeders) and the days to that decision. Verdict bands at the observed rate: FEASIBLE (5-day buffer for scoring/gating/submission), TIGHT, INFEASIBLE — plus a separate silence line: 0 runs/day across the silence window means the queue never drains until the box resumes. Exit 0 whenever the forecast could be produced (UNKNOWN rate with fewer than two manifests), 2 on an unreadable/malformed `tasks.json`.
- **audit_config_drift.py** (in `scripts/`, step 35): Audit committed results for config drift — whether each config file on disk still hashes to the `config_sha256` its runs recorded in their manifests, score JSONs, and fold files. A config edited after a run was scored silently changes what the queue trains next, and the gate refuses the resulting protocol-unequal comparison after the GPU days are spent. Also flags a manifest/queue disagreement on a run's config (CONFIG-MISMATCH), a missing config file, and self-contradictory recorded hashes (MIXED). Exit 0 when clean (notes allowed), 1 on any finding, 2 on an unreadable/malformed `tasks.json`.
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

### Auto-start after a reboot (step 31)

A manual `nohup` start does not survive a reboot — the step-29 doctor
can only report NOT-RUNNING when that happens. Install the persistent
fix once from the checkout:

```bash
bash scripts/queue/install_autostart.sh
```

It installs `gems-queue.service` as a systemd user unit when a user
manager answers (enabled, started, `loginctl enable-linger` attempted
so it also starts at boot without a login session), and otherwise
falls back to an `@reboot` cron entry. Both paths are idempotent and
`--dry-run` previews the change. `Restart=on-failure`, not `always`:
the runner exits 0 on STOPPED (a task failed — human inspection must
precede any resume) and on ALL-DONE, and both must stay stopped; only
a crash restarts it. Installing while a `nohup` runner is already live
is safe (flock): the service's first start exits immediately and the
unit takes over after that runner exits, or on the next reboot. Check
with `systemctl --user status gems-queue.service` (or `crontab -l`
for the cron path); remove with `install_autostart.sh --uninstall`.
The unit file itself is not part of the doctor's SYNC check — it is
run once from the checkout, never executed by the runner, so it is
not copied into the queue dir. The doctor does verify the install,
though: step 32 adds an AUTOSTART line to its output (installed via
systemd unit / @reboot cron, partial, missing, or unknown), and a
NOT-RUNNING verdict with auto-start installed means the auto-start
itself failed — check `systemctl --user status gems-queue.service`
before re-installing.

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

### Queue order audit

```bash
uv run python scripts/queue_order_audit.py
```

Run this after any edit to `tasks.json` (append, insert, or reorder)
before the box pulls: it verifies every consumer follows its
producers in list order and fails loudly (exit 1) on any
ORDER-VIOLATION, instead of letting a gate decide on stale or
missing inputs after hours of training. The two NO-PRODUCER notes
on the current queue are by design (the spatial-only seed probe);
a new note means a task reads an input nothing in the queue
produces — check whether it is grandfathered/committed before
assuming it is missing.

### Queue forecast

```bash
uv run python scripts/queue_forecast.py
```

Run this when the close approaches or the box has been silent: it
answers whether the queue can drain before the safe cutoff
(2026-12-03 22:00 UTC) at the observed rate, and which decision
gets decidable first once the box resumes. The observed rate comes
from committed manifests only, so box-side-only progress is
invisible here by design — the box should commit before trusting
the forecast. A FEASIBLE verdict at the observed rate does not mean
the queue is draining: the silence line is the binding constraint
while the box is quiet, and resumption, not capacity, is what
unblocks the 9 pending gates.

### Config drift audit

```bash
uv run python scripts/audit_config_drift.py
```

Run this after any edit to a config under `configs/` and before the
box pulls: it verifies every config file still hashes to the
`config_sha256` the committed runs recorded for it. A DRIFT means
pending trains for that config would score under a new protocol and
their gate side would be protocol-unequal to the recorded side — fix
the config or re-record before the box trains it. A CONFIG-MISMATCH
means the queue would re-train a run under a different config than
its manifest records. Gate-dir copies are excluded (they duplicate
their source run's protocol); the step-22 audit already verifies
their content by SHA256.

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
