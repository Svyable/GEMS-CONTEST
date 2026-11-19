"""Tests for the record/score queue wrappers' outputs portability (step 45).

``record.sh`` and ``score_view.sh`` were the last two queue wrappers
never executed on this VM (step 44 scope note). ``record.sh`` honored
``GEMS_WORKSPACE`` for its ``cd`` line but exported run manifests to
the absolute box path ``/workspace/gems/phase2-outputs/runs/<id>``;
``score_view.sh`` invoked the LOFO calibrator at the absolute box path
``/workspace/gems/queue/lofo_calibrate_view.py``. Step 45 puts the
record export dir behind the step-44 ``GEMS_OUTPUTS`` override
(default ``/workspace/gems``, box-identical) and the LOFO path behind
a ``QUEUE_DIR`` override (default ``/workspace/gems/queue``,
box-identical — the same default ``scripts/queue/install_autostart.sh``
already uses, and the doctor's SYNC check already pins
``lofo_calibrate_view.py`` as a queue-dir file the runner executes).

Execution fixtures mirror the committed tree exactly:

* ``record.sh`` runs with the exact queued ``s4-record-ref-spatial-b32``
  task arguments against a copy of the real committed run dir
  (``results/runs/ref-cv-spatial-b32`` — scores + fold-0..4.json), the
  real ``configs/reference_unet.yaml``, and the real
  ``data/manifests/official.json``, inside a git-initialized sandbox
  checkout (``record_run.py`` requires a repo for its git provenance).
  The manifest it writes is a sandbox record, not a box result.
* ``score_view.sh`` runs with the exact queued score arguments for the
  same run, but the truth raster, fold map, and fold predictions are
  synthetic GeoTIFF fixtures (the ``tests/test_score_cv_protocol.py``
  idiom: 24x24, predictions equal the truth) because the real
  ``data/raw/existing_faults.tif`` is competition data that is never
  committed and is absent from this VM. The scores the wrapper
  produces are plumbing output only — they verify the wrapper runs
  end to end and lands its files in the right places, and say nothing
  about the real run.

Safety note: the pre-fix scripts are NOT executed as negative
controls. This VM's filesystem root is writable and ``/workspace``
does not exist, so the old scripts would have created it rather than
failing — the defects are pinned statically instead, and the
execution tests assert ``/workspace`` still does not exist
afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "queue"
RECORD = QUEUE / "record.sh"
SCORE_VIEW = QUEUE / "score_view.sh"

# Exact queued s4-record-ref-spatial-b32 task (scripts/queue/tasks.json).
RECORD_RUN_ID = "ref-cv-spatial-b32"
RECORD_RUN_DIR = "runs/ref-cv-spatial-b32"
RECORD_CONFIG = "configs/reference_unet.yaml"
RECORD_HYPOTHESIS = (
    "Organizer reference U-Net recipe under honest spatial CV at buffer 32 "
    "(cv-spatial-v1, all 5 folds); buffer-32 spatial incumbent for the "
    "CAND-STRUCTURE_TENSOR_COHERENCE-B32 and "
    "CAND-RIDGE_VALLEY_RESPONSE-B32 gates. Closes the provenance gap: the "
    "s4 block shipped train+score with no record, unlike every other "
    "queued reference run."
)
RECORD_NOTES = (
    "train_full_map.py --cv-scheme spatial --buffer-pixels 32 "
    "--seed 20260922; same config/epochs/train_step as the other b32 incumbents."
)

# Exact queued score task for the same run (scripts/queue/tasks.json).
SCORE_SCHEME = "spatial"
SCORE_FOLD_MAP = "data/processed/cv-spatial-v1.tif"
SCORE_GATE_DIR = "runs/gate/ref-b32"

RECORD_OUTPUTS_OVERRIDE = 'OUTPUTS="${GEMS_OUTPUTS:-/workspace/gems}"'
SCORE_QUEUE_OVERRIDE = 'QUEUE_DIR="${QUEUE_DIR:-/workspace/gems/queue}"'


def _write_fake_uv(bin_dir: Path) -> None:
    """A stand-in for `uv` that only supports `uv run python ...`.

    Same shim as tests/test_gate_wrapper.py: the box runs wrappers
    with real `uv`; this VM has none, so the shim strips the `run`
    verb and the `python` command token — both consumed by real `uv` —
    and execs python directly on the script and its args.
    """
    shim = bin_dir / "uv"
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "run" ]; then shift; fi\n'
        'if [ "$1" = "python" ]; then shift; fi\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)


def _env(ws: Path, bin_dir: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["GEMS_WORKSPACE"] = str(ws)
    # record_run.py / score_cv.py / lofo_calibrate_view.py import `gems`,
    # which on the box `uv run` resolves from the installed project; here
    # the repo src dir is the equivalent precondition.
    env["PYTHONPATH"] = str(REPO / "src")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.update(extra)
    return env


def test_record_uses_outputs_override() -> None:
    """The export dir must honor GEMS_OUTPUTS with the box path as default."""
    text = RECORD.read_text(encoding="utf-8")
    assert RECORD_OUTPUTS_OVERRIDE in text, "record.sh lacks the GEMS_OUTPUTS override"
    assert 'O="$OUTPUTS/phase2-outputs/runs/$ID"' in text


def test_record_has_no_bare_box_output_paths() -> None:
    """No output path may bypass the override (the pre-fix defect).

    The only literal ``/workspace/gems`` allowed is inside the
    ``GEMS_OUTPUTS`` default assignment itself. The ``--command``
    string is provenance metadata naming the box queue, not a path
    the script writes, so it stays as recorded.
    """
    text = RECORD.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("OUTPUTS="):
            continue
        assert "/workspace/gems/phase2-outputs" not in line, line


def test_record_passes_bash_syntax_check() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(RECORD)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_score_view_uses_queue_dir_override() -> None:
    """The LOFO path must honor QUEUE_DIR with the box path as default."""
    text = SCORE_VIEW.read_text(encoding="utf-8")
    assert SCORE_QUEUE_OVERRIDE in text, "score_view.sh lacks the QUEUE_DIR override"
    assert 'uv run python "$QUEUE_DIR/lofo_calibrate_view.py" $RUN $SCHEME $FMAP' in text


def test_score_view_has_no_bare_box_queue_path() -> None:
    """No path may bypass the override (the pre-fix defect).

    The only literal ``/workspace/gems/queue`` allowed is inside the
    ``QUEUE_DIR`` default assignment itself.
    """
    text = SCORE_VIEW.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("QUEUE_DIR="):
            continue
        assert "/workspace/gems/queue" not in line, line


def test_score_view_passes_bash_syntax_check() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(SCORE_VIEW)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_record_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute record.sh itself from a non-box checkout.

    Fixture state mirrors the committed tree: a copy of the real
    ``results/runs/ref-cv-spatial-b32`` run dir (scores + fold-0..4
    JSONs), the real config, and the real data manifest, inside a
    git-initialized sandbox checkout. The wrapper must exit 0, write
    a manifest for the run id with the recorded config hash and
    hypothesis, and export the manifest, score JSONs, fold JSONs,
    and config to the overridden outputs dir. All writes land under
    the tmp workspace; ``/workspace`` must not appear on this VM.
    """
    assert not Path("/workspace").exists(), "precondition: /workspace absent"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
    shutil.copytree(
        REPO / "results" / "runs" / RECORD_RUN_ID,
        ws / RECORD_RUN_DIR,
    )
    config_path = ws / RECORD_CONFIG
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(REPO / RECORD_CONFIG, config_path)
    manifest_path = ws / "data" / "manifests" / "official.json"
    manifest_path.parent.mkdir(parents=True)
    shutil.copyfile(REPO / "data" / "manifests" / "official.json", manifest_path)
    outputs = tmp_path / "outputs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)

    # record_run.py records git provenance, so the sandbox must be a
    # repo; one commit of the fixture files keeps it clean.
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.name", "GEMS test"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=ws, check=True)

    proc = subprocess.run(
        [
            "bash",
            "scripts/queue/record.sh",
            RECORD_RUN_ID,
            RECORD_RUN_DIR,
            RECORD_CONFIG,
            RECORD_HYPOTHESIS,
            RECORD_NOTES,
        ],
        cwd=ws,
        env=_env(ws, bin_dir, GEMS_OUTPUTS=str(outputs)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    run_manifest = json.loads((ws / RECORD_RUN_DIR / "manifest.json").read_text())
    assert run_manifest["run_id"] == RECORD_RUN_ID
    assert run_manifest["hypothesis"] == RECORD_HYPOTHESIS
    assert run_manifest["notes"] == RECORD_NOTES
    assert run_manifest["code"]["git_dirty"] is False
    expected_sha = hashlib.sha256((REPO / RECORD_CONFIG).read_bytes()).hexdigest()
    assert run_manifest["config"]["sha256"] == expected_sha
    assert run_manifest["config"]["path"] == RECORD_CONFIG
    artifact_paths = {a["path"] for a in run_manifest["artifacts"]}
    assert "runs/ref-cv-spatial-b32/scores/spatial.json" in artifact_paths
    assert "runs/ref-cv-spatial-b32/scores/lofo-spatial.json" in artifact_paths
    assert "runs/ref-cv-spatial-b32/fold-0.json" in artifact_paths
    assert "runs/ref-cv-spatial-b32/fold-4.json" in artifact_paths

    export_dir = outputs / "phase2-outputs" / "runs" / RECORD_RUN_ID
    assert (export_dir / "manifest.json").exists(), proc.stdout + proc.stderr
    assert (export_dir / "spatial.json").exists()
    assert (export_dir / "lofo-spatial.json").exists()
    assert (export_dir / "fold-0.json").exists()
    assert (export_dir / "fold-4.json").exists()
    exported_config = export_dir / "config.yaml"
    assert exported_config.exists()
    assert exported_config.read_bytes() == (REPO / RECORD_CONFIG).read_bytes()
    assert json.loads((export_dir / "manifest.json").read_text()) == run_manifest

    assert not Path("/workspace").exists(), "record.sh wrote outside the sandbox"


def _write_synthetic_rasters(ws: Path) -> None:
    """Synthetic truth / fold-map / predictions (test_score_cv_protocol idiom)."""
    height = width = 24
    truth = np.zeros((height, width), dtype=np.uint8)
    truth[3, 3] = 1
    truth[15, 18] = 1
    folds = np.zeros((height, width), dtype=np.int16)
    folds[:, width // 2 :] = 1
    common = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": "EPSG:32611",
        "transform": from_origin(500000, 4000000, 100, 100),
    }
    truth_path = ws / "data" / "raw" / "existing_faults.tif"
    truth_path.parent.mkdir(parents=True)
    with rasterio.open(truth_path, "w", dtype=truth.dtype, **common) as dst:
        dst.write(truth, 1)
    fold_map_path = ws / SCORE_FOLD_MAP
    fold_map_path.parent.mkdir(parents=True)
    with rasterio.open(fold_map_path, "w", dtype=folds.dtype, **common) as dst:
        dst.write(folds, 1)
    run_dir = ws / RECORD_RUN_DIR
    run_dir.mkdir(parents=True)
    for fold in (0, 1):
        with rasterio.open(run_dir / f"fold-{fold}.tif", "w", dtype="float32", **common) as dst:
            dst.write(truth.astype(np.float32), 1)
        metrics = {
            "buffer_pixels": 16,
            "seed": 20260922,
            "epochs": [{"optimizer_steps": 3}],
            "train_step": 32,
            "config_sha256": "ab" * 32,
        }
        (run_dir / f"fold-{fold}.json").write_text(json.dumps(metrics))


def test_score_view_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute score_view.sh itself from a non-box checkout.

    Fixture state mirrors the queued score task's arguments for
    ``ref-cv-spatial-b32`` (spatial, the cv-spatial-v1 fold map, the
    ref-b32 gate dir), but the truth raster, fold map, and fold
    predictions are synthetic fixtures — the real truth raster is
    competition data, never committed and absent from this VM — so
    the scores are plumbing output only, not a measured result. The
    LOFO calibrator comes from a separate queue dir, as on the box
    (the doctor's SYNC check pins it as a queue-dir file). The
    wrapper must exit 0, write the schema-2 gate JSON and copy it
    into the run's scores, write the LOFO calibration JSON, and log
    the score_cv run. All writes land under the tmp workspace;
    ``/workspace`` must not appear on this VM.
    """
    assert not Path("/workspace").exists(), "precondition: /workspace absent"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
    _write_synthetic_rasters(ws)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    shutil.copyfile(
        REPO / "scripts" / "queue" / "lofo_calibrate_view.py",
        queue_dir / "lofo_calibrate_view.py",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)

    proc = subprocess.run(
        [
            "bash",
            "scripts/queue/score_view.sh",
            RECORD_RUN_DIR,
            SCORE_SCHEME,
            SCORE_FOLD_MAP,
            SCORE_GATE_DIR,
        ],
        cwd=ws,
        env=_env(ws, bin_dir, QUEUE_DIR=str(queue_dir)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    gate_path = ws / SCORE_GATE_DIR / f"{SCORE_SCHEME}.json"
    assert gate_path.exists(), proc.stdout + proc.stderr
    payload = json.loads(gate_path.read_text())
    assert payload["evaluation_protocol"]["schema_version"] == 2
    assert payload["evaluation_protocol"]["metric"] == "distance_weighted_tversky"
    # Synthetic fixture: predictions equal the truth, so the score is
    # ~1 by construction — plumbing output, not a measured result.
    assert payload["macro_mean"] > 0.99999999

    run_scores = ws / RECORD_RUN_DIR / "scores"
    assert (run_scores / f"{SCORE_SCHEME}.json").exists()
    assert json.loads((run_scores / f"{SCORE_SCHEME}.json").read_text()) == payload
    assert (run_scores / f"score_cv-{SCORE_SCHEME}.log").exists()
    lofo_path = run_scores / f"lofo-{SCORE_SCHEME}.json"
    assert lofo_path.exists()
    lofo = json.loads(lofo_path.read_text())
    assert lofo["scheme"] == SCORE_SCHEME
    assert lofo["calibrated_mean"] > 0.99999999

    assert not Path("/workspace").exists(), "score_view.sh wrote outside the sandbox"
