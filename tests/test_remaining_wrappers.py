"""Tests for the four remaining queue wrappers' first execution (step 46).

Correction to the step-45 record: step 45's log entry and the scope note
in ``tests/test_queue_workspace.py`` claim "every queue wrapper has now
been executed on this VM". That was premature. Steps 28/44/45 executed
``compare_schedule.sh``, ``gate.sh``, ``record.sh``, ``score_view.sh``
(and ``train_folds.sh`` in step 41) — but four wrappers were only ever
statically asserted, never executed end to end on this VM:

* ``compare_seed.sh`` — only its Python script was unit-tested
  (``tests/test_compare_seed.py``); the wrapper's guarded staging copies
  and ``tee`` to ``compare.log`` never ran.
* ``curved_view.sh`` / ``endpoint_view.sh`` — only their Python scripts
  were unit-tested; the wrappers' per-fold LOFO-threshold lookup, the
  ``.tmp.tif`` rename, and the resume-skip (``[ -f ... ] && continue``)
  never ran.
* ``rival_threshold.sh`` — only ``scripts/rival_threshold.py`` was
  executed (``tests/test_rival_threshold.py``); the wrapper itself never
  ran, so its argument passing (no ``--truth``: the script's
  ``data/raw/existing_faults.tif`` default) was unexercised.

This file executes all four real wrappers in sandboxed checkouts with
the fake-``uv`` shim (the step-28 idiom) and the exact queued task
arguments, parsed from ``scripts/queue/tasks.json`` /
``tasks.parked.json`` so the test tracks the queue rather than a
hardcoded copy.

Fixtures are synthetic except where noted; every score these runs
produce is plumbing output only — they verify the wrappers run end to
end and land their files in the right places, and say nothing about
the real runs:

* The view wrappers run on a 5-fold synthetic base run (20x40
  probability rasters: a 0.9 line with a 0.4 support tail, 30 m pixels,
  LOFO thresholds 0.5) — the ``tests/test_curved_continuation.py``
  line/tail idiom.
* ``compare_seed.sh`` runs with the real committed incumbent
  ``results/runs/gate/ref-b16/trace.json`` and a seed-perturbed copy
  (seed 20260923, fold scores +0.0005) as the candidate, plus synthetic
  LOFO summaries for the trace view; spatial/fault are missing on both
  sides and must be SKIPPED.
* ``rival_threshold.sh`` runs with the exact queued s21-spatial args
  against a synthetic 2-fold run (the
  ``tests/test_rival_threshold.py`` idiom), synthetic truth at the
  wrapper's default ``data/raw/existing_faults.tif``, and a synthetic
  fold map at the queued ``data/processed/cv-spatial-v1.tif`` path.
  The s21 tasks stay PARKED: this executes the wrapper on synthetic
  data in a sandbox, it does not run the threshold experiment on real
  data and it changes nothing about the parked queue tasks.

Safety note, as in steps 44/45: the pre-fix scripts are NOT executed
as negative controls, and the execution tests assert ``/workspace``
still does not exist afterwards (this VM's filesystem root is
writable, so an absolute box path would have created it).
"""

from __future__ import annotations

import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "queue"

LIVE = json.loads((QUEUE / "tasks.json").read_text())
PARKED = json.loads((QUEUE / "tasks.parked.json").read_text())

# Exact queued commands, parsed at test time so the tests track the queue.
S3_ENDPOINT_SPATIAL = next(t["cmd"] for t in LIVE if t["id"] == "s3-endpoint-spatial")
S13_CURVED_SPATIAL = next(t["cmd"] for t in LIVE if t["id"] == "s13-curved-spatial")
S14_COMPARE_SEED = next(t["cmd"] for t in LIVE if t["id"] == "s14-compare-seed20260923-b16")
S21_RIVAL_SPATIAL = next(t["cmd"] for t in PARKED if t["id"] == "s21-rival-threshold-spatial-b16")


def _write_fake_uv(bin_dir: Path) -> None:
    """A stand-in for `uv` that only supports `uv run python ...`.

    Same shim as tests/test_queue_workspace.py: the box runs wrappers
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


def _sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """A workspace with the repo scripts symlinked in, plus a fake-uv bin."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)
    return ws, bin_dir


def _env(ws: Path, bin_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["GEMS_WORKSPACE"] = str(ws)
    # The wrapper-invoked scripts import `gems`, which on the box `uv
    # run` resolves from the installed project; here the repo src dir is
    # the equivalent precondition (the step-44/45 idiom).
    env["PYTHONPATH"] = str(REPO / "src")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return env


def _run_wrapper(ws: Path, bin_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", *args],
        cwd=ws,
        env=_env(ws, bin_dir),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_raster(path: Path, array: np.ndarray, dtype: str) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "transform": from_origin(0, 600, 30, 30),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def _view_base_run(ws: Path, run_rel: str = "runs/ref-cv-spatial-full") -> Path:
    """5-fold synthetic base run: 0.9 line + 0.4 support tail, LOFO 0.5."""
    base = ws / run_rel
    (base / "scores").mkdir(parents=True)
    for k in range(5):
        prob = np.zeros((20, 40), dtype=np.float32)
        prob[10, 5:26] = 0.9
        prob[10, 26:32] = 0.4
        _write_raster(base / f"fold-{k}.tif", prob, "float32")
    doc = {
        "scheme": "spatial",
        "method": "fixture (step 46)",
        "folds": [
            {"fold": k, "threshold": 0.5, "raw_score": 0.1, "calibrated_score": 0.4}
            for k in range(5)
        ],
        "raw_mean": 0.1,
        "calibrated_mean": 0.4,
    }
    (base / "scores" / "lofo-spatial.json").write_text(json.dumps(doc))
    return base


def _view_wrapper_args(queued_cmd: str) -> list[str]:
    """First command of a queued `view && score` chain, minus its path."""
    first = queued_cmd.split(" && ")[0]
    parts = shlex.split(first)
    assert parts[0].endswith(".sh"), parts
    return ["scripts/queue/" + Path(parts[0]).name, *parts[1:]]


def _assert_view_outputs_changed(ws: Path, base: Path, out_rel: str) -> None:
    """Every fold output exists, is a valid float32 raster, and differs."""
    out = ws / out_rel
    assert not list(out.glob("*.tmp.tif")), "tmp rename leftovers in output dir"
    for k in range(5):
        with rasterio.open(base / f"fold-{k}.tif") as src:
            before = src.read(1)
        with rasterio.open(out / f"fold-{k}.tif") as src:
            after = src.read(1)
            assert src.dtypes == ("float32",)
        assert after.shape == before.shape
        assert np.any(after != before), f"fold {k}: output identical to input"


def test_endpoint_view_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute endpoint_view.sh with the exact queued s3 args."""
    ws, bin_dir = _sandbox(tmp_path)
    base = _view_base_run(ws)
    args = _view_wrapper_args(S3_ENDPOINT_SPATIAL)
    assert args[1:] == ["runs/ref-cv-spatial-full", "runs/endpoint-cv-spatial-full", "spatial"]
    proc = _run_wrapper(ws, bin_dir, args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for k in range(5):
        assert f"fold {k} threshold=0.5" in proc.stdout
    assert "Extended 14 pixels" in proc.stdout
    _assert_view_outputs_changed(ws, base, "runs/endpoint-cv-spatial-full")
    assert not Path("/workspace").exists()


def test_endpoint_view_wrapper_resume_skips_existing(tmp_path: Path) -> None:
    """A pre-existing fold output is left untouched (the resume idiom)."""
    ws, bin_dir = _sandbox(tmp_path)
    _view_base_run(ws)
    out = ws / "runs" / "endpoint-cv-spatial-full"
    out.mkdir(parents=True)
    sentinel = out / "fold-2.tif"
    sentinel.write_bytes(b"SENTINEL-NOT-A-RASTER")
    proc = _run_wrapper(ws, bin_dir, _view_wrapper_args(S3_ENDPOINT_SPATIAL))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sentinel.read_bytes() == b"SENTINEL-NOT-A-RASTER"
    assert "fold 2 threshold" not in proc.stdout
    for k in (0, 1, 3, 4):
        assert (out / f"fold-{k}.tif").is_file()
        assert f"fold {k} threshold=0.5" in proc.stdout
    assert not Path("/workspace").exists()


def test_curved_view_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute curved_view.sh with the exact queued s13 args."""
    ws, bin_dir = _sandbox(tmp_path)
    base = _view_base_run(ws)
    args = _view_wrapper_args(S13_CURVED_SPATIAL)
    assert args[1:] == ["runs/ref-cv-spatial-full", "runs/curved-cv-spatial-full", "spatial"]
    proc = _run_wrapper(ws, bin_dir, args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for k in range(5):
        assert f"fold {k} threshold=0.5" in proc.stdout
    assert "Painted 2 continuation pixels" in proc.stdout
    _assert_view_outputs_changed(ws, base, "runs/curved-cv-spatial-full")
    assert not Path("/workspace").exists()


def _lofo_doc(scores: list[float]) -> dict:
    return {
        "scheme": "trace",
        "method": "fixture (step 46)",
        "folds": [
            {"fold": k, "threshold": 0.5, "raw_score": 0.1, "calibrated_score": s}
            for k, s in enumerate(scores)
        ],
        "raw_mean": 0.1,
        "calibrated_mean": float(statistics.fmean(scores)),
    }


def test_compare_seed_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute compare_seed.sh with the exact queued s14 args.

    Incumbent trace.json is the real committed gate JSON; the candidate
    is the same document with only the seed changed (20260923) and fold
    scores +0.0005 — the one protocol difference the comparison allows.
    """
    ws, bin_dir = _sandbox(tmp_path)
    inc_gate = ws / "runs" / "gate" / "ref-b16"
    inc_gate.mkdir(parents=True)
    inc = json.loads((REPO / "results" / "runs" / "gate" / "ref-b16" / "trace.json").read_text())
    (inc_gate / "trace.json").write_text(json.dumps(inc))
    cand_gate = ws / "runs" / "gate" / "ref-seed20260923-b16"
    cand_gate.mkdir(parents=True)
    cand = json.loads(json.dumps(inc))
    cand["evaluation_protocol"]["seed"] = 20260923
    for row in cand["folds"]:
        row["score"] = min(1.0, row["score"] + 0.0005)
    (cand_gate / "trace.json").write_text(json.dumps(cand))
    inc_cal = [0.0200, 0.0210, 0.0190, 0.0220, 0.0205]
    cand_cal = [c + 0.0004 for c in inc_cal]
    inc_lofo = ws / "runs" / "ref-cv-trace-b16" / "scores"
    inc_lofo.mkdir(parents=True)
    (inc_lofo / "lofo-trace.json").write_text(json.dumps(_lofo_doc(inc_cal)))
    cand_lofo = ws / "runs" / "ref-cv-spatial-b16-seed20260923" / "scores"
    cand_lofo.mkdir(parents=True)
    (cand_lofo / "lofo-trace.json").write_text(json.dumps(_lofo_doc(cand_cal)))

    parts = shlex.split(S14_COMPARE_SEED)
    assert parts[0].endswith("compare_seed.sh"), parts
    args = ["scripts/queue/compare_seed.sh", *parts[1:]]
    proc = _run_wrapper(ws, bin_dir, args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cmp_dir = ws / "runs" / "compare" / "seed20260923-b16"
    log = (cmp_dir / "compare.log").read_text(encoding="utf-8")
    assert "spatial: SKIPPED (missing gate JSON on at least one side)" in log
    assert "fault: SKIPPED (missing gate JSON on at least one side)" in log
    assert "[trace] incumbent seed 20260922 vs candidate seed 20260923" in log
    assert "mean delta +0.00050 (seed-effect std 0.00000)" in log
    assert "mean delta +0.00040 (seed-effect std 0.00000)" in log
    for rel in (
        "inc-raw/trace.json",
        "cand-raw/trace.json",
        "inc-lofo/lofo-trace.json",
        "cand-lofo/lofo-trace.json",
    ):
        assert (cmp_dir / rel).is_file(), rel
    assert not list((cmp_dir / "inc-lofo").glob("lofo-spatial.json"))
    assert not Path("/workspace").exists()


def _rival_run(ws: Path) -> tuple[Path, Path, Path]:
    """2-fold spatial fixture (the tests/test_rival_threshold.py idiom)."""
    truth = np.zeros((12, 12), dtype=np.uint8)
    truth[2:4, 2:4] = 1
    truth[8:10, 8:10] = 1
    truth_path = ws / "data" / "raw" / "existing_faults.tif"
    truth_path.parent.mkdir(parents=True)
    _write_raster(truth_path, truth, "uint8")
    folds = np.zeros((12, 12), dtype=np.int16)
    folds[:, 6:] = 1
    folds_path = ws / "data" / "processed" / "cv-spatial-v1.tif"
    folds_path.parent.mkdir(parents=True)
    _write_raster(folds_path, folds, "int16")
    run_dir = ws / "runs" / "ref-cv-spatial-full"
    (run_dir / "scores").mkdir(parents=True)
    rng = np.random.default_rng(20261026)
    for k in (0, 1):
        pred = rng.uniform(0.0, 0.05, size=(12, 12)).astype(np.float64)
        pred[2:4, 2:4] = 0.95
        pred[8:10, 8:10] = 0.60
        _write_raster(run_dir / f"fold-{k}.tif", pred.astype("float32"), "float32")
    cal = {0: 0.40, 1: 0.35}
    doc = {
        "scheme": "spatial",
        "method": "LOFO exact threshold (test fixture)",
        "folds": [
            {"fold": k, "threshold": 0.9, "raw_score": 0.1, "calibrated_score": cal[k]}
            for k in (0, 1)
        ],
        "raw_mean": 0.1,
        "calibrated_mean": float(np.mean([cal[0], cal[1]])),
    }
    (run_dir / "scores" / "lofo-spatial.json").write_text(json.dumps(doc))
    return run_dir, truth_path, folds_path


def test_rival_threshold_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute rival_threshold.sh with the exact queued s21-spatial args.

    The s21 tasks stay PARKED (see tests/test_rival_threshold.py):
    this runs the wrapper on synthetic data in a sandbox, not the
    threshold experiment on real data.
    """
    ws, bin_dir = _sandbox(tmp_path)
    run_dir, _, _ = _rival_run(ws)
    parts = shlex.split(S21_RIVAL_SPATIAL)
    assert parts[0].endswith("rival_threshold.sh"), parts
    assert parts[1:] == [
        "runs/ref-cv-spatial-full",
        "spatial",
        "data/processed/cv-spatial-v1.tif",
        "0.12296",
    ]
    args = ["scripts/queue/rival_threshold.sh", *parts[1:]]
    proc = _run_wrapper(ws, bin_dir, args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads((run_dir / "scores" / "rival_threshold-spatial.json").read_text())
    assert out["scheme"] == "spatial"
    assert out["factor"] == 0.2
    assert out["expected_score"] == 0.12296
    assert out["threshold"] == pytest.approx(0.2 * 0.12296)
    assert [row["fold"] for row in out["folds"]] == [0, 1]
    assert [row["lofo_calibrated_score"] for row in out["folds"]] == [0.40, 0.35]
    assert out["rival_mean"] == pytest.approx(
        float(np.mean([row["rival_score"] for row in out["folds"]]))
    )
    assert out["mean_delta"] == pytest.approx(
        float(np.mean([row["delta"] for row in out["folds"]]))
    )
    assert not Path("/workspace").exists()
