"""Tests for scripts/rival_threshold.py and the parked s21 experiment tasks.

The threshold experiment (rival closed-form rule vs LOFO-calibrated cutoffs)
is UNAPPROVED: it must be fully specified and tested here, but its queue
tasks must stay parked in tasks.parked.json until Sven green-lights it.
These tests enforce both halves — the script is correct, and nothing in the
live queue can run it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rival_threshold.py"
PARKED = json.loads((REPO / "scripts" / "queue" / "tasks.parked.json").read_text())
LIVE = json.loads((REPO / "scripts" / "queue" / "tasks.json").read_text())

SPEC = importlib.util.spec_from_file_location("rival_threshold", SCRIPT)
rival = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rival)


def parked_s21() -> list[dict]:
    return [t for t in PARKED if t["id"].startswith("s21-")]


def test_rival_threshold_formula() -> None:
    assert rival.rival_threshold(0.12296) == pytest.approx(0.2 * 0.12296)
    assert rival.rival_threshold(0.12296, 0.5) == pytest.approx(0.5 * 0.12296)
    assert rival.rival_threshold(0.05469) == pytest.approx(0.010938)
    assert rival.rival_threshold(0.01856) == pytest.approx(0.003712)


def test_three_parked_tasks_one_per_view() -> None:
    assert len(parked_s21()) == 3
    schemes = sorted(t["id"].split("-")[3] for t in parked_s21())
    assert schemes == ["fault", "spatial", "trace"]
    for task in parked_s21():
        assert task["step"] == 21
        assert "PARKED" in task["note"] and "go-ahead" in task["note"]
        assert "no gate" in task["note"]


def test_parked_tasks_match_committed_lofo() -> None:
    """Each parked task pins its view's committed LOFO-calibrated mean.

    The expected score is the committed mean rounded to 5 decimals (the
    script derives the threshold from the passed value; the rounding moves
    it by < 1e-6). The run dir must be the run that carries that LOFO file.
    """
    for task in parked_s21():
        m = re.fullmatch(
            r"/workspace/gems/queue/rival_threshold\.sh "
            r"(\S+) (\S+) (\S+) (\S+)",
            task["cmd"],
        )
        assert m, f"{task['id']}: unparsable cmd: {task['cmd']}"
        run_dir, scheme, fold_map, expected = m.groups()
        # Committed runs live under results/; the queue's box-side run dir
        # (runs/...) is the same run without the results/ prefix.
        lofo = rival.find_lofo_file(REPO / "results" / run_dir, scheme)
        doc = json.loads(lofo.read_text())
        assert doc["scheme"] == scheme
        assert float(expected) == round(doc["calibrated_mean"], 5), (
            f"{task['id']}: expected-score {expected} != "
            f"round(committed LOFO mean {doc['calibrated_mean']}, 5)"
        )
        # The fold map must be the one the reference run was scored with
        # (the data file itself lives only on the box — never committed).
        live_maps = {}
        for live in LIVE:
            sm = re.search(r"score_view\.sh (\S+) (\S+) (\S+)", live["cmd"])
            if sm:
                live_maps[(sm.group(1), sm.group(2))] = sm.group(3)
        assert live_maps.get((run_dir, scheme)) == fold_map, (
            f"{task['id']}: fold map {fold_map} != the live score task's "
            f"{live_maps.get((run_dir, scheme))}"
        )
        assert (REPO / "scripts" / "queue" / "rival_threshold.sh").exists()


def test_experiment_tasks_parked_not_live() -> None:
    """The unapproved experiment cannot run on the box: no s21 task is live."""
    live_ids = {t["id"] for t in LIVE}
    assert not any(t["id"].startswith("s21-") for t in LIVE)
    parked_ids = [t["id"] for t in PARKED]
    assert len(parked_ids) == len(set(parked_ids)), "duplicate parked task ids"
    assert not (set(parked_ids) & live_ids), "parked id collides with a live task"
    wrapper = (REPO / "scripts" / "queue" / "rival_threshold.sh").read_text()
    assert "scripts/rival_threshold.py" in wrapper
    assert "--expected-score" in wrapper


def run_script(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


def test_cli_rejects_bad_inputs() -> None:
    run = REPO / "results" / "runs" / "ref-cv-spatial-full"
    assert run.is_dir()
    # Non-positive expected score / factor, and a threshold above 1.
    for extra in (
        ["--expected-score", "0"],
        ["--expected-score", "-0.1"],
        ["--expected-score", "0.12296", "--factor", "0"],
        ["--expected-score", "6.0"],
    ):
        result = run_script(str(run), "spatial", "data/processed/cv-spatial-v1.tif", *extra)
        assert result.returncode == 2, (extra, result.stdout, result.stderr)
    # No LOFO file for a run dir that has none.
    result = run_script(
        "no-such-run",
        "spatial",
        "data/processed/cv-spatial-v1.tif",
        "--expected-score",
        "0.12296",
    )
    assert result.returncode == 2, (result.stdout, result.stderr)


def write_raster(path: Path, array: np.ndarray, dtype: str) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def synthetic_run(tmp_path: Path, lofo_folds: list[int]) -> tuple[Path, Path, Path]:
    """Two-fold spatial fixture: truth, fold map, fold predictions, LOFO file."""
    truth = np.zeros((12, 12), dtype=np.uint8)
    truth[2:4, 2:4] = 1
    truth[8:10, 8:10] = 1
    write_raster(tmp_path / "truth.tif", truth, "uint8")
    folds = np.zeros((12, 12), dtype=np.int16)
    folds[:, 6:] = 1
    write_raster(tmp_path / "folds.tif", folds, "int16")
    run_dir = tmp_path / "run"
    (run_dir / "scores").mkdir(parents=True)
    rng = np.random.default_rng(20261026)
    for k in (0, 1):
        pred = rng.uniform(0.0, 0.05, size=(12, 12)).astype(np.float64)
        pred[2:4, 2:4] = 0.95  # fold 0's truth block, high confidence
        pred[8:10, 8:10] = 0.60  # fold 1's truth block, mid confidence
        write_raster(run_dir / f"fold-{k}.tif", pred.astype("float32"), "float32")
    cal = {0: 0.40, 1: 0.35}
    doc = {
        "scheme": "spatial",
        "method": "LOFO exact threshold (test fixture)",
        "folds": [
            {"fold": k, "threshold": 0.9, "raw_score": 0.1, "calibrated_score": cal[k]}
            for k in lofo_folds
        ],
        "raw_mean": 0.1,
        "calibrated_mean": float(np.mean([cal[k] for k in lofo_folds])),
    }
    (run_dir / "scores" / "lofo-spatial.json").write_text(json.dumps(doc))
    return run_dir, tmp_path / "truth.tif", tmp_path / "folds.tif"


def expected_rival_scores(
    run_dir: Path, truth_path: Path, folds_path: Path, threshold: float
) -> dict[int, float]:
    """Independent recomputation of the per-fold rival scores."""
    from gems.calibration import binary_threshold
    from gems.metric import distance_weighted_tversky

    with rasterio.open(truth_path) as t:
        truth = t.read(1) > 0
        valid = t.dataset_mask() > 0
    folds = rasterio.open(folds_path).read(1).astype(np.int16)
    out = {}
    for k in (0, 1):
        with rasterio.open(run_dir / f"fold-{k}.tif") as s:
            pred = s.read(1).astype(np.float64)
        mask = valid & (folds == k) & np.isfinite(pred)
        out[k] = distance_weighted_tversky(
            binary_threshold(pred, threshold), truth, valid_mask=mask
        )
    return out


def test_synthetic_end_to_end(tmp_path: Path) -> None:
    run_dir, truth_path, folds_path = synthetic_run(tmp_path, [0, 1])
    result = run_script(
        str(run_dir),
        "spatial",
        str(folds_path),
        "--expected-score",
        "0.12296",
        "--truth",
        str(truth_path),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    out = json.loads((run_dir / "scores" / "rival_threshold-spatial.json").read_text())
    assert out["scheme"] == "spatial"
    assert out["factor"] == 0.2
    assert out["expected_score"] == 0.12296
    assert out["threshold"] == 0.2 * 0.12296
    assert out["lofo_file"].endswith("scores/lofo-spatial.json")
    expected = expected_rival_scores(run_dir, truth_path, folds_path, out["threshold"])
    for row in out["folds"]:
        k = row["fold"]
        assert row["rival_score"] == expected[k]
        assert row["lofo_calibrated_score"] == {0: 0.40, 1: 0.35}[k]
        assert row["delta"] == row["rival_score"] - row["lofo_calibrated_score"]
    assert out["rival_mean"] == float(np.mean([expected[0], expected[1]]))
    assert out["mean_delta"] == float(np.mean([expected[0] - 0.40, expected[1] - 0.35]))


def test_factor_plumbing_changes_threshold(tmp_path: Path) -> None:
    run_dir, truth_path, folds_path = synthetic_run(tmp_path, [0, 1])
    result = run_script(
        str(run_dir),
        "spatial",
        str(folds_path),
        "--expected-score",
        "0.12296",
        "--factor",
        "0.5",
        "--truth",
        str(truth_path),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    out = json.loads((run_dir / "scores" / "rival_threshold-spatial.json").read_text())
    assert out["threshold"] == 0.5 * 0.12296


def test_lofo_fold_mismatch_is_hard_error(tmp_path: Path) -> None:
    run_dir, truth_path, folds_path = synthetic_run(tmp_path, [0])  # LOFO lacks fold 1
    result = run_script(
        str(run_dir),
        "spatial",
        str(folds_path),
        "--expected-score",
        "0.12296",
        "--truth",
        str(truth_path),
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "folds" in result.stderr
