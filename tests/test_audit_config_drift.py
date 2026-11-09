"""Config-drift audit tests (step 35).

The audit compares the config hash each committed result records
against the config file on disk, so a config edited after a run was
scored is caught before the box trains the next run under the new
protocol and its gate refuses the comparison as protocol-unequal.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_config_drift import audit, main

REPO = Path(__file__).parent.parent


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_run(root: Path, run: str, config: str, sha: str, *, manifest: bool = True) -> None:
    run_dir = root / "results" / "runs" / run
    (run_dir / "scores").mkdir(parents=True)
    if manifest:
        (run_dir / "manifest.json").write_text(
            json.dumps({"config": {"path": config, "sha256": sha}}), encoding="utf-8"
        )
    (run_dir / "scores" / "spatial.json").write_text(
        json.dumps({"evaluation_protocol": {"config_sha256": sha, "schema_version": 2}}),
        encoding="utf-8",
    )


def _write_tasks(root: Path, tasks: list[dict]) -> None:
    queue_dir = root / "scripts" / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")


def _base_tree(tmp_path: Path, config_text: str = "epochs: 5\n") -> tuple[str, str]:
    """A minimal repo: one config, one train task, one recorded run."""
    config = "configs/reference_unet.yaml"
    (tmp_path / "configs").mkdir()
    (tmp_path / config).write_text(config_text, encoding="utf-8")
    _write_tasks(
        tmp_path,
        [
            {
                "id": "s1-train-ref-spatial",
                "cmd": "/workspace/gems/queue/train_folds.sh runs/ref-cv-spatial-b16 "
                f"{config} spatial data/processed/cv-spatial-v1.tif 16",
            }
        ],
    )
    sha = _sha(config_text)
    _write_run(tmp_path, "ref-cv-spatial-b16", config, sha)
    return config, sha


def test_clean_run_matches(tmp_path: Path) -> None:
    _base_tree(tmp_path)
    report = audit(tmp_path)
    assert report["findings"] == 0
    assert report["checked"] == 1
    assert report["runs"][0]["status"] == "OK"


def test_drift_detected_and_names_trains(tmp_path: Path) -> None:
    config, sha = _base_tree(tmp_path)
    # The config file changed after the run was recorded.
    (tmp_path / config).write_text("epochs: 10\n", encoding="utf-8")
    report = audit(tmp_path)
    assert report["findings"] == 1
    entry = report["runs"][0]
    assert entry["status"] == "DRIFT"
    assert sha[:12] in entry["detail"]
    assert "s1-train-ref-spatial" in entry["detail"]
    assert main(["--root", str(tmp_path)]) == 1


def test_manifest_queue_mismatch(tmp_path: Path) -> None:
    config, _sha = _base_tree(tmp_path)
    other = "configs/other.yaml"
    (tmp_path / other).write_text("epochs: 5\n", encoding="utf-8")
    _write_tasks(
        tmp_path,
        [
            {
                "id": "s1-train-ref-spatial",
                "cmd": f"/workspace/gems/queue/train_folds.sh runs/ref-cv-spatial-b16 {other} "
                "spatial data/processed/cv-spatial-v1.tif 16",
            }
        ],
    )
    report = audit(tmp_path)
    assert report["findings"] == 1
    assert report["runs"][0]["status"] == "CONFIG-MISMATCH"
    assert config in report["runs"][0]["detail"]


def test_missing_config_file(tmp_path: Path) -> None:
    config, _ = _base_tree(tmp_path)
    (tmp_path / config).unlink()
    report = audit(tmp_path)
    assert report["findings"] == 1
    assert report["runs"][0]["status"] == "CONFIG-MISSING"


def test_mixed_recorded_hashes(tmp_path: Path) -> None:
    _config, sha = _base_tree(tmp_path)
    run_dir = tmp_path / "results" / "runs" / "ref-cv-spatial-b16"
    (run_dir / "fold-0.json").write_text(
        json.dumps({"config_sha256": _sha("something else")}), encoding="utf-8"
    )
    report = audit(tmp_path)
    assert report["findings"] == 1
    assert report["runs"][0]["status"] == "MIXED"
    assert sha[:12] in report["runs"][0]["detail"]


def test_schema1_score_without_manifest_is_note(tmp_path: Path) -> None:
    _write_tasks(tmp_path, [])
    run_dir = tmp_path / "results" / "runs" / "ref-cv-spatial-full"
    (run_dir).mkdir(parents=True)
    (run_dir / "spatial.json").write_text(
        json.dumps({"evaluation_protocol": {"schema_version": 1}}), encoding="utf-8"
    )
    report = audit(tmp_path)
    assert report["findings"] == 0
    assert report["notes"] == 1
    assert report["runs"][0]["status"] == "NO-RECORDED-HASH"


def test_scored_run_without_train_task_is_note(tmp_path: Path) -> None:
    _write_tasks(tmp_path, [])
    _write_run(
        tmp_path, "ref-cv-spatial-b16", "configs/reference_unet.yaml", _sha("x\n"), manifest=False
    )
    report = audit(tmp_path)
    assert report["findings"] == 0
    assert report["notes"] == 1
    assert report["runs"][0]["status"] == "NO-TRAIN-TASK"


def test_malformed_tasks_json_exits_2(tmp_path: Path) -> None:
    queue_dir = tmp_path / "scripts" / "queue"
    queue_dir.mkdir(parents=True)
    (queue_dir / "tasks.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        audit(tmp_path)
    assert main(["--root", str(tmp_path)]) == 2


def test_gate_dir_copies_excluded(tmp_path: Path) -> None:
    _base_tree(tmp_path)
    gate_dir = tmp_path / "results" / "runs" / "gate" / "ref-b16"
    gate_dir.mkdir(parents=True)
    gate_dir.joinpath("trace.json").write_text(
        json.dumps({"evaluation_protocol": {"config_sha256": _sha("stale copy")}}),
        encoding="utf-8",
    )
    report = audit(tmp_path)
    assert report["checked"] == 1  # the gate copy is not a run
    assert report["findings"] == 0


def test_real_repo_no_drift() -> None:
    """Lock the committed record: 7 runs carry a recorded config hash
    (6 manifests + ref-cv-fault-b32 via scores/folds) and every one
    still matches the config file on disk; no manifest/queue mismatch.
    Update when the box commits new scored runs or a config changes.
    """
    report = audit(REPO)
    assert report["findings"] == 0, report["runs"]
    checked = {entry["run"] for entry in report["runs"]}
    assert {
        "ref-cv-spatial-b32",
        "ref-cv-fault-b32",
        "ref-cv-trace-b16",
        "ref-cv-spatial-full",
        "ref-cv-fault-full",
        "ref-cv-trace-full",
        "cand-mumford_shah_log-spatial-b16",
    } <= checked
    assert all(entry["status"] == "OK" for entry in report["runs"])
