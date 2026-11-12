"""Tests for scripts/audit_queue_refs.py (step 38).

The box runner stops on the first failing task, so a queued command that
names a file the repo no longer has — a config renamed after the task was
queued, a wrapper moved, a script deleted — fails on the box, not here.
These tests verify the audit catches each reference shape missing, stays
silent on the committed queue, and exits 0/1/2 as documented.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_queue_refs import audit, extract_refs, main, render

REPO = Path(__file__).resolve().parents[1]


def _write_repo(root: Path, live: list[dict], parked: list[dict] | None = None) -> None:
    queue_dir = root / "scripts" / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "tasks.json").write_text(json.dumps(live))
    (queue_dir / "tasks.parked.json").write_text(json.dumps(parked or []))
    (queue_dir / "train_folds.sh").write_text("uv run python scripts/train_full_map.py\n")
    (queue_dir / "score_view.sh").write_text("uv run python scripts/score_cv.py\n")
    (queue_dir / "record.sh").write_text("uv run python scripts/record_run.py\n")
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "reference_unet.yaml").write_text("loss: tversky\n")
    (root / "configs" / "cand.yaml").write_text("loss: focal\n")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    for name in ("train_full_map.py", "score_cv.py", "record_run.py"):
        (root / "scripts" / name).write_text("# stub\n")


def _train_task(task_id: str, config: str = "configs/reference_unet.yaml") -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/train_folds.sh runs/r1 {config} spatial "
        "data/processed/cv-spatial-v1.tif 16",
    }


def _record_task(task_id: str, config: str = "configs/cand.yaml") -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/record.sh cand-x runs/x {config} "hyp" "notes"',
    }


def test_missing_config_is_finding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root, [_train_task("t1", "configs/gone.yaml")])
    result = audit(root)
    assert result["findings"] == [
        {"source": "live:t1", "kind": "MISSING-REF", "ref": "configs/gone.yaml"}
    ]
    assert "MISSING-REF live:t1: configs/gone.yaml" in render(result)
    assert main(["--root", str(root)]) == 1


def test_missing_wrapper_is_finding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    task = {
        "id": "t1",
        "cmd": "/workspace/gems/queue/train_folds_v2.sh runs/r1 "
        "configs/reference_unet.yaml spatial data/processed/cv-spatial-v1.tif 16",
    }
    _write_repo(root, [task])
    result = audit(root)
    assert result["findings"] == [
        {"source": "live:t1", "kind": "MISSING-REF", "ref": "queue/train_folds_v2.sh"}
    ]


def test_missing_script_ref_in_task_is_finding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    task = {
        "id": "t1",
        "cmd": "uv run python scripts/score_missing.py runs/r1 spatial",
    }
    _write_repo(root, [task])
    result = audit(root)
    assert result["findings"] == [
        {"source": "live:t1", "kind": "MISSING-REF", "ref": "scripts/score_missing.py"}
    ]


def test_parked_tasks_checked_same_way(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root, [_train_task("t1")], [_train_task("p1", "configs/gone.yaml")])
    result = audit(root)
    assert result["findings"] == [
        {"source": "parked:p1", "kind": "MISSING-REF", "ref": "configs/gone.yaml"}
    ]
    assert result["per_file"]["parked"]["tasks"] == 1


def test_wrapper_calling_missing_script_is_finding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root, [_train_task("t1")])
    (root / "scripts" / "queue" / "score_view.sh").write_text(
        "uv run python scripts/score_vanished.py\n"
    )
    result = audit(root)
    assert result["findings"] == [
        {
            "source": "wrapper:score_view.sh",
            "kind": "MISSING-REF",
            "ref": "scripts/score_vanished.py",
        }
    ]


def test_quoted_hypothesis_is_not_a_ref(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    task = {
        "id": "r1",
        "cmd": "/workspace/gems/queue/record.sh cand-x runs/x configs/cand.yaml "
        '"see configs/gone.yaml for the old loss" "notes"',
    }
    _write_repo(root, [task])
    result = audit(root)
    assert result["findings"] == []
    assert result["refs"] == 2  # record.sh wrapper + configs/cand.yaml


def test_extract_refs_ignores_data_and_runs_paths() -> None:
    refs = extract_refs(
        "/workspace/gems/queue/score_view.sh runs/r1 spatial "
        "data/processed/cv-spatial-v1.tif runs/gate/cand"
    )
    assert refs == {"wrappers": ["score_view.sh"], "configs": [], "scripts": []}


def test_wait_and_assemble_tasks_are_unparsed_notes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    tasks = [
        {"id": "s1-wait", "cmd": "while kill -0 1234 2>/dev/null; do sleep 30; done"},
        {
            "id": "s5b-assemble",
            "cmd": "mkdir -p runs/gate/ref-b16 && cp runs/gate/x/spatial.json runs/gate/ref-b16/",
        },
        _train_task("t1"),
    ]
    _write_repo(root, tasks)
    result = audit(root)
    assert result["findings"] == []
    assert result["unparsed"] == ["live:s1-wait", "live:s5b-assemble"]
    assert "OK:" in render(result)


def test_clean_fabricated_repo_exits_zero(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_repo(root, [_train_task("t1"), _record_task("r1")])
    result = audit(root)
    assert result["tasks"] == 2
    assert result["findings"] == []
    assert main(["--root", str(root)]) == 0


def test_committed_queue_has_zero_missing_refs() -> None:
    """Regression lock: every file the real 134-task queue names exists."""
    result = audit(REPO)
    assert result["tasks"] == 134
    assert result["per_file"]["live"]["tasks"] == 129
    assert result["per_file"]["parked"]["tasks"] == 5
    assert result["refs"] == 218
    assert result["wrapper_refs"] == 9
    assert result["findings"] == []
    assert result["unparsed"] == [
        "live:s1-wait-existing-spatial-loop",
        "live:s26-reassemble-ref-b16",
        "live:s5b-assemble-ref-b16",
        "live:s7-assemble-ref-b16",
    ]


def test_main_exit_codes(tmp_path: Path) -> None:
    # Clean fabricated repo -> exit 0.
    root = tmp_path / "repo"
    _write_repo(root, [_train_task("t1")])
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_queue_refs.py"), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "0 MISSING-REF(s)" in proc.stdout
    # Missing config -> exit 1.
    _write_repo(root, [_train_task("t1", "configs/gone.yaml")])
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_queue_refs.py"), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "MISSING-REF live:t1: configs/gone.yaml" in proc.stdout
    # Malformed tasks.json -> exit 2.
    (root / "scripts" / "queue" / "tasks.json").write_text("{not json")
    assert main(["--root", str(root)]) == 2


def test_main_on_real_repo_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_queue_refs.py"), "--root", str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "134 tasks" in proc.stdout
    assert "0 MISSING-REF(s), 0 other finding(s)" in proc.stdout
