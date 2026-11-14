"""Tests for scripts/audit_queue_duplicates.py (step 40).

The refs audit (step 38) checks that files a queued command names exist
and the args audit (step 39) checks that a command's arguments are
consistent — neither checks whether the queue asks for the same work
twice. The runner keys on task ids with .done markers, gate.sh writes one
verdict file per gate name, and score_view.sh writes one file per
gate-dir/view, so a duplicated id silently drops work, a duplicated gate
name clobbers a verdict, and two runs scored into the same view file let
the later score silently win. These tests verify the audit catches each
duplication shape, exempts only the documented step-26 replays, stays
silent on the committed queue, and exits 0/1/2 as documented.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_queue_duplicates import audit, main

REPO = Path(__file__).resolve().parents[1]


def _write_queue(root: Path, live: list[dict], parked: list[dict] | None = None) -> None:
    queue_dir = root / "scripts" / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "tasks.json").write_text(json.dumps(live))
    (queue_dir / "tasks.parked.json").write_text(json.dumps(parked or []))


def _train(
    task_id: str,
    run: str = "runs/ref-cv-spatial-b16",
    config: str = "configs/reference_unet.yaml",
    scheme: str = "spatial",
) -> dict:
    return {
        "id": task_id,
        "cmd": (
            f"/workspace/gems/queue/train_folds.sh {run} {config} {scheme} "
            "data/processed/cv-spatial-v1.tif 16"
        ),
    }


def _score(
    task_id: str,
    run: str = "runs/ref-cv-spatial-b16",
    scheme: str = "spatial",
    gate_dir: str = "runs/gate/ref-b16",
) -> dict:
    fmap = f"data/processed/cv-{scheme}-v1.tif"
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/score_view.sh {run} {scheme} {fmap} {gate_dir}",
    }


def _record(task_id: str, run: str = "runs/ref-cv-spatial-b16") -> dict:
    return {
        "id": task_id,
        "cmd": (
            f"/workspace/gems/queue/record.sh {Path(run).name} {run} "
            'configs/reference_unet.yaml "hypothesis"'
        ),
    }


def _gate(
    task_id: str, name: str = "CAND-FOO-B16", cand_dir: str = "runs/gate/cand-foo-b16"
) -> dict:
    slug = Path(cand_dir).name.removeprefix("cand-").split("-")[0]
    return {
        "id": task_id,
        "cmd": (
            f"/workspace/gems/queue/gate.sh {name} runs/gate/ref-b16 "
            f'{cand_dir} configs/candidates/cand_{slug}.yaml "hypothesis"'
        ),
    }


def _view(task_id: str, base: str, out: str) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/endpoint_view.sh {base} {out} spatial",
    }


def _kinds(result: dict) -> list[str]:
    return [f["kind"] for f in result["findings"]]


def test_clean_queue(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_train("t1"), _score("s1"), _record("r1"), _gate("g1")])
    result = audit(root)
    assert result["findings"] == []
    assert result["notes"] == []
    assert main(["--root", str(root)]) == 0


def test_dup_task_id(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_train("t1"), _score("t1")])
    result = audit(root)
    assert _kinds(result) == ["DUP-TASK-ID"]
    assert "t1" in result["findings"][0]["ref"]
    assert main(["--root", str(root)]) == 1


def test_dup_task_id_across_live_and_parked(tmp_path: Path) -> None:
    # Parked tasks go live later; an id parked against a live id would
    # still be skipped on its .done marker when unparked.
    root = tmp_path / "repo"
    _write_queue(root, [_train("t1")], parked=[_score("t1")])
    result = audit(root)
    assert _kinds(result) == ["DUP-TASK-ID"]


def test_dup_command(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_train("t1"), _train("t2")])
    result = audit(root)
    # Same command also means the same train run dir; both are findings.
    assert "DUP-COMMAND" in _kinds(result)
    assert "DUP-TRAIN-RUN" in _kinds(result)
    assert main(["--root", str(root)]) == 1


def test_s26_replay_command_is_note_not_finding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_score("s1-score-ref-spatial"), _score("s26-rescore-ref-spatial-full")])
    result = audit(root)
    assert result["findings"] == []
    assert [n["kind"] for n in result["notes"]] == ["REPLAY", "REPLAY"]
    assert "s26-rescore-ref-spatial-full" in result["notes"][0]["source"]
    assert main(["--root", str(root)]) == 0


def test_s26_replay_does_not_hide_third_duplicate(tmp_path: Path) -> None:
    # A replay exempts itself, not a second non-replay copy of the command.
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _score("s1-score-ref-spatial"),
            _score("s26-rescore-ref-spatial-full"),
            _score("s99-score-ref-spatial"),
        ],
    )
    result = audit(root)
    assert "DUP-COMMAND" in _kinds(result)
    assert "DUP-SCORE-COVERAGE" not in _kinds(result)  # same run, not a collision


def test_dup_gate_name(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    # Same gate name, different candidate dirs: the commands differ, but
    # both verdicts land in phase2-outputs/gates/CAND-FOO-B16.json.
    _write_queue(
        root,
        [_gate("g1"), _gate("g2", cand_dir="runs/gate/cand-bar-b16")],
    )
    result = audit(root)
    assert _kinds(result) == ["DUP-GATE-NAME"]
    assert "CAND-FOO-B16" in result["findings"][0]["ref"]


def test_dup_score_coverage_different_runs(tmp_path: Path) -> None:
    # Two different runs scored into the same gate dir + view: the later
    # score silently wins, so the gate compares the wrong candidate.
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _score("s1", run="runs/cand-foo-spatial-b16", gate_dir="runs/gate/cand-foo-b16"),
            _score("s2", run="runs/cand-bar-spatial-b16", gate_dir="runs/gate/cand-foo-b16"),
        ],
    )
    result = audit(root)
    assert _kinds(result) == ["DUP-SCORE-COVERAGE"]
    assert "spatial.json" in result["findings"][0]["ref"]


def test_dup_train_run(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _train(
                "t1", run="runs/cand-foo-spatial-b16", config="configs/candidates/cand_foo.yaml"
            ),
            _train(
                "t2",
                run="runs/cand-foo-spatial-b16",
                config="configs/candidates/cand_foo.yaml",
                scheme="fault",
            ),
        ],
    )
    result = audit(root)
    # Different scheme args, so the commands differ — only the run dir collides.
    assert _kinds(result) == ["DUP-TRAIN-RUN"]


def test_dup_record_id(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    # Same run-id basename under different parent dirs: the record
    # commands differ, but both write the same run manifest.
    _write_queue(
        root,
        [
            _record("r1", run="runs/a/ref-cv-spatial-b16"),
            _record("r2", run="runs/b/ref-cv-spatial-b16"),
        ],
    )
    result = audit(root)
    assert _kinds(result) == ["DUP-RECORD-ID"]
    assert "ref-cv-spatial-b16" in result["findings"][0]["ref"]


def test_dup_view_out_different_bases(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _view("v1", "runs/ref-cv-spatial-full", "runs/endpoint-full"),
            _view("v2", "runs/cand-foo-spatial-b16", "runs/endpoint-full"),
        ],
    )
    result = audit(root)
    assert _kinds(result) == ["DUP-VIEW-OUT"]


def test_compound_command_segments_checked(tmp_path: Path) -> None:
    # A &&-chained command contributes each wrapper segment to the audit.
    root = tmp_path / "repo"
    chained = {
        "id": "c1",
        "cmd": (
            "/workspace/gems/queue/endpoint_view.sh runs/ref-cv-spatial-full "
            "runs/endpoint-full spatial && /workspace/gems/queue/score_view.sh "
            "runs/endpoint-full spatial data/processed/cv-spatial-v1.tif runs/gate/endpoint-full"
        ),
    }
    _write_queue(
        root,
        [chained, _score("s1", run="runs/other-spatial-b16", gate_dir="runs/gate/endpoint-full")],
    )
    result = audit(root)
    assert _kinds(result) == ["DUP-SCORE-COVERAGE"]


def test_unreadable_queue_exit_2(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    queue_dir = root / "scripts" / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "tasks.json").write_text("{not json")
    (queue_dir / "tasks.parked.json").write_text("[]")
    result = audit(root)
    assert _kinds(result) == ["UNREADABLE-QUEUE"]
    assert main(["--root", str(root)]) == 2


def test_main_exit_codes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_train("t1")])
    assert main(["--root", str(root)]) == 0
    _write_queue(root, [_train("t1"), _train("t1")])
    assert main(["--root", str(root)]) == 1


def test_committed_queue_is_clean() -> None:
    """Regression lock: the committed queue (129 live + 5 parked) audits
    0 findings. The only duplication is the documented step-26 replays:
    3 replayed commands (2 re-scores + 1 re-assemble) and the 2 re-score
    coverage notes those re-scores produce."""
    result = audit(REPO)
    assert result["findings"] == []
    assert result["tasks"] == 134
    assert [n["kind"] for n in result["notes"]] == ["REPLAY"] * 5
    assert sorted(n["source"] for n in result["notes"]) == [
        "live:s26-reassemble-ref-b16",
        "live:s26-rescore-ref-fault-full",
        "live:s26-rescore-ref-fault-full",
        "live:s26-rescore-ref-spatial-full",
        "live:s26-rescore-ref-spatial-full",
    ]


def test_committed_queue_via_cli() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_queue_duplicates.py"), "--root", str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "134 tasks (129 live, 5 parked), 5 replay note(s)" in proc.stdout
    assert "0 finding(s)" in proc.stdout
