"""Tests for scripts/audit_queue_args.py (step 39).

The refs audit (step 38) checks that files a queued command names still
exist; it cannot catch arguments that are all real files but wrong for
each other — a trace train on the superseded pre-repair fold map, a
buffer-16 run dir fed buffer 32, a gate hypothesis mangled by `\"`
escapes. Any of those fails (or mis-records) on the box only after
resumption, where the runner stops the whole queue on the first
non-zero exit. These tests verify the audit catches each argument shape,
stays silent on the committed queue, and exits 0/1/2 as documented.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_queue_args import audit, check_command, main

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
    fmap: str = "data/processed/cv-spatial-v1.tif",
    buf: str = "16",
    seed: str | None = None,
) -> dict:
    cmd = f"/workspace/gems/queue/train_folds.sh {run} {config} {scheme} {fmap} {buf}"
    if seed is not None:
        cmd += f" {seed}"
    return {"id": task_id, "cmd": cmd}


def _score(
    task_id: str,
    run: str = "runs/ref-cv-spatial-b16",
    scheme: str = "spatial",
    fmap: str = "data/processed/cv-spatial-v1.tif",
    gate: str = "runs/gate/ref-b16",
) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/score_view.sh {run} {scheme} {fmap} {gate}",
    }


def _record(
    task_id: str,
    run_id: str = "ref-cv-spatial-b16",
    run: str = "runs/ref-cv-spatial-b16",
    config: str = "configs/reference_unet.yaml",
) -> dict:
    return {
        "id": task_id,
        "cmd": f'/workspace/gems/queue/record.sh {run_id} {run} {config} "hyp" "notes"',
    }


def _gate(
    task_id: str,
    name: str = "CAND-FOO-B16",
    cand_gate: str = "runs/gate/cand-foo-b16",
    config: str = "configs/candidates/cand_foo.yaml",
) -> dict:
    return {
        "id": task_id,
        "cmd": f"/workspace/gems/queue/gate.sh {name} runs/gate/ref-b16 {cand_gate} "
        f'{config} "hypothesis"',
    }


def _kinds(result: dict) -> list[str]:
    return [f["kind"] for f in result["findings"]]


def test_clean_queue_passes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _train("t1"),
            _train("t2", run="runs/ref-cv-spatial-b16-seed20260923", seed="20260923"),
            _train(
                "t3",
                run="runs/cand-foo-trace-b16",
                config="configs/candidates/cand_foo.yaml",
                scheme="trace",
                fmap="data/processed/cv-trace-v1b.tif",
            ),
            _score("s1"),
            _score(
                "s2",
                run="runs/cand-foo-trace-b16",
                scheme="trace",
                fmap="data/processed/cv-trace-v1b.tif",
                gate="runs/gate/cand-foo-b16",
            ),
            _record("r1"),
            _gate("g1"),
            {
                "id": "c1",
                "cmd": "/workspace/gems/queue/compare_seed.sh runs/gate/ref-b16 "
                "runs/gate/cand-foo-b16 runs/ref-cv runs/cand-foo "
                '"seed probe note, ignored by the wrapper"',
            },
            {
                "id": "v1",
                "cmd": "/workspace/gems/queue/rival_threshold.sh "
                "runs/ref-cv-trace-b16 trace data/processed/cv-trace-v1b.tif 0.01856",
            },
        ],
    )
    result = audit(root)
    assert result["findings"] == []
    assert result["tasks"] == 9
    assert result["segments"] == 9
    assert result["unparsed"] == []


def test_stale_trace_fold_map_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _train(
                "t1",
                run="runs/cand-foo-trace-b16",
                config="configs/candidates/cand_foo.yaml",
                scheme="trace",
                fmap="data/processed/cv-trace-v1.tif",
            ),
            _score(
                "s1",
                run="runs/cand-foo-trace-b16",
                scheme="trace",
                fmap="data/processed/cv-trace-v1.tif",
                gate="runs/gate/cand-foo-b16",
            ),
        ],
    )
    result = audit(root)
    assert _kinds(result) == ["STALE-FOLD-MAP", "STALE-FOLD-MAP"]
    assert "cv-trace-v1b.tif" in result["findings"][0]["ref"]
    # The pre-repair map is the defect only for trace: v1 is current
    # for spatial and fault.
    _write_queue(
        root,
        [
            _train("t1"),
            _train(
                "t2",
                run="runs/ref-cv-fault-b16",
                scheme="fault",
                fmap="data/processed/cv-fault-v1.tif",
            ),
        ],
    )
    assert audit(root)["findings"] == []


def test_bad_scheme_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_train("t1", scheme="spatia")])
    result = audit(root)
    assert _kinds(result) == ["BAD-SCHEME"]


def test_buffer_mismatch_and_bad_buffer_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _train("t1", run="runs/ref-cv-spatial-b32", buf="16"),
            _train("t2", buf="xx"),
        ],
    )
    result = audit(root)
    assert _kinds(result) == ["BUFFER-SUFFIX-MISMATCH", "BAD-BUFFER"]


def test_bad_seed_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_train("t1", seed="2026")])
    assert _kinds(audit(root)) == ["BAD-SEED"]


def test_config_dir_mismatch_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _train("t1", run="runs/cand-foo-spatial-b16", config="configs/reference_unet.yaml"),
            _train("t2", config="configs/candidates/cand_foo.yaml"),
        ],
    )
    assert _kinds(audit(root)) == ["CONFIG-DIR-MISMATCH", "CONFIG-DIR-MISMATCH"]


def test_record_id_mismatch_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(root, [_record("r1", run_id="ref-cv-spatial-b32")])
    assert _kinds(audit(root)) == ["RECORD-ID-MISMATCH"]


def test_gate_name_and_buffer_mismatch_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _gate("g1", name="CAND-FOO"),
            _gate("g2", cand_gate="runs/gate/cand-foo-b32"),
            _gate("g3", name="CAND-FOO-TRACE", cand_gate="runs/gate/cand-foo-b32"),
        ],
    )
    result = audit(root)
    # g1: bad name shape; g2: name says B16, dir says b32; g3: TRACE
    # names carry no buffer suffix, so any b16/b32 dir is fine.
    assert _kinds(result) == ["BAD-GATE-NAME", "GATE-DIR-MISMATCH"]


def test_escaped_quote_flagged_and_args_still_checked() -> None:
    findings: list[dict] = []
    cmd = (
        "/workspace/gems/queue/gate.sh CAND-FOO-B16 runs/gate/ref-b16 "
        "runs/gate/cand-other-b16 configs/candidates/cand_other.yaml "
        '\\"Reference U-Net recipe beats the incumbent.\\"'
    )
    assert check_command("live:g1", cmd, findings) == 1
    assert [f["kind"] for f in findings] == ["ESCAPED-QUOTE", "GATE-DIR-MISMATCH"]
    # The escaped-quote form is exactly what the s10/s11 gate tasks
    # carried; the plain-quote form is clean.
    findings = []
    assert check_command("live:g2", cmd.replace('\\"', '"'), findings) == 1
    assert [f["kind"] for f in findings] == ["GATE-DIR-MISMATCH"]


def test_quoted_hypothesis_never_read_as_args(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            _record("r1"),
            {
                "id": "r2",
                "cmd": "/workspace/gems/queue/record.sh ref-cv-spatial-b16 "
                "runs/ref-cv-spatial-b16 configs/reference_unet.yaml "
                '"mentions configs/fake.yaml and runs/gate/nope and a bogus scheme" "notes"',
            },
            _gate("g1"),
        ],
    )
    assert audit(root)["findings"] == []


def test_compound_command_checks_each_segment(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            {
                "id": "e1",
                "cmd": "/workspace/gems/queue/endpoint_view.sh runs/ref-cv-spatial-full "
                "runs/endpoint-cv-spatial-full spatial && /workspace/gems/queue/score_view.sh "
                "runs/endpoint-cv-spatial-full spatia data/processed/cv-spatial-v1.tif "
                "runs/gate/endpoint-full",
            }
        ],
    )
    result = audit(root)
    assert _kinds(result) == ["BAD-SCHEME"]
    assert result["segments"] == 2


def test_rival_threshold_expected_score_checked(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            {
                "id": "v1",
                "cmd": "/workspace/gems/queue/rival_threshold.sh "
                "runs/ref-cv-spatial-b16 spatial data/processed/cv-spatial-v1.tif 1.5",
            },
            {
                "id": "v2",
                "cmd": "/workspace/gems/queue/rival_threshold.sh "
                "runs/ref-cv-spatial-b16 spatial data/processed/cv-spatial-v1.tif not-a-number",
            },
        ],
    )
    assert _kinds(audit(root)) == ["BAD-EXPECTED-SCORE", "BAD-EXPECTED-SCORE"]


def test_arg_count_flagged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            {
                "id": "t1",
                "cmd": "/workspace/gems/queue/train_folds.sh runs/x "
                "configs/reference_unet.yaml spatial",
            },
        ],
    )
    assert _kinds(audit(root)) == ["ARG-COUNT"]


def test_unparsed_wait_and_assemble_tasks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_queue(
        root,
        [
            {"id": "w1", "cmd": "while kill -0 1234 2>/dev/null; do sleep 30; done"},
            {
                "id": "a1",
                "cmd": "mkdir -p runs/gate/ref-b16 && cp runs/gate/ref-full/spatial.json "
                "runs/gate/ref-b16/ && ls runs/gate/ref-b16",
            },
            _train("t1"),
        ],
    )
    result = audit(root)
    assert result["findings"] == []
    assert result["unparsed"] == ["live:w1", "live:a1"]
    assert main(["--root", str(root)]) == 0


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
    _write_queue(root, [_train("t1", scheme="spatia")])
    assert main(["--root", str(root)]) == 1


def test_committed_queue_is_clean() -> None:
    """Regression lock: after the step-39 queue fix, the committed queue
    (129 live + 5 parked) audits 0 findings — 136 wrapper segments, with
    only the wait loop and the 3 assemble tasks unparsed."""
    result = audit(REPO)
    assert result["findings"] == []
    assert result["tasks"] == 134
    assert result["segments"] == 136
    assert result["unparsed"] == [
        "live:s1-wait-existing-spatial-loop",
        "live:s26-reassemble-ref-b16",
        "live:s5b-assemble-ref-b16",
        "live:s7-assemble-ref-b16",
    ]


def test_committed_queue_via_cli() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_queue_args.py"), "--root", str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "134 tasks (129 live, 5 parked), 136 wrapper segments checked" in proc.stdout
    assert "0 finding(s)" in proc.stdout
