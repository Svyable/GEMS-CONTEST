"""Tests for the gate wrapper's outputs portability and execution (step 44).

``gate.sh`` is the one queue wrapper the step-28 portability pass left
behind: it honored ``GEMS_WORKSPACE`` for the ``cd`` line but wrote its
verdict JSON and trial ledger to absolute box paths
(``/workspace/gems/phase2-outputs/gates``, ``/workspace/gems/phase1-outputs``),
so it could never run from a different checkout — including this VM,
which is why the decision step itself, unlike every other wrapper, had
never been executed here. Step 44 puts those paths behind a
``GEMS_OUTPUTS`` override (default ``/workspace/gems``, box-identical)
and executes the real wrapper end to end against the committed gate
dirs for the queued s5b mumford_shah_log gate.

Scope note: the execution fixture mirrors the committed tree exactly —
the incumbent dir holds trace only, the candidate dir spatial only —
so the verdict is INCONCLUSIVE with zero comparable views. That is the
finding, not a fixture artifact: it pins what the queued s5b gate
would record if the box ran it before the step-26 re-scores and the
s5b assemble fill the incumbent dir. ``record.sh`` / ``score_view.sh``
still write to absolute box output paths and are out of scope here.

Safety note: the pre-fix script is NOT executed as a negative control.
This VM's filesystem root is writable and ``/workspace`` does not
exist, so the old script would have created it rather than failing —
the defect is pinned statically instead, and the execution test
asserts ``/workspace`` still does not exist afterwards.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "queue"
GATE = QUEUE / "gate.sh"

# The exact queued s5b gate task (scripts/queue/tasks.json).
CANDIDATE = "CAND-MUMFORD_SHAH_LOG-B16"
INCUMBENT_DIR = "runs/gate/ref-b16"
CANDIDATE_DIR = "runs/gate/cand-mumford_shah_log-b16"
CONFIG = "configs/candidates/cand_mumford_shah_log.yaml"
HYPOTHESIS = (
    "Incumbent + mumford_shah_log lineament channels (buffer 16) beats "
    "the buffer-16 reference on pooled spatial with fault/trace "
    "non-regression."
)

OUTPUTS_OVERRIDE = 'OUTPUTS="${GEMS_OUTPUTS:-/workspace/gems}"'


def test_gate_uses_outputs_override() -> None:
    """The outputs root must honor GEMS_OUTPUTS with the box path as default."""
    text = GATE.read_text(encoding="utf-8")
    assert OUTPUTS_OVERRIDE in text, "gate.sh lacks the GEMS_OUTPUTS override"
    assert '"$OUTPUTS/phase2-outputs/gates"' in text
    assert '"$OUTPUTS/phase1-outputs/candidate-trials.jsonl"' in text


def test_gate_has_no_bare_box_output_paths() -> None:
    """No output path may bypass the override (the pre-fix defect).

    The only literal ``/workspace/gems`` allowed is inside the
    ``GEMS_OUTPUTS`` default assignment itself.
    """
    text = GATE.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("OUTPUTS="):
            continue
        assert "/workspace/gems/phase1-outputs" not in line, line
        assert "/workspace/gems/phase2-outputs" not in line, line


def test_gate_passes_bash_syntax_check() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(GATE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def _write_fake_uv(bin_dir: Path) -> None:
    """A stand-in for `uv` that only supports `uv run python ...`.

    Same shim as tests/test_queue_workspace.py: the box runs wrappers
    with real `uv`; this VM has none, so the shim strips the `run`
    verb and the `python` command token — both consumed by real `uv` —
    and execs python directly on the script and its args. It never
    touches the network or the real uv.
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


def test_gate_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute gate.sh itself from a non-box checkout.

    Fixture state mirrors the committed tree: the incumbent gate dir
    holds trace only (the real committed schema-2
    ``results/runs/gate/ref-b16/trace.json``), the candidate gate dir
    holds spatial only (``results/runs/gate/cand-mumford_shah_log-b16/
    spatial.json``), and the candidate config is the real committed
    file. The wrapper must exit 0, write the verdict JSON under the
    overridden outputs dir, append exactly one trial to the overridden
    ledger, and report INCONCLUSIVE — zero comparable views — with a
    reason on every view. All writes land under the tmp workspace;
    ``/workspace`` must not appear on this VM.
    """
    assert not Path("/workspace").exists(), "precondition: /workspace absent"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
    inc_gate = ws / INCUMBENT_DIR
    inc_gate.mkdir(parents=True)
    shutil.copyfile(
        REPO / "results" / "runs" / "gate" / "ref-b16" / "trace.json",
        inc_gate / "trace.json",
    )
    cand_gate = ws / CANDIDATE_DIR
    cand_gate.mkdir(parents=True)
    shutil.copyfile(
        REPO / "results" / "runs" / "gate" / "cand-mumford_shah_log-b16" / "spatial.json",
        cand_gate / "spatial.json",
    )
    config_path = ws / CONFIG
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(REPO / CONFIG, config_path)
    outputs = tmp_path / "outputs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["GEMS_WORKSPACE"] = str(ws)
    env["GEMS_OUTPUTS"] = str(outputs)
    # verify_candidate.py imports `gems`, which on the box `uv run`
    # resolves from the installed project; here the repo src dir is
    # the equivalent precondition.
    env["PYTHONPATH"] = str(REPO / "src")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    proc = subprocess.run(
        [
            "bash",
            "scripts/queue/gate.sh",
            CANDIDATE,
            INCUMBENT_DIR,
            CANDIDATE_DIR,
            CONFIG,
            HYPOTHESIS,
        ],
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    verdict_path = outputs / "phase2-outputs" / "gates" / f"{CANDIDATE}.json"
    assert verdict_path.exists(), proc.stdout + proc.stderr
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["candidate"] == CANDIDATE
    assert verdict["hypothesis"] == HYPOTHESIS
    assert verdict["config"] == CONFIG
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["trials_before"] == 0
    assert verdict["trials_after"] == 1
    assert {v["view"] for v in verdict["views"]} == {"spatial", "fault", "trace"}
    for view in verdict["views"]:
        assert view["comparable"] is False
        assert view["reason"], f"{view['view']} has no reason"

    ledger_path = outputs / "phase1-outputs" / "candidate-trials.jsonl"
    assert ledger_path.exists()
    lines = [ln for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["candidate"] == CANDIDATE
    assert entry["verdict"] == "INCONCLUSIVE"
    assert entry["trials_after"] == 1

    # The tee'd verdict is also the wrapper's stdout.
    assert '"verdict": "INCONCLUSIVE"' in proc.stdout
    assert not Path("/workspace").exists(), "gate.sh wrote outside the sandbox"
