"""Tests for queue-wrapper workspace portability (step 28).

``train_folds.sh`` already ran from any checkout via the
``GEMS_WORKSPACE`` override (``${GEMS_WORKSPACE:-/workspace/gems/GEMS-CONTEST}``),
but the other eight queue wrappers hardcoded ``cd /workspace/gems/GEMS-CONTEST``,
so none of them could run from a different checkout — including this VM,
which is why step 27 could only test the s12 compare wrapper's guarded
copies, never execute the wrapper itself. Step 28 generalizes the
``train_folds.sh`` idiom to every wrapper with a ``cd`` line. The default
is byte-identical to the old hardcoded path, so box behavior is
unchanged; only the checkout location becomes overridable.

Scope boundary, stated so it is not mistaken for an oversight: this
changes only the ``cd`` line. ``gate.sh`` / ``record.sh`` / ``score_view.sh``
still write to absolute box output paths (``phase1-outputs``,
``phase2-outputs``, the box-side ``queue/`` dir) because those outputs
live outside the repo on the box by design; the compare wrappers, which
use only repo-relative paths after ``cd``, are the fully portable ones.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "queue"

# Every wrapper with a cd line, including the train_folds.sh precedent.
WRAPPERS = (
    "compare_schedule.sh",
    "compare_seed.sh",
    "curved_view.sh",
    "endpoint_view.sh",
    "gate.sh",
    "record.sh",
    "rival_threshold.sh",
    "score_view.sh",
    "train_folds.sh",
)

OVERRIDE = 'WORKSPACE="${GEMS_WORKSPACE:-/workspace/gems/GEMS-CONTEST}"'
CD = 'cd "$WORKSPACE"'


@pytest.mark.parametrize("name", WRAPPERS)
def test_wrapper_uses_workspace_override(name: str) -> None:
    """The cd line must honor GEMS_WORKSPACE with the box path as default."""
    text = (QUEUE / name).read_text(encoding="utf-8")
    assert OVERRIDE in text, f"{name} lacks the GEMS_WORKSPACE override"
    assert CD in text, f"{name} does not cd to $WORKSPACE"
    bare = [ln for ln in text.splitlines() if ln.strip() == "cd /workspace/gems/GEMS-CONTEST"]
    assert not bare, f"{name} still hardcodes its checkout path: {bare}"


@pytest.mark.parametrize("name", WRAPPERS)
def test_wrapper_passes_bash_syntax_check(name: str) -> None:
    proc = subprocess.run(
        ["bash", "-n", str(QUEUE / name)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def _write_fake_uv(bin_dir: Path) -> None:
    """A stand-in for `uv` that only supports `uv run python ...`.

    The box runs wrappers with real `uv`; this VM has none, so the
    end-to-end test below shims it with the test interpreter (which has
    the repo's venv deps). The shim strips the `run` verb and the
    `python` command token — both consumed by real `uv` — and execs
    python directly on the script and its args. It never touches the
    network or the real uv.
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


def test_compare_schedule_wrapper_runs_outside_box_path(tmp_path: Path) -> None:
    """Execute compare_schedule.sh itself from a non-box checkout.

    Fixture state mirrors the committed tree after step 26: the
    incumbent gate dir holds trace only (the real committed
    schema-2 ``results/runs/gate/ref-b16/trace.json``), the candidate
    gate dir is empty, and no LOFO summaries exist. The wrapper must
    exit 0, stage only the incumbent trace JSON, and report SKIPPED
    for all three views — the guarded-copy path step 27 tested only
    by static assertion, now exercised end to end. All writes land
    under the tmp workspace, never under the real repo.
    """
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
    inc_gate = ws / "runs" / "gate" / "ref-b16"
    inc_gate.mkdir(parents=True)
    shutil.copyfile(
        REPO / "results" / "runs" / "gate" / "ref-b16" / "trace.json",
        inc_gate / "trace.json",
    )
    (ws / "runs" / "gate" / "cand-longer_schedule-b16").mkdir(parents=True)
    (ws / "runs" / "ref-cv").mkdir(parents=True)
    (ws / "runs" / "cand-longer_schedule").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["GEMS_WORKSPACE"] = str(ws)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    proc = subprocess.run(
        [
            "bash",
            "scripts/queue/compare_schedule.sh",
            "runs/gate/ref-b16",
            "runs/gate/cand-longer_schedule-b16",
            "runs/ref-cv",
            "runs/cand-longer_schedule",
        ],
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cmp_dir = ws / "runs" / "compare" / "longer_schedule-b16"
    log = (cmp_dir / "compare.log").read_text(encoding="utf-8")
    for view in ("spatial", "fault", "trace"):
        assert f"{view}: SKIPPED (missing gate JSON on at least one side)" in log
    assert (cmp_dir / "inc-raw" / "trace.json").exists()
    assert not list((cmp_dir / "cand-raw").iterdir())
    assert not list((cmp_dir / "inc-lofo").iterdir())
    assert not list((cmp_dir / "cand-lofo").iterdir())
