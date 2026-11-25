"""Tests for the committed sprint queue's freshness (step 50).

Step 48 generated ``scripts/queue/tasks.sprint.json`` and locked the
regenerated order in ``test_queue_sprint.py`` — but nothing compared the
*committed* file against a fresh regeneration. The sprint file is now the
live plan (the full drain flipped INFEASIBLE on 2026-11-24), so a queue
edit that does not regenerate it would leave a stale file committed and
the box-side resumption refused on 2026-11-28. These tests fail in CI the
moment the committed sprint drifts from what the live queue regenerates,
and re-run both queue audits against the committed file itself.

Step 51: the committed file is now the *chained* sprint (``--chain`` —
every pending decision cheapest-first after s7), so regeneration passes
``--chain``. The chain's first 21 tasks are byte-identical to the
step-48 single-decision sprint prefix.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / "scripts" / "queue" / "tasks.sprint.json"
PARKED = REPO / "scripts" / "queue" / "tasks.parked.json"
SCRIPT = REPO / "scripts" / "queue_sprint.py"
ORDER_AUDIT = REPO / "scripts" / "queue_order_audit.py"
DUPS_AUDIT = REPO / "scripts" / "audit_queue_duplicates.py"


def test_committed_sprint_matches_regeneration(tmp_path: Path) -> None:
    out = tmp_path / "tasks.sprint.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(REPO), "--chain", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # Full task dicts, not just ids: a regenerated sprint that reorders
    # or rewrites a command must fail here, not just on the box.
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
        SPRINT.read_text(encoding="utf-8")
    )


def _audit_root_with_committed_sprint(tmp_path: Path) -> Path:
    root = tmp_path / "audit-root"
    (root / "scripts" / "queue").mkdir(parents=True)
    (root / "scripts" / "queue" / "tasks.json").write_text(SPRINT.read_text(encoding="utf-8"))
    (root / "scripts" / "queue" / "tasks.parked.json").write_text(
        PARKED.read_text(encoding="utf-8")
    )
    return root


def test_committed_sprint_passes_order_audit(tmp_path: Path) -> None:
    root = _audit_root_with_committed_sprint(tmp_path)
    result = subprocess.run(
        [sys.executable, str(ORDER_AUDIT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 ORDER-VIOLATION(s)" in result.stdout


def test_committed_sprint_passes_duplicates_audit(tmp_path: Path) -> None:
    root = _audit_root_with_committed_sprint(tmp_path)
    result = subprocess.run(
        [sys.executable, str(DUPS_AUDIT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 finding(s)" in result.stdout
