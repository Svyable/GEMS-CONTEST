"""Tests for the bounded train memory gate (step 41).

The train wrapper's memory gate waited unboundedly for MemAvailable to
reach GEMS_MIN_AVAIL_MB, so a memory-hogging job elsewhere on the shared
box stalled the queue silently and indefinitely — the doctor could only
diagnose RUNNING-WAIT-MEM. Step 41 bounds the wait at
GEMS_WAIT_MEM_TIMEOUT_S (default 21600s = 6h): past the bound the fold
gives up with exit 3, so the runner records a .failed marker and stops
for inspection instead of stalling forever. GEMS_MEMINFO and
GEMS_WAIT_MEM_SLEEP_S are test hooks (defaults /proc/meminfo and 60s);
box behavior is unchanged except for the bound.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "scripts" / "queue" / "train_folds.sh"

LOW_MEMINFO = "MemTotal:       16384000 kB\nMemAvailable:    1000000 kB\n"
HIGH_MEMINFO = "MemTotal:       16384000 kB\nMemAvailable:   99999999 kB\n"


def _run_wrapper(
    tmp_path: Path, meminfo: str, timeout_s: int, sleep_s: int = 1
) -> subprocess.CompletedProcess[str]:
    """Run train_folds.sh against a fake meminfo in a tmp workspace."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text(meminfo, encoding="utf-8")
    env = dict(os.environ)
    env["GEMS_WORKSPACE"] = str(ws)
    env["GEMS_MEMINFO"] = str(meminfo_path)
    env["GEMS_WAIT_MEM_TIMEOUT_S"] = str(timeout_s)
    env["GEMS_WAIT_MEM_SLEEP_S"] = str(sleep_s)
    return subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "runs/mem-gate-test",
            "configs/reference_unet.yaml",
            "spatial",
            "cv-spatial-v1.tif",
            "16",
        ],
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_mem_gate_times_out_loudly(tmp_path: Path) -> None:
    """Low memory + zero timeout fails fast with exit 3, not a hang."""
    proc = _run_wrapper(tmp_path, LOW_MEMINFO, timeout_s=0)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "WAIT-MEM-TIMEOUT fold 0" in proc.stdout
    assert "limit 0s" in proc.stdout
    run_dir = tmp_path / "ws" / "runs" / "mem-gate-test"
    assert not (run_dir / "fold-0.json").exists()
    assert not (run_dir / "fold-0.tif").exists()


def test_mem_gate_waits_up_to_timeout(tmp_path: Path) -> None:
    """The gate actually waits out the bound (~2s here), not 0s or 60s."""
    start = time.monotonic()
    proc = _run_wrapper(tmp_path, LOW_MEMINFO, timeout_s=2, sleep_s=1)
    elapsed = time.monotonic() - start
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "WAIT-MEM fold 0" in proc.stdout
    assert "WAIT-MEM-TIMEOUT fold 0" in proc.stdout
    assert 1.0 <= elapsed < 15, f"elapsed {elapsed:.2f}s is not ~2s"


def test_mem_gate_missing_memavailable_line(tmp_path: Path) -> None:
    """A meminfo with no MemAvailable line waits (as 0MB), never crashes."""
    proc = _run_wrapper(tmp_path, "MemTotal:       16384000 kB\n", timeout_s=0)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "WAIT-MEM-TIMEOUT fold 0" in proc.stdout
    assert "integer expression expected" not in proc.stderr


def test_mem_gate_passes_when_memory_plentiful(tmp_path: Path) -> None:
    """Plenty of memory skips the wait and starts the fold (rc != 3)."""
    proc = _run_wrapper(tmp_path, HIGH_MEMINFO, timeout_s=0)
    assert "WAIT-MEM-TIMEOUT" not in proc.stdout
    assert "START runs/mem-gate-test fold 0" in proc.stdout
    # The fold itself fails on this VM (no uv / no competition data) —
    # the point is the gate passed, so the exit is the train failure,
    # never the memory-timeout code 3.
    assert proc.returncode != 3, proc.stdout + proc.stderr


def test_wrapper_declares_timeout_defaults() -> None:
    """Box defaults are the old behavior plus the 6h bound, nothing else."""
    text = WRAPPER.read_text(encoding="utf-8")
    assert "GEMS_MEMINFO:-/proc/meminfo" in text
    assert "GEMS_WAIT_MEM_TIMEOUT_S:-21600" in text
    assert "GEMS_WAIT_MEM_SLEEP_S:-60" in text
    assert "exit 3" in text
    assert "WAIT-MEM-TIMEOUT" in text
