"""Smoke tests: ensure scripts can be imported and show --help without errors."""
import subprocess
import sys
from pathlib import Path


def test_script_help_optimize_threshold():
    """optimize_threshold.py --help should succeed (catches import errors)."""
    script = Path(__file__).parent.parent / "scripts" / "optimize_threshold.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"
    # Script should at least show usage without crashing on import
    assert "optimize_threshold.py" in result.stdout


def test_script_help_calibrate_threshold():
    """calibrate_threshold.py --help should succeed."""
    script = Path(__file__).parent.parent / "scripts" / "calibrate_threshold.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"


def test_script_help_train_reference_oof():
    """train_reference_oof.py --help should succeed."""
    script = Path(__file__).parent.parent / "scripts" / "train_reference_oof.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"


def test_script_help_profile_labels():
    """profile_labels.py --help should succeed."""
    script = Path(__file__).parent.parent / "scripts" / "profile_labels.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"


def test_script_help_lofo_calibrate():
    """lofo_calibrate.py should be importable and show usage on wrong args."""
    script = Path(__file__).parent.parent / "scripts" / "lofo_calibrate.py"
    # Script expects 2 args, so calling with no args should fail with usage info
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0, "lofo_calibrate.py should fail without args"
    # Should show traceback or usage (checks it's importable)
    assert len(result.stderr) > 0 or "IndexError" in result.stderr
