"""Tests for queue auto-start (step 31).

The step-29 doctor can only diagnose NOT-RUNNING ("nothing
auto-starts the runner after a reboot"); step 31 ships the fix:
``scripts/queue/gems-queue.service`` (a systemd user unit) plus
``scripts/queue/install_autostart.sh`` (systemd install, @reboot cron
fallback, dry-run, uninstall). The box is unreachable from this VM,
so the installer is verified here with stubbed ``systemctl`` /
``crontab`` / ``loginctl`` on a hermetic PATH and a temp HOME — the
step-28 fake-``uv`` idiom. No real systemd unit or crontab is ever
touched.
"""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "queue"
UNIT = QUEUE / "gems-queue.service"
INSTALLER = QUEUE / "install_autostart.sh"
BASH = shutil.which("bash") or "/bin/bash"

DEFAULT_QUEUE = "/workspace/gems/queue"
DEFAULT_WORKSPACE = "/workspace/gems/GEMS-CONTEST"


def _parse_unit(path: Path) -> configparser.ConfigParser:
    # strict=False: systemd units legally repeat keys (Environment=
    # appears twice here); configparser would raise DuplicateOptionError.
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # keep key case: ExecStart, not execstart
    parser.read(path, encoding="utf-8")
    return parser


def test_unit_file_sections_and_restart_policy() -> None:
    """The unit must be a valid systemd unit with on-failure restart.

    Restart=on-failure, not always, is the load-bearing choice: the
    runner exits 0 on STOPPED (a task failed — human inspection must
    precede any resume) and on ALL-DONE, and both must stay stopped.
    Only a crash (non-zero exit) restarts the runner.
    """
    parser = _parse_unit(UNIT)
    assert set(parser.sections()) == {"Unit", "Service", "Install"}
    service = parser["Service"]
    assert service["Restart"] == "on-failure"
    assert service["Restart"] != "always"
    assert int(service["RestartSec"]) >= 30
    assert service["ExecStart"].endswith(f"{DEFAULT_QUEUE}/runner.py")
    assert service["StandardOutput"] == f"append:{DEFAULT_QUEUE}/queue.log"
    assert service["StandardError"] == f"append:{DEFAULT_QUEUE}/queue.log"
    assert parser["Install"]["WantedBy"] == "default.target"
    # Environment= repeats, so configparser keeps only the last one;
    # check both lines in the raw text instead.
    text = UNIT.read_text(encoding="utf-8")
    assert f"Environment=QUEUE_DIR={DEFAULT_QUEUE}" in text
    assert f"Environment=GEMS_WORKSPACE={DEFAULT_WORKSPACE}" in text


def test_installer_passes_bash_syntax_check() -> None:
    proc = subprocess.run([BASH, "-n", str(INSTALLER)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def _hermetic_bin(tmp_path: Path, *, stubs: dict[str, str]) -> Path:
    """A PATH dir with the host tools except systemctl/crontab/loginctl.

    Symlinks every executable from /usr/bin and /bin other than the
    three the installer probes, then adds the given stub scripts, so
    the test never sees (or touches) the real systemd/cron.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for src_dir in ("/usr/bin", "/bin"):
        for entry in Path(src_dir).iterdir():
            if entry.name in ("systemctl", "crontab", "loginctl"):
                continue
            if entry.is_file() and os.access(entry, os.X_OK):
                link = bin_dir / entry.name
                if not link.exists():
                    link.symlink_to(entry)
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    return bin_dir


def _systemctl_stub(stub_dir: Path) -> str:
    # The installer always calls `systemctl --user ...`: shift the
    # --user flag off before dispatching, or no branch ever matches
    # and the fallthrough exit 0 masks real behavior.
    return (
        "#!/bin/sh\n"
        f'echo "$@" >> "{stub_dir}/systemctl.log"\n'
        'if [ "$1" = "--user" ]; then shift; fi\n'
        'case "$1" in\n'
        "  show-environment) exit ${STUB_SYSTEMCTL_FAIL_SHOW:-0} ;;\n"
        f'  start) touch "{stub_dir}/active"; exit 0 ;;\n'
        f'  disable) rm -f "{stub_dir}/active"; exit 0 ;;\n'
        f'  is-active) [ -f "{stub_dir}/active" ]; exit $? ;;\n'
        "  daemon-reload|enable) exit 0 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )


def _loginctl_stub(stub_dir: Path) -> str:
    return f'#!/bin/sh\necho "$@" >> "{stub_dir}/loginctl.log"\nexit 0\n'


def _crontab_stub(stub_dir: Path) -> str:
    return (
        "#!/bin/sh\n"
        f'echo "$@" >> "{stub_dir}/crontab.log"\n'
        f'store="{stub_dir}/crontab.store"\n'
        'if [ "$1" = "-l" ]; then\n'
        '  if [ -f "$store" ]; then cat "$store"; exit 0; else exit 1; fi\n'
        "fi\n"
        'if [ "$1" = "-r" ]; then rm -f "$store"; exit 0; fi\n'
        'cp "$1" "$store"\n'
        "exit 0\n"
    )


def _installer_env(tmp_path: Path, bin_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(exist_ok=True)
    env.pop("XDG_CONFIG_HOME", None)
    env["PATH"] = f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin"
    env["USER"] = "testuser"
    env["STUB_SYSTEMCTL_FAIL_SHOW"] = "0"
    return env


def _run_installer(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(INSTALLER), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_systemd_install_writes_unit_enables_and_starts(tmp_path: Path) -> None:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    bin_dir = _hermetic_bin(
        tmp_path,
        stubs={
            "systemctl": _systemctl_stub(stub_dir),
            "loginctl": _loginctl_stub(stub_dir),
        },
    )
    env = _installer_env(tmp_path, bin_dir)

    proc = _run_installer(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    unit_dest = tmp_path / "home" / ".config" / "systemd" / "user" / "gems-queue.service"
    assert unit_dest.exists()
    installed = _parse_unit(unit_dest)
    assert installed["Service"]["ExecStart"].endswith(f"{DEFAULT_QUEUE}/runner.py")

    systemctl_log = (stub_dir / "systemctl.log").read_text(encoding="utf-8")
    assert "--user daemon-reload" in systemctl_log
    assert "--user enable gems-queue.service" in systemctl_log
    assert "--user start gems-queue.service" in systemctl_log
    assert "enable-linger testuser" in (stub_dir / "loginctl.log").read_text(encoding="utf-8")
    assert "managed by systemd" in proc.stdout

    # Second run is idempotent: same bytes, no duplicate unit.
    before = unit_dest.read_bytes()
    proc = _run_installer(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert unit_dest.read_bytes() == before
    assert "already installed and identical" in proc.stdout


def test_systemd_install_respects_queue_dir_and_workspace(tmp_path: Path) -> None:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    bin_dir = _hermetic_bin(
        tmp_path,
        stubs={
            "systemctl": _systemctl_stub(stub_dir),
            "loginctl": _loginctl_stub(stub_dir),
        },
    )
    env = _installer_env(tmp_path, bin_dir)
    proc = _run_installer(env, "--queue-dir", "/tmp/q", "--workspace", "/tmp/w", "--no-start")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    unit_dest = tmp_path / "home" / ".config" / "systemd" / "user" / "gems-queue.service"
    text = unit_dest.read_text(encoding="utf-8")
    assert "Environment=QUEUE_DIR=/tmp/q" in text
    assert "Environment=GEMS_WORKSPACE=/tmp/w" in text
    assert "ExecStart=/usr/bin/env python3 /tmp/q/runner.py" in text
    assert "StandardOutput=append:/tmp/q/queue.log" in text
    assert "start gems-queue.service" not in (stub_dir / "systemctl.log").read_text(
        encoding="utf-8"
    )


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    bin_dir = _hermetic_bin(
        tmp_path,
        stubs={
            "systemctl": _systemctl_stub(stub_dir),
            "loginctl": _loginctl_stub(stub_dir),
            "crontab": _crontab_stub(stub_dir),
        },
    )
    env = _installer_env(tmp_path, bin_dir)
    proc = _run_installer(env, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dry-run" in proc.stdout
    assert not (tmp_path / "home" / ".config").exists()
    assert not (stub_dir / "crontab.store").exists()
    systemctl_log = (stub_dir / "systemctl.log").read_text(encoding="utf-8")
    assert "daemon-reload" not in systemctl_log
    assert "enable" not in systemctl_log
    assert "start" not in systemctl_log


def test_cron_fallback_installs_single_reboot_line(tmp_path: Path) -> None:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    # systemctl present but its user manager does not answer, so the
    # installer must fall back to cron rather than failing.
    bin_dir = _hermetic_bin(
        tmp_path,
        stubs={
            "systemctl": _systemctl_stub(stub_dir),
            "crontab": _crontab_stub(stub_dir),
        },
    )
    env = _installer_env(tmp_path, bin_dir)
    env["STUB_SYSTEMCTL_FAIL_SHOW"] = "1"

    for _ in range(2):  # run twice: the entry must not duplicate
        proc = _run_installer(env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "falling back to @reboot cron entry" in proc.stdout

    store = (stub_dir / "crontab.store").read_text(encoding="utf-8")
    lines = [ln for ln in store.splitlines() if ln.strip()]
    assert len(lines) == 2, store
    assert lines[0].startswith("# GEMS queue autostart")
    assert lines[1].startswith("@reboot sleep 30;")
    assert f"{DEFAULT_QUEUE}/runner.py" in lines[1]
    assert f">> {DEFAULT_QUEUE}/queue.log 2>&1 &" in lines[1]
    assert not (tmp_path / "home" / ".config").exists()


def test_uninstall_removes_unit_and_cron_entry(tmp_path: Path) -> None:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    bin_dir = _hermetic_bin(
        tmp_path,
        stubs={
            "systemctl": _systemctl_stub(stub_dir),
            "loginctl": _loginctl_stub(stub_dir),
            "crontab": _crontab_stub(stub_dir),
        },
    )
    env = _installer_env(tmp_path, bin_dir)
    assert _run_installer(env).returncode == 0
    unit_dest = tmp_path / "home" / ".config" / "systemd" / "user" / "gems-queue.service"
    assert unit_dest.exists()

    proc = _run_installer(env, "--uninstall")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not unit_dest.exists()
    assert "--user disable --now gems-queue.service" in (stub_dir / "systemctl.log").read_text(
        encoding="utf-8"
    )

    # Cron path: force the fallback, install, then uninstall.
    env["STUB_SYSTEMCTL_FAIL_SHOW"] = "1"
    assert _run_installer(env).returncode == 0
    assert (stub_dir / "crontab.store").exists()
    proc = _run_installer(env, "--uninstall")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    store = (stub_dir / "crontab.store").read_text(encoding="utf-8")
    assert "GEMS queue autostart" not in store
    assert "runner.py" not in store


def test_no_mechanism_exits_2_with_manual_command(tmp_path: Path) -> None:
    # Neither systemctl nor crontab nor loginctl on PATH: the installer
    # must fail loudly with the manual start command, not silently.
    bin_dir = _hermetic_bin(tmp_path, stubs={})
    env = _installer_env(tmp_path, bin_dir)
    proc = _run_installer(env)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "nohup python3" in proc.stderr
    assert f"{DEFAULT_QUEUE}/runner.py" in proc.stderr
    assert not (tmp_path / "home" / ".config").exists()
