#!/usr/bin/env bash
# Install box-side auto-start for the GEMS queue runner (step 31).
#
# The step-29 doctor can only diagnose NOT-RUNNING ("nothing
# auto-starts the runner after a reboot"); this script is the fix.
# Two mechanisms, in preference order:
#
#   1. systemd user unit — gems-queue.service (same dir as this
#      script) installed to ~/.config/systemd/user/, enabled, and
#      started. Restart=on-failure, NOT always: the runner exits 0
#      on STOPPED (a task failed — human inspection required) and on
#      ALL-DONE, and both must stay stopped; only a crash restarts.
#      `loginctl enable-linger` is attempted so the unit also starts
#      at boot without a login session (best effort: warns and
#      continues when loginctl is unavailable or refuses).
#   2. @reboot cron entry — when no user systemd manager answers.
#      Idempotent (marker comment), guarded by the runner's own flock.
#
# The runner's own flock on runner.lock means installing while a nohup
# runner is already live is safe: the service's first start exits
# immediately ("runner already active") and the unit takes over after
# that runner exits, or on the next reboot.
#
# Usage:
#   bash scripts/queue/install_autostart.sh [--dry-run] [--no-start]
#       [--uninstall] [--queue-dir DIR] [--workspace DIR]
#       [--unit-source FILE]
# Exit 0 on success, 2 when no supported mechanism is available
# (prints the manual start command instead).
set -euo pipefail

QUEUE_DIR="${QUEUE_DIR:-/workspace/gems/queue}"
WORKSPACE="${GEMS_WORKSPACE:-/workspace/gems/GEMS-CONTEST}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SOURCE="$SCRIPT_DIR/gems-queue.service"
DRY_RUN=0
NO_START=0
UNINSTALL=0

usage() {
    cat <<'EOF'
Usage: install_autostart.sh [--dry-run] [--no-start] [--uninstall]
       [--queue-dir DIR] [--workspace DIR] [--unit-source FILE]

Installs box-side auto-start for the GEMS queue runner: a systemd
user unit when a user manager answers, otherwise an @reboot cron
entry. --dry-run prints what would change and changes nothing.
--no-start installs and enables without starting now. --uninstall
removes the unit (or cron entry) again.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --no-start) NO_START=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --queue-dir) QUEUE_DIR="$2"; shift ;;
        --workspace) WORKSPACE="$2"; shift ;;
        --unit-source) UNIT_SOURCE="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

UNIT_NAME="gems-queue.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_DEST="$UNIT_DIR/$UNIT_NAME"
CRON_MARKER="# GEMS queue autostart (installed by scripts/queue/install_autostart.sh)"

say() { echo "[install_autostart] $*"; }

# Escape a value for use as a sed replacement with | as delimiter.
sed_escape() { printf '%s' "$1" | sed -e 's/[&|\\]/\\&/g'; }

systemd_available() {
    command -v systemctl >/dev/null 2>&1 \
        && systemctl --user show-environment >/dev/null 2>&1
}

# Render the unit file with the chosen queue/workspace paths.
render_unit() {
    local q_esc w_esc
    q_esc="$(sed_escape "$QUEUE_DIR")"
    w_esc="$(sed_escape "$WORKSPACE")"
    sed -e "s|^Environment=QUEUE_DIR=.*|Environment=QUEUE_DIR=$q_esc|" \
        -e "s|^Environment=GEMS_WORKSPACE=.*|Environment=GEMS_WORKSPACE=$w_esc|" \
        -e "s|^ExecStart=.*runner.py|ExecStart=/usr/bin/env python3 $q_esc/runner.py|" \
        -e "s|^StandardOutput=append:.*|StandardOutput=append:$q_esc/queue.log|" \
        -e "s|^StandardError=append:.*|StandardError=append:$q_esc/queue.log|" \
        "$UNIT_SOURCE"
}

cron_line() {
    printf '@reboot sleep 30; QUEUE_DIR=%s GEMS_WORKSPACE=%s nohup python3 %s/runner.py >> %s/queue.log 2>&1 &' \
        "$QUEUE_DIR" "$WORKSPACE" "$QUEUE_DIR" "$QUEUE_DIR"
}

install_systemd() {
    if [ "$DRY_RUN" -eq 1 ]; then
        say "dry-run: would install $UNIT_SOURCE -> $UNIT_DEST"
        say "dry-run: would run: systemctl --user daemon-reload; systemctl --user enable $UNIT_NAME"
        [ "$NO_START" -eq 0 ] && say "dry-run: would run: systemctl --user start $UNIT_NAME"
        say "dry-run: would attempt: loginctl enable-linger $USER"
        return 0
    fi
    mkdir -p "$UNIT_DIR"
    local tmp rendered
    tmp="$(mktemp)"
    render_unit > "$tmp"
    if [ -f "$UNIT_DEST" ] && cmp -s "$tmp" "$UNIT_DEST"; then
        say "unit already installed and identical: $UNIT_DEST"
        rm -f "$tmp"
    else
        cp "$tmp" "$UNIT_DEST"
        rm -f "$tmp"
        say "installed unit: $UNIT_DEST"
    fi
    systemctl --user daemon-reload
    systemctl --user enable "$UNIT_NAME"
    say "enabled $UNIT_NAME (starts at boot / login)"
    if command -v loginctl >/dev/null 2>&1; then
        if loginctl enable-linger "$USER" 2>/dev/null; then
            say "linger enabled: unit starts at boot without a login session"
        else
            say "warning: loginctl enable-linger failed — the unit starts at login, not at boot, until linger is enabled"
        fi
    else
        say "warning: no loginctl — the unit starts at login, not at boot, until linger is enabled"
    fi
    if [ "$NO_START" -eq 0 ]; then
        systemctl --user start "$UNIT_NAME" || true
        if systemctl --user is-active --quiet "$UNIT_NAME"; then
            say "runner is now managed by systemd (systemctl --user status $UNIT_NAME)"
        else
            say "service not active after start — a nohup runner is probably already live (flock), so the unit takes over after it exits or on reboot"
        fi
    fi
}

uninstall_systemd() {
    if [ "$DRY_RUN" -eq 1 ]; then
        say "dry-run: would run: systemctl --user disable --now $UNIT_NAME; rm -f $UNIT_DEST; systemctl --user daemon-reload"
        return 0
    fi
    systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
    rm -f "$UNIT_DEST"
    systemctl --user daemon-reload 2>/dev/null || true
    say "removed $UNIT_DEST"
}

install_cron() {
    local line
    line="$(cron_line)"
    if [ "$DRY_RUN" -eq 1 ]; then
        say "dry-run: would ensure this @reboot line in the crontab:"
        say "dry-run: $CRON_MARKER"
        say "dry-run: $line"
        return 0
    fi
    local current tmp
    current="$(mktemp)"; tmp="$(mktemp)"
    crontab -l 2>/dev/null > "$current" || true
    # Drop any previous GEMS autostart lines so re-runs and path changes
    # never accumulate duplicates.
    grep -v -F "$CRON_MARKER" "$current" | grep -v -F "/runner.py" > "$tmp" || true
    printf '%s\n%s\n' "$CRON_MARKER" "$line" >> "$tmp"
    crontab "$tmp"
    rm -f "$current" "$tmp"
    say "installed @reboot cron entry (runner starts 30s after boot)"
}

uninstall_cron() {
    if [ "$DRY_RUN" -eq 1 ]; then
        say "dry-run: would remove the GEMS @reboot line from the crontab"
        return 0
    fi
    local current tmp
    current="$(mktemp)"; tmp="$(mktemp)"
    crontab -l 2>/dev/null > "$current" || true
    grep -v -F "$CRON_MARKER" "$current" | grep -v -F "/runner.py" > "$tmp" || true
    crontab "$tmp"
    rm -f "$current" "$tmp"
    say "removed the GEMS @reboot cron entry"
}

if [ "$UNINSTALL" -eq 1 ]; then
    if systemd_available; then
        uninstall_systemd
    elif command -v crontab >/dev/null 2>&1; then
        uninstall_cron
    else
        say "nothing to uninstall: no systemd user manager and no crontab"
    fi
    exit 0
fi

if [ ! -f "$UNIT_SOURCE" ]; then
    echo "unit file not found: $UNIT_SOURCE" >&2
    exit 2
fi

if systemd_available; then
    say "systemd user manager available — installing user unit"
    install_systemd
elif command -v crontab >/dev/null 2>&1; then
    say "no systemd user manager — falling back to @reboot cron entry"
    install_cron
else
    cat >&2 <<EOF
No systemd user manager and no crontab on this host, so auto-start
cannot be installed here. Start the runner manually after each reboot:
  nohup python3 $QUEUE_DIR/runner.py >> $QUEUE_DIR/queue.log 2>&1 &
EOF
    exit 2
fi
