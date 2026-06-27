#!/usr/bin/env bash
# OS-LEVEL resurrection heartbeat — the watcher of the watchers.
#
# Every supervisor (ladder, keepalive, conductor, caffeinate, verifier) is a userland process.
# If the chain breaks — OOM-kill, crash, accidental kill, sleep, reboot — nothing userland brings
# it back, so an overnight failure stays dead. This script is invoked by a launchd LaunchAgent on
# a fixed timer (managed by PID 1), so it runs regardless of any terminal/IDE/login state. If the
# ladder OR the keepalive is down it relaunches the whole stack (run_7b_frontier.sh is idempotent:
# it adopts a live run and re-arms any missing supervisor). The only thing that stops this loop is
# the machine being powered off — exactly the requested guarantee.
set -uo pipefail
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd "$HOME/Downloads/hawking" || exit 2
mkdir -p reports/cron
LOG="$HOME/Library/Application Support/hawking/cronguard.log"
LOCK=reports/cron/7b_frontier.lock

# FDA probe: a launchd agent can't write to ~/Downloads until Full Disk Access is granted to
# /bin/bash. Until then, stay quiet and do nothing (the userland keepalive — which HAS permission
# — is the active guard). Once FDA is granted this probe passes and full resurrection switches on.
if ! ( : > reports/cron/.cronguard_probe ) 2>/dev/null; then
  echo "$(date '+%F %T') awaiting Full Disk Access for /bin/bash (no ~/Downloads access yet)" \
    >> "$HOME/Library/Application Support/hawking/cronguard.log"
  exit 0
fi
rm -f reports/cron/.cronguard_probe 2>/dev/null

_alive() { [ -f "$LOCK" ] || return 1; local p; p=$(cat "$LOCK" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
_ka_alive() { pgrep -f frontier_keepalive.sh >/dev/null 2>&1; }

# don't relaunch a run that genuinely finished (all configs done, autopilot dry)
_truly_done() {
  tail -120 reports/cron/7b_frontier_run.log 2>/dev/null | grep -q '# done ->' \
    && [ ! -f reports/cron/7b_frontier_inject.py ] \
    && ! _ka_alive
}

if _alive && _ka_alive; then
  echo "$(date '+%F %T') ok: frontier(pid $(cat "$LOCK" 2>/dev/null)) + keepalive alive" >> "$LOG"
elif _truly_done; then
  echo "$(date '+%F %T') done: frontier finished, autopilot dry — not relaunching" >> "$LOG"
else
  echo "$(date '+%F %T') RESURRECT: frontier/keepalive down — relaunching full stack" >> "$LOG"
  ./run_7b_frontier.sh >> "$LOG" 2>&1 || true
fi
