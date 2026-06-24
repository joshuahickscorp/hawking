#!/usr/bin/env bash
# Health monitor + OOM backstop for the detached 7B ladder.
#
# Samples disk/swap/RSS every 30s into reports/cron/7b_ladder_health.log so the run can be
# watched "from behind the glass" without attaching. EMERGENCY BACKSTOP: if free disk falls
# below FLOOR_GB the baker is SIGTERM'd — killing it collapses macOS dynamic swap instantly
# (proven 2026-06-23: 39GB→3GB on kill), which frees the disk and averts a system-wide
# ENOSPC. The ladder's _ensure_disk guard then logs that config as skipped and CONTINUES.
# This is a backstop only: with BAKE_THREADS=4 the baker resident set stays ~12GB and swap
# should never grow enough to trigger it.
#
# Self-terminates when the ladder lock disappears (run finished). Launch detached:
#   nohup tools/condense/ladder_health.sh >/dev/null 2>&1 &
set -uo pipefail
cd "$HOME/Downloads/hawking" || exit 2
# per-run paths via env (defaults = the original 7B ladder)
HEALTH="${LADDER_HEALTH:-reports/cron/7b_ladder_health.log}"
LOCK="${LADDER_LOCK:-reports/cron/7b_ladder.lock}"
FLOOR_GB=6
mkdir -p reports/cron

free_gb(){ df -g / | awk 'NR==2{print $4}'; }
swap_used_mb(){ sysctl -n vm.swapusage | awk '{print $6}' | tr -d 'M'; }

echo "# ladder health monitor start $(date)  FLOOR=${FLOOR_GB}GB" >> "$HEALTH"
# wait up to 90s for the ladder to write its lock (it spends ~10s importing torch first) so we
# don't race-exit before monitoring even begins
for _ in $(seq 1 90); do [ -f "$LOCK" ] && break; sleep 1; done
while [ -f "$LOCK" ]; do
  fg=$(free_gb); sw=$(swap_used_mb)
  bpid=$(pgrep -f 'quantize-model' | head -1)
  brss=$(ps -o rss= -p "${bpid:-0}" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
  cfg=$(tail -3 reports/cron/7b_ladder_run.log 2>/dev/null | grep -oE '[0-9]-(AWQ|RHT)|res[0-9]\+[0-9]' | tail -1)
  printf '%s diskGB=%s swapMB=%s bakerRSS_GB=%s cfg=%s\n' "$(date +%H:%M:%S)" "$fg" "$sw" "${brss:-0}" "${cfg:-?}" >> "$HEALTH"
  # emergency backstop
  if [ -n "$fg" ] && [ "$fg" -lt "$FLOOR_GB" ] 2>/dev/null; then
    if [ -n "${bpid:-}" ]; then
      echo "!! EMERGENCY $(date +%H:%M:%S) diskGB=$fg < $FLOOR_GB — SIGTERM baker $bpid (swap will collapse, config skips)" >> "$HEALTH"
      kill "$bpid" 2>/dev/null
    fi
  fi
  sleep 30
done
echo "# ladder health monitor exit $(date) (lock gone — run finished)" >> "$HEALTH"
