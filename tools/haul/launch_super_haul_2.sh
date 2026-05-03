#!/usr/bin/env bash
# Single-command launcher for Phase-2 super-haul-1.
#
# Usage:   bash tools/haul/launch_super_haul_2.sh
#
# Why this script exists:
#   The multi-line `nohup env ... \\\n ./tools/haul/coexist.sh ...` invocation
#   breaks under some terminal apps' paste handling — line continuations
#   get mangled, the shell ends up in multi-line input mode, and bash
#   shows `>` continuation prompts that look like the haul is "stuck"
#   when really it's the interactive shell waiting for closing quotes.
#
#   This script collapses everything into one self-contained file. The
#   user runs ONE short command; bash never sees a multi-line paste; no
#   stdin reaches the haul; output goes to /tmp/super_haul_2.log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG=/tmp/super_haul_2.log
ATTEMPT_LOG=/tmp/super_haul_2_attempt$(date -u +%Y%m%dT%H%M%SZ).log

cd "$REPO_ROOT"

# Hand off to coexist.sh fully detached: nohup survives terminal SIGHUP,
# `< /dev/null` ensures the haul cannot read from any tty (so terminal
# typos can never reach it), `&` backgrounds it, `disown` removes it
# from the shell's job table.
nohup env \
    SLM_PID="" \
    HAUL=1 \
    PER_VALIDATOR_TIMEOUT_S=2400 \
    HAUL_COOLDOWN_S=0 \
    ./tools/haul/coexist.sh launch phase2 \
    > "$LOG" 2>&1 < /dev/null &

HAUL_PID=$!
disown

# Symlink for the timestamped log so old runs are preserved.
cp "$LOG" "$ATTEMPT_LOG" 2>/dev/null || true
ln -sf "$LOG" "$ATTEMPT_LOG.live" 2>/dev/null || true

cat <<EOF
─────────────────────────────────────────────────────────────────
super-haul-2 launched (detached, slm-aware)

  pid       : $HAUL_PID
  log (live): $LOG
  log (snap): $ATTEMPT_LOG (snapshot at launch)

watch progress (safe to Ctrl-C — only kills the tail):
  tail -f $LOG

check completion:
  tail -20 $LOG | head     # last 20 lines

stop the haul (only if needed):
  pkill -f 'coexist.sh launch phase2'
  pkill -f 'run-gates.sh phase2'

The haul is fully detached. Closing this terminal, accidental
typing, paste mishaps — none of it affects the running haul.

Expected wall clock: ~30-60 min depending on slm contention.
Expected outcome: HALT at W3.4 with [perf_below_threshold] —
that IS the success state for this haul; the dec_tps numbers
in _evidence/W3.{1,2,3}/result.json are the data point.
─────────────────────────────────────────────────────────────────
EOF
