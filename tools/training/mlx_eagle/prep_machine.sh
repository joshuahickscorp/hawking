#!/usr/bin/env bash
# prep_machine.sh — operational checklist before long training runs on M3 Pro.
#
# Run BEFORE starting a multi-hour MLX training session. Most steps are
# free wins (Spotlight off on the data dir, memory purge, plug in laptop).
# A few require sudo and are flagged.
#
# Reverts: there's a `--undo` flag that re-enables Spotlight on the data dir.

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRAINING_DATA_DIR="$PROJECT_ROOT/training_data"

UNDO=0
if [ "${1:-}" = "--undo" ]; then UNDO=1; fi

ts() { date "+%H:%M:%S"; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

if [ "$UNDO" -eq 1 ]; then
  log "RE-ENABLING Spotlight on $TRAINING_DATA_DIR"
  sudo mdutil -i on "$TRAINING_DATA_DIR" || true
  log "done."
  exit 0
fi

log "=== M3 Pro training prep ==="
log

# 1. Spotlight off on data dir
if [ -d "$TRAINING_DATA_DIR" ]; then
  log "1. Disabling Spotlight indexing on $TRAINING_DATA_DIR (saves CPU)"
  if sudo -n true 2>/dev/null; then
    sudo mdutil -i off "$TRAINING_DATA_DIR" 2>&1 | sed 's/^/   /'
  else
    log "   (needs sudo: 'sudo mdutil -i off $TRAINING_DATA_DIR')"
  fi
else
  log "1. SKIP: $TRAINING_DATA_DIR not present"
fi
log

# 2. Memory purge
log "2. Purging inactive memory pages (clean unified-memory start)"
if sudo -n true 2>/dev/null; then
  sudo purge 2>&1 | sed 's/^/   /'
  log "   done"
else
  log "   (needs sudo: 'sudo purge')"
fi
log

# 3. Disable Power Nap during training
log "3. Power Nap status (recommend disable during training):"
pmset -g | grep -E "powernap|sleep" | sed 's/^/   /'
log "   To disable: 'sudo pmset -a powernap 0'"
log

# 4. Power-source check
log "4. Power source:"
pmset -g batt | head -1 | sed 's/^/   /'
log "   (training should be on AC, not battery)"
log

# 5. Thermal sensor read
log "5. Current thermal state:"
if command -v powermetrics >/dev/null 2>&1; then
  log "   (powermetrics requires sudo; skipping detailed read)"
  log "   tip: 'sudo powermetrics -i 1000 -n 1 --samplers smc' for fan + temp"
else
  log "   powermetrics not available"
fi
log

# 6. Background processes worth killing
log "6. Other heavy processes consuming GPU/CPU (kill if not needed):"
ps -axc -o pid,pcpu,pmem,comm | sort -nrk2 | head -10 | sed 's/^/   /'
log

# 7. Verify capture is or isn't running (depending on intent)
log "7. dismantle capture-hidden process:"
if pgrep -f "dismantle capture-hidden" >/dev/null; then
  pgrep -lf "dismantle capture-hidden" | sed 's/^/   ALIVE: /'
  log "   ⚠️  capture is running — it will compete with MLX training for GPU"
  log "   to pause: 'pkill -TERM -f \"dismantle capture-hidden\"'"
  log "   to resume later: re-run with --resume flag"
else
  log "   not running — safe to start MLX training"
fi
log

log "=== done. Operational tips:"
log "  • plug in laptop (clamshell mode is fine)"
log "  • close non-essential apps (Chrome, Slack, IDE)"
log "  • disable Spotlight on the data dir (done above if sudo)"
log "  • leave the lid open OR use a fan stand — thermal throttling is real"
log "  • disable Power Nap during the run"
log
log "Undo Spotlight change later: $0 --undo"
