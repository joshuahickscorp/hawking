#!/usr/bin/env bash
# Odyssey-I one-tick driver: reap finished lanes, then launch next (self-gated).
# Launchd template: workspace/campaign/odyssey/com.hawking.odyssey.plist
# Do not load that plist from this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# launchd runs with a minimal environment: restore HOME + a full PATH so grok-run
# can find `grok`, and git/date/python3 resolve.
export HOME="${HOME:-/Users/scammermike}"
export PATH="$HOME/.grok/bin:$HOME/.claude-grok/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

PY="${ODYSSEY_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
if [ ! -x "$PY" ]; then
  PY=python3
fi

LOG="$ROOT/workspace/campaign/odyssey/driver.log"
mkdir -p "$(dirname "$LOG")"
ts="$(date '+%Y-%m-%dT%H:%M:%S%z')"

{
  echo "== odyssey driver tick $ts =="
  echo "repo=$ROOT py=$PY"
  # --max-lanes = MODEL concurrency ceiling (memgate is the real limiter under it);
  # --grok-lanes 0 = NO grok in the autonomous loop (deterministic science only,
  # zero grok usage). Grok novelty is deliberate, watched scaffolding, dispatched
  # by hand — never an every-tick reflex.
  ODYSSEY_HEADROOM_ADMIT=1 "$PY" tools/odyssey_ctl.py cycle --go --max-lanes 6 --grok-lanes 0
  echo "-- cycle rc=$? --"
  # Commit DATA progress only (receipts/completions/state/packets/matrix). NEVER tools/ code
  # (code-editing lanes land in REVIEW_QUEUE for human review, uncommitted).
  git add -- receipts/odyssey-i \
    workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json \
    workspace/campaign/odyssey/ODYSSEY_STATE.json \
    workspace/campaign/odyssey/RUN_LOG.jsonl \
    workspace/campaign/odyssey/TRANSFER_MATRIX.json \
    workspace/campaign/odyssey/GRAVITY_RULEBASE.json \
    workspace/campaign/odyssey/NEGATIVE_SCIENCE.json \
    workspace/campaign/odyssey/patients >/dev/null 2>&1 || true
  git commit -q -m "odyssey-i driver: autonomous cycle $ts" >/dev/null 2>&1 && echo "-- committed data --" || echo "-- nothing to commit --"
  echo "== tick done $ts =="
} >>"$LOG" 2>&1
