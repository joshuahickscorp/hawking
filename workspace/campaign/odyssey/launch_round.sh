#!/bin/bash
# ONE launch path for an autonomy round (G006).
#
# Two rounds were invalidated by launching them by hand:
#
#   round 21   plain `python3.12 -m hcli ...`   -- no HCLI_MAX_TOOL_OBSERVATIONS,
#              so the round got a ONE-observation budget for a four-step plan and
#              closed after reading the ledger.
#   round 21b  same command, and `python3.12` resolves to /Users/.../.local/bin
#              (a bare uv interpreter whose site-packages contains only pip).
#              odyssey.dense_anatomy raised ModuleNotFoundError: numpy. The round
#              correctly diagnosed "numpy missing" -- on an environment the launch
#              had broken.
#
# The campaign interpreter is /usr/local/bin/python3.12 (numpy 2.2.6), which is
# what rounds through 16 used. Neither defect was a harness regression and
# neither was the round's fault; both were the launch. So the launch is a file.
set -euo pipefail

PY=/usr/local/bin/python3.12
CHALLENGE="$(dirname "$0")/round_challenge.txt"
LOG="${2:-/tmp/odyssey_round.log}"

"$PY" -c 'import numpy' 2>/dev/null || {
  echo "REFUSING: $PY cannot import numpy -- odyssey.dense_anatomy needs it and the round will"
  echo "diagnose a missing dependency instead of measuring anything." >&2
  exit 1
}
[ -f "$CHALLENGE" ] || { echo "REFUSING: no challenge at $CHALLENGE" >&2; exit 1; }

export HCLI_MAX_TOOL_OBSERVATIONS="${HCLI_MAX_TOOL_OBSERVATIONS:-8}"
echo "round: $PY, observations=$HCLI_MAX_TOOL_OBSERVATIONS, log=$LOG"
exec timeout 3600 "$PY" -m hcli \
  --model hcli/hawking-native.sealed-3.14.json \
  --max-cycles 1 --max-turns 20 \
  --task-file "$CHALLENGE" > "$LOG" 2>&1
