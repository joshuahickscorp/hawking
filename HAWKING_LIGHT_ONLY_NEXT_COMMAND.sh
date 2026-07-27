#!/usr/bin/env bash
# Re-derive the campaign's state. Safe to run at any time; changes nothing.
set -euo pipefail
cd "$(dirname "$0")"
python3.12 tools/campaign/light_governor.py --watch 3
python3.12 tools/campaign/light_status.py
echo
echo 'If the governor says HEAVY_WINDOW_AVAILABLE, the next action is H1 in'
echo 'HAWKING_NEXT_HEAVY_TOURNAMENT.json -- representation arms on a real SMALL parent.'
echo 'It is never started automatically. A human decides.'
