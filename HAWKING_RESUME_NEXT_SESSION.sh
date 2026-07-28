#!/usr/bin/env bash
# Resume the heavy continuation. Safe to run at any time; changes nothing.
set -uo pipefail
cd "$(dirname "$0")"

echo "=== machine ==="
python3.12 tools/campaign/light_governor.py | head -3

echo
echo "=== the long job: activation-aware full pack ==="
if pgrep -f activation_aware_pack >/dev/null; then
  n=$(grep -c "VERIFIED in" /tmp/glm_pilot/fullpack2.log 2>/dev/null || echo 0)
  res=$(ls "$HOME/Library/Application Support/Hawking/GLM52Gravity/pilot_source"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  echo "RUNNING: ${n}/282 shards measured, ${res} resident"
  echo "  ~93 s per shard measured -> measure ~7.3 h, pack pass ~7 h more"
else
  echo "NOT RUNNING."
  OUT="$HOME/Library/Application Support/Hawking/GLM52Gravity/activation_aware_pack"
  if [ -f "$OUT/MEASUREMENT.json" ]; then
    echo "  MEASUREMENT.json exists -- resume from allocation:"
    echo "    .venv/glm52/bin/python tools/condense/glm52_activation_aware_pack.py \\"
    echo "      --allocate-from \"$OUT/MEASUREMENT.json\" --target-bpw 49/50 --out \"$OUT\""
  else
    echo "  restart the full run:"
    echo "    .venv/glm52/bin/python tools/condense/glm52_activation_aware_pack.py --full \\"
    echo "      --shards 1-282 --target-bpw 49/50 --ranks 16,64 --fetch --evict \\"
    echo "      --disk-floor-gib 105 --out \"$OUT\""
  fi
fi

echo
echo "=== fences (must stay these values) ==="
echo "  ODYSSEY_LAUNCH_AUTHORIZED = $(cat odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED)"
python3.12 tools/odyssey/substrate_capability.py --check >/dev/null 2>&1
echo "  substrate capability gate exit = $? (1 = correctly refusing)"


echo
echo "=== IMPORTANT: the running pack has OLD code in memory ==="
echo "It launched at 11:09:59, before the prefetch and --workers commits (87187f91,"
echo "fb68ae88). Python loaded the module at start, so its PACK phase will run serially"
echo "with no prefetch and no workers -- roughly 2-3 h instead of well under one."
echo
OUT="$HOME/Library/Application Support/Hawking/GLM52Gravity/activation_aware_pack"
if [ -f "$OUT/MEASUREMENT.json" ]; then
  echo "MEASUREMENT.json EXISTS -- measurement finished."
  echo "Do NOT let the old process continue into packing. Kill it and run the pack phase"
  echo "with the optimised code:"
  echo
  echo "  pkill -f activation_aware_pack"
  echo "  .venv/glm52/bin/python tools/condense/glm52_activation_aware_pack.py \\"
  echo "    --pack-from \"$OUT/ALLOCATION.json\" --shards 1-282 --fetch --evict \\"
  echo "    --workers 4 --disk-floor-gib 105 --out \"$OUT\""
else
  echo "measurement still in progress; leave it alone. Kill only once MEASUREMENT.json exists."
fi

echo
echo "=== THE decisive next action, once a packed artifact exists ==="
echo "  .venv/glm52/bin/python tools/condense/glm52_capability_gate.py --artifact <packed-dir> --run --out CAPABILITY.json"
echo "  G_math is ONE forward pass on '2 + 2 ='. It decides whether this campaign has a substrate."
echo "  Only an artifact passing G_math and G_live gets its hash bound APPROVED, and only"
echo "  then does Odyssey become a live question."
