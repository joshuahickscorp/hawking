#!/bin/bash
# Launch the G016 dense packer lane as soon as a Grok slot frees.
# Waits for < 6 running lanes and >= 60 GiB free, then launches once and exits.
C="$1"
for i in $(seq 1 240); do   # up to ~4h at 60s
  n=$(~/.claude-grok/bin/grok-run status 2>/dev/null | grep -c "^running")
  free=$(df -g / | tail -1 | awk '{print $4}')
  if [ "$n" -lt 6 ] && [ "$free" -ge 60 ]; then
    ~/.claude-grok/bin/grok-run delegate --task qwen38-mixed-pack \
      --contract "$C" --repo /Users/scammermike/Downloads/hawking \
      --profile gate --background
    echo "launched at $(date -u +%H:%M:%S)Z with n=$n free=${free}GiB"
    exit 0
  fi
  sleep 60
done
echo "gave up: no slot in 4h"
