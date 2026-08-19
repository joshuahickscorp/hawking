#!/bin/bash
# Detached Odyssey patient downloader. Sequential (no bandwidth thrash — network is
# the wall, ~4 concurrent files is the knee), disk-floor guarded. Runs in parallel
# with grok builds + GPU science. hf reads the HF token from env/cache automatically.
set -u
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/hf
DL=/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/downloads
FLOOR_GB=45   # stop before free disk drops below this
mkdir -p "$DL"
MASTER="$DL/queue.log"
echo "=== downloader start $(date) ===" >> "$MASTER"

# small -> medium, highest compiler-value / GB first. giants (O007/O008/O009/O010) deferred.
QUEUE=(
  "O000 google/gemma-3-1b-it"
  "O002 google/gemma-3-4b-it"
  "O003 moonshotai/Kimi-VL-A3B-Instruct"
  "O006 Qwen/Qwen3-VL-30B-A3B-Instruct"
  "O004 mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)

free_gb() { df -g /Users/scammermike | awk 'NR==2{print $4}'; }

for entry in "${QUEUE[@]}"; do
  oid=${entry%% *}; repo=${entry#* }
  fg=$(free_gb)
  if [ "$fg" -lt "$FLOOR_GB" ]; then
    echo "$(date) STOP: free ${fg}GB < floor ${FLOOR_GB}GB, skipping $oid $repo and rest" >> "$MASTER"
    break
  fi
  log="$DL/${oid}_$(echo "$repo" | tr '/' '_').log"
  echo "$(date) START $oid $repo (free ${fg}GB)" >> "$MASTER"
  if HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" download "$repo" > "$log" 2>&1; then
    echo "$(date) DONE  $oid $repo (free $(free_gb)GB)" >> "$MASTER"
  else
    echo "$(date) FAIL  $oid $repo (see $(basename "$log") — gated/auth or network)" >> "$MASTER"
  fi
done
echo "=== downloader end $(date) (free $(free_gb)GB) ===" >> "$MASTER"
