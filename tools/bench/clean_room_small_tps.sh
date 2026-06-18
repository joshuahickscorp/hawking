#!/usr/bin/env bash
# Clean-room single-stream dec_tps: dismantle vs llama.cpp on the smallest-coherent
# models. RUN WITH CLAUDE QUIT (a live Claude session inflates/perturbs absolute tps).
# Usage: bash tools/bench/clean_room_small_tps.sh [llama_bench_path]
set -euo pipefail
cd "$(dirname "$0")/../.."
DM=./target/release/dismantle
LB="${1:-/tmp/llamacpp/build/bin/llama-bench}"
PROMPT="Write a haiku about the ocean, then list 3 prime numbers."
N=128
echo "model,engine,dec_tps"
for m in Qwen2.5-0.5B-Instruct-Q4_K_M Qwen2.5-3B-Instruct-Q4_K_M Llama-3.2-1B-Instruct-Q4_K_M; do
  g="models/$m.gguf"; [ -f "$g" ] || { echo "$m,MISSING,-"; continue; }
  # dismantle (median of 3)
  for i in 1 2 3; do
    "$DM" generate --weights "$g" --prompt "$PROMPT" --max-new-tokens $N --temperature 0 2>&1 \
      | sed -n 's/.*dec_tps=\([0-9.]*\).*/'"$m"',dismantle,\1/p'
  done
  # llama.cpp (tg128, median of 3) if available
  if [ -x "$LB" ]; then
    "$LB" -m "$g" -p 0 -n $N -ngl 99 -r 3 2>/dev/null \
      | sed -n 's/.*tg'"$N"' *| *\([0-9.]*\).*/'"$m"',llamacpp,\1/p'
  fi
done
echo "# Compare dismantle vs llamacpp per model; the dismantle/llamacpp ratio = the real small-model gap."
