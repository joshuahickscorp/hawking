#!/usr/bin/env bash
# path-to-125 clean-window bench.
#
# Run this AFTER quitting Claude.app (Cmd-Q) — contended GPU produces
# 4-5x inflated dec_tps numbers (per memory bench_contamination). The
# script will refuse to run if it sees Claude.app still alive.
#
# Captures three configs:
#   1. Off baseline                                            (sequential profile)
#   2. ngram-spec K=4 + parallel-k                             (proves K-batched verify amortization)
#   3. Eagle4 chain K=4 + parallel-k (current v3 head)         (current head accept rate)
#
# After training a v4_chain head, re-run with:
#   EAGLE4_CKPT=eagle4/checkpoints/eagle4_v4_chain/latest.npz ./tools/bench/path_to_125_bench.sh
# to compare against post-Branch-1 baseline.
#
# Writes raw.jsonl + summary.txt under reports/path_to_90/_bench_<timestamp>/.

set -euo pipefail

if pgrep -i "Claude" >/dev/null 2>&1; then
  echo "ERROR: Claude is still running. Quit Claude.app (Cmd-Q) before benching." >&2
  echo "       Contended GPU produces 4-5x inflated dec_tps — useless data." >&2
  exit 2
fi

if pgrep -f "slm" >/dev/null 2>&1; then
  echo "WARN: slm process detected — pause it (kill -STOP <pid>) or wait until idle." >&2
  echo "      Sleeping 30s for any background activity to drain..." >&2
  sleep 30
fi

# Recommended memory headroom (one-time per boot; needs sudo).
CUR=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
if [[ "$CUR" -lt 14336 ]]; then
  echo "WARN: iogpu.wired_limit_mb=$CUR; recommend 14336 for sustained bench:" >&2
  echo "  sudo sysctl iogpu.wired_limit_mb=14336" >&2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

WEIGHTS="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
PROFILE_SEQ="$REPO_ROOT/profiles/deepseek-v2-lite-q4.m3pro18.json"
PROFILE_PK="/tmp/path_to_125_bench_profile_parallelk.json"
FROZEN_NPZ="$REPO_ROOT/eagle4/v2lite_frozen.npz"
DRAFT_NPZ="${EAGLE4_CKPT:-$REPO_ROOT/eagle4/checkpoints/eagle4_v3/best.npz}"
DISMANTLE="$REPO_ROOT/target/release/dismantle"

if [[ ! -x "$DISMANTLE" ]]; then
  echo "ERROR: $DISMANTLE missing. Build first: cargo build --release -p dismantle" >&2
  exit 3
fi
for f in "$WEIGHTS" "$PROFILE_SEQ" "$FROZEN_NPZ" "$DRAFT_NPZ"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing artifact: $f" >&2
    exit 4
  fi
done

# Build the parallel-k variant of the profile.
python3 -c "
import json, sys
p = json.load(open('$PROFILE_SEQ'))
p['selected']['verify_kernels'] = 'parallel-k'
json.dump(p, open('$PROFILE_PK','w'), indent=2)
"

TS="$(date +%Y%m%dT%H%M%S)"
OUTDIR="$REPO_ROOT/reports/path_to_90/_bench_${TS}"
mkdir -p "$OUTDIR"
RAW="$OUTDIR/raw.jsonl"
SUMMARY="$OUTDIR/summary.txt"

PROMPTS=(
  "The quick brown fox"
  "Write a Python function to compute Fibonacci numbers"
  "Summarize the plot of Hamlet in three sentences"
  "The capital of France is"
)
TOKENS=64
TRIALS=3

run_trial() {
  local mode="$1"          # off | ngram | eagle4
  local profile="$2"       # sequential | parallel-k
  local profile_path="$3"
  local prompt="$4"
  local trial="$5"
  local chain_k="${6:-1}"

  local out
  if [[ "$mode" == "eagle4" ]]; then
    out=$(EAGLE4_CHAIN_K="$chain_k" nice -n 19 "$DISMANTLE" generate \
            --weights "$WEIGHTS" --kernel-profile "$profile_path" \
            --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 \
            --speculate eagle4 --draft-head "$DRAFT_NPZ" --eagle4-frozen "$FROZEN_NPZ" \
            2>&1)
  elif [[ "$mode" == "ngram" ]]; then
    out=$(nice -n 19 "$DISMANTLE" generate \
            --weights "$WEIGHTS" --kernel-profile "$profile_path" \
            --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 \
            --speculate ngram \
            2>&1)
  else
    # off — no spec-decode flags
    out=$(nice -n 19 "$DISMANTLE" generate \
            --weights "$WEIGHTS" --kernel-profile "$profile_path" \
            --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 \
            2>&1)
  fi

  local dec_tps accepted rejected
  dec_tps=$(echo "$out" | grep -oE 'dec_tps=[0-9.]+' | head -1 | cut -d= -f2)
  accepted=$(echo "$out" | grep -oE 'draft_accepted=[0-9]+' | head -1 | cut -d= -f2)
  rejected=$(echo "$out" | grep -oE 'draft_rejected=[0-9]+' | head -1 | cut -d= -f2)
  dec_tps="${dec_tps:-0}"; accepted="${accepted:-0}"; rejected="${rejected:-0}"

  printf '{"mode":"%s","profile":"%s","chain_k":%d,"prompt":%s,"trial":%d,"dec_tps":%s,"draft_accepted":%s,"draft_rejected":%s}\n' \
    "$mode" "$profile" "$chain_k" "$(printf '%s' "$prompt" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
    "$trial" "$dec_tps" "$accepted" "$rejected"
}

echo "=== path-to-125 bench @ $TS ===" | tee "$SUMMARY"
echo "draft head: $DRAFT_NPZ" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# Configs: (mode, profile_label, profile_path, chain_k)
CONFIGS=(
  "off       sequential   $PROFILE_SEQ 1"
  "ngram     parallel-k   $PROFILE_PK  1"
  "eagle4    sequential   $PROFILE_SEQ 1"
  "eagle4    parallel-k   $PROFILE_PK  4"
)

> "$RAW"
for cfg in "${CONFIGS[@]}"; do
  read -r mode profile_label profile_path chain_k <<<"$cfg"
  echo "--- $mode / $profile_label / chain_k=$chain_k ---" | tee -a "$SUMMARY"
  config_label="${mode}/${profile_label}/K${chain_k}"
  sum=0; cnt=0; min=9999; max=0
  for prompt in "${PROMPTS[@]}"; do
    for t in $(seq 1 $TRIALS); do
      line=$(run_trial "$mode" "$profile_label" "$profile_path" "$prompt" "$t" "$chain_k")
      echo "$line" >> "$RAW"
      dec_tps=$(echo "$line" | python3 -c 'import sys,json; print(json.load(sys.stdin)["dec_tps"])')
      sum=$(python3 -c "print($sum + $dec_tps)")
      cnt=$((cnt+1))
      min=$(python3 -c "print(min($min, $dec_tps))")
      max=$(python3 -c "print(max($max, $dec_tps))")
      printf '  trial=%d prompt=%-30s dec_tps=%s\n' "$t" "$(echo "$prompt" | cut -c1-30)" "$dec_tps" | tee -a "$SUMMARY"
    done
  done
  mean=$(python3 -c "print($sum / $cnt if $cnt else 0)")
  median_line=$(grep -F "\"mode\":\"$mode\",\"profile\":\"$profile_label\",\"chain_k\":$chain_k" "$RAW" | \
                python3 -c '
import sys, json
xs = sorted(json.loads(l)["dec_tps"] for l in sys.stdin)
print(xs[len(xs)//2])')
  printf '  [%s] median=%s mean=%s min=%s max=%s n=%d\n' \
    "$config_label" "$median_line" "$mean" "$min" "$max" "$cnt" | tee -a "$SUMMARY"
  echo "" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== RESULT ===" | tee -a "$SUMMARY"
echo "raw: $RAW" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# Try to surface a brief summary of dec_tps medians.
python3 - "$RAW" <<'PY' | tee -a "$SUMMARY"
import sys, json, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["mode"], r["profile"], r["chain_k"])].append(r["dec_tps"])
print("config                            median   mean    min     max     n")
for (mode, prof, ck), xs in groups.items():
    xs = sorted(xs)
    med = xs[len(xs)//2]
    mean = sum(xs) / len(xs)
    label = f"{mode}/{prof}/K{ck}"
    print(f"{label:34s} {med:6.2f}  {mean:6.2f}  {min(xs):6.2f}  {max(xs):6.2f}  {len(xs)}")
PY

if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "path-to-125 bench complete" with title "dismantle"' || true
fi
echo ""
echo "Bench complete. Summary above; raw JSON at $RAW"
