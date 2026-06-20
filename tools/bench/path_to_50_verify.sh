#!/usr/bin/env bash
# Verify + paired-bench the uncommitted path-to-50 Stage-1 work on Qwen-3B.
#   A = predec base, fusions OFF   (pre-path-to-50 baseline)
#   B = dirty-tree defaults        (gate+up fuse + ffn_down predec ON)
#   C = B + 2-row ILP              (HAWKING_QWEN_PREDEC_2R=1)
# Parity: greedy output of B and C must be bit-identical to A.
# Bench: interleaved paired dec_tps (cancels thermal drift); Claude-open is
# fine for paired deltas (memory/feedback_bench_with_claude_open.md).
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN="./target/release/hawking"
WEIGHTS="models/qwen2.5-3b-instruct-q4_k_m.gguf"
PROFILE="profiles/qwen3b-instruct-q4k.m3pro18.json"
TOKENS="${TOKENS:-32}"
TRIALS="${TRIALS:-5}"
PROMPT='fn fibonacci(n: u64) -> u64 {'

# locked Qwen fast-path (same as quick_bench.sh)
export HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 \
       HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 \
       HAWKING_QWEN_Q4K_PREDEC=1

env_for () { case "$1" in
  A) echo "HAWKING_QWEN_FFN_GATEUP_FUSE=0 HAWKING_QWEN_FFN_DOWN_PREDEC=0 HAWKING_QWEN_PREDEC_2R=0";;
  B) echo "HAWKING_QWEN_FFN_GATEUP_FUSE=1 HAWKING_QWEN_FFN_DOWN_PREDEC=1 HAWKING_QWEN_PREDEC_2R=0";;
  C) echo "HAWKING_QWEN_FFN_GATEUP_FUSE=1 HAWKING_QWEN_FFN_DOWN_PREDEC=1 HAWKING_QWEN_PREDEC_2R=1";;
esac; }

mode="${1:-all}"

if [[ "$mode" == "parity" || "$mode" == "all" ]]; then
  echo "=== PARITY (greedy ${TOKENS}tok, bit-identical gate) ==="
  for c in A B C; do
    env $(env_for $c) nice -n 19 taskpolicy -b "$BIN" generate \
      --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
      --prompt "$PROMPT" --max-new-tokens "$TOKENS" --temperature 0 --seed 0 \
      > "/tmp/p50_$c.txt" 2>/dev/null
  done
  ok=1
  for c in B C; do
    if diff -q /tmp/p50_A.txt "/tmp/p50_$c.txt" >/dev/null; then
      echo "  $c vs A: IDENTICAL ✓"
    else echo "  $c vs A: DIFF ✗"; ok=0; fi
  done
  [[ $ok == 1 ]] && echo "  PARITY: PASS (bit-identical)" || echo "  PARITY: FAIL"
fi

if [[ "$mode" == "bench" || "$mode" == "all" ]]; then
  echo "=== PAIRED BENCH (${TRIALS} trials × ${TOKENS}tok, interleaved) ==="
  TPS_A=""; TPS_B=""; TPS_C=""
  for t in $(seq 1 "$TRIALS"); do
    for c in A B C; do
      j="/tmp/p50_bench_${c}_${t}.json"
      env $(env_for $c) nice -n 19 taskpolicy -b "$BIN" bench --trace-dispatch \
        --backend dismantle --suite decode --weights "$WEIGHTS" --trials 1 \
        --max-new-tokens "$TOKENS" --kernel-profile "$PROFILE" --json "$j" \
        >/dev/null 2>&1
      v=$(jq -r '.results.trial_stats[0].decode_tps // "0"' "$j" 2>/dev/null)
      eval "TPS_$c=\"\$TPS_$c $v\""
    done
  done
  med () { printf '%s\n' $1 | sort -n | awk '{a[NR]=$0} END{print a[int((NR+1)/2)]}'; }
  mA=$(med "$TPS_A"); mB=$(med "$TPS_B"); mC=$(med "$TPS_C")
  printf "  A (base)     trials:%s  median=%.2f\n" "$TPS_A" "$mA"
  printf "  B (fuses)    trials:%s  median=%.2f\n" "$TPS_B" "$mB"
  printf "  C (fuses+2r) trials:%s  median=%.2f\n" "$TPS_C" "$mC"
  awk -v a="$mA" -v b="$mB" -v c="$mC" 'BEGIN{
    printf "  B/A = %.3fx  (%+.1f%%)\n", b/a, (b/a-1)*100;
    printf "  C/A = %.3fx  (%+.1f%%)\n", c/a, (c/a-1)*100;
    printf "  C/B = %.3fx  (%+.1f%%)\n", c/b, (c/b-1)*100; }'
fi
