#!/usr/bin/env bash
# tools/bench/quick_bench.sh — 3-trial median bench for dev-loop iteration.
#
# NOT authoritative — bench numbers are contaminated when run from inside
# an active Claude Code session (4-5x slower than truth, see
# reports/v0.3.5_clean_rebench.md). Use tools/bench/clean_bench.sh from a
# Cmd+Q-Claude terminal for shippable numbers.
#
# This wrapper exists to give fast structural-metric feedback (counters
# from --trace-dispatch are load-independent and accurate even contaminated).

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TOKENS="${TOKENS:-32}"
BIN="./target/release/hawking"

CLAUDE_RUNNING=0
if pgrep -f "Claude.app" > /dev/null 2>&1; then
    CLAUDE_RUNNING=1
    echo "⚠️  Claude desktop app is running — dec_tps will be contaminated." >&2
    echo "   Use tools/bench/clean_bench.sh for shippable numbers." >&2
fi

# `--strict` (or STRICT=1 env) blocks contaminated runs entirely. Use when
# you intend to gate a wedge on absolute dec_tps — past sessions wasted
# hours treating contaminated numbers as truth (see bench_contamination.md
# and feedback_bench_with_claude_open.md). Paired-delta runs don't need
# this; absolute-tps gates do.
STRICT_FLAG="${STRICT:-}"
if [[ "${1:-}" == "--strict" ]]; then STRICT_FLAG=1; shift || true; fi
if [[ -n "$STRICT_FLAG" ]] && [[ "$CLAUDE_RUNNING" == "1" ]]; then
    echo "❌ STRICT mode: refusing to bench with Claude running." >&2
    echo "   Cmd+Q the Claude desktop app and any CLI sessions, then re-run." >&2
    exit 64
fi

if [[ "$(basename "$WEIGHTS")" == qwen2.5-3b-instruct-q4_k_m.gguf ]]; then
    # README's Qwen headline is the locked fast stack, not the raw
    # CPU/Metal-hybrid fallback path. Keep explicit user overrides intact.
    : "${HAWKING_QWEN_TCB:=1}"
    : "${HAWKING_QWEN_VOCAB_PRUNE:=32000}"
    : "${HAWKING_QWEN_Q4K_LMHEAD:=1}"
    : "${HAWKING_QWEN_FFN_DOWN_Q4K:=1}"
    export HAWKING_QWEN_TCB
    export HAWKING_QWEN_VOCAB_PRUNE
    export HAWKING_QWEN_Q4K_LMHEAD
    export HAWKING_QWEN_FFN_DOWN_Q4K
    echo "Qwen locked fast-path env: TCB=$HAWKING_QWEN_TCB vocab=$HAWKING_QWEN_VOCAB_PRUNE q4k_lmhead=$HAWKING_QWEN_Q4K_LMHEAD ffn_down_q4k=$HAWKING_QWEN_FFN_DOWN_Q4K" >&2
fi

echo "=== quick_bench: 3 trials × ${TOKENS} tokens ==="
TRIALS_TPS=()
for i in 1 2 3; do
    OUT_JSON="/tmp/quick_bench_t${i}.json"
    echo "trial $i..."
    perl -e 'alarm 600; exec @ARGV' "$BIN" bench --trace-dispatch \
        --backend hawking --suite decode \
        --weights "$WEIGHTS" --trials 1 --max-new-tokens "$TOKENS" \
        --kernel-profile "$PROFILE" \
        --json "$OUT_JSON" >/dev/null 2>&1
    TPS=$(jq -r '.results.trial_stats[0].decode_tps // "0"' "$OUT_JSON")
    TRIALS_TPS+=("$TPS")
done

MEDIAN=$(printf '%s\n' "${TRIALS_TPS[@]}" | sort -n | sed -n '2p')
SPREAD=$(awk -v vals="${TRIALS_TPS[*]}" 'BEGIN {
    n = split(vals, a, " "); min = a[1]; max = a[1];
    for (i = 2; i <= n; i++) { if (a[i] < min) min = a[i]; if (a[i] > max) max = a[i]; }
    printf "%.2f", (max - min) / a[1] * 100;
}')

echo ""
echo "=== results ==="
printf "trials:    %s %s %s\n" "${TRIALS_TPS[@]}"
printf "median:    %s dec_tps\n" "$MEDIAN"
printf "spread:    %s%%  (>=10%% suggests other contaminating processes)\n" "$SPREAD"
echo ""
echo "structural metrics from trial 3:"
jq '.results.trial_stats[0] | {
    metal_buffers_created_per_token,
    cpu_alloc_bytes_per_token,
    dispatch_commits_per_token
}' /tmp/quick_bench_t3.json
