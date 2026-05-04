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
BIN="./target/release/dismantle"

if pgrep -f "Claude.app" > /dev/null 2>&1; then
    echo "⚠️  Claude desktop app is running — dec_tps will be contaminated." >&2
    echo "   Use tools/bench/clean_bench.sh for shippable numbers." >&2
fi

echo "=== quick_bench: 3 trials × ${TOKENS} tokens ==="
TRIALS_TPS=()
for i in 1 2 3; do
    OUT_JSON="/tmp/quick_bench_t${i}.json"
    echo "trial $i..."
    perl -e 'alarm 600; exec @ARGV' "$BIN" bench --trace-dispatch \
        --backend dismantle --suite decode \
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
