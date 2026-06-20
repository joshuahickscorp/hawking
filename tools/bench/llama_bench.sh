#!/usr/bin/env bash
# tools/bench/llama_bench.sh — decode bench for a Llama-family GGUF
# (Llama-2 / Llama-3.x / Mistral), served by the `llama` dense engine.
#
# The kernel profile is GENERATED from the GGUF on first run (its
# tensor-layout + shader hashes must match the model, so it cannot be
# hand-written) via `dismantle autotune`, then reused. Delete the
# profile to regenerate after a shader-source change.
#
# NOT authoritative — bench numbers are contaminated when run from inside
# an active Claude Code session (4-5x slower than truth, see
# reports/v0.3.5_clean_rebench.md and memory/bench_contamination.md).
# Use a Cmd+Q-Claude terminal for shippable absolute numbers; paired
# deltas are fine contaminated.
#
# Usage:
#   WEIGHTS=models/llama-3.2-3b-instruct-q4_k_m.gguf ./tools/bench/llama_bench.sh
#   # optional: TOKENS=64  PROFILE=profiles/my.json  ./tools/bench/llama_bench.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/llama-3.2-3b-instruct-q4_k_m.gguf}"
TOKENS="${TOKENS:-32}"
BIN="./target/release/hawking"

if [[ ! -f "$WEIGHTS" ]]; then
    echo "❌ weights not found: $WEIGHTS" >&2
    echo "   Pull a Llama GGUF into models/ and set WEIGHTS=, e.g.:" >&2
    echo "   WEIGHTS=models/llama-3.2-3b-instruct-q4_k_m.gguf $0" >&2
    exit 66
fi
if [[ ! -x "$BIN" ]]; then
    echo "❌ $BIN not built. Run: cargo build --release -p hawking" >&2
    exit 69
fi

# Default profile path derives from the weights filename so multiple
# Llama variants (1B / 3B / 8B) each get their own.
WEIGHTS_STEM="$(basename "$WEIGHTS")"
WEIGHTS_STEM="${WEIGHTS_STEM%.gguf}"
PROFILE="${PROFILE:-profiles/${WEIGHTS_STEM}.m3pro18.json}"

# Generate the profile from the GGUF if absent. autotune writes the
# deterministic profile (model_arch=llama, correct layout/shader hashes)
# the runtime then validates at load.
if [[ ! -f "$PROFILE" ]]; then
    echo "=== generating kernel profile (first run): $PROFILE ===" >&2
    "$BIN" autotune \
        --weights "$WEIGHTS" \
        --profile m3-pro-18gb \
        --max-hours 0.05 \
        --out "$PROFILE"
fi

CLAUDE_RUNNING=0
if pgrep -f "Claude.app" > /dev/null 2>&1; then
    CLAUDE_RUNNING=1
    echo "⚠️  Claude desktop app is running — dec_tps will be contaminated." >&2
    echo "   Cmd+Q for shippable absolute numbers; paired deltas are fine." >&2
fi

STRICT_FLAG="${STRICT:-}"
if [[ "${1:-}" == "--strict" ]]; then STRICT_FLAG=1; shift || true; fi
if [[ -n "$STRICT_FLAG" ]] && [[ "$CLAUDE_RUNNING" == "1" ]]; then
    echo "❌ STRICT mode: refusing to bench with Claude running." >&2
    exit 64
fi

echo "=== llama_bench: 3 trials × ${TOKENS} tokens ==="
echo "    weights: $WEIGHTS" >&2
echo "    profile: $PROFILE" >&2
TRIALS_TPS=()
for i in 1 2 3; do
    OUT_JSON="/tmp/llama_bench_t${i}.json"
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
}' /tmp/llama_bench_t3.json
