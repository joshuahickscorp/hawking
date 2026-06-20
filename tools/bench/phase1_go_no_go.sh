#!/usr/bin/env bash
# tools/bench/phase1_go_no_go.sh — Phase 1 internal checkpoint.
#
# Plan: dismantle-execution-plan-enchanted-salamander.md, "Phase 1 internal
# go/no-go checkpoint" (task #19).
#
# Trigger: after MLA flash-attn (#8) + ICB (#9) + mixed-precision quant
# runtime (#12) all land + parity-pass. Run this on a clean bench (Claude
# Code quit) and read the verdict.
#
#   exit 0  →  GO. Combined dec_tps ≥ 40 tps. Continue Phase 1: AMX,
#              vocab-pruned LM, eagle5 sparse-FFN, Q8 KV, RoPE/MLA minors.
#   exit 1  →  NO-GO. dec_tps < 40. Profile and decide whether to keep
#              grinding Phase 1 levers or jump to Phase 2 (the levers
#              that DON'T gate Phase 2: AMX, vocab pruning, minors).
#              Phase 2 prerequisites from Phase 1: MLA kernel, ICB
#              plumbing, mixed-precision runtime, eagle5 inference path.
#
# Usage:
#   tools/bench/phase1_go_no_go.sh                # uses defaults below
#   THRESHOLD_TPS=45 tools/bench/phase1_go_no_go.sh   # tighter gate
#
# Env:
#   WEIGHTS         model GGUF
#   PROFILE         kernel profile JSON (must enable MLA + ICB + mixed-prec)
#   TOKENS          tokens per trial (default: 128 for stable signal)
#   TRIALS          number of trials (default: 5; reports trimmed median)
#   THRESHOLD_TPS   GO/NO-GO cutoff (default: 40)
#   BIN             dismantle binary (default: target/release/hawking)
#   REPORTS_DIR     where to write JSON traces (default: reports/)

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TOKENS="${TOKENS:-128}"
TRIALS="${TRIALS:-5}"
THRESHOLD_TPS="${THRESHOLD_TPS:-40}"
BIN="${BIN:-./target/release/hawking}"
REPORTS_DIR="${REPORTS_DIR:-reports}"

# Bench hygiene — refuse if Claude is running (4-5× contamination,
# see memory/bench_contamination.md).
if pgrep -f "Claude.app" >/dev/null 2>&1; then
    echo "❌ Claude desktop app is running. Cmd+Q Claude and rerun." >&2
    echo "   Phase 1 GO/NO-GO requires shippable numbers." >&2
    exit 2
fi
if pgrep -f "claude" 2>/dev/null | grep -vq "$$"; then
    # Some claude CLI session may also be running — best-effort warn.
    echo "⚠️  A 'claude' process appears to be running. Verify it's not a session." >&2
fi

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not built. Run: cargo build --release -p dismantle" >&2
    exit 2
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "error: weights not found: $WEIGHTS" >&2
    exit 2
fi
if [ ! -f "$PROFILE" ]; then
    echo "error: profile not found: $PROFILE" >&2
    exit 2
fi

# Verify the profile enables the three Phase-1 prerequisites.
# These keys must exist; their actual schedule values vary by lever.
for required_lever in mla_flash icb mixed_precision; do
    if ! grep -q "\"${required_lever}" "$PROFILE" 2>/dev/null; then
        echo "⚠️  profile $PROFILE does not mention '$required_lever'. The" >&2
        echo "    go/no-go is gated on MLA flash-attn + ICB + mixed-prec all" >&2
        echo "    being ON. If they're enabled under a different key name, ignore." >&2
    fi
done

mkdir -p "$REPORTS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPORTS_DIR/phase1_go_no_go_${STAMP}"
mkdir -p "$OUT_DIR"

echo "=== Phase 1 GO/NO-GO ==="
echo "  binary:        $BIN"
echo "  weights:       $WEIGHTS"
echo "  profile:       $PROFILE"
echo "  tokens/trial:  $TOKENS"
echo "  trials:        $TRIALS"
echo "  threshold:     $THRESHOLD_TPS dec_tps"
echo "  output:        $OUT_DIR"
echo

TRIAL_TPS=()
for i in $(seq 1 "$TRIALS"); do
    OUT_JSON="$OUT_DIR/trial_${i}.json"
    echo "--- trial $i / $TRIALS ---"
    "$BIN" bench \
        --backend dismantle --suite decode \
        --weights "$WEIGHTS" \
        --kernel-profile "$PROFILE" \
        --trials 1 --max-new-tokens "$TOKENS" \
        --json "$OUT_JSON" >/dev/null
    TPS=$(jq -r '.results.trial_stats[0].decode_tps // "0"' "$OUT_JSON")
    echo "  dec_tps = $TPS"
    TRIAL_TPS+=("$TPS")
done

# Trimmed median: sort, drop top + bottom, take middle.
SORTED=$(printf '%s\n' "${TRIAL_TPS[@]}" | sort -n)
N="${#TRIAL_TPS[@]}"
if [ "$N" -ge 5 ]; then
    TRIMMED=$(printf '%s\n' "$SORTED" | sed -n "2,$((N-1))p")
    MEDIAN=$(printf '%s\n' "$TRIMMED" | awk 'BEGIN{c=0;s=0} {a[c++]=$1; s+=$1} END{ if(c%2) print a[int(c/2)]; else print (a[c/2-1]+a[c/2])/2 }')
else
    MEDIAN=$(printf '%s\n' "$SORTED" | awk 'BEGIN{c=0} {a[c++]=$1} END{ if(c%2) print a[int(c/2)]; else print (a[c/2-1]+a[c/2])/2 }')
fi

echo
echo "=== Verdict ==="
echo "  trials:           ${TRIAL_TPS[*]}"
echo "  trimmed median:   $MEDIAN dec_tps"
echo "  threshold:        $THRESHOLD_TPS dec_tps"

# Write structured verdict file for downstream automation.
VERDICT_FILE="$OUT_DIR/verdict.json"
PASS=$(awk -v m="$MEDIAN" -v t="$THRESHOLD_TPS" 'BEGIN{print (m+0 >= t+0) ? "true" : "false"}')
cat > "$VERDICT_FILE" <<EOF
{
  "trials": [$(IFS=,; echo "${TRIAL_TPS[*]}")],
  "trimmed_median_tps": $MEDIAN,
  "threshold_tps": $THRESHOLD_TPS,
  "pass": $PASS,
  "stamp": "$STAMP",
  "weights": "$WEIGHTS",
  "profile": "$PROFILE"
}
EOF
echo "  verdict file:     $VERDICT_FILE"

if [ "$PASS" = "true" ]; then
    echo
    echo "✅ GO. Continue Phase 1: AMX (#16), vocab-prune LM (#17),"
    echo "   eagle5 sparse-FFN (#14), Q8/layer-diff KV (#15), RoPE minors (#18)."
    exit 0
else
    echo
    echo "❌ NO-GO. Below ${THRESHOLD_TPS} dec_tps."
    echo "   Capture an MST trace next: tools/bench/mst_capture.sh"
    echo "   Identify the bottleneck before deciding Phase 2 jump vs more Phase 1."
    exit 1
fi
