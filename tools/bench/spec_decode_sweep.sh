#!/usr/bin/env bash
# Spec-decode sweep: off vs eagle4-K=4 vs eagle5-K=4/8/16 across 3 prompts.
#
# Pre-staged for the post-pipeline phase. Run after eagle5 v2 quantize
# lands so we have the head to plug in.
#
# Outputs: artifacts/runs/overnight/spec_decode_sweep.md
# Usage: bash tools/bench/spec_decode_sweep.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

OUT="artifacts/runs/overnight/spec_decode_sweep.md"
WEIGHTS="models/deepseek-v2-lite-q4.gguf"
PROFILE="profiles/deepseek-v2-lite-q4.m3pro18.json"
BIN="./target/release/dismantle"

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN missing" | tee -a "$OUT"
    exit 1
fi

{
    echo "# Spec-decode sweep — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo ""
    echo "Each cell: dec_tps (draft_accepted / draft_rejected). Single trial per cell"
    echo "for speed; rerun with --trials 3 if signal is borderline."
    echo ""
    echo "| Prompt | off | eagle4 K=4 |"
    echo "|---|---|---|"
} > "$OUT"

run_one() {
    local prompt="$1"
    local mode="$2"
    local vw="$3"
    local extra=""
    if [ "$mode" != "off" ]; then
        extra="--speculate $mode --verify-window $vw"
    fi
    "$BIN" generate \
        --weights "$WEIGHTS" \
        --kernel-profile "$PROFILE" \
        --prompt "$prompt" \
        --max-new-tokens 64 \
        --seed 0 \
        $extra 2>&1 | grep -E "^\[stats\]" | head -1
}

parse_tps() {
    # Extract dec_tps + accept/reject from a [stats] line
    awk '{
        for (i=1; i<=NF; i++) {
            if ($i ~ /dec_tps=/) tps=substr($i, 9)
            if ($i ~ /draft_accepted=/) acc=substr($i, 16)
            if ($i ~ /draft_rejected=/) rej=substr($i, 16)
        }
        printf "%.2f (%s/%s)", tps, acc, rej
    }'
}

PROMPTS=(
    "Once upon a time, there"
    "The capital of France is"
    "def fibonacci(n):"
)

for prompt in "${PROMPTS[@]}"; do
    short="$(echo "$prompt" | cut -c1-30)..."
    off_stats=$(run_one "$prompt" "off" "4" | parse_tps)
    e4_stats=$(run_one "$prompt" "exact-shared" "4" | parse_tps)
    echo "| \`$short\` | $off_stats | $e4_stats |" >> "$OUT"
done

# Eagle5 head-swap (TODO once eagle5 v2 weights are landed): the runtime
# currently loads eagle4 weights from eagle4/v2lite_frozen.npz at startup
# by default. To swap to eagle5 v2, either:
#   (a) update eagle4/v2lite_frozen.npz symlink to checkpoints/eagle5_v2/q4_head.npz
#   (b) add a --draft-head CLI flag (not yet wired — needs Rust change)
echo "" >> "$OUT"
echo "## Eagle5 v2 swap: NOT YET WIRED" >> "$OUT"
echo "" >> "$OUT"
echo "Once eagle5_quantize lands a q4 head, the runtime needs either a" >> "$OUT"
echo "symlink swap of \`eagle4/v2lite_frozen.npz\` or a new \`--draft-head\`" >> "$OUT"
echo "CLI flag to point at \`checkpoints/eagle5_v2/q4_head.npz\`." >> "$OUT"
echo "Add to this sweep when wiring is complete." >> "$OUT"

echo "done — see $OUT"
