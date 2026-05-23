#!/usr/bin/env bash
# Standalone fallback: runs just the vocab-prune paired bench.
# Used when the main overnight pipeline died before reaching stage 6.
# Needs: target/release/dismantle built, vocab_whitelist_995.json present,
# model + profile present. No corpus required.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight"
mkdir -p "$LOG_DIR"
BENCH_OUT="$LOG_DIR/vocab_bench_results_rescue.md"

echo "=== rescue vocab-prune bench ===" > "$BENCH_OUT"
date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$BENCH_OUT"
echo "" >> "$BENCH_OUT"

if [ ! -x ./target/release/dismantle ]; then
    echo "FAIL: ./target/release/dismantle missing — cannot bench" >> "$BENCH_OUT"
    exit 1
fi
if [ ! -f artifacts/calibration/analysis/vocab_whitelist_995.json ]; then
    echo "FAIL: vocab_whitelist_995.json missing" >> "$BENCH_OUT"
    exit 1
fi

echo "## Baseline (no prune)" >> "$BENCH_OUT"
for trial in 1 2 3; do
    echo "### trial $trial" >> "$BENCH_OUT"
    ./target/release/dismantle generate \
        --weights models/deepseek-v2-lite-q4.gguf \
        --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
        --prompt "Once upon a time" \
        --max-new-tokens 64 \
        --seed 0 2>&1 | tail -10 >> "$BENCH_OUT" || echo "(trial $trial failed)" >> "$BENCH_OUT"
    echo "" >> "$BENCH_OUT"
done

echo "" >> "$BENCH_OUT"
echo "## Pruned (--vocab-prune-path)" >> "$BENCH_OUT"
for trial in 1 2 3; do
    echo "### trial $trial" >> "$BENCH_OUT"
    ./target/release/dismantle generate \
        --weights models/deepseek-v2-lite-q4.gguf \
        --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
        --vocab-prune-path artifacts/calibration/analysis/vocab_whitelist_995.json \
        --prompt "Once upon a time" \
        --max-new-tokens 64 \
        --seed 0 2>&1 | tail -10 >> "$BENCH_OUT" || echo "(trial $trial failed)" >> "$BENCH_OUT"
    echo "" >> "$BENCH_OUT"
done

echo "done — see $BENCH_OUT"
