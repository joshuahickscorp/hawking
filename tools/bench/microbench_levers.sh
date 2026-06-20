#!/usr/bin/env bash
# tools/bench/microbench_levers.sh
#
# RAM-light single-binary microbench for path-to-50 levers.
# Uses the existing ./target/release/hawking binary as-is — does NOT
# rebuild — so it doesn't compete with running worktree agents for RAM.
#
# Each lever is N trials of `dismantle generate` with different flags.
# Output: artifacts/runs/microbench/<utc>/report.md with paired-delta tps.
#
# Honors artifacts/runs/PAUSE between trials.
#
# Usage:
#   bash tools/bench/microbench_levers.sh                 # 3 trials per lever
#   TRIALS=2 bash tools/bench/microbench_levers.sh        # smaller, faster
#   TOKENS=128 bash tools/bench/microbench_levers.sh      # longer generation

set -o pipefail
# Note: `set -u` removed because macOS bash 3.2 trips on
# `${empty_array[@]}` even when the array is intentionally empty.
cd "$(dirname "$0")/../.."

BIN="./target/release/hawking"
WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
PROMPT="${PROMPT:-Once upon a time}"
TOKENS="${TOKENS:-64}"
TRIALS="${TRIALS:-3}"
SEED="${SEED:-0}"

if [[ ! -x "$BIN" ]]; then
    echo "❌ binary missing at $BIN — run 'cargo build --release -p dismantle' first"
    exit 1
fi

RUN_DIR="artifacts/runs/microbench/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
LATEST="artifacts/runs/microbench/latest"
rm -f "$LATEST"
ln -s "$(basename "$RUN_DIR")" "$LATEST"
REPORT="$RUN_DIR/report.md"
RAW="$RUN_DIR/raw"
mkdir -p "$RAW"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
wait_if_paused() {
    if [[ -f artifacts/runs/PAUSE ]]; then
        echo "  ⏸  PAUSED — touch artifacts/runs/RESUME to continue"
        while [[ -f artifacts/runs/PAUSE ]]; do
            sleep 5
            if [[ -f artifacts/runs/RESUME ]]; then
                rm -f artifacts/runs/PAUSE artifacts/runs/RESUME
                echo "  ▶  resumed"
                break
            fi
        done
    fi
}

# Probe binary for what flags actually exist (defensive: lever skipped if its
# flag isn't accepted).
HAS_VOCAB_PRUNE=$($BIN generate --help 2>&1 | grep -q '\-\-vocab-prune-path' && echo 1 || echo 0)
HAS_TIER_MAP=$($BIN generate --help 2>&1 | grep -q '\-\-quant-tier-map-path' && echo 1 || echo 0)
HAS_Q8_KV=$($BIN generate --help 2>&1 | grep -qE '\-\-q8-kv|\-\-q8kv' && echo 1 || echo 0)
HAS_SPECULATE=$($BIN generate --help 2>&1 | grep -q '\-\-speculate' && echo 1 || echo 0)

VOCAB_WHITELIST="artifacts/calibration/analysis/vocab_whitelist_995.json"
TIER_DEFAULT="artifacts/calibration/tier_maps/v2_lite_default.json"
TIER_AGGRESSIVE="artifacts/calibration/tier_maps/v2_lite_aggressive_down_q4.json"

# Q8 KV flag name uncertainty — probe what the binary actually accepts.
Q8_KV_FLAG=""
if [[ "$HAS_Q8_KV" = "1" ]]; then
    if $BIN generate --help 2>&1 | grep -q '\-\-q8-kv'; then Q8_KV_FLAG="--q8-kv"
    elif $BIN generate --help 2>&1 | grep -q '\-\-q8kv'; then Q8_KV_FLAG="--q8kv"
    fi
fi

cat > "$REPORT" <<EOF
# Path-to-50 microbench — $(stamp)

**Binary:** $BIN ($(stat -f '%Sm' "$BIN" 2>/dev/null))
**Weights:** $WEIGHTS
**Profile:** $PROFILE
**Prompt:** "$PROMPT"
**Tokens/trial:** $TOKENS  ·  **Trials/lever:** $TRIALS  ·  **Seed:** $SEED
**Mode:** paired-delta — Claude may be live; absolute numbers contaminated, relative deltas valid (per memory feedback_bench_with_claude_open.md).

**Flag probe:**
- vocab-prune: $HAS_VOCAB_PRUNE
- tier-map: $HAS_TIER_MAP
- q8-kv: $HAS_Q8_KV (flag: ${Q8_KV_FLAG:-N/A})
- speculate: $HAS_SPECULATE

| Lever | Avg dec_tps | Trials | Δ vs L0 |
|---|---:|:---:|---:|
EOF

# LEVER_AVG stored as files under $RAW/avg_<id>.txt — portable to bash 3.2.

run_lever() {
    local id="$1" label="$2"
    shift 2
    local extra_args=("$@")
    echo "[$(stamp)] === $id: $label ==="
    local tps_sum=0
    local n=0
    local raw_log="$RAW/${id}.log"
    : > "$raw_log"

    for t in $(seq 1 "$TRIALS"); do
        wait_if_paused
        echo "  trial $t/$TRIALS..."
        local out
        out=$($BIN generate \
            --weights "$WEIGHTS" \
            --kernel-profile "$PROFILE" \
            --prompt "$PROMPT" \
            --max-new-tokens "$TOKENS" \
            --seed "$SEED" \
            "${extra_args[@]}" 2>&1 | tee -a "$raw_log")
        # Parse "dec_tps=NN.NN" from stats line.
        local tps
        tps=$(echo "$out" | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
        if [[ -n "$tps" ]]; then
            echo "    tps=$tps"
            tps_sum=$(echo "$tps_sum + $tps" | bc -l)
            n=$((n + 1))
        else
            echo "    (failed to parse tps)"
        fi
    done

    if [[ "$n" -gt 0 ]]; then
        local avg
        avg=$(echo "scale=2; $tps_sum / $n" | bc -l)
        echo "$avg" > "$RAW/avg_${id}.txt"
        echo "[$(stamp)]   avg=$avg over $n trials"
    else
        echo "N/A" > "$RAW/avg_${id}.txt"
        echo "[$(stamp)]   no successful trials"
    fi
}

# Lever sequence. Skip any with missing inputs.

run_lever "L0_baseline" "Baseline (no levers)"

if [[ "$HAS_VOCAB_PRUNE" = "1" ]] && [[ -f "$VOCAB_WHITELIST" ]]; then
    run_lever "L1_vocab_prune" "Vocab-prune (whitelist=$(basename "$VOCAB_WHITELIST"))" \
        --vocab-prune-path "$VOCAB_WHITELIST"
fi

if [[ "$HAS_TIER_MAP" = "1" ]] && [[ -f "$TIER_DEFAULT" ]]; then
    run_lever "L2_tier_default" "Mixed-precision (tier=v2_lite_default)" \
        --quant-tier-map-path "$TIER_DEFAULT"
fi
if [[ "$HAS_TIER_MAP" = "1" ]] && [[ -f "$TIER_AGGRESSIVE" ]]; then
    run_lever "L2_tier_aggro" "Mixed-precision (tier=aggressive_down_q4)" \
        --quant-tier-map-path "$TIER_AGGRESSIVE"
fi

if [[ -n "$Q8_KV_FLAG" ]]; then
    run_lever "L3_q8_kv" "Q8 KV cache ($Q8_KV_FLAG)" "$Q8_KV_FLAG"
fi

if [[ "$HAS_SPECULATE" = "1" ]]; then
    run_lever "L4_spec_exact_K4" "Spec-decode exact-shared K=4" \
        --speculate exact-shared --verify-window 4
    run_lever "L4_spec_exact_K16" "Spec-decode exact-shared K=16 (Session B target)" \
        --speculate exact-shared --verify-window 16
    run_lever "L4b_spec_ngram_K4" "Spec-decode n-gram K=4 (FREE drafts)" \
        --speculate ngram --verify-window 4
    run_lever "L4b_spec_ngram_K8" "Spec-decode n-gram K=8" \
        --speculate ngram --verify-window 8
fi

# STACK: everything available enabled at once
STACK_ARGS=()
[[ "$HAS_VOCAB_PRUNE" = "1" ]] && [[ -f "$VOCAB_WHITELIST" ]] && \
    STACK_ARGS+=(--vocab-prune-path "$VOCAB_WHITELIST")
[[ "$HAS_TIER_MAP" = "1" ]] && [[ -f "$TIER_AGGRESSIVE" ]] && \
    STACK_ARGS+=(--quant-tier-map-path "$TIER_AGGRESSIVE")
[[ -n "$Q8_KV_FLAG" ]] && STACK_ARGS+=("$Q8_KV_FLAG")
if [[ ${#STACK_ARGS[@]} -gt 0 ]]; then
    run_lever "STACK_no_spec" "STACK (no spec-decode): ${STACK_ARGS[*]}" "${STACK_ARGS[@]}"
fi

# Emit table rows + report (read avg from $RAW/avg_<id>.txt files).
BASELINE_TPS="N/A"
[[ -f "$RAW/avg_L0_baseline.txt" ]] && BASELINE_TPS=$(cat "$RAW/avg_L0_baseline.txt")
for id in L0_baseline L1_vocab_prune L2_tier_default L2_tier_aggro L3_q8_kv L4_spec_exact_K4 L4_spec_exact_K16 L4b_spec_ngram_K4 L4b_spec_ngram_K8 STACK_no_spec; do
    avg_file="$RAW/avg_${id}.txt"
    [[ ! -f "$avg_file" ]] && continue
    avg=$(cat "$avg_file")
    delta="—"
    if [[ "$avg" != "N/A" ]] && [[ "$BASELINE_TPS" != "N/A" ]] && [[ "$id" != "L0_baseline" ]]; then
        delta=$(echo "scale=2; $avg - $BASELINE_TPS" | bc -l)
        [[ "${delta:0:1}" != "-" ]] && delta="+$delta"
    fi
    echo "| $id | $avg | $TRIALS | $delta |" >> "$REPORT"
done

echo "" >> "$REPORT"
echo "**Raw output per lever:** $RAW/" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Notes" >> "$REPORT"
echo "" >> "$REPORT"
echo "- Absolute tps is contaminated 4-5× by Claude (per memory bench_contamination.md). Δ vs L0 is the trustworthy column." >> "$REPORT"
[[ "$HAS_Q8_KV" = "0" ]] && echo "- **Q8 KV skipped:** binary lacks the flag. Session C may have landed Rust changes but the binary needs \`cargo build --release -p dismantle\` to surface them." >> "$REPORT"

echo
echo "=== microbench complete ==="
echo "report: $REPORT"
echo
echo "--- summary ---"
tail -n 40 "$REPORT"
