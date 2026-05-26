#!/usr/bin/env bash
# tools/bench/w4a8_lmhead_per_channel_bench.sh — paired bench for the
# Track E W4A8 per-channel LM_HEAD ship/hold decision.
#
# Compares three W4A8 LM_HEAD configurations on Qwen-3B:
#   1. Baseline           — predec default-on, no W4A8
#                           (DISMANTLE_QWEN_Q4K_PREDEC=1 is default)
#   2. W4A8 per-block     — held-but-not-shipped path
#                           (DISMANTLE_QWEN_W4A8=1)
#   3. W4A8 per-channel   — Track E new path
#                           (DISMANTLE_QWEN_W4A8=1
#                            DISMANTLE_QWEN_W4A8_PER_CHANNEL=1)
#
# For each condition: N trials of `dismantle generate`, report
# median dec_tps + ms/tok. Quality check: greedy 16-token output from
# each W4A8 config is compared token-by-token to the BASELINE
# (config #1), reported as "match=K/16".
#
# Per `memory/feedback_bench_with_claude_open.md`, paired-delta runs
# work with Claude open because contamination cancels in the relative
# delta. Absolute-tps gates need `tools/bench/clean_bench.sh` from a
# Cmd+Q-Claude terminal.
#
# Usage:
#   ./tools/bench/w4a8_lmhead_per_channel_bench.sh
#   TOKENS=128 TRIALS=7 PROMPT="The quick brown fox" \
#       ./tools/bench/w4a8_lmhead_per_channel_bench.sh
#
# Env:
#   TOKENS    : max_new_tokens per perf trial (default 64)
#   TRIALS    : trials per condition         (default 5)
#   PROMPT    : prompt string                (default "Once upon a time")
#   WEIGHTS   : path to GGUF                 (default models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   PROFILE   : optional kernel-profile path (default: none / auto)

set -euo pipefail
cd "$(dirname "$0")/../.."

TOKENS="${TOKENS:-64}"
TRIALS="${TRIALS:-5}"
PROMPT="${PROMPT:-Once upon a time}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-}"
BIN="./target/release/dismantle"

QUALITY_TOKENS=16

if [[ ! -x "$BIN" ]]; then
    echo "binary not found: $BIN — run 'cargo build --release -p dismantle' first" >&2
    exit 2
fi
if [[ ! -f "$WEIGHTS" ]]; then
    echo "weights not found: $WEIGHTS" >&2
    exit 2
fi

PROFILE_FLAG=()
if [[ -n "$PROFILE" ]]; then
    if [[ ! -f "$PROFILE" ]]; then
        echo "profile not found: $PROFILE" >&2
        exit 2
    fi
    PROFILE_FLAG=(--kernel-profile "$PROFILE")
fi

# Run one generate trial with the given env overrides. Echoes
# "<dec_tps> <decode_ms> <completion>" parsed from the [stats] line.
# $1 is a space-separated string of "KEY=VAL" env pairs (may be empty).
run_one_perf() {
    local env_pairs="$1"
    local stats
    # shellcheck disable=SC2086
    stats=$(
        env $env_pairs \
            nice -n 19 taskpolicy -b "$BIN" generate \
                --weights "$WEIGHTS" \
                --prompt "$PROMPT" \
                --max-new-tokens "$TOKENS" \
                --temperature 0.0 \
                "${PROFILE_FLAG[@]}" \
                2>&1 1>/dev/null \
            | grep -E '^\[stats\]' | tail -1
    )
    if [[ -z "$stats" ]]; then
        echo "0 0 0"
        return
    fi
    local tps dms comp
    tps=$(echo "$stats" | sed -n 's/.*dec_tps=\([0-9.]*\).*/\1/p')
    dms=$(echo "$stats" | sed -n 's/.*decode_ms=\([0-9.]*\).*/\1/p')
    comp=$(echo "$stats" | sed -n 's/.*completion=\([0-9]*\).*/\1/p')
    echo "${tps:-0} ${dms:-0} ${comp:-0}"
}

# Run one generate trial for QUALITY: capture stdout (the generated
# text), greedy temp=0, QUALITY_TOKENS tokens. Echoes the generated
# text (single line, may be empty on failure).
run_one_quality() {
    local env_pairs="$1"
    local out
    # shellcheck disable=SC2086
    out=$(
        env $env_pairs \
            nice -n 19 taskpolicy -b "$BIN" generate \
                --weights "$WEIGHTS" \
                --prompt "$PROMPT" \
                --max-new-tokens "$QUALITY_TOKENS" \
                --temperature 0.0 \
                "${PROFILE_FLAG[@]}" \
                2>/dev/null || true
    )
    # Collapse to a single line for stable diffing. The CLI may emit
    # the prompt prefix as well; we keep the whole stdout so the
    # token-match heuristic compares like-for-like across conditions.
    printf '%s' "$out" | tr '\n' ' '
}

# Median of a space-separated list of numbers (sort + middle element).
median() {
    awk '{
        n = NF;
        for (i = 1; i <= n; i++) a[i] = $i;
        # insertion sort (small n)
        for (i = 2; i <= n; i++) {
            v = a[i]; j = i - 1;
            while (j > 0 && a[j] > v) { a[j+1] = a[j]; j--; }
            a[j+1] = v;
        }
        if (n == 0) { print "0"; exit }
        if (n % 2 == 1) printf "%.3f\n", a[(n+1)/2];
        else            printf "%.3f\n", (a[n/2] + a[n/2+1]) / 2;
    }'
}

# Count how many of the first QUALITY_TOKENS whitespace-separated
# tokens of $2 match $1 position-by-position. Tokens are
# whitespace-delimited words (not BPE pieces — this is a stdout
# string-token check, not a tokenizer-id check). Reports
# "K/QUALITY_TOKENS".
count_token_match() {
    local baseline="$1"
    local candidate="$2"
    awk -v base="$baseline" -v cand="$candidate" -v lim="$QUALITY_TOKENS" '
        BEGIN {
            nb = split(base, ba, /[ \t]+/);
            nc = split(cand, ca, /[ \t]+/);
            n = nb < nc ? nb : nc;
            if (n > lim) n = lim;
            match_count = 0;
            for (i = 1; i <= n; i++) if (ba[i] == ca[i]) match_count++;
            printf "%d/%d", match_count, lim;
        }
    '
}

# Drive one condition: N perf trials, report median dec_tps and ms/tok.
# $1=label, $2=env_pairs. Sets global LAST_MEDIAN_TPS, LAST_MS_PER_TOK.
LAST_MEDIAN_TPS=0
LAST_MS_PER_TOK=0
run_condition_perf() {
    local label="$1"
    local env_pairs="$2"
    local tps_list="" dms_list="" comp_list=""
    for t in $(seq 1 "$TRIALS"); do
        read -r tps dms comp <<< "$(run_one_perf "$env_pairs")"
        tps_list+=" $tps"
        dms_list+=" $dms"
        comp_list+=" $comp"
        printf "  [%s] trial %d/%d: dec_tps=%s decode_ms=%s tokens=%s\n" \
            "$label" "$t" "$TRIALS" "$tps" "$dms" "$comp" >&2
    done
    local tps_med dms_med comp_med ms_per_tok
    tps_med=$(echo "$tps_list" | median)
    dms_med=$(echo "$dms_list" | median)
    comp_med=$(echo "$comp_list" | median)
    ms_per_tok=$(awk -v dms="$dms_med" -v c="$comp_med" \
        'BEGIN { if (c > 0) printf "%.2f", dms/c; else print "0" }')
    LAST_MEDIAN_TPS="$tps_med"
    LAST_MS_PER_TOK="$ms_per_tok"
}

echo "=== W4A8 LM_HEAD per-channel paired bench (Qwen-3B) ==="
echo "weights  = $WEIGHTS"
echo "profile  = ${PROFILE:-<none>}"
echo "prompt   = $PROMPT"
echo "tokens   = $TOKENS  (per perf trial)"
echo "trials   = $TRIALS  per condition"
echo "quality  = ${QUALITY_TOKENS}-token greedy stdout diff vs baseline"
echo

# Condition env pairs. Baseline relies on predec being default-on
# in the current build (per memory: composition_decision_matrix
# 2026-05-26, predec flipped default-on in 6f0209e).
ENV_BASELINE=""
ENV_W4A8_PER_BLOCK="DISMANTLE_QWEN_W4A8=1"
ENV_W4A8_PER_CHANNEL="DISMANTLE_QWEN_W4A8=1 DISMANTLE_QWEN_W4A8_PER_CHANNEL=1"

echo "--- perf trials ---" >&2

run_condition_perf "baseline"             "$ENV_BASELINE"
BASE_TPS="$LAST_MEDIAN_TPS"
BASE_MS="$LAST_MS_PER_TOK"

run_condition_perf "w4a8 per-block"       "$ENV_W4A8_PER_BLOCK"
PERBLK_TPS="$LAST_MEDIAN_TPS"
PERBLK_MS="$LAST_MS_PER_TOK"

run_condition_perf "w4a8 per-channel LM"  "$ENV_W4A8_PER_CHANNEL"
PERCH_TPS="$LAST_MEDIAN_TPS"
PERCH_MS="$LAST_MS_PER_TOK"

echo >&2
echo "--- quality probe (${QUALITY_TOKENS}-tok greedy vs baseline) ---" >&2

BASE_QUAL=$(run_one_quality "$ENV_BASELINE")
PERBLK_QUAL=$(run_one_quality "$ENV_W4A8_PER_BLOCK")
PERCH_QUAL=$(run_one_quality "$ENV_W4A8_PER_CHANNEL")

BASE_MATCH="${QUALITY_TOKENS}/${QUALITY_TOKENS}"  # baseline vs itself
PERBLK_MATCH=$(count_token_match "$BASE_QUAL" "$PERBLK_QUAL")
PERCH_MATCH=$(count_token_match "$BASE_QUAL" "$PERCH_QUAL")

# Relative deltas vs baseline.
delta_pct() {
    awk -v a="$1" -v b="$2" \
        'BEGIN { if (b > 0) printf "%+.2f%%", (a-b)/b*100; else print "n/a" }'
}
PERBLK_DELTA=$(delta_pct "$PERBLK_TPS" "$BASE_TPS")
PERCH_DELTA=$(delta_pct "$PERCH_TPS" "$BASE_TPS")

echo
echo "=== results ==="
printf "%-26s  %10s  %10s  %10s  %s\n" \
    "condition" "med_tps" "ms/tok" "vs_base" "match"
printf "%-26s  %10s  %10s  %10s  %s\n" \
    "--------------------------" "----------" "----------" "----------" "------"
printf "%-26s  %10s  %10s  %10s  %s\n" \
    "baseline (predec only)"   "$BASE_TPS"   "$BASE_MS"   "    -"          "$BASE_MATCH"
printf "%-26s  %10s  %10s  %10s  %s\n" \
    "w4a8 per-block"           "$PERBLK_TPS" "$PERBLK_MS" "$PERBLK_DELTA"  "$PERBLK_MATCH"
printf "%-26s  %10s  %10s  %10s  %s\n" \
    "w4a8 per-channel LM_HEAD" "$PERCH_TPS"  "$PERCH_MS"  "$PERCH_DELTA"   "$PERCH_MATCH"
echo
echo "Note: paired-delta numbers are valid with Claude running."
echo "      Absolute dec_tps may be contaminated — see bench_contamination.md."
