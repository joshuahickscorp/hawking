#!/usr/bin/env bash
# tools/bench/eagle5_paired_bench.sh — paired bench for the eagle5 v2
# spec-decode runtime path.
#
# Runs `dismantle generate` with --speculate off (baseline) and with
# --speculate eagle5 at K∈{2,4,8} on the same prompt + tokens, n trials
# each, and prints dec_tps + draft_accept/reject + ms/token.
#
# Per `memory/feedback_bench_with_claude_open.md`, paired-delta runs
# work with Claude open because contamination cancels in the relative
# delta. Absolute-tps gates would need `tools/bench/clean_bench.sh`
# from a Cmd+Q-Claude terminal.
#
# Env:
#   WEIGHTS        : path to GGUF (default models/deepseek-v2-lite-q4.gguf)
#   PROFILE        : path to kernel profile (default profiles/.../m3pro18.json)
#   PROMPT         : prompt string (default "Once upon a time")
#   TOKENS         : max_new_tokens per trial (default 64)
#   TRIALS         : n trials per condition (default 5)
#   EAGLE5_HEAD    : optional trained-head safetensors path
#                    (omit → mock head fallback)

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
PROMPT="${PROMPT:-Once upon a time}"
TOKENS="${TOKENS:-64}"
TRIALS="${TRIALS:-5}"
EAGLE5_HEAD="${EAGLE5_HEAD:-}"
BIN="./target/release/hawking"

if [[ ! -x "$BIN" ]]; then
    echo "❌ binary not found: $BIN — run 'cargo build --release -p dismantle' first" >&2
    exit 2
fi
if [[ ! -f "$WEIGHTS" ]]; then
    echo "❌ weights not found: $WEIGHTS" >&2
    exit 2
fi

# Build the optional --eagle5-head flag (empty if no checkpoint).
EAGLE5_FLAG=""
if [[ -n "$EAGLE5_HEAD" ]]; then
    EAGLE5_FLAG="--eagle5-head $EAGLE5_HEAD"
fi

# One-trial driver: emits "<dec_tps> <accepted> <rejected> <decode_ms> <completion>".
run_one() {
    local spec_args="$1"
    local out
    out=$(
        "$BIN" generate \
            --weights "$WEIGHTS" \
            --prompt "$PROMPT" \
            --max-new-tokens "$TOKENS" \
            --temperature 0.0 \
            --kernel-profile "$PROFILE" \
            $spec_args \
            >/dev/null 2>&1 || true
        # The stats line goes to stderr; capture it via 2>&1 in a fresh run.
    )
    # Redo with stderr capture (CLI prints generated text to stdout, stats
    # to stderr). We only care about the [stats] line.
    local stats
    stats=$(
        "$BIN" generate \
            --weights "$WEIGHTS" \
            --prompt "$PROMPT" \
            --max-new-tokens "$TOKENS" \
            --temperature 0.0 \
            --kernel-profile "$PROFILE" \
            $spec_args \
            2>&1 1>/dev/null \
        | grep -E '^\[stats\]' | tail -1
    )
    if [[ -z "$stats" ]]; then
        echo "0 0 0 0 0"
        return
    fi
    local tps acc rej dms comp
    tps=$(echo "$stats" | sed -n 's/.*dec_tps=\([0-9.]*\).*/\1/p')
    acc=$(echo "$stats" | sed -n 's/.*draft_accepted=\([0-9]*\).*/\1/p')
    rej=$(echo "$stats" | sed -n 's/.*draft_rejected=\([0-9]*\).*/\1/p')
    dms=$(echo "$stats" | sed -n 's/.*decode_ms=\([0-9.]*\).*/\1/p')
    comp=$(echo "$stats" | sed -n 's/.*completion=\([0-9]*\).*/\1/p')
    echo "${tps:-0} ${acc:-0} ${rej:-0} ${dms:-0} ${comp:-0}"
}

# Compute mean of a space-separated list of numbers.
mean() {
    awk '{ for (i=1; i<=NF; i++) s+=$i; n+=NF } END { if (n>0) printf "%.3f\n", s/n; else print "0" }'
}

# Condition driver: prints "<label> dec_tps_mean acc_rate ms_per_tok"
# given a label and the --speculate args.
run_condition() {
    local label="$1"
    local spec_args="$2"
    local tps_list="" acc_list="" rej_list="" dms_list="" comp_list=""
    for t in $(seq 1 "$TRIALS"); do
        read tps acc rej dms comp <<< "$(run_one "$spec_args")"
        tps_list="$tps_list $tps"
        acc_list="$acc_list $acc"
        rej_list="$rej_list $rej"
        dms_list="$dms_list $dms"
        comp_list="$comp_list $comp"
    done
    local tps_mean acc_mean rej_mean dms_mean comp_mean
    tps_mean=$(echo "$tps_list" | mean)
    acc_mean=$(echo "$acc_list" | mean)
    rej_mean=$(echo "$rej_list" | mean)
    dms_mean=$(echo "$dms_list" | mean)
    comp_mean=$(echo "$comp_list" | mean)
    # Accept rate = accepted / (accepted+rejected); guard div0 for no-spec.
    local acc_rate ms_per_tok
    acc_rate=$(awk -v a="$acc_mean" -v r="$rej_mean" 'BEGIN { if (a+r > 0) printf "%.3f", a/(a+r); else print "n/a" }')
    ms_per_tok=$(awk -v dms="$dms_mean" -v c="$comp_mean" 'BEGIN { if (c > 0) printf "%.2f", dms/c; else print "0" }')
    printf "%-28s dec_tps=%7.2f  ms/tok=%6.2f  accept=%s  acc=%5.1f  rej=%5.1f  tokens=%4.0f  trials=%d\n" \
        "$label" "$tps_mean" "$ms_per_tok" "$acc_rate" "$acc_mean" "$rej_mean" "$comp_mean" "$TRIALS"
}

echo "=== eagle5 paired bench ==="
echo "weights  = $WEIGHTS"
echo "profile  = $PROFILE"
echo "prompt   = $PROMPT"
echo "tokens   = $TOKENS  (per trial)"
echo "trials   = $TRIALS  per condition"
if [[ -n "$EAGLE5_HEAD" ]]; then
    echo "head     = $EAGLE5_HEAD  (trained-head loader)"
else
    echo "head     = (mock random weights — near-1/vocab accept rate)"
fi
echo

run_condition "no-spec greedy"              ""
run_condition "eagle5 K=2 (mock head)"      "--speculate eagle5 --verify-window 2 $EAGLE5_FLAG"
run_condition "eagle5 K=4 (mock head)"      "--speculate eagle5 --verify-window 4 $EAGLE5_FLAG"
run_condition "eagle5 K=8 (mock head)"      "--speculate eagle5 --verify-window 8 $EAGLE5_FLAG"
