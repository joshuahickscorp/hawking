#!/usr/bin/env bash
#
# tools/bench/coexist_bench.sh — bench WITHOUT requiring the agent to quit.
#
# Trades absolute accuracy for the ability to run while the agent app is open.
# Use when:
#   - You have an overnight agent session you don't want to interrupt.
#   - You want fast relative-change feedback between two commits.
#
# Do NOT use for:
#   - Shippable v-version baselines (use clean_bench.sh from an agent-quit terminal).
#   - Flipping defaults via the bench-first gate.
#
# Mitigations vs naked quick_bench:
#   1. Sets best throughput + latency tiers (taskpolicy -t 0 -l 0).
#   2. caffeinate -di prevents idle/display sleep + CPU throttling.
#   3. Explicitly does NOT use the haul-mode 'nice 19 / taskpolicy -b'
#      that yields to slm — for benching we want the OPPOSITE of yield.
#   4. 6 trials (vs quick_bench's 3) for tighter median.
#   5. Computes contamination ratio against the most recent clean anchor.
#   6. Stamps "COEXIST MODE" in every output so it can't be confused with clean.
#   7. Warmup trial (throwaway) to prime OS page cache + Metal pipeline cache.
#   8. Trimmed mean (25%) + IQR + 95% CI for robust statistics.
#   9. Appends every run to bench_results/bench_history.jsonl for cross-commit diff.
#
# Usage:
#   bash tools/bench/coexist_bench.sh
#   TOKENS=64 bash tools/bench/coexist_bench.sh    # longer per-trial
#   ANCHOR=2.207 bash tools/bench/coexist_bench.sh # override clean anchor
#   bash tools/bench/coexist_bench.sh --quiet --profile /tmp/profile.json
#
# Standardized bench parameters (v1.1.0 roadmap):
#   dev iteration bench:    TRIALS=4 TOKENS=24  (~3-5 min, default)
#   sub-phase commit gate:  TRIALS=4 TOKENS=24  (same as dev)
#   phase close bench:      TRIALS=6 TOKENS=64  (~15 min)
#   authoritative ship:     TRIALS=10 TOKENS=64 with clean_bench.sh (~25 min)

set -euo pipefail
cd "$(dirname "$0")/../.."

_agent_env="$(git rev-parse --show-toplevel 2>/dev/null)/.agent_env"
[ -f "$_agent_env" ] && source "$_agent_env"
unset _agent_env

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TOKENS="${TOKENS:-32}"
TRIALS="${TRIALS:-6}"
ANCHOR="${ANCHOR:-2.207}"   # last clean-bench dec_tps measured
BIN="./target/release/hawking"
QUIET=0

# ─── ARGUMENT PARSING ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quiet|-q) QUIET=1; shift ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --weights) WEIGHTS="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --anchor) ANCHOR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

OUT_DIR="bench_results/coexist_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

ts()  { date -u +%FT%TZ; }
log() {
    if [[ "$QUIET" -eq 0 ]]; then
        printf '%s %s\n' "$(ts)" "$*"
    fi
}

log "=== COEXIST MODE bench — the agent app may be running ==="
log "  trials=${TRIALS}  tokens=${TOKENS}  anchor=${ANCHOR} dec_tps"
log "  output: $OUT_DIR"

# Pre-flight: build sanity (no agent gate, that's the point).
if ! "$BIN" --version >/dev/null 2>&1; then
    log "binary missing or broken — running cargo build --release --workspace..."
    cargo build --release --workspace >/dev/null 2>&1
fi

# Confirm the agent app is actually running (else just use clean_bench).
if pgrep -f "${AGENT_APP_PGREP:?see .agent_env.example}" >/dev/null 2>&1; then
    log "agent app: present (expected for coexist mode)"
    log "  → mitigations active: best-tier scheduling, caffeinate, ${TRIALS}-trial median"
else
    log "agent app: NOT running — consider clean_bench.sh for higher-fidelity numbers"
fi

# Spotlight check (don't gate, just warn).
TOP=$(ps -axo %cpu,comm | sort -nr | awk -v agent_pat="${AGENT_APP_PGREP:?see .agent_env.example}" '$2 !~ /dismantle/ && $2 !~ agent_pat {print; exit}')
log "top non-bench non-agent process: $TOP"

# ─── WARMUP TRIAL ─────────────────────────────────────────────────────────────
# One throwaway 4-token run to warm OS page cache + Metal pipeline cache +
# tokenizer init. Eliminates the "first trial is always slow" outlier pattern.
log "=== WARMUP (throwaway — 4 tokens, not counted) ==="
WARMUP_JSON="$OUT_DIR/warmup.json"
set +e
perl -e 'alarm 120; exec @ARGV' \
    caffeinate -di \
    taskpolicy -t 0 -l 0 \
    "$BIN" bench \
        --backend hawking --suite decode \
        --weights "$WEIGHTS" --trials 1 --max-new-tokens 4 \
        --kernel-profile "$PROFILE" \
        --json "$WARMUP_JSON" \
    >"$OUT_DIR/warmup.stdout" 2>"$OUT_DIR/warmup.stderr"
set -e
log "warmup done — caches primed"

# ─── TRIAL LOOP ───────────────────────────────────────────────────────────────
# Note: NO `nice -n 19 taskpolicy -b` here. That's the Phase 1 cooperative
# scheduling rule for hauls running ALONGSIDE slm — for this bench we want
# the OPPOSITE: elevate priority so the agent gives us cycles, not the other way.
TRIALS_TPS=()
for i in $(seq 1 "$TRIALS"); do
    OUT_JSON="$OUT_DIR/trial_${i}.json"
    TRACE_JSON="$OUT_DIR/trial_${i}.trace.json"

    # 30s settle between trials so we don't catch an agent GC pause.
    if [[ $i -gt 1 ]]; then
        log "settling 30s between trials..."
        sleep 30
    fi

    log "=== trial $i / $TRIALS ==="
    set +e
    # Priority elevation, no-sudo:
    #   taskpolicy -t 0 -l 0 → best throughput + latency tier
    #   caffeinate -di       → no idle/display sleep, no throttle
    HAWKING_TRACE_DISPATCH=1 \
    perl -e 'alarm 600; exec @ARGV' \
        caffeinate -di \
        taskpolicy -t 0 -l 0 \
        "$BIN" bench --trace-dispatch \
            --backend hawking --suite decode \
            --weights "$WEIGHTS" --trials 1 --max-new-tokens "$TOKENS" \
            --kernel-profile "$PROFILE" \
            --json "$OUT_JSON" \
            --trace-json "$TRACE_JSON" \
            >"$OUT_DIR/trial_${i}.stdout" 2>"$OUT_DIR/trial_${i}.stderr"
    EC=$?
    set -e

    if [[ $EC -ne 0 ]] || [[ ! -f "$OUT_JSON" ]]; then
        log "DISCARD trial $i: command failed (exit=$EC)"
        TRIALS_TPS+=("0")
        continue
    fi

    TPS=$(jq -r '.results.trial_stats[0].decode_tps // 0' "$OUT_JSON")
    TOK=$(jq -r '.results.trial_stats[0].completion_tokens // 0' "$OUT_JSON")
    log "trial $i: ${TPS} dec_tps  (${TOK} tokens)"
    TRIALS_TPS+=("$TPS")
done

# ─── STATISTICS ───────────────────────────────────────────────────────────────
log ""
log "=== SUMMARY (COEXIST MODE — not authoritative) ==="
printf 'all trials:   %s\n' "${TRIALS_TPS[*]}"

# Drop zeros (= discarded trials), sort ascending.
SORTED=$(printf '%s\n' "${TRIALS_TPS[@]}" | awk '$1 > 0' | sort -n)
COUNT=$(printf '%s\n' "$SORTED" | wc -l | tr -d ' ')
if [[ "$COUNT" -lt 3 ]]; then
    log "FAIL: only $COUNT valid trials (need >= 3). See $OUT_DIR/*.stderr"
    exit 1
fi

# Median (50th percentile)
MEDIAN_IDX=$(( (COUNT + 1) / 2 ))
MEDIAN=$(printf '%s\n' "$SORTED" | sed -n "${MEDIAN_IDX}p")

MIN=$(printf '%s\n' "$SORTED" | head -1)
MAX=$(printf '%s\n' "$SORTED" | tail -1)

# Trimmed mean: for N>=4, drop bottom 25% and top 25%; mean the middle.
# For N<3 (already guarded above) or N<4, fall back to median.
if [[ "$COUNT" -ge 4 ]]; then
    TRIM_N=$(( COUNT / 4 ))
    TRIMMED_VALS=$(printf '%s\n' "$SORTED" | tail -n "+$((TRIM_N + 1))" | head -n "$(( COUNT - 2 * TRIM_N ))")
    TRIMMED_MEAN=$(printf '%s\n' "$TRIMMED_VALS" | awk '{ s+=$1; n++ } END { printf "%.3f", s/n }')
    ESTIMATOR="trimmed_mean_25pct"
else
    TRIMMED_MEAN="$MEDIAN"
    ESTIMATOR="median_fallback_N_lt_4"
fi

# IQR (Q3 - Q1).
Q1_IDX=$(( (COUNT + 3) / 4 ))
Q3_IDX=$(( (3 * COUNT + 3) / 4 ))
[[ "$Q1_IDX" -lt 1 ]] && Q1_IDX=1
[[ "$Q3_IDX" -gt "$COUNT" ]] && Q3_IDX=$COUNT
Q1=$(printf '%s\n' "$SORTED" | sed -n "${Q1_IDX}p")
Q3=$(printf '%s\n' "$SORTED" | sed -n "${Q3_IDX}p")
IQR=$(awk -v q1="$Q1" -v q3="$Q3" 'BEGIN { printf "%.3f", q3-q1 }')

# 95% CI via ±1.96 × σ/√N (normal approximation).
MEAN=$(printf '%s\n' "$SORTED" | awk '{ s+=$1; n++ } END { printf "%.6f", s/n }')
if [[ "$COUNT" -ge 2 ]]; then
    STDDEV=$(printf '%s\n' "$SORTED" | awk -v mean="$MEAN" \
        '{ s+=($1-mean)^2; n++ } END { printf "%.6f", sqrt(s/(n-1)) }')
else
    STDDEV="0"
fi
CI_HALF=$(awk -v sd="$STDDEV" -v n="$COUNT" 'BEGIN { printf "%.3f", 1.96*sd/sqrt(n) }')
CI_LO=$(awk -v m="$MEDIAN" -v h="$CI_HALF" 'BEGIN { printf "%.3f", m-h }')
CI_HI=$(awk -v m="$MEDIAN" -v h="$CI_HALF" 'BEGIN { printf "%.3f", m+h }')

# Spread flag.
SPREAD_FLAG=""
if awk -v iqr="$IQR" -v med="$MEDIAN" 'BEGIN { exit (med>0 && iqr/med>0.15) ? 0 : 1 }'; then
    SPREAD_FLAG="  ⚠ SPREAD HIGH (IQR/median > 15%)"
fi

printf 'valid trials: %d / %d\n' "$COUNT" "$TRIALS"
printf 'median:       %s dec_tps (95%% CI: [%s, %s], IQR: %s)%s\n' \
    "$MEDIAN" "$CI_LO" "$CI_HI" "$IQR" "$SPREAD_FLAG"
printf 'trimmed_mean: %s dec_tps (%s)\n' "$TRIMMED_MEAN" "$ESTIMATOR"
printf 'min..max:     %s .. %s\n' "$MIN" "$MAX"

# Contamination ratio.
RATIO=$(awk -v c="$MEDIAN" -v a="$ANCHOR" 'BEGIN { if (c>0) printf "%.2f", a/c; else print "n/a" }')
ESTIMATED_CLEAN_FROM_RATIO=$(awk -v c="$MEDIAN" -v r="$RATIO" 'BEGIN { printf "%.2f", c*r }')
ESTIMATED_CLEAN_3X=$(awk -v c="$MEDIAN" 'BEGIN { printf "%.2f", c*3.0 }')

printf '\n%s\n' "--- contamination analysis ---"
printf '%s\n' "clean anchor:                       $ANCHOR dec_tps"
printf '%s\n' "observed coexist median:            $MEDIAN dec_tps"
printf '%s\n' "observed contamination factor:      ${RATIO}x  (anchor / observed)"
printf '%s\n' "estimated clean (median × ratio):   ~$ESTIMATED_CLEAN_FROM_RATIO dec_tps"
printf '%s\n' "worst-case estimated clean (× 3):   ~$ESTIMATED_CLEAN_3X dec_tps"
printf '%s\n' "(use the FROM_RATIO number when comparing successive coexist runs;"
printf '%s\n' " use × 3 only when the anchor is stale / from a different commit)"

# Structural metrics from trial 3 (load-independent).
log ""
log "=== STRUCTURAL METRICS (load-independent — can compare across modes) ==="
STRUCT_JSON=""
for try_trial in 3 2 1; do
    if [[ -f "$OUT_DIR/trial_${try_trial}.json" ]]; then
        STRUCT_JSON=$(jq '.results.trial_stats[0] | {
            metal_buffers_created_per_token,
            cpu_alloc_bytes_per_token,
            dispatch_commits_per_token,
            gpu_resident_bytes_per_token
        }' "$OUT_DIR/trial_${try_trial}.json" 2>/dev/null || true)
        [[ -n "$STRUCT_JSON" ]] && break
    fi
done
[[ -n "$STRUCT_JSON" ]] && printf '%s\n' "$STRUCT_JSON" || log "(structural metrics unavailable)"

# ─── BENCH HISTORY JSONL ──────────────────────────────────────────────────────
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
MODEL_TAG=$(basename "$WEIGHTS" .gguf 2>/dev/null || echo "unknown")

# Build trials JSON array (valid only).
VALID_TRIALS_JSON=$(printf '%s\n' "${TRIALS_TPS[@]}" | \
    awk '$1>0 { printf "%s,", $1 }' | sed 's/,$//')

# Structural metrics for JSONL (scalars from the first good trial json).
STRUCT_CPT="null"
STRUCT_BPT="null"
STRUCT_APT="null"
for try_trial in 3 2 1; do
    if [[ -f "$OUT_DIR/trial_${try_trial}.json" ]]; then
        STRUCT_CPT=$(jq -r '.results.trial_stats[0].dispatch_commits_per_token // "null"' \
            "$OUT_DIR/trial_${try_trial}.json" 2>/dev/null || echo "null")
        STRUCT_BPT=$(jq -r '.results.trial_stats[0].metal_buffers_created_per_token // "null"' \
            "$OUT_DIR/trial_${try_trial}.json" 2>/dev/null || echo "null")
        STRUCT_APT=$(jq -r '.results.trial_stats[0].cpu_alloc_bytes_per_token // "null"' \
            "$OUT_DIR/trial_${try_trial}.json" 2>/dev/null || echo "null")
        [[ "$STRUCT_CPT" != "null" ]] && break
    fi
done

mkdir -p bench_results
cat >> bench_results/bench_history.jsonl <<EOF
{"timestamp":"$(ts)","commit":"${COMMIT}","branch":"${BRANCH}","tool":"coexist_bench","config":{"trials":${TRIALS},"tokens":${TOKENS},"model":"${MODEL_TAG}"},"results":{"median":${MEDIAN},"trimmed_mean":${TRIMMED_MEAN},"ci_95_lo":${CI_LO},"ci_95_hi":${CI_HI},"iqr":${IQR},"estimator":"${ESTIMATOR}","trials":[${VALID_TRIALS_JSON}]},"structural":{"commits_per_token":${STRUCT_CPT},"buffers_per_token":${STRUCT_BPT},"alloc_bytes_per_token":${STRUCT_APT}}}
EOF
log "appended to bench_results/bench_history.jsonl"

# ─── SUMMARY.MD ───────────────────────────────────────────────────────────────
{
    printf '# coexist bench — %s\n\n' "$(ts)"
    printf '**Mode:** COEXIST (the agent app may be present, not closed)\n'
    printf '**Authoritative:** No — use only for relative comparison.\n\n'
    printf '## Trials\n'
    for i in $(seq 1 "$TRIALS"); do
        printf '%s\n' "- trial $i: ${TRIALS_TPS[$((i-1))]}"
    done
    printf '\n## Stats\n'
    printf '%s\n' "- valid trials: $COUNT / $TRIALS"
    printf '%s\n' "- median: **$MEDIAN dec_tps** (95% CI: [$CI_LO, $CI_HI], IQR: $IQR)${SPREAD_FLAG}"
    printf '%s\n' "- trimmed_mean: $TRIMMED_MEAN dec_tps ($ESTIMATOR)"
    printf '%s\n' "- min..max: $MIN .. $MAX"
    printf '\n## Contamination\n'
    printf '%s\n' "- clean anchor: $ANCHOR dec_tps"
    printf '%s\n' "- contamination factor: ${RATIO}x"
    printf '%s\n' "- estimated clean (× ratio): ~$ESTIMATED_CLEAN_FROM_RATIO dec_tps"
    printf '%s\n' "- worst-case estimated clean (× 3): ~$ESTIMATED_CLEAN_3X dec_tps"
    printf '\n## History\n'
    printf '%s\n' "- appended to bench_results/bench_history.jsonl"
    printf '%s\n' "- commit: $COMMIT  branch: $BRANCH"
} > "$OUT_DIR/summary.md"

log ""
log "results: $OUT_DIR/summary.md"
log "exit 0"
