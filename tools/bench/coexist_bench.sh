#!/usr/bin/env bash
#
# tools/bench/coexist_bench.sh — bench WITHOUT requiring Claude to quit.
#
# Trades absolute accuracy for the ability to run while Claude.app is open.
# Use when:
#   - You have an overnight Claude session you don't want to interrupt.
#   - You want fast relative-change feedback between two commits.
#
# Do NOT use for:
#   - Shippable v-version baselines (use clean_bench.sh from a Claude-quit terminal).
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
#
# Usage:
#   bash tools/bench/coexist_bench.sh
#   TOKENS=64 bash tools/bench/coexist_bench.sh    # longer per-trial
#   ANCHOR=2.207 bash tools/bench/coexist_bench.sh # override clean anchor

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TOKENS="${TOKENS:-32}"
TRIALS="${TRIALS:-6}"
ANCHOR="${ANCHOR:-2.207}"   # last clean-bench dec_tps measured
BIN="./target/release/dismantle"

OUT_DIR="bench_results/coexist_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

ts()  { date -u +%FT%TZ; }
log() { printf '%s %s\n' "$(ts)" "$*"; }

log "=== COEXIST MODE bench — Claude.app may be running ==="
log "  trials=${TRIALS}  tokens=${TOKENS}  anchor=${ANCHOR} dec_tps"
log "  output: $OUT_DIR"

# Pre-flight: build sanity (no Claude gate, that's the point).
if ! "$BIN" --version >/dev/null 2>&1; then
    log "binary missing or broken — running cargo build --release --workspace..."
    cargo build --release --workspace >/dev/null 2>&1
fi

# Confirm Claude.app is actually running (else just use clean_bench).
if pgrep -f "Claude.app" >/dev/null 2>&1; then
    log "Claude.app: present (expected for coexist mode)"
    log "  → mitigations active: best-tier scheduling, caffeinate, 6-trial median"
else
    log "Claude.app: NOT running — consider clean_bench.sh for higher-fidelity numbers"
fi

# Spotlight check (don't gate, just warn).
TOP=$(ps -axo %cpu,comm | sort -nr | awk '$2 !~ /dismantle|Claude/ {print; exit}')
log "top non-bench non-Claude process: $TOP"

# ─── TRIAL LOOP ───────────────────────────────────────────────────────────────
# Note: NO `nice -n 19 taskpolicy -b` here. That's the Phase 1 cooperative
# scheduling rule for hauls running ALONGSIDE slm — for this bench we want
# the OPPOSITE: elevate priority so Claude gives us cycles, not the other way.
TRIALS_TPS=()
for i in $(seq 1 "$TRIALS"); do
    OUT_JSON="$OUT_DIR/trial_${i}.json"
    TRACE_JSON="$OUT_DIR/trial_${i}.trace.json"

    # 30s settle between trials so we don't catch a Claude GC pause.
    if [[ $i -gt 1 ]]; then
        log "settling 30s between trials..."
        sleep 30
    fi

    log "=== trial $i / $TRIALS ==="
    set +e
    # Priority elevation, no-sudo:
    #   taskpolicy -t 0 -l 0 → best throughput + latency tier
    #   caffeinate -di       → no idle/display sleep, no throttle
    DISMANTLE_TRACE_DISPATCH=1 \
    perl -e 'alarm 600; exec @ARGV' \
        caffeinate -di \
        taskpolicy -t 0 -l 0 \
        "$BIN" bench --trace-dispatch \
            --backend dismantle --suite decode \
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

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
log ""
log "=== SUMMARY (COEXIST MODE — not authoritative) ==="
printf 'all trials:   %s\n' "${TRIALS_TPS[*]}"

# Compute median (drop zeros = discards).
VALID=$(printf '%s\n' "${TRIALS_TPS[@]}" | awk '$1 > 0' | sort -n)
COUNT=$(printf '%s\n' "$VALID" | wc -l | tr -d ' ')
if [[ "$COUNT" -lt 3 ]]; then
    log "FAIL: only $COUNT valid trials (need >= 3). See $OUT_DIR/*.stderr"
    exit 1
fi
MEDIAN_IDX=$(( (COUNT + 1) / 2 ))
MEDIAN=$(printf '%s\n' "$VALID" | sed -n "${MEDIAN_IDX}p")
MIN=$(printf '%s\n' "$VALID" | head -1)
MAX=$(printf '%s\n' "$VALID" | tail -1)
SPREAD=$(awk -v min="$MIN" -v max="$MAX" -v med="$MEDIAN" \
    'BEGIN { if (med > 0) printf "%.1f", (max - min) / med * 100; else print "n/a" }')

printf 'valid trials: %d / %d\n' "$COUNT" "$TRIALS"
printf 'median:       %s dec_tps\n' "$MEDIAN"
printf 'min..max:     %s .. %s\n' "$MIN" "$MAX"
printf 'spread:       %s%%\n' "$SPREAD"

# Contamination ratio — coexist median should be ~1/3 of clean anchor.
RATIO=$(awk -v c="$MEDIAN" -v a="$ANCHOR" 'BEGIN { if (c>0) printf "%.2f", a/c; else print "n/a" }')
ESTIMATED_CLEAN=$(awk -v c="$MEDIAN" -v r=3.0 'BEGIN { printf "%.2f", c*r }')

printf '\n--- contamination analysis ---\n'
printf 'clean anchor: %s dec_tps\n' "$ANCHOR"
printf 'observed/clean ratio: 1/%s  (clean is %sx faster)\n' "$RATIO" "$RATIO"
printf 'expected ratio at typical Claude load: 1/3 to 1/4\n'
printf 'estimated clean dec_tps (median * 3): ~%s\n' "$ESTIMATED_CLEAN"

# Structural metrics from trial 3 (load-independent).
log ""
log "=== STRUCTURAL METRICS (load-independent — can compare across modes) ==="
jq '.results.trial_stats[0] | {
    metal_buffers_created_per_token,
    cpu_alloc_bytes_per_token,
    dispatch_commits_per_token,
    gpu_resident_bytes_per_token
}' "$OUT_DIR/trial_3.json" 2>/dev/null || \
    jq '.results.trial_stats[0] | {
        metal_buffers_created_per_token,
        cpu_alloc_bytes_per_token,
        dispatch_commits_per_token
    }' "$OUT_DIR/trial_1.json"

# Write summary.md
{
    printf '# coexist bench — %s\n\n' "$(ts)"
    printf '**Mode:** COEXIST (Claude.app present, not closed)\n'
    printf '**Authoritative:** No — use only for relative comparison.\n\n'
    printf '## Trials\n'
    for i in $(seq 1 "$TRIALS"); do
        printf '- trial %d: %s\n' "$i" "${TRIALS_TPS[$((i-1))]}"
    done
    printf '\n## Stats\n'
    printf '- valid trials: %d / %d\n' "$COUNT" "$TRIALS"
    printf '- median: **%s dec_tps**\n' "$MEDIAN"
    printf '- spread: %s%%\n' "$SPREAD"
    printf '\n## Contamination\n'
    printf '- clean anchor: %s dec_tps\n' "$ANCHOR"
    printf '- contamination factor: %sx\n' "$RATIO"
    printf '- estimated clean: ~%s dec_tps\n' "$ESTIMATED_CLEAN"
} > "$OUT_DIR/summary.md"

log ""
log "results: $OUT_DIR/summary.md"
log "exit 0"
