#!/usr/bin/env bash
# Smart overnight chain v3 — applies Q8 KV patch + rebuilds + runs targeted
# bench modules answering the open questions from the prior chain.
#
# Key changes vs v2:
#   - M0 applies reports/patches/session_C_q8_kv_wiring.patch + rebuilds
#     binary so --q8-kv actually exists (prior chains found binary unflagged)
#   - M1 stack-combination matrix to find why STACK<L1 in prior data
#   - M2 long-form Q8 KV at 256/512/1024/2048 tokens
#   - M3 multi-prompt with TRIALS=10 for tight error bars
#   - M9 variance hunt — 30 trials on best config for confidence
#   - RAM watcher in addition to disk watcher
#   - All idempotent; nohup + disown survives Claude crash
#
# Estimated: ~6-8 h compute. No commits made.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight_chain_v3"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/chain.log"
STATUS="$LOG_DIR/status.json"
echo "$$" > "$LOG_DIR/pid"
START_EPOCH=$(date +%s)

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

write_status() {
    local module="$1" state="$2" note="${3:-}"
    local uptime=$(( $(date +%s) - START_EPOCH ))
    local free_gb=$(df -g . | awk 'NR==2 {print $4}')
    cat > "$STATUS.tmp" <<EOF
{
  "ts": "$(stamp)",
  "pid": $$,
  "current_module": "$module",
  "state": "$state",
  "note": "$note",
  "uptime_seconds": $uptime,
  "free_disk_gb": $free_gb
}
EOF
    mv "$STATUS.tmp" "$STATUS"
}

wait_if_paused() {
    if [[ -f artifacts/runs/PAUSE ]]; then
        log "⏸  PAUSED at $1"; write_status "$1" "paused"
        while [[ -f artifacts/runs/PAUSE ]]; do
            sleep 15
            if [[ -f artifacts/runs/RESUME ]]; then
                rm -f artifacts/runs/PAUSE artifacts/runs/RESUME
                log "▶  resumed"; break
            fi
        done
    fi
}

# --- Background watchers ---
{ while true; do sleep 60
    FREE=$(df -g . | awk 'NR==2 {print $4}')
    [[ "$FREE" -lt 10 ]] && { echo "[$(stamp)] DISK CRITICAL ($FREE GB)" >> "$LOG"; touch artifacts/runs/PAUSE; break; }
    PAGES=$(vm_stat | grep "Pages free:" | awk '{print $3}' | tr -d '.')
    RAM_MB=$((PAGES * 16384 / 1024 / 1024))
    if [[ "$RAM_MB" -lt 300 ]]; then
        echo "[$(stamp)] RAM CRITICAL (${RAM_MB} MB free)" >> "$LOG"
        # Don't auto-PAUSE; just log — pause might leave a stuck mid-trial
    fi
done; } &
DW_PID=$!
trap "kill $DW_PID 2>/dev/null || true" EXIT

log "=== smart chain v3 starting (pid=$$) ==="

# ============================================================
# M0 — Apply Q8 KV patch + rebuild binary
# ============================================================
log "=== M0: Q8 KV patch + rebuild ==="
write_status "M0_patch_rebuild" "running"
M0_DONE="$LOG_DIR/M0.done"
PATCH=reports/patches/session_C_q8_kv_wiring.patch
if [[ -f "$M0_DONE" ]]; then
    log "M0 marker present, skipping"
elif [[ -f "$PATCH" ]]; then
    if git apply --check "$PATCH" 2>/dev/null; then
        git apply "$PATCH" 2>>"$LOG" && log "patch applied" || log "patch apply FAILED"
    else
        log "patch already applied or conflicts; continuing"
    fi
    log "rebuilding..."
    if cargo build --release -p dismantle >>"$LOG" 2>&1; then
        log "rebuild OK"
        # Verify --q8-kv now present
        if ./target/release/dismantle generate --help 2>&1 | grep -q '\-\-q8-kv'; then
            log "✓ --q8-kv flag NOW available"
            touch "$M0_DONE"
        else
            log "⚠️  --q8-kv flag still missing after rebuild"
        fi
    else
        log "❌ rebuild failed; chain continues with old binary"
    fi
else
    log "no patch file at $PATCH; skipping"
    touch "$M0_DONE"
fi

# Detect Q8 KV availability for downstream modules
Q8_KV_ARG=""
if ./target/release/dismantle generate --help 2>&1 | grep -q '\-\-q8-kv'; then
    Q8_KV_ARG="--q8-kv"
    log "Q8 KV available: $Q8_KV_ARG"
fi

# Common flags
WEIGHTS=models/deepseek-v2-lite-q4.gguf
PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json
VOCAB=artifacts/calibration/analysis/vocab_whitelist_995.json
TIER_AGG=artifacts/calibration/tier_maps/v2_lite_aggressive_down_q4.json
TIER_DEF=artifacts/calibration/tier_maps/v2_lite_default.json

run_trial() {
    local prompt="$1"; local tokens="$2"; local seed="$3"; shift 3
    ./target/release/dismantle generate \
        --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
        --prompt "$prompt" --max-new-tokens "$tokens" --seed "$seed" \
        "$@" 2>&1 | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2
}

# ============================================================
# M1 — Stack-combination matrix (why STACK<L1?)
# ============================================================
wait_if_paused "M1_stack_matrix"
log "=== M1: stack-combination matrix ==="
write_status "M1_stack_matrix" "running"
M1_OUT="$LOG_DIR/M1_stack_matrix.md"
if [[ -f "$LOG_DIR/M1.done" ]]; then
    log "M1 done, skipping"
else
    {
        echo "# Stack-combination matrix — $(stamp)"
        echo ""
        echo "Investigates why STACK (vocab+tier+ngram) was less than L1 alone in prior runs. 8 combinations × 5 trials."
        echo ""
        echo "| Config | t1 | t2 | t3 | t4 | t5 |"
        echo "|---|---:|---:|---:|---:|---:|"
        declare -a CFGS=(
            "baseline:"
            "L1_vocab:--vocab-prune-path $VOCAB"
            "L2_tier_agg:--quant-tier-map-path $TIER_AGG"
            "L4_ngram:--speculate ngram --verify-window 4"
            "L1+L2:--vocab-prune-path $VOCAB --quant-tier-map-path $TIER_AGG"
            "L1+L4:--vocab-prune-path $VOCAB --speculate ngram --verify-window 4"
            "L2+L4:--quant-tier-map-path $TIER_AGG --speculate ngram --verify-window 4"
            "ALL:--vocab-prune-path $VOCAB --quant-tier-map-path $TIER_AGG --speculate ngram --verify-window 4"
        )
        for entry in "${CFGS[@]}"; do
            label=${entry%%:*}
            flags=${entry#*:}
            row="| $label |"
            for t in 1 2 3 4 5; do
                tps=$(run_trial "Once upon a time" 64 $t $flags)
                [[ -z "$tps" ]] && tps="fail"
                row="$row $tps |"
            done
            echo "$row"
        done
    } > "$M1_OUT"
    touch "$LOG_DIR/M1.done"
fi
write_status "M1_stack_matrix" "done"

# ============================================================
# M2 — Long-form Q8 KV (256/512/1024/2048 tok × on/off × 5 trials)
# ============================================================
wait_if_paused "M2_long_form_q8kv"
log "=== M2: long-form Q8 KV scaling ==="
write_status "M2_long_form_q8kv" "running"
M2_OUT="$LOG_DIR/M2_long_form_q8kv.md"
if [[ -f "$LOG_DIR/M2.done" ]]; then
    log "M2 done, skipping"
else
    {
        echo "# Long-form Q8 KV scaling — $(stamp)"
        echo ""
        echo "Per memory q8_kv_production: gain grows with context. Tested at 4 lengths × 2 modes × 5 trials."
        echo ""
        if [[ -z "$Q8_KV_ARG" ]]; then
            echo "**⚠️ --q8-kv flag NOT available in binary. Only 'off' column populated; 'on' column blank.**"
            echo ""
        fi
        for tokens in 256 512 1024 2048; do
            echo "## $tokens tokens"
            echo "| Mode | t1 | t2 | t3 | t4 | t5 |"
            echo "|---|---:|---:|---:|---:|---:|"
            for mode in off on; do
                arg=""
                [[ "$mode" = "on" ]] && arg="$Q8_KV_ARG"
                row="| Q8 KV $mode |"
                for t in 1 2 3 4 5; do
                    tps=$(run_trial "Once upon a time" $tokens $t $arg)
                    [[ -z "$tps" ]] && tps="fail"
                    row="$row $tps |"
                done
                echo "$row"
            done
            echo ""
        done
    } > "$M2_OUT"
    touch "$LOG_DIR/M2.done"
fi
write_status "M2_long_form_q8kv" "done"

# ============================================================
# M3 — Multi-prompt × top stacks × 10 trials (tight error bars)
# ============================================================
wait_if_paused "M3_multi_prompt_high_conf"
log "=== M3: multi-prompt high-confidence ==="
write_status "M3_multi_prompt_high_conf" "running"
M3_OUT="$LOG_DIR/M3_multi_prompt_high_conf.md"
if [[ -f "$LOG_DIR/M3.done" ]]; then
    log "M3 done, skipping"
else
    {
        echo "# Multi-prompt high-confidence — $(stamp)"
        echo ""
        echo "5 prompts × 4 configs × 10 trials. Confirms stack delta across prompt types with tight error bars."
        echo ""
        declare -a PROMPTS=(
            "Once upon a time"
            "def fibonacci(n):"
            "The capital of France is"
            "Explain photosynthesis briefly:"
            "Solve 17 × 23 step by step:"
        )
        declare -a CFGS=(
            "baseline:"
            "L1:--vocab-prune-path $VOCAB"
            "L1+ngram:--vocab-prune-path $VOCAB --speculate ngram --verify-window 4"
            "L1+ngram+Q8KV:--vocab-prune-path $VOCAB --speculate ngram --verify-window 4 $Q8_KV_ARG"
        )
        for p in "${PROMPTS[@]}"; do
            echo "## \"$p\""
            for entry in "${CFGS[@]}"; do
                label=${entry%%:*}
                flags=${entry#*:}
                vals=""
                for t in 1 2 3 4 5 6 7 8 9 10; do
                    tps=$(run_trial "$p" 64 $t $flags)
                    [[ -z "$tps" ]] && tps="fail"
                    vals="$vals $tps"
                done
                # Compute mean of numeric vals
                mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "fail"}')
                echo "- **$label** mean=$mean trials=[$vals ]"
            done
            echo ""
        done
    } > "$M3_OUT"
    touch "$LOG_DIR/M3.done"
fi
write_status "M3_multi_prompt_high_conf" "done"

# ============================================================
# M4 — Kernel microbench survey (refresh autotune-related data)
# ============================================================
wait_if_paused "M4_kernel_survey"
log "=== M4: kernel microbench survey ==="
write_status "M4_kernel_survey" "running"
M4_OUT="$LOG_DIR/M4_kernel_survey.md"
if [[ -f "$LOG_DIR/M4.done" ]]; then
    log "M4 done, skipping"
else
    {
        echo "# Kernel survey — $(stamp)"
        echo ""
        if ./target/release/dismantle bench-kernel --help 2>&1 | head -1 | grep -q "bench-kernel\|usage"; then
            ./target/release/dismantle bench-kernel 2>&1 | head -80
        else
            echo "bench-kernel subcommand not available; survey skipped."
        fi
    } > "$M4_OUT"
    touch "$LOG_DIR/M4.done"
fi
write_status "M4_kernel_survey" "done"

# ============================================================
# M5 — Spec-decode K sweep on best stack
# ============================================================
wait_if_paused "M5_spec_K_sweep"
log "=== M5: spec-decode K sweep ==="
write_status "M5_spec_K_sweep" "running"
M5_OUT="$LOG_DIR/M5_spec_K_sweep.md"
if [[ -f "$LOG_DIR/M5.done" ]]; then
    log "M5 done, skipping"
else
    {
        echo "# Spec-decode K sweep on best stack — $(stamp)"
        echo ""
        echo "| K | mode | t1 | t2 | t3 | t4 | t5 |"
        echo "|---|---|---:|---:|---:|---:|---:|"
        for K in 2 4 8 16; do
            for mode in ngram exact-shared; do
                row="| $K | $mode |"
                for t in 1 2 3 4 5; do
                    tps=$(run_trial "Once upon a time" 64 $t \
                        --vocab-prune-path "$VOCAB" \
                        --speculate "$mode" --verify-window "$K")
                    [[ -z "$tps" ]] && tps="fail"
                    row="$row $tps |"
                done
                echo "$row"
            done
        done
    } > "$M5_OUT"
    touch "$LOG_DIR/M5.done"
fi
write_status "M5_spec_K_sweep" "done"

# ============================================================
# M6 — System diagnostic
# ============================================================
wait_if_paused "M6_sys_diag"
log "=== M6: system diagnostic ==="
write_status "M6_sys_diag" "running"
M6_OUT="$LOG_DIR/M6_sys_diag.md"
if [[ -f "$LOG_DIR/M6.done" ]]; then
    log "M6 done, skipping"
else
    {
        echo "# System diagnostic — $(stamp)"
        echo ""
        echo "## RAM (vm_stat)"
        vm_stat | head -8
        echo ""
        echo "## Disk"
        df -h . | head -2
        echo ""
        echo "## Top RAM consumers"
        ps -axrm | head -10
        echo ""
        echo "## Worktree state"
        git worktree list 2>/dev/null
        echo ""
        echo "## Uncommitted file count (main)"
        git status --short 2>/dev/null | wc -l
    } > "$M6_OUT"
    touch "$LOG_DIR/M6.done"
fi
write_status "M6_sys_diag" "done"

# ============================================================
# M7 — Final TRIALS=15 on winning configs
# ============================================================
wait_if_paused "M7_final_confidence"
log "=== M7: final TRIALS=15 ==="
write_status "M7_final_confidence" "running"
M7_OUT="$LOG_DIR/M7_final_confidence.md"
if [[ -f "$LOG_DIR/M7.done" ]]; then
    log "M7 done, skipping"
else
    {
        echo "# Final confidence — TRIALS=15 — $(stamp)"
        echo ""
        declare -a CFGS=(
            "baseline:"
            "L1:--vocab-prune-path $VOCAB"
            "L1+ngram:--vocab-prune-path $VOCAB --speculate ngram --verify-window 4"
            "L1+ngram+Q8KV:--vocab-prune-path $VOCAB --speculate ngram --verify-window 4 $Q8_KV_ARG"
        )
        for entry in "${CFGS[@]}"; do
            label=${entry%%:*}; flags=${entry#*:}
            echo "## $label"
            vals=""
            for t in $(seq 1 15); do
                tps=$(run_trial "Once upon a time" 64 $t $flags)
                [[ -z "$tps" ]] && tps="fail"
                vals="$vals $tps"
            done
            mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "fail"}')
            stdev=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk -v m="$mean" '{d=$1-m; s+=d*d; n++} END {if(n>1) printf "%.2f", sqrt(s/(n-1)); else print "NA"}')
            echo "- mean=$mean stdev=$stdev"
            echo "- trials=[$vals ]"
            echo ""
        done
    } > "$M7_OUT"
    touch "$LOG_DIR/M7.done"
fi
write_status "M7_final_confidence" "done"

# ============================================================
# M8 — Ultra-long bench (4096 tok) — only if Q8 KV available
# ============================================================
wait_if_paused "M8_ultra_long"
log "=== M8: ultra-long 4096 tok ==="
write_status "M8_ultra_long" "running"
M8_OUT="$LOG_DIR/M8_ultra_long.md"
if [[ -f "$LOG_DIR/M8.done" ]]; then
    log "M8 done, skipping"
else
    {
        echo "# Ultra-long bench (4096 tok) — $(stamp)"
        echo ""
        echo "| Mode | t1 | t2 | t3 |"
        echo "|---|---:|---:|---:|"
        for mode in off on; do
            arg=""
            [[ "$mode" = "on" ]] && arg="$Q8_KV_ARG"
            row="| Q8 KV $mode |"
            for t in 1 2 3; do
                tps=$(run_trial "Once upon a time" 4096 $t $arg)
                [[ -z "$tps" ]] && tps="fail"
                row="$row $tps |"
            done
            echo "$row"
        done
    } > "$M8_OUT"
    touch "$LOG_DIR/M8.done"
fi
write_status "M8_ultra_long" "done"

# ============================================================
# M9 — Variance hunt — 30 trials on winning config
# ============================================================
wait_if_paused "M9_variance_hunt"
log "=== M9: variance hunt — 30 trials on winning config ==="
write_status "M9_variance_hunt" "running"
M9_OUT="$LOG_DIR/M9_variance_hunt.md"
if [[ -f "$LOG_DIR/M9.done" ]]; then
    log "M9 done, skipping"
else
    {
        echo "# Variance hunt — 30 trials on L1+ngram — $(stamp)"
        echo ""
        echo "Tight error bars on the most-likely-to-ship stack."
        echo ""
        vals=""
        for t in $(seq 1 30); do
            tps=$(run_trial "Once upon a time" 64 $t \
                --vocab-prune-path "$VOCAB" \
                --speculate ngram --verify-window 4)
            [[ -z "$tps" ]] && tps="fail"
            vals="$vals $tps"
        done
        mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n; else print "fail"}')
        stdev=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk -v m="$mean" '{d=$1-m; s+=d*d; n++} END {if(n>1) printf "%.3f", sqrt(s/(n-1)); else print "NA"}')
        echo "- **mean** = $mean dec_tps"
        echo "- **stdev** = $stdev"
        echo "- 95%CI ≈ ±$(echo "$stdev" | awk '{printf "%.3f", $1 * 1.96 / sqrt(30)}')"
        echo "- trials=[$vals ]"
    } > "$M9_OUT"
    touch "$LOG_DIR/M9.done"
fi
write_status "M9_variance_hunt" "done"

# ============================================================
# M10 — Commit-readiness audit + morning wrap
# ============================================================
log "=== M10: commit audit + wrap memo ==="
M10_OUT="$LOG_DIR/M10_morning_wrap.md"
{
    echo "# Smart chain v3 — morning wrap"
    echo ""
    echo "**Started:** $(stat -f '%Sm' "$LOG_DIR/pid" 2>/dev/null || echo unknown)"
    echo "**Finished:** $(stamp)"
    echo "**Total uptime:** $(( ($(date +%s) - START_EPOCH) / 60 )) min"
    echo "**Q8 KV available?** $([[ -n "$Q8_KV_ARG" ]] && echo YES || echo NO)"
    echo ""
    echo "## Module artifacts"
    for f in "$LOG_DIR"/M*.md; do
        [[ -f "$f" ]] && echo "- \`${f#$LOG_DIR/}\`"
    done
    echo ""
    echo "## Read in this order in the morning"
    echo "1. \`M9_variance_hunt.md\` → tight estimate on best stack"
    echo "2. \`M1_stack_matrix.md\` → which combinations actually stack"
    echo "3. \`M7_final_confidence.md\` → ship-decision data"
    echo "4. \`M2_long_form_q8kv.md\` → Q8 KV scaling truth"
    echo "5. \`M5_spec_K_sweep.md\` → optimal verify window"
    echo "6. Other modules → background data"
    echo ""
    echo "## Uncommitted state (working tree)"
    git status --short 2>/dev/null | wc -l | xargs echo "files changed:"
    echo ""
    echo "## Suggested next actions"
    echo "1. If M9 mean is +3 tps over baseline with low stdev → ship L1+ngram as default"
    echo "2. If M2 shows Q8 KV positive at 1024+ tok → flip --q8-kv default ON for long contexts"
    echo "3. Decide which uncommitted hunks to commit (manual git review)"
} > "$M10_OUT"
log "M10: $M10_OUT"
write_status "complete" "done"

TOTAL_MIN=$(( ($(date +%s) - START_EPOCH) / 60 ))
log "=== smart chain v3 COMPLETE in $TOTAL_MIN min ==="
exit 0
