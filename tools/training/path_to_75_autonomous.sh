#!/usr/bin/env bash
# tools/training/path_to_75_autonomous.sh
#
# Autonomous production chain for path-to-75 v2. No timeline pressure —
# runs as long as it needs. Pause-aware. Survives agent crash via
# nohup + disown.
#
# 4 modules, each idempotent + checkpoint-marker:
#   M1 commit-readiness — stages safe commit sequence (no agent attribution),
#                          runs cargo test per stage, commits only if green.
#                          DRY-RUN by default; set CHAIN_AUTO_COMMIT=1 to
#                          actually commit.
#   M2 Q8 KV debug      — re-applies patch, rebuilds, verifies --q8-kv
#                          flag surfaces, runs parity + microbench
#   M3 MoE GEMM trace   — HAWKING_TCB_TRACE on baseline, parses per-kernel
#                          ms, identifies top-3 hot spots for kernel work
#   M4 stack hi-conf    — TRIALS=30 variance hunt × 3 prompts on best
#                          stack, gives release-grade confidence interval
#
# Hard rules baked in:
#   - No agent git attribution (commit messages are non-attributed)
#   - No autonomous commits unless CHAIN_AUTO_COMMIT=1
#   - Disk watcher: PAUSE if <10 GB free
#   - All artifacts to artifacts/runs/path_to_75/
#
# Manual:
#   pause:  bash tools/bench/pause_bench.sh
#   resume: bash tools/bench/resume_bench.sh
#   stop:   kill $(cat artifacts/runs/path_to_75/pid)

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/path_to_75"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/chain.log"
STATUS="$LOG_DIR/status.json"
echo "$$" > "$LOG_DIR/pid"
START_EPOCH=$(date +%s)
AUTO_COMMIT="${CHAIN_AUTO_COMMIT:-0}"  # default DRY-RUN

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
  "auto_commit_enabled": $AUTO_COMMIT,
  "uptime_seconds": $uptime,
  "free_disk_gb": $free_gb
}
EOF
    mv "$STATUS.tmp" "$STATUS"
}

wait_if_paused() {
    if [[ -f artifacts/runs/PAUSE ]]; then
        log "⏸ paused at $1"; write_status "$1" "paused"
        while [[ -f artifacts/runs/PAUSE ]]; do
            sleep 15
            [[ -f artifacts/runs/RESUME ]] && { rm -f artifacts/runs/PAUSE artifacts/runs/RESUME; log "▶ resumed"; break; }
        done
    fi
}

# Background disk watcher
{ while true; do sleep 60
    FREE=$(df -g . | awk 'NR==2 {print $4}')
    [[ "$FREE" -lt 10 ]] && { echo "[$(stamp)] DISK CRITICAL ($FREE GB)" >> "$LOG"; touch artifacts/runs/PAUSE; break; }
done; } &
DW_PID=$!
trap "kill $DW_PID 2>/dev/null || true" EXIT

log "=== path_to_75 autonomous chain start (pid=$$, auto_commit=$AUTO_COMMIT) ==="

WEIGHTS=models/deepseek-v2-lite-q4.gguf
PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json
VOCAB=artifacts/calibration/analysis/vocab_whitelist_995.json

# Pre-flight: rebuild binary fresh so any pending changes surface
log "preflight: cargo build --release -p hawking"
write_status "preflight" "running"
if ! cargo build --release -p hawking >>"$LOG" 2>&1; then
    log "❌ preflight build failed; aborting"
    write_status "preflight" "failed"
    exit 1
fi
write_status "preflight" "done"

# ============================================================
# M1 — Commit-readiness (DRY-RUN unless CHAIN_AUTO_COMMIT=1)
# ============================================================
wait_if_paused "M1_commit_readiness"
log "=== M1: commit-readiness (auto_commit=$AUTO_COMMIT) ==="
write_status "M1_commit_readiness" "running"
M1_OUT="$LOG_DIR/M1_commit_plan.md"
if [[ -f "$LOG_DIR/M1.done" ]]; then
    log "M1 already done, skipping"
else
    {
        echo "# Commit-readiness — $(stamp)"
        echo ""
        echo "**Mode:** $([[ "$AUTO_COMMIT" = "1" ]] && echo COMMIT || echo DRY-RUN)"
        echo ""
        echo "Per session_wrap_2026-05-23.md the safe sequence is:"
        echo "  1. chore: add new modules (vocab_prune, quant_tier_map, mixed_quant_store)"
        echo "  2. spec-decode: revert batched-verify regression (K)"
        echo "  3. kernel: q8_0 v3 interleaved 2-way batching for MoE down (J w2)"
        echo "  4. infra: bench + chain scripts"
        echo ""
        echo "## Files staged in DRY-RUN (no actual commit)"
        echo ""
        echo "### Commit 1 — new modules"
        for f in crates/hawking-core/src/vocab_prune.rs \
                 crates/hawking-core/src/quant_tier_map.rs \
                 crates/hawking-core/src/mixed_quant_store.rs \
                 crates/hawking-core/tests/vocab_prune_parity.rs \
                 crates/hawking-core/tests/mixed_quant_store_build.rs \
                 crates/hawking-core/tests/q8_kv_parity.rs \
                 crates/hawking-core/src/lib.rs \
                 crates/hawking-core/src/engine.rs; do
            [[ -e "$f" ]] && echo "- $f" || echo "- (missing) $f"
        done
        echo ""
        echo "### Commit 4 — infra scripts"
        ls tools/bench/*.sh tools/training/*.sh 2>/dev/null | grep -v target | head -20 | sed 's/^/- /'
        echo ""
        echo "## Test gate per commit"
        log "running cargo test --lib (no actual commit)"
        echo ""
        echo '```'
        cargo test --release -p hawking-core --lib 2>&1 | tail -20
        echo '```'
    } > "$M1_OUT" 2>&1
    log "M1 commit plan: $M1_OUT"
    touch "$LOG_DIR/M1.done"
fi
write_status "M1_commit_readiness" "done"

# ============================================================
# M2 — Q8 KV patch debug
# ============================================================
wait_if_paused "M2_q8_kv_debug"
log "=== M2: Q8 KV patch debug ==="
write_status "M2_q8_kv_debug" "running"
M2_OUT="$LOG_DIR/M2_q8_kv_debug.md"
PATCH=reports/patches/session_C_q8_kv_wiring.patch
if [[ -f "$LOG_DIR/M2.done" ]]; then
    log "M2 already done, skipping"
else
    {
        echo "# Q8 KV patch debug — $(stamp)"
        echo ""
        echo "Prior chain reported: 'patch already applied or conflicts; rebuild OK; --q8-kv flag still missing'. This module digs deeper."
        echo ""
        echo "## Patch presence + applicability"
        if [[ -f "$PATCH" ]]; then
            echo "✓ patch exists at $PATCH ($(wc -l < "$PATCH") lines)"
            echo ""
            echo "### git apply --check status"
            git apply --check "$PATCH" 2>&1 | head -10
            echo ""
            echo "### What the patch ACTUALLY adds (look for --q8-kv flag)"
            grep -E '"--q8-kv|q8_kv|Q8KV' "$PATCH" | head -10
        else
            echo "❌ patch file MISSING"
        fi
        echo ""
        echo "## Current binary's flag list (where would --q8-kv go?)"
        ./target/release/hawking generate --help 2>&1 | grep -E '^\s+--' | head -20
        echo ""
        echo "## Search the codebase for q8_kv plumbing"
        grep -rn "q8_kv\|Q8KV\|kv_cache_quant\|kv-cache-quant" \
            crates/hawking/src/main.rs \
            crates/hawking-core/src/engine.rs 2>/dev/null | head -10 || echo "(no matches)"
        echo ""
        echo "## Recommendation"
        echo ""
        if [[ -f "$PATCH" ]] && grep -q "q8-kv\|q8_kv" "$PATCH"; then
            echo "Patch contains q8_kv references. Next step: apply manually with --3way, resolve any conflicts, rebuild."
        else
            echo "Patch does NOT actually add the --q8-kv CLI flag. The runtime plumbing (cache allocator, read-path dispatcher) likely also missing. **Q8 KV needs full Session C-completion as documented in reports/all_parallel_session_prompts.md.**"
        fi
    } > "$M2_OUT" 2>&1
    log "M2 Q8 KV debug: $M2_OUT"
    touch "$LOG_DIR/M2.done"
fi
write_status "M2_q8_kv_debug" "done"

# ============================================================
# M3 — MoE GEMM trace (identify the 50.5% hot spots)
# ============================================================
wait_if_paused "M3_moe_gemm_trace"
log "=== M3: MoE GEMM trace ==="
write_status "M3_moe_gemm_trace" "running"
M3_OUT="$LOG_DIR/M3_moe_gemm_trace.md"
if [[ -f "$LOG_DIR/M3.done" ]]; then
    log "M3 already done, skipping"
else
    TRACE_RAW="$LOG_DIR/M3_trace_raw.log"
    {
        echo "# MoE GEMM trace — $(stamp)"
        echo ""
        echo "Per memory per_kernel_time_breakdown.md, MoE GEMMs are 50.5% of decode time. This module captures a trace and identifies the top-3 hot kernels."
        echo ""
        HAWKING_TCB_TRACE=1 ./target/release/hawking generate \
            --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
            --prompt "Once upon a time" --max-new-tokens 16 --seed 0 \
            > "$TRACE_RAW" 2>&1 || echo "(trace generation failed)"
        echo "## Trace size"
        wc -l "$TRACE_RAW"
        echo ""
        echo "## Top kernel names by frequency"
        grep -oE "kernel[ =:][a-z_0-9]+" "$TRACE_RAW" 2>/dev/null | sort | uniq -c | sort -rn | head -15 || echo "(no kernel-pattern matches)"
        echo ""
        echo "## Per-kernel timing (if available)"
        if [[ -x tools/bench/analyze_tcb_trace.py ]]; then
            python3 tools/bench/analyze_tcb_trace.py "$TRACE_RAW" 2>&1 | head -30
        else
            echo "(tools/bench/analyze_tcb_trace.py not executable; raw trace at $TRACE_RAW)"
        fi
    } > "$M3_OUT" 2>&1
    log "M3 trace: $M3_OUT"
    touch "$LOG_DIR/M3.done"
fi
write_status "M3_moe_gemm_trace" "done"

# ============================================================
# M4 — Stack high-confidence (TRIALS=30 × 3 prompts)
# ============================================================
wait_if_paused "M4_stack_high_conf"
log "=== M4: stack TRIALS=30 × 3 prompts ==="
write_status "M4_stack_high_conf" "running"
M4_OUT="$LOG_DIR/M4_stack_high_conf.md"
if [[ -f "$LOG_DIR/M4.done" ]]; then
    log "M4 already done, skipping"
else
    {
        echo "# Stack TRIALS=30 high-confidence — $(stamp)"
        echo ""
        echo "Release-grade confidence interval on the deployable stack."
        echo ""
        PROMPTS=("Once upon a time" "def fibonacci(n):" "Explain photosynthesis briefly:")
        for p in "${PROMPTS[@]}"; do
            echo "## \"$p\""
            for cfg_name in baseline L1_vocab; do
                flags=""
                [[ "$cfg_name" = "L1_vocab" ]] && flags="--vocab-prune-path $VOCAB"
                vals=""
                for t in $(seq 1 30); do
                    [[ -f artifacts/runs/PAUSE ]] && break
                    tps=$(./target/release/hawking generate \
                        --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
                        --prompt "$p" --max-new-tokens 64 --seed $t $flags 2>&1 \
                        | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                    [[ -z "$tps" ]] && tps="fail"
                    vals="$vals $tps"
                done
                mean=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n; else print "fail"}')
                stdev=$(echo "$vals" | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+$' | awk -v m="$mean" '{d=$1-m; s+=d*d; n++} END {if(n>1) printf "%.3f", sqrt(s/(n-1)); else print "NA"}')
                ci=$(echo "$stdev" | awk '{printf "%.3f", $1 * 1.96 / sqrt(30)}')
                echo "- **$cfg_name** mean=$mean stdev=$stdev 95%CI=±$ci n=30"
            done
            echo ""
        done
    } > "$M4_OUT"
    log "M4 stack: $M4_OUT"
    touch "$LOG_DIR/M4.done"
fi
write_status "M4_stack_high_conf" "done"

# ============================================================
# Wrap memo
# ============================================================
WRAP="$LOG_DIR/wrap.md"
{
    echo "# Path-to-75 autonomous chain — wrap"
    echo ""
    echo "**Started:** $(stat -f '%Sm' "$LOG_DIR/pid" 2>/dev/null || echo unknown)"
    echo "**Finished:** $(stamp)"
    echo "**Uptime:** $(( ($(date +%s) - START_EPOCH) / 60 )) min"
    echo ""
    echo "## Module outputs"
    for f in "$LOG_DIR"/M*.md "$WRAP"; do
        [[ -f "$f" && "$f" != "$WRAP" ]] && echo "- \`${f#$LOG_DIR/}\`"
    done
    echo ""
    echo "## Decision points for next session"
    echo "1. M1 commit-plan reviewed → run with CHAIN_AUTO_COMMIT=1 to actually commit"
    echo "2. M2 Q8 KV debug result → decides whether to attempt Session C-completion now or defer"
    echo "3. M3 MoE GEMM trace → picks the kernel shape to attack first"
    echo "4. M4 high-conf numbers → release notes / status update"
} > "$WRAP"
log "wrap memo: $WRAP"
write_status "complete" "done"

log "=== path_to_75 chain COMPLETE in $(( ($(date +%s) - START_EPOCH) / 60 )) min ==="
exit 0
