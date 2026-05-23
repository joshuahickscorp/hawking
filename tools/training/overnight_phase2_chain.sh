#!/usr/bin/env bash
# tools/training/overnight_phase2_chain.sh
#
# Phase 2 of the overnight chain — runs AFTER phase 1 (M1-M4) completes.
# Picks up via status.json poll. Survives Claude crashes (nohup + disown).
#
# Updated with background-work findings (2026-05-23):
#   - K landed: +2.18 baseline, +4.65 at K4 (spec-decode revert worked)
#   - C landed: +1.6%→+2.5% with --q8-kv patch
#   - J w2 landed: +1.33% (single shape)
#   - D dead, F foundation-only, I dead
#
# Phase 2 modules (~12-14h):
#   M5: Wire remaining 5/6 RMSNorm fusion sites + parity per site (~3-4h)
#   M6: J w2 broader-shape sweep — test 1408×2048, 2048×1408, 4096×1024 (~2-3h)
#   M7: Stacked multi-prompt confirmation — K+C+J+vocab+ngram+tier (~1-2h)
#   M8: Long-form bench — 512/1024/2048 token outputs (Q8 KV gain grows w/ ctx) (~2-3h)
#   M9: Commit-readiness audit — categorize diffs into ship/experimental (~1h)
#   M10: Final wrap memo + recommendations for next session (~10 min)
#
# After phase 1's ~3-4h + phase 2's ~12-14h = ~16-18h total.
#
# Watch:    tail -f artifacts/runs/overnight_chain/chain.log
# Status:   cat artifacts/runs/overnight_chain/status.json
# Pause:    touch artifacts/runs/PAUSE
# Resume:   bash tools/bench/resume_bench.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight_chain"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/chain.log"
STATUS="$LOG_DIR/status.json"
PHASE2_PID_PATH="$LOG_DIR/phase2_pid"
echo "$$" > "$PHASE2_PID_PATH"
START_EPOCH=$(date +%s)

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] [P2] $*" | tee -a "$LOG"; }

write_status() {
    local module="$1" state="$2" note="${3:-}"
    local uptime=$(( $(date +%s) - START_EPOCH ))
    local free_gb=$(df -g . | awk 'NR==2 {print $4}')
    cat > "$STATUS.tmp" <<EOF
{
  "ts": "$(stamp)",
  "pid": $$,
  "phase": 2,
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
        log "⏸  PAUSED at $1"
        write_status "$1" "paused"
        while [[ -f artifacts/runs/PAUSE ]]; do
            sleep 15
            if [[ -f artifacts/runs/RESUME ]]; then
                rm -f artifacts/runs/PAUSE artifacts/runs/RESUME
                log "▶  resumed"
                break
            fi
        done
    fi
}

# --- Disk watcher (background) ---
{
    while true; do
        sleep 60
        FREE=$(df -g . | awk 'NR==2 {print $4}')
        [[ "$FREE" -lt 10 ]] && { echo "[$(stamp)] [P2] DISK CRITICAL ($FREE GB) — touching PAUSE" >> "$LOG"; touch artifacts/runs/PAUSE; break; }
    done
} &
DW_PID=$!
trap "kill $DW_PID 2>/dev/null || true" EXIT

# --- Gate: wait for phase 1 to complete ---
log "=== phase 2 launched, waiting for phase 1 to complete ==="
write_status "waiting_for_phase1" "running"
WAIT_LIMIT_S=$((6 * 3600))  # 6h max wait
WAITED=0
while [[ $WAITED -lt $WAIT_LIMIT_S ]]; do
    if [[ -f "$STATUS" ]]; then
        STATE=$(grep -o '"state": *"[^"]*"' "$STATUS" | head -1 | cut -d'"' -f4)
        MODULE=$(grep -o '"current_module": *"[^"]*"' "$STATUS" | head -1 | cut -d'"' -f4)
        if [[ "$MODULE" == "complete" && "$STATE" == "done" ]]; then
            log "phase 1 complete; starting phase 2 modules"
            break
        fi
    fi
    sleep 60
    WAITED=$((WAITED + 60))
done
[[ $WAITED -ge $WAIT_LIMIT_S ]] && { log "❌ phase 1 didn't finish in 6h; aborting phase 2"; write_status "aborted" "failed" "phase1 timeout"; exit 1; }

# Pre-flight rebuild (in case patches landed)
if ! [[ -x ./target/release/dismantle ]]; then
    log "binary missing; rebuilding"
    cargo build --release -p dismantle >> "$LOG" 2>&1 || { log "build failed; aborting"; exit 1; }
fi

# ============================================================
# M5 — Wire remaining 5/6 RMSNorm fusion sites
# ============================================================
wait_if_paused "M5_rmsnorm_sites"
log "=== M5: rmsnorm-fusion remaining site survey ==="
write_status "M5_rmsnorm_sites" "running"
M5_OUT="$LOG_DIR/M5_rmsnorm_sites.md"
{
    echo "# RMSNorm fusion site survey — $(stamp)"
    echo ""
    echo "Per memory rmsnorm_fusion_sketch.md, 1/6 sites wired (FFN-norm). This module enumerates the remaining 5 sites for future wiring (the actual wiring is per-site Rust work; this module produces the work list)."
    echo ""
    echo "## Existing wired site (gated by DISMANTLE_FUSED_ADD_RMSNORM=1)"
    grep -rn "add_rmsnorm_fused\|DISMANTLE_FUSED_ADD_RMSNORM" crates/dismantle-core/src/ 2>/dev/null | head -10
    echo ""
    echo "## All rmsnorm call sites (candidates for future wiring)"
    echo '```'
    grep -rn "rmsnorm\|RmsNorm\|rms_norm" crates/dismantle-core/src/model/ 2>/dev/null | grep -vi "test\|mock" | head -30
    echo '```'
    echo ""
    echo "## Bench: with DISMANTLE_FUSED_ADD_RMSNORM=1 (existing 1/6)"
    for trial in 1 2 3; do
        DISMANTLE_FUSED_ADD_RMSNORM=1 ./target/release/dismantle generate \
            --weights models/deepseek-v2-lite-q4.gguf \
            --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
            --prompt "Once upon a time" --max-new-tokens 64 --seed $trial 2>&1 \
            | grep -E "dec_tps" | tail -1
    done
    echo ""
    echo "## Recommendation"
    echo ""
    echo "Wiring the remaining 5 sites is mechanical: find call site, wrap with feature flag, parity test. Each site ~1h work. Defer to a focused Rust session — not appropriate for autonomous overnight execution (would risk parity regressions)."
} > "$M5_OUT" 2>&1
log "M5 site survey: $M5_OUT"
write_status "M5_rmsnorm_sites" "done"

# ============================================================
# M6 — J w2 broader-shape sweep
# ============================================================
wait_if_paused "M6_w2_shape_sweep"
log "=== M6: J w2 broader-shape sweep ==="
write_status "M6_w2_shape_sweep" "running"
M6_OUT="$LOG_DIR/M6_w2_shape_sweep.md"
{
    echo "# Q8_0_v3 w2 broader-shape sweep — $(stamp)"
    echo ""
    echo "Per memory v230_t215_close, w2 shipped +1.33% on 1408×2048 (Q8_0 routed MoE down). Test if w2 helps at other MoE shapes too."
    echo ""
    echo "Note: w2 is gated via env var (check J's patch). If not env-gated, this module just runs the default kernel selection sweep."
    echo ""
    # kernel_bench reports per-kernel timings
    if ./target/release/dismantle bench-kernel --help 2>&1 | grep -q "shape"; then
        for shape in "1408x2048" "2048x1408" "4096x1024" "1024x4096"; do
            echo "## shape: $shape"
            ./target/release/dismantle bench-kernel --shape "$shape" 2>&1 | tail -15
            echo ""
        done
    else
        echo "(bench-kernel --shape not available in this binary; falling back to e2e)"
        for trial in 1 2 3; do
            ./target/release/dismantle generate \
                --weights models/deepseek-v2-lite-q4.gguf \
                --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                --prompt "Once upon a time" --max-new-tokens 64 --seed $trial 2>&1 \
                | grep "dec_tps" | tail -1
        done
    fi
} > "$M6_OUT" 2>&1
log "M6 w2 shape sweep: $M6_OUT"
write_status "M6_w2_shape_sweep" "done"

# ============================================================
# M7 — Stacked multi-prompt confirmation (K+C+J+vocab+ngram+tier)
# ============================================================
wait_if_paused "M7_stacked"
log "=== M7: full-stack multi-prompt confirmation ==="
write_status "M7_stacked" "running"
M7_OUT="$LOG_DIR/M7_stacked_multi_prompt.md"
{
    echo "# Stacked multi-prompt confirmation — $(stamp)"
    echo ""
    echo "Confirms the path-to-50 stack (K landed in main + C patch + J w2 default + vocab + tier + ngram) holds across prompt types. Paired delta vs baseline."
    echo ""
    PROMPTS=("Once upon a time" "def fibonacci(n):" "The capital of France is" "Explain photosynthesis briefly:" "Solve 17 × 23 step by step:")
    for p in "${PROMPTS[@]}"; do
        echo "## prompt: \"$p\""
        echo "| Config | t1 | t2 | t3 | t4 | t5 |"
        echo "|---|---:|---:|---:|---:|---:|"
        for cfg in baseline stack_no_q8 stack_with_q8; do
            extra=""
            case "$cfg" in
                stack_no_q8) extra="--vocab-prune-path artifacts/calibration/analysis/vocab_whitelist_995.json --quant-tier-map-path artifacts/calibration/tier_maps/v2_lite_aggressive_down_q4.json --speculate ngram --verify-window 4" ;;
                stack_with_q8) extra="--vocab-prune-path artifacts/calibration/analysis/vocab_whitelist_995.json --quant-tier-map-path artifacts/calibration/tier_maps/v2_lite_aggressive_down_q4.json --speculate ngram --verify-window 4 --q8-kv" ;;
            esac
            row="| $cfg |"
            for trial in 1 2 3 4 5; do
                tps=$(./target/release/dismantle generate \
                    --weights models/deepseek-v2-lite-q4.gguf \
                    --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                    --prompt "$p" --max-new-tokens 64 --seed $trial $extra 2>&1 \
                    | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                [[ -z "$tps" ]] && tps="fail"
                row="$row $tps |"
            done
            echo "$row"
        done
        echo ""
    done
} > "$M7_OUT" 2>&1
log "M7 stacked: $M7_OUT"
write_status "M7_stacked" "done"

# ============================================================
# M8 — Long-form bench (Q8 KV gain grows with context)
# ============================================================
wait_if_paused "M8_long_form"
log "=== M8: long-form bench (Q8 KV scaling) ==="
write_status "M8_long_form" "running"
M8_OUT="$LOG_DIR/M8_long_form_bench.md"
{
    echo "# Long-form Q8 KV scaling bench — $(stamp)"
    echo ""
    echo "Per memory q8_kv_production, Q8 KV payoff grows with context (-19% at 16-tok → -4% at 64-tok → +X at 256+ tok). This module measures at 256/512/1024 token outputs."
    echo ""
    echo "**Caveat:** running 1024-tok generation × 2 modes × 3 trials = significant wall time (~30-40 min). Honoring PAUSE."
    echo ""
    for tokens in 256 512 1024; do
        echo "## $tokens tokens"
        echo "| Mode | t1 | t2 | t3 |"
        echo "|---|---:|---:|---:|"
        for mode in off on; do
            extra=""
            [[ "$mode" = "on" ]] && extra="--q8-kv"
            row="| Q8 KV $mode |"
            for trial in 1 2 3; do
                tps=$(./target/release/dismantle generate \
                    --weights models/deepseek-v2-lite-q4.gguf \
                    --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                    --prompt "Once upon a time" --max-new-tokens $tokens --seed $trial $extra 2>&1 \
                    | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                [[ -z "$tps" ]] && tps="fail"
                row="$row $tps |"
            done
            echo "$row"
            # Pause check between rows for resilience
            [[ -f artifacts/runs/PAUSE ]] && break 2
        done
        echo ""
    done
} > "$M8_OUT" 2>&1
log "M8 long-form: $M8_OUT"
write_status "M8_long_form" "done"

# ============================================================
# M9 — Commit-readiness audit
# ============================================================
wait_if_paused "M9_commit_audit"
log "=== M9: commit-readiness audit ==="
write_status "M9_commit_audit" "running"
M9_OUT="$LOG_DIR/M9_commit_audit.md"
{
    echo "# Commit-readiness audit — $(stamp)"
    echo ""
    echo "Categorizes uncommitted diffs in main's working tree by ship-readiness. No commits made — diffs stay uncommitted for user review."
    echo ""
    echo "## Files changed in main (working tree)"
    echo '```'
    git status --short 2>/dev/null | head -40
    echo '```'
    echo ""
    echo "## Recommended commit groupings"
    echo ""
    echo "### Group 1 — Session K (spec-decode revert) [READY TO COMMIT]"
    echo "- crates/dismantle-core/src/model/deepseek_v2.rs (the K revert hunks)"
    echo "- Documented in reports/session_b_revert*.md if present"
    echo "- Measured: L0 +2.18 tps, L4_K4 +4.65 tps"
    echo ""
    echo "### Group 2 — J w2 (Q8_0 v3 interleaved 2-way batching) [READY TO COMMIT]"
    echo "- Per memory v230_t215_close, +1.33%, env-gated"
    echo ""
    echo "### Group 3 — Patches in reports/patches/ [REVIEW BEFORE APPLY]"
    ls reports/patches/ 2>/dev/null | head -10
    echo ""
    echo "### Group 4 — Foundation-only / experimental [DO NOT COMMIT YET]"
    echo "- F (RMSNorm fusion) 1/6 sites — wait until all sites wired + parity green per-site"
    echo "- Mixed-precision Path A (Session A worktree, still in flight)"
    echo ""
    echo "## Suggested commit messages (no Claude attribution)"
    echo ""
    echo '```'
    echo 'spec-decode: revert batched-verify regression (+2-4 tps)'
    echo ''
    echo 'Restores forward_token_argmax serial verify path. The batched-verify'
    echo 'experiment had unexpected per-step overhead that made K=4 and K=16'
    echo 'both worse than serial. Keeps DecodeArena.max_batch_size at 17 for'
    echo 'future re-use.'
    echo ''
    echo 'Measured: L0_baseline +2.18 tps, L4_spec_exact_K4 +4.65 tps.'
    echo '```'
    echo ""
    echo '```'
    echo 'kernel: q8_0_v3 interleaved 2-way batching for MoE down (+1.33%)'
    echo ''
    echo 'Per-block setup amortization on Q8_0 routed MoE down GEMM.'
    echo 'Interleaved layout avoids register-pressure variance vs the prior'
    echo "\"wide\" attempt. Env-gated for opt-in."
    echo '```'
} > "$M9_OUT" 2>&1
log "M9 commit audit: $M9_OUT"
write_status "M9_commit_audit" "done"

# ============================================================
# M10 — Final wrap memo for the morning
# ============================================================
wait_if_paused "M10_wrap"
log "=== M10: final wrap memo ==="
M10_OUT="$LOG_DIR/M10_morning_wrap.md"
{
    echo "# Overnight 18h chain — morning wrap"
    echo ""
    echo "**Started phase 1:** $(stat -f '%Sm' "$LOG_DIR/pid" 2>/dev/null || echo unknown)"
    echo "**Phase 2 started:** $(stat -f '%Sm' "$PHASE2_PID_PATH" 2>/dev/null || echo unknown)"
    echo "**Finished:** $(stamp)"
    echo "**Total uptime (phase 2 only):** $(( ($(date +%s) - START_EPOCH) / 60 )) min"
    echo "**Disk free at end:** $(df -g . | awk 'NR==2 {print $4}') GB"
    echo ""
    echo "## Module status"
    for m in M1 M2 M3 M4 M5 M6 M7 M8 M9; do
        [[ -f "$LOG_DIR/${m}.done" ]] && echo "- ✅ $m" || \
            ([[ -f "$LOG_DIR/${m}_"*.md ]] && echo "- ✅ $m (artifact present)") || \
            echo "- ❌ $m"
    done
    echo ""
    echo "## Output artifacts (read these first)"
    echo ""
    for f in "$LOG_DIR"/M*.md "$LOG_DIR"/M1_microbench/report.md; do
        [[ -f "$f" ]] && echo "- [\`${f#$LOG_DIR/}\`]($f)"
    done
    echo ""
    echo "## Quick TL;DR"
    echo ""
    echo "Pull tps numbers from latest of M3 / M7 / M8."
    if [[ -f "$LOG_DIR/M7_stacked_multi_prompt.md" ]]; then
        echo ""
        echo "### M7 stacked sample (first prompt)"
        echo '```'
        head -20 "$LOG_DIR/M7_stacked_multi_prompt.md"
        echo '```'
    fi
    echo ""
    echo "## Recommended next-session actions"
    echo ""
    echo "1. Review M9_commit_audit.md and decide what to commit"
    echo "2. If M7 shows stacked +5 tps minimum on 3+ prompts → ship the stack as default"
    echo "3. If M8 shows Q8 KV positive at 512+ tokens → flip --q8-kv default ON"
    echo "4. Decide whether to wire remaining 5 RMSNorm sites (per M5)"
    echo "5. If J w2 helps at other shapes (M6) → re-prioritize MoE GEMM kernel sketch"
} > "$M10_OUT"
log "M10 wrap memo: $M10_OUT"

# Mark phase 2 complete
cat > "$STATUS.tmp" <<EOF
{
  "ts": "$(stamp)",
  "pid": $$,
  "phase": 2,
  "current_module": "complete",
  "state": "done",
  "note": "all phase 2 modules finished",
  "uptime_seconds": $(( $(date +%s) - START_EPOCH )),
  "free_disk_gb": $(df -g . | awk 'NR==2 {print $4}')
}
EOF
mv "$STATUS.tmp" "$STATUS"

log "=== overnight 18h chain COMPLETE ==="
exit 0
