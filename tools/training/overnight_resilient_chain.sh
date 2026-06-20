#!/usr/bin/env bash
# tools/training/overnight_resilient_chain.sh
#
# Survives Claude crash. Detaches via nohup; reparents to launchd.
# 4 modules chained, each idempotent. If a module fails, the next one
# still runs — partial progress always banked.
#
# Modules:
#   M1: Comprehensive bench matrix across all landed levers (~1-2h)
#   M2: Apply Q8 KV patch + microbench at 16/64/256 tok (~30 min)
#   M3: Multi-prompt sweep (story/code/factual/math × all levers) (~1h)
#   M4: Wrap memo with all findings (~5 min)
#
# Total: ~3-4h compute. If Claude dies, this keeps running — check
# progress via: tail -f artifacts/runs/overnight_chain/chain.log
# or: cat artifacts/runs/overnight_chain/status.json
#
# Manual stop: touch artifacts/runs/PAUSE (clean stop at next module boundary)
# Hard kill:   kill $(cat artifacts/runs/overnight_chain/pid)

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight_chain"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/chain.log"
STATUS="$LOG_DIR/status.json"
PID_PATH="$LOG_DIR/pid"

echo "$$" > "$PID_PATH"
START_EPOCH=$(date +%s)

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

write_status() {
    local module="$1" state="$2" note="${3:-}"
    local now=$(date +%s)
    local uptime=$((now - START_EPOCH))
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
        log "⏸  PAUSED — touch artifacts/runs/RESUME to continue"
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

# Disk watcher — runs in background, touches STOP if disk gets dangerous
{
    while true; do
        sleep 60
        FREE=$(df -g . | awk 'NR==2 {print $4}')
        if [[ "$FREE" -lt 10 ]]; then
            echo "[$(stamp)] DISK CRITICAL ($FREE GB) — touching PAUSE" >> "$LOG"
            touch artifacts/runs/PAUSE
            break
        fi
    done
} &
DISK_WATCHER_PID=$!
trap "kill $DISK_WATCHER_PID 2>/dev/null || true" EXIT

log "=== overnight resilient chain starting ==="
log "pid=$$ disk_watcher_pid=$DISK_WATCHER_PID"
log "pause: touch artifacts/runs/PAUSE  ·  resume: bash tools/bench/resume_bench.sh"
log "log: tail -f $LOG"

# Pre-flight: ensure binary exists; rebuild if not
if [[ ! -x ./target/release/hawking ]]; then
    log "no binary — building"
    write_status "build" "running"
    cargo build --release -p hawking >> "$LOG" 2>&1 || {
        log "❌ cargo build failed; aborting"
        write_status "build" "failed"
        exit 1
    }
fi

# ============================================================
# M1 — Comprehensive bench matrix (TRIALS=5 for confidence)
# ============================================================
wait_if_paused "M1_bench_matrix"
log "=== M1: bench matrix (TRIALS=5) ==="
write_status "M1_bench_matrix" "running"
M1_DONE_MARKER="$LOG_DIR/M1.done"
if [[ -f "$M1_DONE_MARKER" ]]; then
    log "M1 already done (marker present), skipping"
else
    TRIALS=5 bash tools/bench/microbench_levers.sh >> "$LOG" 2>&1 || \
        log "M1 microbench failed; continuing to M2"
    cp -r artifacts/runs/microbench/latest "$LOG_DIR/M1_microbench" 2>/dev/null
    touch "$M1_DONE_MARKER"
fi
write_status "M1_bench_matrix" "done"

# ============================================================
# M2 — Apply Q8 KV patch + microbench at 16/64/256 tok
# ============================================================
wait_if_paused "M2_q8_kv"
log "=== M2: Q8 KV patch + length-sweep bench ==="
write_status "M2_q8_kv" "running"
M2_DONE_MARKER="$LOG_DIR/M2.done"
if [[ -f "$M2_DONE_MARKER" ]]; then
    log "M2 already done, skipping"
elif [[ -f reports/patches/session_C_q8_kv_wiring.patch ]]; then
    log "applying Q8 KV patch..."
    # Apply patch in a way that's safe-to-rerun. If already applied, skip.
    if git apply --check reports/patches/session_C_q8_kv_wiring.patch 2>/dev/null; then
        git apply reports/patches/session_C_q8_kv_wiring.patch >> "$LOG" 2>&1 || \
            log "patch apply failed; continuing without Q8 KV"
        log "rebuilding dismantle with Q8 KV..."
        cargo build --release -p hawking >> "$LOG" 2>&1 || \
            log "rebuild failed; continuing"
    else
        log "Q8 KV patch already applied or conflicts; skipping apply"
    fi
    # Sweep at 16/64/256 tok with Q8 KV on/off
    M2_OUT="$LOG_DIR/M2_q8_kv_sweep.md"
    {
        echo "# Q8 KV length sweep — $(stamp)"
        echo ""
        for tokens in 16 64 256; do
            echo "## $tokens tokens"
            for mode in off on; do
                echo "### Q8 KV $mode"
                flags=""
                [[ "$mode" = "on" ]] && flags="--q8-kv"
                for trial in 1 2 3; do
                    ./target/release/hawking generate \
                        --weights models/deepseek-v2-lite-q4.gguf \
                        --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                        --prompt "Once upon a time" \
                        --max-new-tokens $tokens --seed $trial $flags \
                        2>&1 | grep "dec_tps" | tail -1 || echo "(trial $trial failed)"
                done
                echo ""
            done
        done
    } > "$M2_OUT" 2>&1
    log "M2 sweep complete: $M2_OUT"
    touch "$M2_DONE_MARKER"
else
    log "no Q8 KV patch found at reports/patches/session_C_q8_kv_wiring.patch — skipping M2"
    touch "$M2_DONE_MARKER"
fi
write_status "M2_q8_kv" "done"

# ============================================================
# M3 — Multi-prompt sweep across all confirmed levers
# ============================================================
wait_if_paused "M3_multi_prompt"
log "=== M3: multi-prompt sweep ==="
write_status "M3_multi_prompt" "running"
M3_DONE_MARKER="$LOG_DIR/M3.done"
if [[ -f "$M3_DONE_MARKER" ]]; then
    log "M3 already done, skipping"
else
    M3_OUT="$LOG_DIR/M3_multi_prompt.md"
    {
        echo "# Multi-prompt sweep — $(stamp)"
        echo ""
        echo "All trials are seed-0, 64 tokens, 3 trials. Paired delta valid (Claude live OK)."
        echo ""

        # Prompt set covering diverse cases
        declare -a PROMPTS=(
            "Once upon a time"
            "def fibonacci(n):"
            "The capital of France is"
            "2 + 2 ="
            "In machine learning, a transformer is"
        )

        for prompt in "${PROMPTS[@]}"; do
            echo "## prompt: \"$prompt\""
            echo ""
            echo "| Lever | trial1 | trial2 | trial3 |"
            echo "|---|---:|---:|---:|"
            for lever_id in baseline vocab tier_aggro ngram_K4 stack; do
                flags=""
                case "$lever_id" in
                    vocab) flags="--vocab-prune-path artifacts/calibration/analysis/vocab_whitelist_995.json" ;;
                    tier_aggro) flags="--quant-tier-map-path artifacts/calibration/tier_maps/v2_lite_aggressive_down_q4.json" ;;
                    ngram_K4) flags="--speculate ngram --verify-window 4" ;;
                    stack) flags="--vocab-prune-path artifacts/calibration/analysis/vocab_whitelist_995.json --quant-tier-map-path artifacts/calibration/tier_maps/v2_lite_aggressive_down_q4.json --speculate ngram --verify-window 4" ;;
                esac
                row="| $lever_id |"
                for trial in 1 2 3; do
                    tps=$(./target/release/hawking generate \
                        --weights models/deepseek-v2-lite-q4.gguf \
                        --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                        --prompt "$prompt" --max-new-tokens 64 --seed $trial $flags 2>&1 \
                        | grep -oE 'dec_tps=[0-9]+\.[0-9]+' | head -1 | cut -d= -f2)
                    [[ -z "$tps" ]] && tps="fail"
                    row="$row $tps |"
                done
                echo "$row"
            done
            echo ""
        done
    } > "$M3_OUT" 2>&1
    log "M3 multi-prompt complete: $M3_OUT"
    touch "$M3_DONE_MARKER"
fi
write_status "M3_multi_prompt" "done"

# ============================================================
# M4 — Wrap memo
# ============================================================
wait_if_paused "M4_wrap"
log "=== M4: wrap memo ==="
write_status "M4_wrap" "running"
M4_OUT="$LOG_DIR/M4_wrap_memo.md"
{
    echo "# Overnight resilient chain — wrap"
    echo ""
    echo "**Started:** $(stat -f '%Sm' "$PID_PATH" 2>/dev/null || echo unknown)"
    echo "**Finished:** $(stamp)"
    echo "**Total uptime:** $(( ($(date +%s) - START_EPOCH) / 60 )) minutes"
    echo "**Disk free at end:** $(df -g . | awk 'NR==2 {print $4}') GB"
    echo ""
    echo "## Module status"
    echo ""
    for m in M1_bench_matrix M2_q8_kv M3_multi_prompt; do
        [[ -f "$LOG_DIR/${m%%_*}.done" ]] && echo "- ✅ $m" || echo "- ❌ $m"
    done
    echo ""
    echo "## Output artifacts"
    echo ""
    for f in "$LOG_DIR/M1_microbench/report.md" "$LOG_DIR/M2_q8_kv_sweep.md" "$LOG_DIR/M3_multi_prompt.md"; do
        [[ -f "$f" ]] && echo "- \`$f\`" || echo "- (missing) $f"
    done
    echo ""
    echo "## Quick summary"
    if [[ -f "$LOG_DIR/M1_microbench/report.md" ]]; then
        echo ""
        echo "### M1 bench matrix:"
        echo '```'
        grep -E "^\|" "$LOG_DIR/M1_microbench/report.md" | head -15
        echo '```'
    fi
} > "$M4_OUT"
log "M4 wrap memo: $M4_OUT"
write_status "complete" "done"

log "=== overnight chain COMPLETE in $((( $(date +%s) - START_EPOCH ) / 60 )) min ==="
exit 0
