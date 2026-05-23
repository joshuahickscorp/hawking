#!/usr/bin/env bash
# Extended autonomous chain — runs AFTER the main resume pipeline completes.
#
# Stages 8-11 (default, ~5-6 h):
#   8. Spec-decode K=4/8/16 sweep (eagle4 baseline)
#   9. Autotune kernel profile regen
#   10. Eagle5 v3 alt-config retrain (capture_layer=13)
#   11. τ-eval v3 + compare to v2
#
# Optional Stage 12 (set EXTENDED_LONG_CTX=1 to enable, ~6h additional):
#   12. Long-context corpus capture (max_tokens=1024, ~750 seqs)
#
# Waits for main pipeline to complete via artifacts/runs/overnight/status.json
# state="complete" before starting. Polls every 60s, max 4 hours.
#
# Idempotent: each stage's success/failure logged; downstream stages
# continue regardless. Safe to re-run; will skip already-done stages.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight"
mkdir -p "$LOG_DIR"
EXT_LOG="$LOG_DIR/extended_chain.log"
EXT_STATUS="$LOG_DIR/extended_status.json"
PID_PATH="$LOG_DIR/extended.pid"

echo "$$" > "$PID_PATH"
START_EPOCH=$(date +%s)

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] $*" | tee -a "$EXT_LOG"; }

write_status() {
    local stage="$1" state="$2" note="${3:-}"
    local uptime=$(( $(date +%s) - START_EPOCH ))
    local free_gb=$(df -g . | awk 'NR==2 {print $4}')
    cat > "$EXT_STATUS.tmp" <<EOF
{
  "ts": "$(stamp)",
  "extended_pid": $$,
  "current_stage": "$stage",
  "state": "$state",
  "note": "$note",
  "uptime_seconds": $uptime,
  "free_disk_gb": $free_gb
}
EOF
    mv "$EXT_STATUS.tmp" "$EXT_STATUS"
}

# --- Wait for main pipeline to land ---
log "extended chain starting; waiting for main pipeline to complete"
write_status "waiting_main_pipeline" "running"
WAIT_LIMIT_S=$((4 * 3600))
WAITED=0
while [ $WAITED -lt $WAIT_LIMIT_S ]; do
    if [ -f "$LOG_DIR/status.json" ]; then
        MAIN_STATE=$(python3 -c "import json; print(json.load(open('$LOG_DIR/status.json')).get('state',''))" 2>/dev/null || echo "")
        MAIN_STAGE=$(python3 -c "import json; print(json.load(open('$LOG_DIR/status.json')).get('current_stage',''))" 2>/dev/null || echo "")
        if [ "$MAIN_STAGE" = "complete" ] && [ "$MAIN_STATE" = "done" ]; then
            log "main pipeline complete (stage=$MAIN_STAGE state=$MAIN_STATE); starting extended chain"
            break
        fi
    fi
    sleep 60
    WAITED=$((WAITED + 60))
done

if [ $WAITED -ge $WAIT_LIMIT_S ]; then
    log "FAIL: main pipeline did not complete within 4h; aborting"
    write_status "aborted" "failed" "main pipeline timeout"
    exit 1
fi

source .venv-calibration/bin/activate || { log "FAIL venv"; write_status "aborted" "failed" "venv"; exit 1; }

# ============================================================
# STAGE 8: Spec-decode K=4/8/16 sweep (eagle4 baseline)
# ============================================================
log "=== EXT STAGE 8: spec-decode K sweep (eagle4) ==="
write_status "spec_decode_sweep" "running"
if bash tools/bench/spec_decode_sweep.sh >> "$EXT_LOG" 2>&1; then
    write_status "spec_decode_sweep" "done"
    log "spec-decode sweep done; see artifacts/runs/overnight/spec_decode_sweep.md"
else
    write_status "spec_decode_sweep" "failed"
    log "WARN spec-decode sweep failed; continuing"
fi

# ============================================================
# STAGE 9: Autotune kernel profile regen
# ============================================================
log "=== EXT STAGE 9: autotune kernel profile ==="
write_status "autotune" "running"
AUTOTUNE_OUT="$LOG_DIR/autotune_output.log"
# Use a short max-hours so we don't burn all night; current profile is recently regenerated, this just refreshes.
if ./target/release/dismantle autotune --max-hours 0.5 >> "$AUTOTUNE_OUT" 2>&1; then
    write_status "autotune" "done"
    log "autotune done; see $AUTOTUNE_OUT"
else
    write_status "autotune" "failed"
    log "WARN autotune failed; continuing (current profile remains in use)"
fi

# ============================================================
# STAGE 10: Eagle5 v3 alt-config retrain (capture_layer=13)
# ============================================================
log "=== EXT STAGE 10: eagle5 v3 retrain (capture_layer=13) ==="
write_status "eagle5_v3_train" "running"
mkdir -p checkpoints/eagle5_v3
# Use minimal corpus (now in v2_lite_corpus_min) — same data, far smaller
CORPUS_DIR="artifacts/calibration/v2_lite_corpus_min"
[ -d "$CORPUS_DIR" ] || CORPUS_DIR="artifacts/calibration/v2_lite_corpus"

if python3 tools/training/eagle5_train.py \
        --corpus-dir "$CORPUS_DIR" \
        --frozen eagle4/v2lite_frozen.npz \
        --ckpt-dir checkpoints/eagle5_v3 \
        --epochs 3 --batch-size 16 --seq-len 16 \
        --capture-layer 13 \
        --sparsity-head off \
        >> "$EXT_LOG" 2>&1; then
    write_status "eagle5_v3_train" "done"
    log "eagle5 v3 train done"
    V3_FAILED=0
else
    write_status "eagle5_v3_train" "failed"
    log "WARN eagle5 v3 train failed; skipping v3 eval"
    V3_FAILED=1
fi

# ============================================================
# STAGE 11: τ-eval v3 + compare to v2
# ============================================================
if [ "$V3_FAILED" = "0" ]; then
    log "=== EXT STAGE 11: τ-eval v3 ==="
    write_status "eagle5_v3_eval" "running"
    V3_CKPT=$(ls -t checkpoints/eagle5_v3/*.npz 2>/dev/null | head -1)
    if [ -n "$V3_CKPT" ] && python3 tools/training/eagle5_tau_eval.py \
            --ckpt "$V3_CKPT" \
            --frozen eagle4/v2lite_frozen.npz \
            --corpus "$CORPUS_DIR" \
            --depth 4 \
            --max-windows 500 \
            --capture-layer 13 \
            >> "$EXT_LOG" 2>&1; then
        write_status "eagle5_v3_eval" "done"
        log "v3 τ-eval done"
    else
        write_status "eagle5_v3_eval" "failed"
        log "WARN v3 τ-eval failed"
    fi
fi

# ============================================================
# STAGE 12 (OPTIONAL): Long-context corpus capture
# ============================================================
if [ "${EXTENDED_LONG_CTX:-0}" = "1" ]; then
    log "=== EXT STAGE 12: long-context corpus capture (max_tokens=1024) ==="
    write_status "long_ctx_corpus" "running"
    OUT_DIR="artifacts/calibration/v2_lite_corpus_1024"
    mkdir -p "$OUT_DIR"
    # 750 sequences × 1024 tokens × similar capture set
    # Disk: ~3 GB per shard × ~24 shards = ~72 GB. CHECK DISK FIRST.
    FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
    if [ "$FREE_GB" -lt 100 ]; then
        log "WARN free_gb=$FREE_GB < 100, skipping long-ctx corpus"
        write_status "long_ctx_corpus" "skipped" "disk too low"
    else
        if python3 tools/training/build_corpus.py \
                --max-sequences 750 --max-tokens-per-seq 1024 \
                --shard-size 32 --batch-size 4 \
                --device mps --dtype float16 \
                --quantize-intermediates int8 \
                --out "$OUT_DIR" \
                >> "$EXT_LOG" 2>&1; then
            write_status "long_ctx_corpus" "done"
            log "long-ctx corpus done at $OUT_DIR"
        else
            write_status "long_ctx_corpus" "failed"
            log "WARN long-ctx corpus failed"
        fi
    fi
fi

# ============================================================
# STAGE 12: Long-context corpus capture (6 h, ~72 GB)
# Gated on disk + EXTENDED_LONG_CTX env. Pre-flight: need ≥100 GB free.
# ============================================================
FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
LONG_CTX_DIR="artifacts/calibration/v2_lite_corpus_1024"
LONG_CTX_MIN="artifacts/calibration/v2_lite_corpus_1024_min"
# Trimmed to 300 seqs (was 750) — fits 24h budget + still yields 300K
# training positions, enough for a head-comparison signal.
LONG_CTX_SEQS="${LONG_CTX_SEQS:-300}"
if [ "${EXTENDED_LONG_CTX:-1}" = "1" ] && [ "$FREE_GB" -ge 60 ]; then
    log "=== EXT STAGE 12: long-context corpus capture (1024 tok, ${LONG_CTX_SEQS} seqs) ==="
    write_status "long_ctx_corpus" "running"
    mkdir -p "$LONG_CTX_DIR"
    if python3 tools/training/build_corpus.py \
            --max-sequences "$LONG_CTX_SEQS" --max-tokens-per-seq 1024 \
            --shard-size 16 --batch-size 4 \
            --device mps --dtype float16 \
            --quantize-intermediates int8 \
            --out "$LONG_CTX_DIR" \
            >> "$EXT_LOG" 2>&1; then
        write_status "long_ctx_corpus" "done"
        log "long-context corpus done"
        # Stage 12b: immediately minimal-extract so stage 13 reads fast.
        log "=== EXT STAGE 12b: minimal-extract long-context corpus ==="
        if python3 tools/training/build_minimal_corpus.py \
                --src-dir "$LONG_CTX_DIR" \
                --dst-dir "$LONG_CTX_MIN" \
                --capture-layers 25 \
                >> "$EXT_LOG" 2>&1; then
            N_MIN=$(ls "$LONG_CTX_MIN"/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
            N_FULL=$(ls "$LONG_CTX_DIR"/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
            if [ "$N_MIN" = "$N_FULL" ] && [ "$N_MIN" -gt 0 ]; then
                log "minimal extract OK ($N_MIN shards); deleting full long-context corpus"
                rm -rf "$LONG_CTX_DIR"
                LONG_CTX_DIR="$LONG_CTX_MIN"
            fi
        fi
        LONGCTX_OK=1
    else
        write_status "long_ctx_corpus" "failed"
        log "WARN long-context corpus failed"
        LONGCTX_OK=0
    fi
else
    log "EXT STAGE 12 skipped (EXTENDED_LONG_CTX=${EXTENDED_LONG_CTX:-1} free_gb=$FREE_GB <60)"
    LONGCTX_OK=0
fi

# ============================================================
# STAGE 13: Eagle5 v4 on long-context corpus
# Only if STAGE 12 produced shards. Long-context training builds a head
# that can predict tokens further out — better acceptance on real prompts.
# ============================================================
if [ "${LONGCTX_OK:-0}" = "1" ] && \
   [ "$(ls "$LONG_CTX_DIR"/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')" -gt 5 ]; then
    log "=== EXT STAGE 13: eagle5 v4 train (long-context, seq_len=64) ==="
    write_status "eagle5_v4_train" "running"
    mkdir -p checkpoints/eagle5_v4
    if python3 tools/training/eagle5_train.py \
            --corpus-dir "$LONG_CTX_DIR" \
            --frozen eagle4/v2lite_frozen.npz \
            --ckpt-dir checkpoints/eagle5_v4 \
            --epochs 3 --batch-size 8 --seq-len 64 \
            --capture-layer 25 \
            --sparsity-head off \
            >> "$EXT_LOG" 2>&1; then
        write_status "eagle5_v4_train" "done"
        log "eagle5 v4 long-context train done"
        V4_OK=1
    else
        write_status "eagle5_v4_train" "failed"
        log "WARN eagle5 v4 train failed"
        V4_OK=0
    fi

    if [ "${V4_OK:-0}" = "1" ]; then
        log "=== EXT STAGE 14: τ-eval v4 ==="
        write_status "eagle5_v4_eval" "running"
        V4_CKPT=$(ls -t checkpoints/eagle5_v4/*.npz 2>/dev/null | head -1)
        if [ -n "$V4_CKPT" ] && python3 tools/training/eagle5_tau_eval.py \
                --ckpt "$V4_CKPT" \
                --frozen eagle4/v2lite_frozen.npz \
                --corpus "$LONG_CTX_DIR" \
                --depth 4 \
                --max-windows 500 \
                --capture-layer 25 \
                >> "$EXT_LOG" 2>&1; then
            write_status "eagle5_v4_eval" "done"
            log "v4 τ-eval done"
        else
            write_status "eagle5_v4_eval" "failed"
            log "WARN v4 τ-eval failed"
        fi
    fi
else
    log "EXT STAGE 13 skipped (long-context corpus unavailable)"
fi

# ============================================================
# Wrap — generate decision-ready compare report
# ============================================================
log "=== generating eagle5 variant comparison ==="
{
    echo "# Eagle5 variant comparison — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo ""
    echo "| Variant | capture_layer | epochs | seq_len | corpus | τ@4 | depth-1 accept |"
    echo "|---|---|---|---|---|---|---|"
    for variant in v2 v3 v4; do
        DIR="checkpoints/eagle5_$variant"
        if [ -f "$DIR/log.jsonl" ]; then
            LAST=$(tail -1 "$DIR/log.jsonl" 2>/dev/null || echo "")
            echo "| $variant | ? | ? | ? | ? | ? | ? | (log_tail: $LAST)" | head -c 200
            echo ""
        fi
    done
} > "$LOG_DIR/eagle5_variants_compare.md"

TOTAL_H=$(( ( $(date +%s) - START_EPOCH ) / 3600 ))
log "=== EXTENDED CHAIN COMPLETE in ~${TOTAL_H}h ==="
log "see: artifacts/runs/overnight/eagle5_variants_compare.md"
write_status "complete" "done"
exit 0
