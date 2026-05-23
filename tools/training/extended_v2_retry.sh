#!/usr/bin/env bash
# Retry failed extended stages (10-14) with fixes applied.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="artifacts/runs/overnight"
EXT_LOG="$LOG_DIR/extended_chain.log"
STATUS_PATH="$LOG_DIR/extended_status.json"
echo "$$" > "$LOG_DIR/extended.pid"
START_EPOCH=$(date +%s)
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] $*" | tee -a "$EXT_LOG"; }
write_status() {
    local stage="$1" state="$2"
    local uptime=$(( $(date +%s) - START_EPOCH ))
    local free_gb=$(df -g . | awk 'NR==2 {print $4}')
    cat > "$STATUS_PATH.tmp" <<EOF
{
  "ts": "$(stamp)",
  "extended_pid": $$,
  "current_stage": "$stage",
  "state": "$state",
  "uptime_seconds": $uptime,
  "free_disk_gb": $free_gb
}
EOF
    mv "$STATUS_PATH.tmp" "$STATUS_PATH"
}

source .venv-calibration/bin/activate

# Stage 10 retry: v3 with sparsity off
log "=== RETRY STAGE 10: eagle5 v3 train (capture_layer=13, sparsity off) ==="
write_status "eagle5_v3_train" "running"
mkdir -p checkpoints/eagle5_v3
rm -f checkpoints/eagle5_v3/log.jsonl
if python3 tools/training/eagle5_train.py \
        --corpus-dir artifacts/calibration/v2_lite_corpus \
        --frozen eagle4/v2lite_frozen.npz \
        --ckpt-dir checkpoints/eagle5_v3 \
        --epochs 3 --batch-size 16 --seq-len 16 \
        --capture-layer 13 --sparsity-head off \
        >> "$EXT_LOG" 2>&1; then
    write_status "eagle5_v3_train" "done"
    log "v3 train done"
    # τ-eval
    log "=== STAGE 11: τ-eval v3 ==="
    write_status "eagle5_v3_eval" "running"
    V3_CKPT=$(ls -t checkpoints/eagle5_v3/*.npz 2>/dev/null | head -1)
    if [ -n "$V3_CKPT" ]; then
        python3 tools/training/eagle5_tau_eval.py \
            --ckpt "$V3_CKPT" \
            --frozen eagle4/v2lite_frozen.npz \
            --corpus artifacts/calibration/v2_lite_corpus \
            --depth 4 --max-windows 500 \
            --capture-layer 13 --sparsity-head off \
            >> "$EXT_LOG" 2>&1 && write_status "eagle5_v3_eval" "done" || write_status "eagle5_v3_eval" "failed"
    fi
else
    write_status "eagle5_v3_train" "failed"
    log "WARN v3 train failed; aborting stage 11"
fi

# Stage 12: long-context corpus
FREE=$(df -g . | awk 'NR==2 {print $4}')
if [ "$FREE" -ge 60 ]; then
    log "=== STAGE 12: long-context corpus (300 seqs × 1024 tok) ==="
    write_status "long_ctx_corpus" "running"
    LONG_DIR="artifacts/calibration/v2_lite_corpus_1024"
    mkdir -p "$LONG_DIR"
    if python3 tools/training/build_corpus.py \
            --max-sequences 300 --max-tokens-per-seq 1024 \
            --shard-size 16 --batch-size 4 \
            --device mps --dtype float16 \
            --quantize-intermediates int8 \
            --out "$LONG_DIR" \
            >> "$EXT_LOG" 2>&1; then
        write_status "long_ctx_corpus" "done"
        log "=== STAGE 13: eagle5 v4 (long-context) ==="
        write_status "eagle5_v4_train" "running"
        mkdir -p checkpoints/eagle5_v4
        python3 tools/training/eagle5_train.py \
            --corpus-dir "$LONG_DIR" \
            --frozen eagle4/v2lite_frozen.npz \
            --ckpt-dir checkpoints/eagle5_v4 \
            --epochs 3 --batch-size 8 --seq-len 64 \
            --capture-layer 25 --sparsity-head off \
            >> "$EXT_LOG" 2>&1 && write_status "eagle5_v4_train" "done" || write_status "eagle5_v4_train" "failed"
    else
        write_status "long_ctx_corpus" "failed"
    fi
fi

write_status "complete" "done"
log "=== EXTENDED V2 RETRY COMPLETE ==="
exit 0
