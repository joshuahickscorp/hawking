#!/usr/bin/env bash
# Resume the path-to-50 pipeline from stage 3 onward.
# Use this when corpus is already built and you just need eagle5 + bench.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight"
mkdir -p "$LOG_DIR"
STATUS_PATH="$LOG_DIR/status.json"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
PID_PATH="$LOG_DIR/overnight.pid"

echo "$$" > "$PID_PATH"
START_EPOCH=$(date +%s)

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
    local stage="$1" state="$2" note="${3:-}"
    local now uptime free_gb
    now=$(date +%s); uptime=$((now - START_EPOCH))
    free_gb=$(df -g . | awk 'NR==2 {print $4}')
    cat > "$STATUS_PATH.tmp" <<EOF
{
  "ts": "$(stamp)",
  "pipeline_pid": $$,
  "current_stage": "$stage",
  "state": "$state",
  "note": "$note",
  "uptime_seconds": $uptime,
  "free_disk_gb": $free_gb
}
EOF
    mv "$STATUS_PATH.tmp" "$STATUS_PATH"
}

log() { echo "[$(stamp)] $*" | tee -a "$PIPELINE_LOG"; }

# --- Stage 3: eagle5 train ---
log "=== RESUME: STAGE 3: eagle5 train ==="
write_status "eagle5_train" "running"
mkdir -p checkpoints/eagle5_v2
source .venv-calibration/bin/activate || { log "FAIL venv"; write_status "eagle5_train" "failed" "venv"; exit 3; }

EAGLE5_FAILED=0
if ! python3 tools/training/eagle5_train.py \
        --corpus-dir artifacts/calibration/v2_lite_corpus \
        --frozen eagle4/v2lite_frozen.npz \
        --ckpt-dir checkpoints/eagle5_v2 \
        --epochs 5 --batch-size 16 --seq-len 16 \
        >> "$PIPELINE_LOG" 2>&1; then
    log "FAIL stage 3 (eagle5_train)."
    write_status "eagle5_train" "failed" "trainer crashed"
    EAGLE5_FAILED=1
else
    write_status "eagle5_train" "done"
fi

# --- Stage 4: tau-eval ---
if [ "$EAGLE5_FAILED" = "0" ]; then
    log "=== STAGE 4: eagle5 tau-eval ==="
    write_status "eagle5_eval" "running"
    LATEST_CKPT=$(ls -t checkpoints/eagle5_v2/*.pt checkpoints/eagle5_v2/*.npz 2>/dev/null | head -1)
    if [ -z "$LATEST_CKPT" ]; then
        log "WARN no checkpoint"
        write_status "eagle5_eval" "failed" "no ckpt"
        EAGLE5_FAILED=1
    elif ! python3 tools/training/eagle5_tau_eval.py \
            --ckpt "$LATEST_CKPT" \
            --frozen eagle4/v2lite_frozen.npz \
            --corpus artifacts/calibration/v2_lite_corpus \
            --depth 4 \
            --max-windows 500 \
            >> "$PIPELINE_LOG" 2>&1; then
        log "WARN tau-eval failed"
        write_status "eagle5_eval" "failed"
    else
        write_status "eagle5_eval" "done"
    fi
fi

# --- Stage 5: quantize + parity ---
if [ "$EAGLE5_FAILED" = "0" ]; then
    log "=== STAGE 5: eagle5 quantize + parity ==="
    write_status "eagle5_quantize" "running"
    LATEST_CKPT=$(ls -t checkpoints/eagle5_v2/*.pt checkpoints/eagle5_v2/*.npz 2>/dev/null | head -1)
    Q4_OUT="checkpoints/eagle5_v2/q4_head.npz"
    if ! python3 tools/training/eagle5_quantize.py quantize \
            --in "$LATEST_CKPT" --out "$Q4_OUT" --bits 4 --group-size 64 \
            >> "$PIPELINE_LOG" 2>&1; then
        log "WARN quantize failed"
        write_status "eagle5_quantize" "failed"
    else
        FIRST_SHARD=$(ls artifacts/calibration/v2_lite_corpus/shard_*.parquet 2>/dev/null | head -1)
        if [ -n "$FIRST_SHARD" ]; then
            python3 tools/training/eagle5_quantize.py parity \
                --bf16 "$LATEST_CKPT" --q4 "$Q4_OUT" \
                --frozen eagle4/v2lite_frozen.npz \
                --shard "$FIRST_SHARD" \
                >> "$PIPELINE_LOG" 2>&1 || log "WARN parity"
        fi
        write_status "eagle5_quantize" "done"
    fi
fi

# --- Stage 6: vocab-prune bench ---
log "=== STAGE 6: vocab-prune paired bench ==="
write_status "vocab_bench" "running"
BENCH_OUT="$LOG_DIR/vocab_bench_results.md"
{
    echo "# Vocab-prune paired bench"
    echo "Run: $(stamp)"
    echo ""
    if [ -x "./target/release/dismantle" ]; then
        echo "## Baseline (no prune)"
        for trial in 1 2; do
            echo "### baseline trial $trial"
            ./target/release/dismantle generate \
                --weights models/deepseek-v2-lite-q4.gguf \
                --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                --prompt "Once upon a time" \
                --max-new-tokens 64 \
                --seed 0 2>&1 | tail -10 || echo "baseline trial $trial failed"
            echo ""
        done
        echo "## Pruned (--vocab-prune-path)"
        for trial in 1 2; do
            echo "### pruned trial $trial"
            ./target/release/dismantle generate \
                --weights models/deepseek-v2-lite-q4.gguf \
                --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                --vocab-prune-path artifacts/calibration/analysis/vocab_whitelist_995.json \
                --prompt "Once upon a time" \
                --max-new-tokens 64 \
                --seed 0 2>&1 | tail -10 || echo "pruned trial $trial failed"
            echo ""
        done
    else
        echo "dismantle binary missing — skipped"
    fi
} > "$BENCH_OUT"
write_status "vocab_bench" "done"

# --- Stage 7: Minimal-corpus extraction (parallel) ---
log "=== STAGE 7: build minimal corpus (parallel) ==="
write_status "minimal_corpus" "running"
if [ -d artifacts/calibration/v2_lite_corpus ] && \
   ls artifacts/calibration/v2_lite_corpus/shard_*.parquet > /dev/null 2>&1; then
    if python3 tools/training/build_minimal_corpus.py >> "$PIPELINE_LOG" 2>&1; then
        write_status "minimal_corpus" "done"
        log "minimal corpus built"
    else
        log "WARN minimal corpus failed"
        write_status "minimal_corpus" "failed"
    fi
else
    log "WARN no source corpus to minimize"
    write_status "minimal_corpus" "skipped" "no source corpus"
fi

# Stage 7b: optional original-corpus delete (gated; only delete if minimal corpus is valid AND >0 shards)
N_MIN=$(ls artifacts/calibration/v2_lite_corpus_min/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
N_ORIG=$(ls artifacts/calibration/v2_lite_corpus/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_MIN" = "$N_ORIG" ] && [ "$N_MIN" -gt 0 ]; then
    log "minimal corpus has $N_MIN shards matching original; deleting original (frees ~76 GB)"
    rm -rf artifacts/calibration/v2_lite_corpus
    write_status "minimal_corpus" "done" "deleted original; $N_MIN minimal shards"
else
    log "minimal corpus shard count mismatch ($N_MIN vs $N_ORIG); KEEPING original for safety"
fi

# --- Wrap ---
TOTAL_H=$(( ( $(date +%s) - START_EPOCH ) / 3600 ))
log "=== RESUME PIPELINE COMPLETE in ~${TOTAL_H}h ==="
write_status "complete" "done"
exit 0
