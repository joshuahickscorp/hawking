#!/usr/bin/env bash
# tools/training/overnight_path_to_50.sh
#
# One-shot overnight chain for the path-to-50 compute pipeline.
# Each stage's success is gated; on failure the chain aborts and writes a
# diagnostic file. Heartbeat written after each major stage transition so
# the cron monitor can report stage-level progress.
#
# Stages (estimated wall clock on M3 Pro 18GB):
#   1. cargo build --release          (~15-30 min from clean)
#   2. corpus rebuild (3000 seqs)     (~4 h)
#   3. eagle5_train (5 epochs)        (~3-4 h)
#   4. eagle5_tau_eval                (~30 min)
#   5. eagle5_quantize + parity       (~15 min)
#   6. vocab-prune paired bench       (~30 min)
# Total: ~8.5-9.5 h
#
# Disk: corpus is ~122 GB. Verify ≥130 GB free before launching.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight"
mkdir -p "$LOG_DIR"
STATUS_PATH="$LOG_DIR/status.json"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
PID_PATH="$LOG_DIR/overnight.pid"

echo "$$" > "$PID_PATH"
START_EPOCH=$(date +%s)

stamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

write_status() {
    # $1 = stage label, $2 = state (running/done/failed), $3 = note
    local stage="$1" state="$2" note="${3:-}"
    local now uptime
    now=$(date +%s)
    uptime=$((now - START_EPOCH))
    local free_gb
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

log() {
    echo "[$(stamp)] $*" | tee -a "$PIPELINE_LOG"
}

# --- Stage 1: cargo build --release ---
log "=== STAGE 1: cargo build --release ==="
write_status "build" "running"
if ! cargo build --release -p hawking >> "$PIPELINE_LOG" 2>&1; then
    log "FAIL stage 1 (cargo build). Aborting."
    write_status "build" "failed" "cargo build failed"
    exit 1
fi
write_status "build" "done"

# --- Stage 2: corpus rebuild ---
log "=== STAGE 2: corpus rebuild (3000 seqs, batch=8) ==="
write_status "corpus" "running"
rm -f artifacts/calibration/STOP artifacts/calibration/LOOP_DONE \
      artifacts/calibration/heartbeat.json artifacts/calibration/overnight.pid

# Run watchdog FOREGROUND so we block until it exits. Watchdog exits 0 when
# build_corpus.py completes (MAX_SEQUENCES reached).
if ! MAX_SEQUENCES=3000 MAX_TOKENS_PER_SEQ=256 SHARD_SIZE=32 BATCH_SIZE=8 \
        ./tools/training/run_corpus_autonomous.sh >> "$PIPELINE_LOG" 2>&1; then
    log "FAIL stage 2 (corpus build). Aborting."
    write_status "corpus" "failed" "watchdog exited non-zero"
    exit 2
fi
n_shards=$(ls artifacts/calibration/v2_lite_corpus/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
log "corpus rebuild complete — ${n_shards} shards"
write_status "corpus" "done" "${n_shards} shards"

# --- Stage 3: eagle5 train ---
log "=== STAGE 3: eagle5 train (5 epochs, batch=16, seq=16) ==="
write_status "eagle5_train" "running"
mkdir -p checkpoints/eagle5_v2
if ! source .venv-calibration/bin/activate; then
    log "FAIL stage 3 (venv activate). Aborting."
    write_status "eagle5_train" "failed" "venv missing"
    exit 3
fi
if ! python3 tools/training/eagle5_train.py \
        --corpus-dir artifacts/calibration/v2_lite_corpus \
        --frozen eagle4/v2lite_frozen.npz \
        --ckpt-dir checkpoints/eagle5_v2 \
        --epochs 5 --batch-size 16 --seq-len 16 \
        >> "$PIPELINE_LOG" 2>&1; then
    log "FAIL stage 3 (eagle5_train). Aborting subsequent eagle5 steps but continuing to vocab bench."
    write_status "eagle5_train" "failed" "trainer crashed; skipping eval/quantize"
    EAGLE5_FAILED=1
else
    write_status "eagle5_train" "done"
    EAGLE5_FAILED=0
fi

# --- Stage 4: eagle5 tau-eval (only if train succeeded) ---
if [ "$EAGLE5_FAILED" = "0" ]; then
    log "=== STAGE 4: eagle5 tau-eval ==="
    write_status "eagle5_eval" "running"
    LATEST_CKPT=$(ls -t checkpoints/eagle5_v2/*.pt checkpoints/eagle5_v2/*.npz 2>/dev/null | head -1)
    if [ -z "$LATEST_CKPT" ]; then
        log "WARN stage 4: no checkpoint found. Skipping eval+quantize."
        write_status "eagle5_eval" "failed" "no checkpoint"
        EAGLE5_FAILED=1
    elif ! python3 tools/training/eagle5_tau_eval.py \
            --ckpt "$LATEST_CKPT" \
            --frozen eagle4/v2lite_frozen.npz \
            --corpus artifacts/calibration/v2_lite_corpus \
            --depth 4 \
            >> "$PIPELINE_LOG" 2>&1; then
        log "WARN stage 4 (tau-eval). Continuing."
        write_status "eagle5_eval" "failed" "eval crashed"
    else
        write_status "eagle5_eval" "done"
    fi
fi

# --- Stage 5: eagle5 quantize + parity (only if train+eval succeeded) ---
if [ "$EAGLE5_FAILED" = "0" ]; then
    log "=== STAGE 5: eagle5 quantize + parity ==="
    write_status "eagle5_quantize" "running"
    LATEST_CKPT=$(ls -t checkpoints/eagle5_v2/*.pt checkpoints/eagle5_v2/*.npz 2>/dev/null | head -1)
    Q4_OUT="checkpoints/eagle5_v2/q4_head.npz"
    if ! python3 tools/training/eagle5_quantize.py quantize \
            --in "$LATEST_CKPT" --out "$Q4_OUT" --bits 4 --group-size 64 \
            >> "$PIPELINE_LOG" 2>&1; then
        log "WARN stage 5 quantize."
        write_status "eagle5_quantize" "failed" "quantize crashed"
    else
        # Parity check
        FIRST_SHARD=$(ls artifacts/calibration/v2_lite_corpus/shard_*.parquet 2>/dev/null | head -1)
        if [ -n "$FIRST_SHARD" ]; then
            python3 tools/training/eagle5_quantize.py parity \
                --bf16 "$LATEST_CKPT" --q4 "$Q4_OUT" \
                --frozen eagle4/v2lite_frozen.npz \
                --shard "$FIRST_SHARD" \
                >> "$PIPELINE_LOG" 2>&1 || log "WARN parity check failed."
        fi
        write_status "eagle5_quantize" "done"
    fi
fi

# --- Stage 6: vocab-prune paired bench ---
log "=== STAGE 6: vocab-prune paired bench ==="
write_status "vocab_bench" "running"

BENCH_OUT="$LOG_DIR/vocab_bench_results.md"
{
    echo "# Vocab-prune paired bench"
    echo "Run: $(stamp)"
    echo ""
    if [ -x "./target/release/hawking" ]; then
        echo "### baseline trials"
        for trial in 1 2 3; do
            echo "#### baseline trial $trial"
            ./target/release/hawking generate \
                --weights models/deepseek-v2-lite-q4.gguf \
                --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
                --prompt "Once upon a time" \
                --max-new-tokens 64 \
                --seed 0 2>&1 | tail -10 || echo "baseline trial $trial failed"
            echo ""
        done
        echo "### pruned trials"
        for trial in 1 2 3; do
            echo "#### pruned trial $trial"
            ./target/release/hawking generate \
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
write_status "vocab_bench" "done" "$(wc -l < "$BENCH_OUT") lines in $BENCH_OUT"

# --- Wrap ---
TOTAL_HOURS=$(( ( $(date +%s) - START_EPOCH ) / 3600 ))
log "=== PIPELINE COMPLETE in ~${TOTAL_HOURS}h ==="
write_status "complete" "done" "all stages finished or failed-as-noted"
echo "$(stamp) overnight pipeline complete" >> "$STATUS_PATH.history"
exit 0
