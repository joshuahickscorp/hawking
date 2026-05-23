#!/usr/bin/env bash
# Standalone post-pipeline trigger.
# Used by the cron or manually if the main pipeline didn't auto-chain to
# stage 7 (minimal-corpus + delete-original).
#
# Idempotent: skips work that's already done; safe to re-run.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/post_pipeline.log"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

# Gate: only run if main pipeline is fully done
STATUS_JSON="artifacts/runs/overnight/status.json"
if [ ! -f "$STATUS_JSON" ]; then
    log "no status.json — main pipeline never ran. exiting."
    exit 0
fi
CURRENT_STAGE=$(python3 -c "import json; print(json.load(open('$STATUS_JSON')).get('current_stage',''))" 2>/dev/null || echo "")
STATE=$(python3 -c "import json; print(json.load(open('$STATUS_JSON')).get('state',''))" 2>/dev/null || echo "")
if [ "$CURRENT_STAGE" != "complete" ] && [ "$CURRENT_STAGE" != "vocab_bench" ]; then
    log "main pipeline still on stage=$CURRENT_STAGE state=$STATE — skipping (run after pipeline lands)"
    exit 0
fi

# Step 1: build minimal corpus
if [ -d artifacts/calibration/v2_lite_corpus_min ] && \
   [ "$(ls artifacts/calibration/v2_lite_corpus_min/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')" -gt 90 ]; then
    log "minimal corpus already complete; skipping extraction"
else
    log "running build_minimal_corpus.py (parallel)…"
    if python3 tools/training/build_minimal_corpus.py >> "$LOG" 2>&1; then
        log "minimal corpus extraction done"
    else
        log "FAIL: minimal_corpus extraction"
        exit 1
    fi
fi

# Step 2: shard count check + delete original
N_MIN=$(ls artifacts/calibration/v2_lite_corpus_min/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
N_ORIG=$(ls artifacts/calibration/v2_lite_corpus/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
log "shards: minimal=$N_MIN original=$N_ORIG"
if [ "$N_MIN" = "$N_ORIG" ] && [ "$N_MIN" -gt 0 ]; then
    log "deleting original corpus (frees ~76 GB)"
    rm -rf artifacts/calibration/v2_lite_corpus
    log "deletion complete"
elif [ "$N_ORIG" = "0" ]; then
    log "original corpus already gone"
else
    log "shard count mismatch — KEEPING original for safety"
fi

FREE=$(df -g . | awk 'NR==2 {print $4}')
log "post-pipeline done. free_disk_gb=$FREE"
echo "$(stamp) post-pipeline done" > "$LOG_DIR/post_pipeline.done"
