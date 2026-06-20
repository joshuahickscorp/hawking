#!/usr/bin/env bash
# Pipelined RWKV-7 post-train: batched teacher capture  ‖  shard-streaming SFT.
#
# Overlaps the two longest phases (optimization #3). The batched capture
# (`dismantle generate --batched-capture`) writes per-group shards as it goes;
# the streaming SFT trainer (`rwkv7_sft_stream.py --watch`) begins on the first
# finished shard while capture keeps producing later ones. Instead of
# capture-THEN-train (sum of the two), wall-clock ≈ max(capture, train) + a
# one-shard lead-in.
#
# ⚠️  GPU step — run ONLY when the perf bench has freed the GPU. This loads the
#     Qwen teacher (~2.3 GB) for capture and the RWKV-7 student (~6–8 GB) for
#     SFT. They run in the SAME process group but the multiseq capture is
#     GPU-bursty while SFT is steady; on 18 GB this is feasible because capture
#     decode is light (Q4_K, B=8) — but if you see memory pressure, run capture
#     to completion first (drop --watch) and train after. Capture-to-disk-first
#     is always the safe fallback.
#
# Usage:
#   tools/training/rwkv7_pipeline.sh \
#       <prompts.txt> <teacher.gguf> <rwkv7-hf-dir> <out-dir> [max_new_tokens] [batch]
#
# Example:
#   tools/training/rwkv7_pipeline.sh \
#       artifacts/rwkv7_posttrain/dpo_prompts.prompts.txt \
#       models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
#       models/rwkv7-g1-04-hf \
#       artifacts/rwkv7_posttrain/sft_out 256 8
set -euo pipefail

PROMPTS="${1:?prompts file}"
TEACHER="${2:?teacher gguf}"
RWKV_HF="${3:?rwkv7 HF dir}"
OUT="${4:?output dir}"
MAXNEW="${5:-256}"
BATCH="${6:-8}"

CAP_OUT="$(dirname "$OUT")/teacher_capture.jsonl"
CAP_GLOB="$(dirname "$OUT")/teacher_capture.shard-*.jsonl"
PY="${PYTHON:-python3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[pipeline] capture -> $CAP_OUT (B=$BATCH, max_new=$MAXNEW)"
echo "[pipeline] SFT will stream from $CAP_GLOB into $OUT"

# Estimate shard count so the trainer knows when capture is complete.
N_PROMPTS=$(grep -cve '^[[:space:]]*$' "$PROMPTS" || echo 0)
EXPECTED_SHARDS=$(( (N_PROMPTS + BATCH - 1) / BATCH ))
echo "[pipeline] $N_PROMPTS prompts -> ~$EXPECTED_SHARDS shards at B=$BATCH"

# ── Phase 1 (background): batched teacher capture ────────────────────────────
# Greedy, --profile exact (clean teacher target). Writes shards as groups finish.
(
  cd "$REPO_ROOT"
  cargo run --release -p hawking -- generate \
    --weights "$TEACHER" \
    --prompts-file "$PROMPTS" \
    --batched-capture \
    --capture-out "$CAP_OUT" \
    --capture-batch "$BATCH" \
    --max-new-tokens "$MAXNEW" \
    --max-seq-len 4096 \
    --profile exact
) &
CAP_PID=$!
echo "[pipeline] capture PID=$CAP_PID"

# ── Phase 2 (foreground): shard-streaming SFT, overlapping capture ───────────
# --watch waits for the first shard, then trains; re-globs to pick up shards
# that landed during model load. The trainer reads completed shards only.
PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" "$REPO_ROOT/tools/training/rwkv7_sft_stream.py" \
  --model "$RWKV_HF" \
  --shards-glob "$CAP_GLOB" \
  --out "$OUT" \
  --watch \
  --expected-shards "$EXPECTED_SHARDS" \
  --max-length 1024 \
  --grad-accum 16 || {
    echo "[pipeline] SFT failed; waiting for capture to finish before exit" >&2
    wait "$CAP_PID" || true
    exit 1
  }

# Make sure capture finished (it should have, since SFT needed all shards).
wait "$CAP_PID" || echo "[pipeline] (capture already finished)"
echo "[pipeline] DONE: capture + SFT complete -> $OUT/final"
