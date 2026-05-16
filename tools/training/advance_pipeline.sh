#!/usr/bin/env bash
# advance_pipeline.sh — state-machine that advances the C2 → C3 pipeline by ONE step.
#
# Run periodically (every 30-60 sec) via pipeline_loop.sh or a launchd plist.
# Each invocation:
#   1. Determines current pipeline state from filesystem markers.
#   2. Executes the next single step if its preconditions are met.
#   3. Writes a per-step result + a top-level state marker.
#   4. Exits — does NOT loop. The caller loops.
#
# This idempotency means the script is safe to run hundreds of times: each
# call checks markers and skips work already done. If the laptop closes and
# reopens, just re-launching the loop picks up wherever the chain left off.
#
# Stages (state machine):
#
#   S0  CAPTURE_RUNNING       capture PID alive AND unique samples < 55000
#       └→ no action (wait)
#   S1  CAPTURE_DONE          unique samples ≥ 55000
#       └→ run to-parquet → write S1_DONE marker
#   S2  PARQUET_DONE          S1_DONE marker exists
#       └→ run pre_shuffle → write S2_DONE marker
#   S3  SHUFFLE_DONE          S2_DONE marker exists
#       └→ run compute_hidden_stats → write S3_DONE marker
#   S4  STATS_DONE            S3_DONE marker exists
#       └→ smoke train 100 steps → write S4_DONE marker (with loss curve summary)
#   S5  SMOKE_DONE            S4_DONE marker exists AND smoke loss decreased
#       └→ full train (epochs=3) → write S5_DONE marker
#   S6  TRAIN_DONE            S5_DONE marker exists
#       └→ eval_acceptance.py prep → write S6_DONE marker
#   S7  HELDOUT_DONE          S6_DONE marker exists
#       └→ eval_acceptance.py eval → write S7_DONE marker (with acceptance %)
#   S8  EVAL_DONE             S7_DONE marker exists
#       └→ spec_decode_stub.py → write S8_DONE marker (with speedup estimate)
#   S9  ALL_DONE              S8_DONE marker exists
#       └→ exit clean, nothing more to do
#
# Halt: touch ${PIPELINE_DIR}/HALT to skip all further steps and exit.

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PIPELINE_DIR="${PIPELINE_DIR:-training_data/c2_hidden/eagle3_v0/pipeline}"
mkdir -p "$PIPELINE_DIR"

# ---- Config (overridable via env) ----
SHARD_BIN="${SHARD_BIN:-training_data/c2_hidden/eagle3_v0/shard_000.bin}"
SHARD_PARQUET="${SHARD_PARQUET:-training_data/c2_hidden/eagle3_v0/shard_000.parquet}"
SHARD_SHUFFLED="${SHARD_SHUFFLED:-training_data/c2_hidden/eagle3_v0/shard_000.shuffled.bin}"
HIDDEN_STATS="${HIDDEN_STATS:-tools/training/mlx_eagle/hidden_stats.npz}"
FROZEN_NPZ="${FROZEN_NPZ:-tools/training/mlx_eagle/v2lite_frozen.npz}"
SMOKE_CKPT_DIR="${SMOKE_CKPT_DIR:-tools/training/mlx_eagle/ckpt_smoke}"
FULL_CKPT_DIR="${FULL_CKPT_DIR:-tools/training/mlx_eagle/ckpt}"
HELDOUT_JSONL="${HELDOUT_JSONL:-tests/data/held_out_500.jsonl}"
HELDOUT_SHARD="${HELDOUT_SHARD:-training_data/c2_hidden/held_out_500.bin}"
EVAL_RESULT="${EVAL_RESULT:-reports/path_to_90/stage3_c2/eval_latest.json}"
STUB_RESULT="${STUB_RESULT:-reports/path_to_90/stage3_c2/spec_stub_latest.json}"
TARGET_SAMPLES="${TARGET_SAMPLES:-55000}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"

PY="${PY:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
DISMANTLE="${DISMANTLE:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
KERNEL_PROFILE="${KERNEL_PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"

# Marker files (sequenced)
M_S1="$PIPELINE_DIR/S1_PARQUET_DONE"
M_S2="$PIPELINE_DIR/S2_SHUFFLE_DONE"
M_S3="$PIPELINE_DIR/S3_STATS_DONE"
M_S4="$PIPELINE_DIR/S4_SMOKE_DONE"
M_S5="$PIPELINE_DIR/S5_TRAIN_DONE"
M_S6="$PIPELINE_DIR/S6_HELDOUT_DONE"
M_S7="$PIPELINE_DIR/S7_EVAL_DONE"
M_S8="$PIPELINE_DIR/S8_STUB_DONE"
M_ALL="$PIPELINE_DIR/ALL_DONE"
M_HALT="$PIPELINE_DIR/HALT"

PIPELINE_LOG="$PIPELINE_DIR/pipeline.log"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$PIPELINE_LOG"; }

# ---- Halt check ----
if [ -f "$M_HALT" ]; then
  log "HALT marker present at $M_HALT; not advancing"
  exit 0
fi
if [ -f "$M_ALL" ]; then
  exit 0  # silent — pipeline complete
fi

# ---- Helper: count unique sample_ids in .bin (fast scan) ----
count_unique_samples() {
  $PY -c "
import struct
seen = set()
try:
    with open('$SHARD_BIN', 'rb') as f:
        hdr = f.read(16)
        if hdr[:4] != b'DCAP':
            print(0); exit()
        hd = struct.unpack('<I', hdr[8:12])[0]
        hb_bytes = hd * 2
        while True:
            lb = f.read(2)
            if not lb: break
            (id_len,) = struct.unpack('<H', lb)
            sid = f.read(id_len).decode()
            f.seek(12 + hb_bytes, 1)
            seen.add(sid)
except FileNotFoundError:
    pass
print(len(seen))
" 2>/dev/null
}

# ---- S1: capture → parquet ----
if [ ! -f "$M_S1" ]; then
  SAMPLES=$(count_unique_samples)
  if [ -z "$SAMPLES" ] || [ "$SAMPLES" -lt "$TARGET_SAMPLES" ]; then
    # Not yet — capture still running. Exit silently.
    exit 0
  fi
  log "S1: capture done ($SAMPLES samples). Running to-parquet…"
  if $PY tools/training/capture_hidden.py to-parquet \
      --src "$SHARD_BIN" --dst "$SHARD_PARQUET" --compression zstd \
      >> "$PIPELINE_LOG" 2>&1; then
    echo "samples=$SAMPLES parquet_bytes=$(stat -f %z "$SHARD_PARQUET")" > "$M_S1"
    log "S1: DONE → $SHARD_PARQUET ($(du -h "$SHARD_PARQUET" | cut -f1))"
    echo "PIPELINE_S1_DONE samples=$SAMPLES"
  else
    log "S1: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  exit 0
fi

# ---- S2: parquet → shuffle ----
if [ ! -f "$M_S2" ]; then
  log "S2: pre-shuffle to $SHARD_SHUFFLED…"
  if $PY tools/training/mlx_eagle/pre_shuffle.py \
      --src "$SHARD_BIN" --dst "$SHARD_SHUFFLED" --hidden-dim "$HIDDEN_DIM" \
      >> "$PIPELINE_LOG" 2>&1; then
    echo "shuffled_bytes=$(stat -f %z "$SHARD_SHUFFLED")" > "$M_S2"
    log "S2: DONE → $SHARD_SHUFFLED"
    echo "PIPELINE_S2_DONE"
  else
    log "S2: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  exit 0
fi

# ---- S3: shuffle → hidden stats ----
if [ ! -f "$M_S3" ]; then
  log "S3: computing hidden stats…"
  if $PY tools/training/mlx_eagle/compute_hidden_stats.py \
      --shard "$SHARD_BIN" --hidden-dim "$HIDDEN_DIM" --out "$HIDDEN_STATS" \
      >> "$PIPELINE_LOG" 2>&1; then
    echo "stats=$HIDDEN_STATS" > "$M_S3"
    log "S3: DONE → $HIDDEN_STATS"
    echo "PIPELINE_S3_DONE"
  else
    log "S3: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  exit 0
fi

# ---- S4: smoke train (100 steps) ----
if [ ! -f "$M_S4" ]; then
  log "S4: smoke train (100 steps, lion, bf16, next-aux, hidden norm)…"
  if $PY tools/training/mlx_eagle/train.py \
      --parquet "$SHARD_PARQUET" \
      --frozen "$FROZEN_NPZ" \
      --max-steps 100 --batch-size 16 --seq-len 16 \
      --log-every 10 --save-every 50 \
      --ckpt-dir "$SMOKE_CKPT_DIR" \
      --log "$PIPELINE_DIR/smoke_train.log" \
      --dtype bf16 --optimizer lion --aux-target-kind next \
      --hidden-stats "$HIDDEN_STATS" \
      >> "$PIPELINE_LOG" 2>&1; then
    # Sanity: loss should have decreased over the run.
    FIRST_LOSS=$(grep '"step":' "$PIPELINE_DIR/smoke_train.log" 2>/dev/null | head -1 | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['loss'])" 2>/dev/null || echo "?")
    LAST_LOSS=$(grep '"step":' "$PIPELINE_DIR/smoke_train.log" 2>/dev/null | tail -1 | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['loss'])" 2>/dev/null || echo "?")
    echo "first_loss=$FIRST_LOSS last_loss=$LAST_LOSS" > "$M_S4"
    log "S4: DONE — first_loss=$FIRST_LOSS last_loss=$LAST_LOSS"
    echo "PIPELINE_S4_DONE first=$FIRST_LOSS last=$LAST_LOSS"
  else
    log "S4: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  exit 0
fi

# ---- S5: full train ----
if [ ! -f "$M_S5" ]; then
  log "S5: full train ($EPOCHS epochs, lion, bf16, next-aux, hidden norm)…"
  if $PY tools/training/mlx_eagle/train.py \
      --parquet "$SHARD_PARQUET" \
      --frozen "$FROZEN_NPZ" \
      --epochs "$EPOCHS" --batch-size 16 --seq-len 16 \
      --lr "$LR" \
      --log-every 50 --save-every 500 \
      --ckpt-dir "$FULL_CKPT_DIR" \
      --log "$PIPELINE_DIR/full_train.log" \
      --dtype bf16 --optimizer lion --aux-target-kind next \
      --hidden-stats "$HIDDEN_STATS" \
      >> "$PIPELINE_LOG" 2>&1; then
    LAST_LOSS=$(grep '"step":' "$PIPELINE_DIR/full_train.log" 2>/dev/null | tail -1 | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['loss'])" 2>/dev/null || echo "?")
    echo "final_loss=$LAST_LOSS ckpt=$FULL_CKPT_DIR/latest.npz" > "$M_S5"
    log "S5: DONE — final_loss=$LAST_LOSS"
    echo "PIPELINE_S5_DONE final_loss=$LAST_LOSS"
  else
    log "S5: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  exit 0
fi

# ---- S6: held-out capture ----
if [ ! -f "$M_S6" ]; then
  log "S6: held-out capture (500 samples disjoint from training)…"
  if $PY tools/training/mlx_eagle/eval_acceptance.py prep \
      --n 500 --seed 42 \
      --out-jsonl "$HELDOUT_JSONL" --out-shard "$HELDOUT_SHARD" \
      --binary "$DISMANTLE" --weights "$WEIGHTS" --kernel-profile "$KERNEL_PROFILE" \
      --resume \
      >> "$PIPELINE_LOG" 2>&1; then
    echo "shard=$HELDOUT_SHARD" > "$M_S6"
    log "S6: DONE → $HELDOUT_SHARD"
    echo "PIPELINE_S6_DONE"
  else
    log "S6: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  exit 0
fi

# ---- S7: eval acceptance ----
if [ ! -f "$M_S7" ]; then
  log "S7: eval acceptance…"
  if $PY tools/training/mlx_eagle/eval_acceptance.py eval \
      --ckpt "$FULL_CKPT_DIR/latest.npz" --shard "$HELDOUT_SHARD" \
      --frozen "$FROZEN_NPZ" \
      --out "$EVAL_RESULT" \
      --batch-size 128 --max-records 50000 \
      >> "$PIPELINE_LOG" 2>&1; then
    ACCEPT_TOP1=$($PY -c "import json; print(json.load(open('$EVAL_RESULT'))['accept_top1'])" 2>/dev/null || echo "?")
    echo "accept_top1=$ACCEPT_TOP1 result=$EVAL_RESULT" > "$M_S7"
    log "S7: DONE — accept_top1=$ACCEPT_TOP1"
    echo "PIPELINE_S7_DONE accept_top1=$ACCEPT_TOP1"
  else
    log "S7: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  exit 0
fi

# ---- S8: spec decode stub ----
if [ ! -f "$M_S8" ]; then
  log "S8: spec_decode_stub (K=4)…"
  if $PY tools/training/mlx_eagle/spec_decode_stub.py \
      --ckpt "$FULL_CKPT_DIR/latest.npz" --shard "$HELDOUT_SHARD" \
      --frozen "$FROZEN_NPZ" \
      --k 4 --max-samples 200 \
      --out "$STUB_RESULT" \
      >> "$PIPELINE_LOG" 2>&1; then
    SPEEDUP=$($PY -c "import json; r=json.load(open('$STUB_RESULT')); print(r['headline_metrics']['speedup_vs_no_spec_K_verify'])" 2>/dev/null || echo "?")
    echo "speedup_k=$SPEEDUP result=$STUB_RESULT" > "$M_S8"
    log "S8: DONE — speedup_vs_no_spec_K_verify=$SPEEDUP"
    echo "PIPELINE_S8_DONE speedup=$SPEEDUP"
  else
    log "S8: FAILED — see $PIPELINE_LOG"
    exit 2
  fi
  # Mark all done.
  touch "$M_ALL"
  log "PIPELINE COMPLETE — all 8 stages done"
  echo "PIPELINE_ALL_DONE"
  exit 0
fi
