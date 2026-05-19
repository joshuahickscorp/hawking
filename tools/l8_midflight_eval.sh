#!/bin/bash
# path-to-125 L8 — mid-flight eval against the latest training checkpoint.
#
# Runs `eagle4/eagle4.py eval` with a reduced max_records (500 vs 5000)
# so we get a quick top1_target rate without paying the full eval cost.
# Training continues uninterrupted; this just reads `latest.npz` from
# disk and runs the head forward against a held-out subset of parquet
# rows.
#
# Output: one-line JSON to stdout AND appended to
# reports/path_to_90/_levers/l8_midflight_evals.jsonl
#
# Usage:
#   tools/l8_midflight_eval.sh             # eval latest.npz
#   tools/l8_midflight_eval.sh step_000200 # eval a specific step checkpoint

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python"
CKPT_DIR="eagle4/checkpoints/eagle4_v4_fromscratch"
LOG_DIR="reports/path_to_90/_levers"

CKPT_NAME="${1:-latest}"
CKPT_FILE="$CKPT_DIR/${CKPT_NAME}.npz"

if [ ! -f "$CKPT_FILE" ]; then
  echo "{\"error\":\"checkpoint $CKPT_FILE not found\",\"checked_at\":\"$(date -u +%FT%TZ)\"}"
  exit 1
fi

mkdir -p "$LOG_DIR"

# Pull the gate value from the latest training log line. Cheap.
GATE=$(grep -E "^step=" "$LOG_DIR/l8_train.log" | tail -1 | grep -oE "gate=[0-9.]+" | cut -d= -f2 || echo "?")
LAST_STEP=$(grep -E "^step=" "$LOG_DIR/l8_train.log" | tail -1 | grep -oE "^step=[0-9]+" | cut -d= -f2 || echo "?")
ALPHA=$(grep -E "^step=" "$LOG_DIR/l8_train.log" | tail -1 | grep -oE "α=[0-9.]+" | cut -d= -f2 || echo "?")

# Reduced max_records for speed; spot-check a subset, not a full eval.
RAW=$("$VENV_PYTHON" eagle4/eagle4.py eval \
  --ckpt "$CKPT_FILE" \
  --frozen eagle4/v2lite_frozen.npz \
  --parquet training_data/c2_hidden/eagle4_v0/shard_00060.parquet \
  --max-records 500 \
  --mask-top-k 8 \
  2>&1 | tee /tmp/l8_eval_raw.txt | grep -E '"n_top1_target"|"n_top1_corpus"|"n":' || true)

# eagle4.py eval prints JSON then per-layer table. Extract the JSON.
JSON=$("$VENV_PYTHON" -c "
import json, sys
raw = open('/tmp/l8_eval_raw.txt').read()
# eval prints a JSON object first, then per-layer lines starting with '  layer'
try:
    start = raw.index('{')
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{': depth += 1
        elif raw[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    obj = json.loads(raw[start:end])
except Exception as e:
    obj = {'parse_error': str(e), 'raw_head': raw[:200]}
obj['ckpt'] = '${CKPT_NAME}'
obj['train_step'] = '${LAST_STEP}'
obj['train_gate'] = '${GATE}'
obj['train_alpha'] = '${ALPHA}'
obj['checked_at'] = '$(date -u +%FT%TZ)'
n = obj.get('n', 0) or 1
obj['top1_target_rate'] = obj.get('n_top1_target', 0) / n
obj['top1_corpus_rate'] = obj.get('n_top1_corpus', 0) / n
print(json.dumps(obj))
")

echo "$JSON"
echo "$JSON" >>"$LOG_DIR/l8_midflight_evals.jsonl"
