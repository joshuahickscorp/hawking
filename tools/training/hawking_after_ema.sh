#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv-rwkv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi

exec "$PY" "$ROOT/tools/training/hawking_after_ema.py" \
  --run-dir "$ROOT/artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8" \
  --artifact-root "$ROOT/artifacts/lowbit_rwkv7/hawking_arc" \
  --threshold "${HAWKING_EMA_TARGET:-6.0}" \
  --checkpoint-interval "${HAWKING_CHECKPOINT_INTERVAL:-5}" \
  --poll-seconds "${HAWKING_POLL_SECONDS:-120}" \
  --seed "${HAWKING_SEED:-1337}" \
  --device "${HAWKING_DEVICE:-mps}" \
  --eval-short-tokens "${HAWKING_EVAL_SHORT_TOKENS:-8192}" \
  --eval-long-tokens "${HAWKING_EVAL_LONG_TOKENS:-32768}" \
  --sample-prompts "${HAWKING_SAMPLE_PROMPTS:-8}" \
  --sample-new-tokens "${HAWKING_SAMPLE_NEW_TOKENS:-160}" \
  "$@"
