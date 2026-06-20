#!/usr/bin/env bash
# Watch one custom RWKV-7 draft run and append eval JSON as checkpoints appear.
#
# Usage:
#   VARIANT=draft_150m bash tools/training/rwkv7_draft_watcher.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  VARIANT=draft_150m bash tools/training/rwkv7_draft_watcher.sh

Environment:
  VARIANT, RUN, HF, TEACHER, DATA, POLL_SECONDS, PYTHON
  PPL_TOKENS, PPL_STRIDE, ACCEPT_SEQS, ACCEPT_MAX_LENGTH, DRAFT_K, EVAL_DEVICE
EOF
    exit 0
fi

VARIANT="${VARIANT:-draft_150m}"
RUN="${RUN:-$ROOT/artifacts/lowbit_rwkv7/runs/custom_${VARIANT}}"
HF="${HF:-$ROOT/models/rwkv7-g1-04-hf}"
TEACHER="${TEACHER:-$HF/model.safetensors}"
DATA="${DATA:-$ROOT/artifacts/rwkv7_posttrain/sft.jsonl}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EVAL_LOG="$RUN/eval_log.jsonl"
WATCH_LOG="$RUN/watcher.log"
PYTHON="${PYTHON:-$ROOT/.venv-rwkv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

mkdir -p "$RUN"

file_sig() {
    local path="$1"
    stat -f "%m:%z" "$path" 2>/dev/null || stat -c "%Y:%s" "$path"
}

eval_ckpt() {
    local ckpt="$1"
    local tmp
    tmp="$(mktemp)"
    echo "[watcher] evaluating $ckpt at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$WATCH_LOG"
    if "$PYTHON" "$ROOT/tools/training/rwkv7_draft_ppl_eval.py" \
        --variant "$VARIANT" \
        --checkpoint "$ckpt" \
        --teacher "$TEACHER" \
        --hf-dir "$HF" \
        --data "$DATA" \
        --tokens "${PPL_TOKENS:-4096}" \
        --stride "${PPL_STRIDE:-4096}" \
        --accept-seqs "${ACCEPT_SEQS:-200}" \
        --accept-max-length "${ACCEPT_MAX_LENGTH:-256}" \
        --draft-k "${DRAFT_K:-4}" \
        --device "${EVAL_DEVICE:-cpu}" \
        > "$tmp" 2>> "$WATCH_LOG"; then
        cat "$tmp" >> "$EVAL_LOG"
        "$PYTHON" - "$tmp" <<'PY'
import json
import sys

record = json.loads(open(sys.argv[1], encoding="utf-8").read())
step = record.get("step")
ppl = record.get("wikitext2_ppl")
accept = 100.0 * record.get("draft_accept_rate", 0.0)
print(f"[step={step}] ppl={ppl:.2f} accept={accept:.2f}%")
PY
    else
        echo "[watcher] eval failed for $ckpt; see $WATCH_LOG" | tee -a "$WATCH_LOG"
    fi
    rm -f "$tmp"
}

last_latest_sig=""
last_final_sig=""

echo "[watcher] started for $VARIANT at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$WATCH_LOG"
echo "[watcher] run=$RUN poll=${POLL_SECONDS}s" | tee -a "$WATCH_LOG"

while true; do
    latest="$RUN/latest/state_dict.pt"
    final="$RUN/final/state_dict.pt"

    if [[ -f "$latest" ]]; then
        sig="$(file_sig "$latest")"
        if [[ "$sig" != "$last_latest_sig" ]]; then
            last_latest_sig="$sig"
            eval_ckpt "$latest"
        fi
    fi

    if [[ -f "$final" ]]; then
        sig="$(file_sig "$final")"
        if [[ "$sig" != "$last_final_sig" ]]; then
            last_final_sig="$sig"
            eval_ckpt "$final"
        fi
        echo "[watcher] final checkpoint observed; exiting." | tee -a "$WATCH_LOG"
        break
    fi

    sleep "$POLL_SECONDS"
done
