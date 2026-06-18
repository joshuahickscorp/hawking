#!/usr/bin/env bash
# Launch custom RWKV-7 draft variants sequentially.
#
# Expected wall time on an M3 Pro is roughly 8-16h total for the default
# 3-epoch sweep, depending on prompt lengths, chunked WKV settings, and whether
# teacher-logit KD shards are supplied.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<'EOF'
Usage:
  bash tools/training/launch_draft_sweep.sh

Environment:
  PYTHON=.venv-rwkv/bin/python
  DEVICE=mps
  EPOCHS=3
  GRAD_ACCUM=16
  LR=5e-4
  POLL_SECONDS=300
  DRAFT_VARIANTS="draft_100m draft_150m draft_200m draft_300m"
  TEACHER_LOGITS=/path/to/teacher_logits   # optional KD shards
EOF
        exit 0
    fi
    echo "unknown argument: $1" >&2
    echo "run with --help for usage" >&2
    exit 2
fi

read -r -a variants <<< "${DRAFT_VARIANTS:-draft_100m draft_150m draft_200m draft_300m}"
if [[ "${#variants[@]}" -eq 0 ]]; then
    echo "DRAFT_VARIANTS resolved to an empty list" >&2
    exit 2
fi
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-5e-4}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PYTHON="${PYTHON:-$ROOT/.venv-rwkv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

teacher_args=()
if [[ -n "${TEACHER_LOGITS:-}" ]]; then
    teacher_args=(--teacher-logits "$TEACHER_LOGITS")
fi

for variant in "${variants[@]}"; do
    out="$ROOT/artifacts/lowbit_rwkv7/runs/custom_${variant}"
    mkdir -p "$out"

    echo "=== [$variant] starting trainer + watcher at $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
    VARIANT="$variant" POLL_SECONDS="$POLL_SECONDS" PYTHON="$PYTHON" \
        bash "$ROOT/tools/training/rwkv7_draft_watcher.sh" &
    watcher_pid=$!

    "$PYTHON" "$ROOT/tools/training/rwkv7_train_draft.py" \
        --variant "$variant" \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --out "$out" \
        "${teacher_args[@]}" &
    train_pid=$!

    wait "$train_pid" && train_rc=0 || train_rc=$?
    if [[ $train_rc -ne 0 ]]; then
        echo "=== [$variant] trainer FAILED (exit $train_rc); killing watcher ===" >&2
        kill "$watcher_pid" 2>/dev/null || true
        wait "$watcher_pid" 2>/dev/null || true
        echo "=== [$variant] aborted at $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
        continue
    fi
    wait "$watcher_pid"
    echo "=== [$variant] complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
done

echo "=== draft sweep complete ==="
