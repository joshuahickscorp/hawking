#!/usr/bin/env bash
# Pause the slm overnight trainer tree, run a command, resume slm.
# Used so dismantle benches don't compete with slm trainer for GPU.
# State-safe: SIGSTOP freezes the process; SIGCONT resumes exactly.
set -euo pipefail

# Find slm overnight_shift controller (top of the tree)
SLM_PID=$(pgrep -f "overnight_shift.py" | head -1 || true)

paused_pids=""
restore() {
    if [[ -n "$paused_pids" ]]; then
        echo "[with_slm_paused] resuming: $paused_pids" >&2
        kill -CONT $paused_pids 2>/dev/null || true
    fi
}
trap restore EXIT INT TERM

if [[ -n "$SLM_PID" ]]; then
    # Capture controller + all descendants (train.sh, mamba_byte_train.py, etc.)
    paused_pids=$(pgrep -g $(ps -o pgid= -p "$SLM_PID" | tr -d ' ') | tr '\n' ' ')
    if [[ -z "$paused_pids" ]]; then
        paused_pids="$SLM_PID"
    fi
    echo "[with_slm_paused] pausing slm tree: $paused_pids" >&2
    kill -STOP $paused_pids 2>/dev/null || true
    # Brief settle so GPU work flushes
    sleep 2
else
    echo "[with_slm_paused] no slm overnight_shift.py found; running unpaused" >&2
fi

"$@"
