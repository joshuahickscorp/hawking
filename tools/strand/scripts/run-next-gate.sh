#!/usr/bin/env bash
# run-next-gate.sh — the durable local gate queue (replaces fragile nohup sequencers).
# Runs the FIRST pending gate in scripts/gates/ (sorted by numeric prefix) when the box
# is idle. Each gate script self-guards via `--pending` (exit 0 = needs running, 1 =
# done), so this is stateless-resumable: it re-derives what's left every call. One gate
# at a time, pid-locked. The conductor calls this from its idle block; adding new dev =
# drop a numbered self-guarding script into scripts/gates/. "Schedule all dev" lives here.
cd /Users/scammermike/Downloads/strand || exit 1
LOCK=scratch/.gate-runner.lock
LOG=scratch/gate-runner.log

# box must be idle — never pile a gate onto a live PV/quant (the MPS co-tenancy rule)
pgrep -f 'strand-qat.py|quantize-model' >/dev/null 2>&1 && exit 0

# a gate already in flight?
if [ -f "$LOCK" ]; then
    pid=$(cat "$LOCK" 2>/dev/null)
    if kill -0 "$pid" 2>/dev/null; then exit 0; else rm -f "$LOCK"; fi
fi

for g in scripts/gates/*.sh; do
    [ -f "$g" ] || continue
    if bash "$g" --pending 2>/dev/null; then
        nohup caffeinate -i bash "$g" >> "$LOG" 2>&1 &
        echo $! > "$LOCK"
        echo "[gate-runner $(date '+%d %H:%M:%S')] launched $g pid $!" >> "$LOG"
        exit 0
    fi
done
exit 0
