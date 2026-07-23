#!/bin/sh
# Bounds every overnight-supervisor tick to a hard wall-clock ceiling.
#
# launchd StartInterval never starts a new instance while the previous one is still running, and it has
# no per-run max-runtime. So a single stuck tick (a transient IO/APFS stall during import, a hung network
# send, a subprocess without a timeout) freezes the ENTIRE handoff chain forever - which is exactly what
# happened once (a tick blocked 2h13m in a kernel open() during import). This wrapper kills any tick that
# runs longer than TIMEOUT, so the worst case is a delayed tick, never a permanently frozen chain.
#
# /bin/sh, sleep, and kill all live on the root filesystem, so this wrapper cannot itself hang on whatever
# stalled the Python tick.
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
SUP=/Users/scammermike/Downloads/hawking/tools/condense/overnight_supervisor.py
# Above every legitimate single-tick subprocess ceiling (readiness 300s, adapter 600s) so a real slow
# tick is never pre-empted, but far below a true hang. Handlers are idempotent, so a killed tick retries.
TIMEOUT=900

"$PY" "$SUP" tick &
child=$!
( sleep "$TIMEOUT"; kill -9 "$child" 2>/dev/null ) &
guard=$!
wait "$child"
rc=$?
kill "$guard" 2>/dev/null   # tick finished in time; retire the watchdog
exit "$rc"
