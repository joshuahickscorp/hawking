#!/bin/sh
# Keep the school running across the death of any single process.
#
# The driver already survives a bad cycle; this survives the driver. But the
# driver is not the only thing that can die: hawkingd owns the model worker and
# nothing supervised IT, so a daemon that fell over took the whole school down
# silently -- the driver kept cycling against a control plane that was gone, and
# an unattended run that ends because one process exited is not unattended.
#
# hawkingd supervises the resident, the driver supervises the goals, and this
# supervises both. Started here rather than assumed running, so a cold boot of
# this script is a cold boot of the school.
cd "$(dirname "$0")/.." || exit 1

STATE="$PWD/.hcli/resident/state.json"
PY=/Users/scammermike/.venvs/hawking-aider/bin/python3
[ -x "$PY" ] || PY=python3

log() { echo "--- $(date -u +%FT%TZ) $*" >> .hcli/school/run.log; }

# hawkingd, restarted whenever it is not there.
(
    while true; do
        if ! pgrep -f "hcli.hawkingd --supervise" > /dev/null 2>&1; then
            log "hawkingd absent, starting"
            "$PY" -m hcli.hawkingd --supervise "$STATE" >> .hcli/school/hawkingd.log 2>&1
            log "hawkingd exited $?, restarting in 15s"
            sleep 15
        else
            sleep 20
        fi
    done
) &

# The driver, restarted whenever it exits.
while true; do
    python3 tools/hcli_school.py --start-level 2 >> .hcli/school/run.log 2>&1
    log "driver exited, restarting in 30s"
    sleep 30
done
