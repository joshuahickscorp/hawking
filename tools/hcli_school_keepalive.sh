#!/bin/sh
# Keep the school driver alive across its own death.
#
# The driver already survives a bad cycle; this survives the driver. An
# unattended run that ends because one process exited is not an unattended run.
cd "$(dirname "$0")/.." || exit 1
while true; do
    python3 tools/hcli_school.py --start-level 2 >> .hcli/school/run.log 2>&1
    echo "--- driver exited $(date -u +%FT%TZ), restarting in 30s" >> .hcli/school/run.log
    sleep 30
done
