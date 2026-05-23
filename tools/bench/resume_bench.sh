#!/usr/bin/env bash
# Resume a paused pipeline. See tools/bench/pause_bench.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
touch artifacts/runs/RESUME
echo "resume signal sent."
echo "(pipeline checks every 10s; will clear PAUSE + RESUME and continue.)"
