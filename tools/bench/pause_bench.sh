#!/usr/bin/env bash
# Pause any running pipeline that honors the artifacts/runs/PAUSE flag.
# Compatible with overnight_path_to_50_bench.sh and path_to_50_matrix.sh.
#
# Usage:
#   bash tools/bench/pause_bench.sh                # pause; pipeline stops between stages
#   bash tools/bench/resume_bench.sh               # resume
#
# Pipeline does NOT kill in-flight trials — it waits for the current
# stage/trial to finish, then pauses before the next. That avoids wasted
# compute and corrupted artifacts.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts/runs
touch artifacts/runs/PAUSE
rm -f artifacts/runs/RESUME
echo "paused — pipeline will halt before next stage/trial."
echo "to resume: bash tools/bench/resume_bench.sh"
