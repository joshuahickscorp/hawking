#!/usr/bin/env bash
# Thin laboratory-harness front-end. Spec: tools/strand/specs/strand_act2_night2.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export LAB_HARNESS_ARGS="${LAB_HARNESS_ARGS-$*}"
export PYTHONPATH="$ROOT/tools/foundry${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/artifacts/runs"
exec python3.12 -m lab_harness run "$ROOT/tools/strand/specs/strand_act2_night2.json"
