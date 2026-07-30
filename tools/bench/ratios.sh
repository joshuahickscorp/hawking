#!/usr/bin/env bash
# Thin laboratory-harness front-end. Spec: tools/bench/specs/ratios.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export LAB_HARNESS_ARGS="${LAB_HARNESS_ARGS-$*}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/artifacts/runs"
exec python3.12 -m lab.bench_harness run "$ROOT/tools/bench/specs/ratios.json"
