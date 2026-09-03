#!/usr/bin/env bash
# Post-capture: null-first then existing family probe. Do not rewrite the probe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="$ROOT/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1"
CAPTURE_RUN="${1:-}"
if [[ -z "$CAPTURE_RUN" ]]; then
  echo "usage: $0 /absolute/path/to/capture-run" >&2
  exit 2
fi
if [[ ! -f "$CAPTURE_RUN/capture-result.json" ]]; then
  echo "missing capture-result.json under $CAPTURE_RUN" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT/null-first" "$OUT_ROOT/family-probe"
python3 "$ROOT/lab/operators/q30_activation_null_first_report.py" \
  --capture-run "$CAPTURE_RUN" \
  --label "broad_activation_v1" \
  --out-json "$OUT_ROOT/null-first/NULL_BROAD_ACTIVATION.json"
python3 "$ROOT/lab/operators/q30_activation_aware_family_probe.py" \
  --capture-run "$CAPTURE_RUN" \
  --out-dir "$OUT_ROOT/family-probe"
echo "NULL_FIRST=$OUT_ROOT/null-first/NULL_BROAD_ACTIVATION.json"
echo "PROBE_DIR=$OUT_ROOT/family-probe"
