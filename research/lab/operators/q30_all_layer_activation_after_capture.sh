#!/usr/bin/env bash
# Post all-layer capture: per-layer null-first, then surplus-first repack.
# Does not claim coherence. Does not touch the live Q30 server.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="$ROOT/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/all-layer-activation-v1"
CANDIDATE_ROOT="$ROOT/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/activation-weighted-svd-all-layer-v1"
CAPTURE_RUN="${1:-}"
if [[ -z "$CAPTURE_RUN" ]]; then
  echo "usage: $0 /absolute/path/to/all-layer-capture-run" >&2
  exit 2
fi
if [[ ! -f "$CAPTURE_RUN/capture-result.json" ]]; then
  echo "missing capture-result.json under $CAPTURE_RUN" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT/null-first" "$CANDIDATE_ROOT"

echo "== null-first (per layer) =="
python3 "$ROOT/lab/operators/q30_activation_null_first_report.py" \
  --capture-run "$CAPTURE_RUN" \
  --label "all_layer_activation_v1" \
  --out-json "$OUT_ROOT/null-first/NULL_ALL_LAYER_ACTIVATION.json"

echo "== surplus-first all-layer repack (complete_physical_bpw <= 1.5) =="
python3 "$ROOT/lab/operators/ascension_qwen30_activation_weighted_svd_repack.py" \
  --capture-run "$CAPTURE_RUN" \
  --root "$CANDIDATE_ROOT" \
  --require-all-layer-capture \
  --workers "${WORKERS:-4}"

echo "NULL_FIRST=$OUT_ROOT/null-first/NULL_ALL_LAYER_ACTIVATION.json"
echo "CANDIDATE_ROOT=$CANDIDATE_ROOT"
echo "Inspect coverage: $CANDIDATE_ROOT/QWEN30_ACTIVATION_WEIGHTED_SVD_V1_COVERAGE_RECEIPT.json"
