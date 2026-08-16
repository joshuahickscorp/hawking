#!/bin/bash
# GPU-locked occupancy compose series. Invoke via:
#   ./tools/gpu_lane_lock.sh dsv-integrate-occupancy receipts/ascent-2026-08-16/run_dsv_integrate_occupancy.sh
# Arms: A = all authority, R = ordered RMSNorm only (default), B = full occupancy (not bit-identical).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BIN=workspace/ops/build/rust/release-fast/examples/gravity_deepseek_v4_native_token_graph
ART=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity
OUTDIR=receipts/ascent-2026-08-16/occ
mkdir -p "$OUTDIR"
export HAWKING_DSV4F_KERNEL_PROBE=0
export HAWKING_DSV4F_VERIFY=admission
export HAWKING_DSV4F_MLA_KV_QAT_SIMD=1
export HAWKING_DSV4F_CB_COLLAPSE=1

run_one() {
  local name="$1" arm="$2" max_layer="$3"
  case "$arm" in
    A) export HAWKING_DSV4F_MLA_RMSNORM_SIMD=0 HAWKING_DSV4F_MLA_WO_A_SIMD=0 HAWKING_DSV4F_MLA_FP8_SIMD=0 ;;
    R) export HAWKING_DSV4F_MLA_RMSNORM_SIMD=1 HAWKING_DSV4F_MLA_WO_A_SIMD=0 HAWKING_DSV4F_MLA_FP8_SIMD=0 ;;
    B) export HAWKING_DSV4F_MLA_RMSNORM_SIMD=1 HAWKING_DSV4F_MLA_WO_A_SIMD=1 HAWKING_DSV4F_MLA_FP8_SIMD=1 ;;
    *) echo "arm must be A, R, or B" >&2; exit 2 ;;
  esac
  echo "===== $name arm=$arm max_layer=$max_layer $(date -u +%H:%M:%S) ====="
  "$BIN" --artifact "$ART" --out "$OUTDIR/${name}.json" --max-layer "$max_layer"
}

# Layer-0 SHA compare first (cheap correctness).
run_one occ-l0-A A 0
run_one occ-l0-B B 0

# Warmup + 3 alternating full-token pairs.
run_one occ-warmup B 42
run_one occ-A1 A 42
run_one occ-B1 B 42
run_one occ-A2 A 42
run_one occ-B2 B 42
run_one occ-A3 A 42
run_one occ-B3 B 42
echo "===== series complete $(date -u +%H:%M:%S) ====="
