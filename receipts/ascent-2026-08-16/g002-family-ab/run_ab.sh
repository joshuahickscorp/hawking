#!/bin/bash
# G002 family-vs-legacy per-facet A/B. One build, one session, toggle only
# HAWKING_DECODE_FAMILY. Recon-fuse / host facets / occupancy stay at shipping
# defaults. GPU time is completed MTLCommandBuffer only (the existing examples
# already use GPUEndTime-GPUStartTime).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/receipts/ascent-2026-08-16/g002-family-ab"
mkdir -p "$OUT"
Q80="$ROOT/target/release-fast/examples/ascension_qwen80_mixed_hybrid_greedy"
DSV="$ROOT/target/release-fast/examples/gravity_deepseek_v4_native_token_graph"
Q38="$ROOT/target/release-fast/examples/ascension_qwen38_token_ns"
Q80_ROOT="/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1"
Q80_TOK="/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next/tokenizer.json"
Q38_ROOT="/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1"
Q38_TOK="/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json"
DSV_ART="/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"

echo "G002 family A/B start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$OUT/run.log"
echo "binaries q80=$(ls -l "$Q80" | awk '{print $5,$6,$7,$8}') dsv=$(ls -l "$DSV" | awk '{print $5,$6,$7,$8}') q38=$(ls -l "$Q38" | awk '{print $5,$6,$7,$8}')" | tee -a "$OUT/run.log"
echo "git=$(git rev-parse HEAD) family_toggle=HAWKING_DECODE_FAMILY" | tee -a "$OUT/run.log"

run_q80() {
  local arm="$1" pair="$2"
  local fam=1
  [ "$arm" = "legacy" ] && fam=0
  echo "=== Q80 $arm pair $pair HAWKING_DECODE_FAMILY=$fam $(date -u +%H:%M:%S) ===" | tee -a "$OUT/run.log"
  HAWKING_DECODE_FAMILY="$fam" "$Q80" \
    --artifact-root "$Q80_ROOT" \
    --tokenizer "$Q80_TOK" \
    --prompt "Write a function that reverses a string." \
    --max-new-tokens 12 \
    --reps 1 \
    --out "$OUT/q80_${arm}_p${pair}.json" \
    2>>"$OUT/q80_${arm}_p${pair}.stderr" | tee -a "$OUT/q80_${arm}_p${pair}.stdout"
}

run_dsv() {
  local arm="$1" pair="$2"
  local fam=1
  [ "$arm" = "legacy" ] && fam=0
  echo "=== DSV4F $arm pair $pair HAWKING_DECODE_FAMILY=$fam $(date -u +%H:%M:%S) ===" | tee -a "$OUT/run.log"
  HAWKING_DECODE_FAMILY="$fam" HAWKING_DSV4F_ARTIFACT="$DSV_ART" "$DSV" \
    --artifact "$DSV_ART" \
    --max-layer 42 \
    --out "$OUT/dsv4f_${arm}_p${pair}.json" \
    2>>"$OUT/dsv4f_${arm}_p${pair}.stderr" | tee -a "$OUT/dsv4f_${arm}_p${pair}.stdout"
  if [ -f "$OUT/DSV4F_TOKEN_NS_LEDGER.json" ]; then
    mv "$OUT/DSV4F_TOKEN_NS_LEDGER.json" "$OUT/dsv4f_${arm}_p${pair}_ledger.json"
  fi
}

run_q38() {
  local arm="$1" pair="$2"
  local fam=1
  [ "$arm" = "legacy" ] && fam=0
  echo "=== Qwen3.8 $arm pair $pair HAWKING_DECODE_FAMILY=$fam $(date -u +%H:%M:%S) ===" | tee -a "$OUT/run.log"
  HAWKING_DECODE_FAMILY="$fam" "$Q38" \
    --artifact-root "$Q38_ROOT" \
    --tokenizer "$Q38_TOK" \
    --prompt "Say hi." \
    --max-new-tokens 16 \
    --reps 1 \
    --out "$OUT/qwen38_${arm}_p${pair}.json" \
    2>>"$OUT/qwen38_${arm}_p${pair}.stderr" | tee -a "$OUT/qwen38_${arm}_p${pair}.stdout"
}

# Alternating pairs: family then legacy, three times, per graph.
for pair in 1 2 3; do
  run_q80 family "$pair"
  run_q80 legacy "$pair"
done
for pair in 1 2 3; do
  run_dsv family "$pair"
  run_dsv legacy "$pair"
done
for pair in 1 2 3; do
  run_q38 family "$pair"
  run_q38 legacy "$pair"
done

echo "G002 family A/B raw runs done $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run.log"
