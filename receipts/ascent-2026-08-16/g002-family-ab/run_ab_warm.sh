#!/bin/bash
# Extra warm pairs 4-6 after the first session already filled the page cache.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/receipts/ascent-2026-08-16/g002-family-ab"
Q80="$ROOT/target/release-fast/examples/ascension_qwen80_mixed_hybrid_greedy"
DSV="$ROOT/target/release-fast/examples/gravity_deepseek_v4_native_token_graph"
Q80_ROOT="/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1"
Q80_TOK="/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next/tokenizer.json"
DSV_ART="/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"

echo "G002 warm pairs 4-6 start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run.log"

for pair in 4 5 6; do
  for arm in family legacy; do
    fam=1; [ "$arm" = "legacy" ] && fam=0
    echo "=== Q80 $arm pair $pair HAWKING_DECODE_FAMILY=$fam $(date -u +%H:%M:%S) ===" | tee -a "$OUT/run.log"
    HAWKING_DECODE_FAMILY="$fam" "$Q80" \
      --artifact-root "$Q80_ROOT" --tokenizer "$Q80_TOK" \
      --prompt "Write a function that reverses a string." \
      --max-new-tokens 12 --reps 1 \
      --out "$OUT/q80_${arm}_p${pair}.json" \
      2>>"$OUT/q80_${arm}_p${pair}.stderr" | tee -a "$OUT/q80_${arm}_p${pair}.stdout"
  done
done

for pair in 4 5 6; do
  for arm in family legacy; do
    fam=1; [ "$arm" = "legacy" ] && fam=0
    echo "=== DSV4F $arm pair $pair HAWKING_DECODE_FAMILY=$fam $(date -u +%H:%M:%S) ===" | tee -a "$OUT/run.log"
    HAWKING_DECODE_FAMILY="$fam" HAWKING_DSV4F_ARTIFACT="$DSV_ART" "$DSV" \
      --artifact "$DSV_ART" --max-layer 42 \
      --out "$OUT/dsv4f_${arm}_p${pair}.json" \
      2>>"$OUT/dsv4f_${arm}_p${pair}.stderr" | tee -a "$OUT/dsv4f_${arm}_p${pair}.stdout"
    if [ -f "$OUT/DSV4F_TOKEN_NS_LEDGER.json" ]; then
      mv "$OUT/DSV4F_TOKEN_NS_LEDGER.json" "$OUT/dsv4f_${arm}_p${pair}_ledger.json"
    fi
  done
done

echo "G002 warm pairs 4-6 done $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run.log"
