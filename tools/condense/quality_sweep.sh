#!/usr/bin/env bash
# REAL perplexity sweep: f16 parent vs TQ-condensed (lowest bits), measured by
# actual inference (not the output-space proxy). Proves "condensation doesn't mean
# loss" in the metric users feel. Optionally runs the doctor (QAT) per bit (heavy).
#
# Usage: tools/condense/quality_sweep.sh [model-dir] [bits-csv]
#   model-dir  default scratch/qwen-05b
#   bits-csv   default "3,2"
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="${1:-scratch/qwen-05b}"
BITS="${2:-3,2}"
ST="$MODEL/model.safetensors"
BAKER="vendor/strand-quant/target/release/quantize-model"
PY=python3.12
OUT="reports/condense/ppl_sweep.jsonl"
mkdir -p reports/condense
: > "$OUT"

[ -f "$BAKER" ] || { echo "baker not built: $BAKER" >&2; exit 2; }
[ -f "$ST" ]    || { echo "model safetensors missing: $ST" >&2; exit 2; }

echo "[ppl] baseline f16 ..." >&2
$PY tools/condense/ppl_bench.py "$MODEL" - f16 2>/dev/null | tee -a "$OUT"

IFS=',' read -ra BARR <<< "$BITS"
for b in "${BARR[@]}"; do
  CST="scratch/$(basename "$MODEL")-tq${b}.safetensors"
  echo "[bake] ${b}-bit -> $CST ..." >&2
  if $BAKER --in "$ST" --out "$CST" --bits "$b" --quality --rht-cols > "/tmp/bake_tq${b}.log" 2>&1; then
    $PY tools/condense/ppl_bench.py "$MODEL" "$CST" "tq${b}" 2>/dev/null | tee -a "$OUT"
  else
    echo "{\"label\":\"tq${b}\",\"error\":\"bake failed (see /tmp/bake_tq${b}.log)\"}" | tee -a "$OUT"
    tail -3 "/tmp/bake_tq${b}.log" >&2
  fi
done

echo "" >&2
echo "============ REAL PPL SWEEP ($MODEL) ============" >&2
$PY - "$OUT" >&2 <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
base = next((r["ppl"] for r in rows if r.get("label")=="f16"), None)
for r in rows:
    if "ppl" in r:
        d = f"  (+{(r['ppl']/base-1)*100:5.1f}% vs f16)" if base and r["label"]!="f16" else ""
        print(f"  {r['label']:6} ppl={r['ppl']:8.3f}{d}")
    else:
        print(f"  {r['label']:6} {r.get('error')}")
PYEOF
echo "=================================================" >&2
