#!/usr/bin/env bash
# Sweep LoRA recovery configs to find the first that BEATS llama Q4_K (+2.1%) at < 4.5 bpw.
# Each config: STRAND base + LoRA(KD) -> held-out ppl vs f16, compared to Q4_K.
# Sequential (single GPU). Live output (PYTHONUNBUFFERED). Big diverse calib by default.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUNBUFFERED=1
export DOCTOR_CALIB="${CALIB:-scratch/calib_corpus_big.txt}"
PY=python3.12
head -c 8000 docs/plans/condense_master_plan_2026_06_22.md > /tmp/ppl8k.txt
jget() { $PY -c "import sys,json;print(json.load(sys.stdin)['ppl'])" 2>/dev/null; }
hf=$(PPL_TEXT=/tmp/ppl8k.txt $PY tools/condense/ppl_bench.py scratch/qwen-05b - f16 2>/dev/null | jget)
echo "f16 reference ppl (3-way text): $hf"
echo "======================================================================"

# base_tag : rank : steps : lr : bpw : label
CONFIGS=(
  "tq3-full:64:400:2e-4:3.7:TQ3+KD-r64"
  "tq3-full:128:400:2e-4:4.1:TQ3+KD-r128"
  "tq2-full:64:400:2e-4:3.0:TQ2+KD-r64"
  "tq2-full:128:500:2e-4:3.4:TQ2+KD-r128"
)
for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r base rank steps lr bpw label <<< "$cfg"
  out="scratch/qwen-05b-sweep-${base}-r${rank}.safetensors"
  echo ""; echo ">>> $label  (base=$base rank=$rank steps=$steps lr=$lr)"
  KD=1 $PY tools/condense/doctor.py lora "scratch/qwen-05b-${base}.safetensors" "$steps" "$lr" "$rank" "$out" 2>&1 \
    | grep -E 'base held|best held|KD:' | sed 's/^/    /'
  hc=$(PPL_TEXT=/tmp/ppl8k.txt $PY tools/condense/ppl_bench.py scratch/qwen-05b "$out" "$label" 2>/dev/null | jget)
  $PY tools/condense/verdict.py "$hf" "${hc:-0}" "$bpw" "$label"
done
echo "======================================================================"
echo "sweep done"
