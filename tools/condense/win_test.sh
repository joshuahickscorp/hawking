#!/usr/bin/env bash
# End-to-end WIN test on any model: STRAND TQ-N base + LoRA-KD recovery vs llama Q4_K,
# relative ppl degradation (same model, same text). Bigger models quantize better, so
# this is where condensation+recovery should WIN (the 0.5B is the pessimistic floor).
# Usage: win_test.sh <hf-model-dir> <tag> [bits] [rank] [steps]
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUNBUFFERED=1
MD="$1"; TAG="$2"; BITS="${3:-3}"; RANK="${4:-32}"; STEPS="${5:-150}"
ST="$MD/model.safetensors"
BAKER=vendor/strand-quant/target/release/quantize-model
export DOCTOR_MODEL="$MD"
export DOCTOR_CALIB="${CALIB:-scratch/calib_corpus.txt}"
jget(){ python3.12 -c "import sys,json;print(json.load(sys.stdin)['ppl'])" 2>/dev/null; }
lppl(){ llama-perplexity -m "$1" -f /tmp/ppl8k.txt -c 512 2>&1 | grep -o 'PPL = [0-9.]*' | tail -1 | grep -o '[0-9.]*'; }

echo "[1] STRAND TQ$BITS bake of $MD ..."
$BAKER --in "$ST" --out "scratch/$TAG-tq$BITS.safetensors" --bits "$BITS" --quality --rht-cols \
  --outlier-channel 1 --outlier-bits 8 2>&1 | tail -1
echo "[2] LoRA-KD recovery (rank $RANK, $STEPS steps) ..."
KD=1 KD_TOPK=64 python3.12 tools/condense/doctor_lora.py "scratch/$TAG-tq$BITS.safetensors" "$STEPS" 3e-4 "$RANK" \
  "scratch/$TAG-healed.safetensors" 2>&1 | grep -E 'base held|best held'
echo "[3] f16 + Q4_K GGUFs ..."
[ -f "scratch/$TAG-f16.gguf" ]  || python3.12 tools/strand/tools/gguf/convert_hf_to_gguf.py "$MD" --outfile "scratch/$TAG-f16.gguf" --outtype f16 >/dev/null 2>&1
[ -f "scratch/$TAG-q4km.gguf" ] || llama-quantize "scratch/$TAG-f16.gguf" "scratch/$TAG-q4km.gguf" Q4_K_M >/dev/null 2>&1
echo "[4] verdict (held-out, same text) ..."
head -c 8000 docs/plans/condense_master_plan_2026_06_22.md > /tmp/ppl8k.txt
hf=$(PPL_TEXT=/tmp/ppl8k.txt python3.12 tools/condense/ppl_bench.py "$MD" - f16 2>/dev/null | jget)
hc=$(PPL_TEXT=/tmp/ppl8k.txt python3.12 tools/condense/ppl_bench.py "$MD" "scratch/$TAG-healed.safetensors" healed 2>/dev/null | jget)
lf=$(lppl "scratch/$TAG-f16.gguf"); lq=$(lppl "scratch/$TAG-q4km.gguf")
python3.12 - "${hf:-0}" "${hc:-0}" "${lf:-0}" "${lq:-0}" "$TAG" "$BITS" <<'PY'
import sys
hf,hc,lf,lq=(float(x) for x in sys.argv[1:5]); tag,bits=sys.argv[5],sys.argv[6]
hd=(hc/hf-1)*100 if hf else 0; ld=(lq/lf-1)*100 if lf else 0
print(f"\n===== WIN TEST: {tag} (TQ{bits}+LoRA-KD vs llama Q4_K) =====")
print(f"  Hawking TQ{bits}+LoRA-KD : +{hd:6.1f}% vs f16  (ppl {hc:.1f} / {hf:.1f})")
print(f"  llama   Q4_K_M        : +{ld:6.1f}% vs f16  (ppl {lq:.1f} / {lf:.1f})")
print("  => " + ("🏆 WIN: better quality at FEWER bpw" if hd < ld else
                 f"Pareto: +{hd:.1f}% vs +{ld:.1f}% (denser, close)"))
PY
