#!/usr/bin/env bash
# Rigorous 3-way QUALITY bench: relative perplexity degradation (quant ppl / f16 ppl)
# per engine, SAME model + SAME text. Relative degradation cancels harness/tokenizer
# differences, so Hawking (transformers) vs llama (llama-perplexity) are comparable.
#
# The honest scoreboard: Hawking only WINS when condensed+DOCTOR degradation < Q4_K's
# (~+2%) at fewer bpw. PTQ alone LOSES (TQ3 ~+44%) — pass a doctor-recovered safetensors.
#
# Usage: quality_3way.sh <hawking-condensed.safetensors> [label]
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

COND="${1:-scratch/qwen-05b-tq2-full.safetensors}"
LBL="${2:-condensed}"
MODEL=scratch/qwen-05b
PY=python3.12
CALIB=/tmp/ppl_3way_calib.txt

cat docs/RESEARCH.md docs/plans/tq_compute_for_memory_appendix_2026_07_14.md 2>/dev/null \
  | head -c 8000 > "$CALIB"
[ -s "$CALIB" ] || { echo "calib text empty" >&2; exit 2; }

# ensure the llama-side GGUFs exist (convert + quantize once)
[ -f scratch/qwen-05b-f16.gguf ] || \
  $PY tools/strand/tools/gguf/convert_hf_to_gguf.py "$MODEL" --outfile scratch/qwen-05b-f16.gguf --outtype f16 >/dev/null 2>&1
[ -f scratch/qwen-05b-q4km.gguf ] || \
  llama-quantize scratch/qwen-05b-f16.gguf scratch/qwen-05b-q4km.gguf Q4_K_M >/dev/null 2>&1

jget() { $PY -c "import sys,json;print(json.load(sys.stdin)['ppl'])" 2>/dev/null; }
lppl() { llama-perplexity -m "$1" -f "$CALIB" -c 512 2>&1 | grep -o 'PPL = [0-9.]*' | tail -1 | grep -o '[0-9.]*'; }

echo "[3way] Hawking f16 + ${LBL} (transformers) ..." >&2
hf=$(PPL_TEXT=$CALIB $PY -m tools.condense legacy ppl_bench "$MODEL" - f16 2>/dev/null | jget)
hc=$(PPL_TEXT=$CALIB $PY -m tools.condense legacy ppl_bench \
  "$MODEL" "$COND" "$LBL" 2>/dev/null | jget)
echo "[3way] llama f16 + Q4_K (llama-perplexity) ..." >&2
lf=$(lppl scratch/qwen-05b-f16.gguf); lq=$(lppl scratch/qwen-05b-q4km.gguf)

$PY - "${hf:-0}" "${hc:-0}" "${lf:-0}" "${lq:-0}" "$LBL" <<'PYEOF'
import sys
hf,hc,lf,lq = (float(x) for x in sys.argv[1:5]); lbl=sys.argv[5]
print("\n===== 3-WAY QUALITY — relative degradation vs each engine's own f16 =====")
print("  (same model, same text, each in its faithful harness; lower % = closer to 1:1)\n")
if hf and hc: print(f"  Hawking {lbl:14} @low-bit : +{(hc/hf-1)*100:7.1f}%   (ppl {hc:.1f} vs f16 {hf:.1f})")
if lf and lq: print(f"  llama.cpp Q4_K_M    @4.5bpw  : +{(lq/lf-1)*100:7.1f}%   (ppl {lq:.1f} vs f16 {lf:.1f})")
win = (hf and hc and lf and lq and (hc/hf-1) < (lq/lf-1))
print(f"\n  => Hawking {'WINS' if win else 'LOSES'} on quality "
      f"({'condensed+doctor beats Q4_K at fewer bpw' if win else 'needs the doctor — PTQ alone is worse than Q4_K'})")
print("==========================================================================")
PYEOF
