#!/usr/bin/env bash
# Watchdog: waits for the 32B .tq bake to finish, then runs the benches automatically
# (your "put a watchdog on the long test"). File-signal wait (not pgrep). Detached.
#
# Runs: (1) 1.5B iso-model 3-way (Hawking TQ3 vs llama Q4_K vs MLX 4-bit) — achievable now;
#       (2) 32B llama Q4_K tps (the contender — swaps on 19GB);
#       (3) 32B Hawking TQ serve ATTEMPT (CPU path Q12-inflates to ~128GB -> expect OOM;
#           confirms the GPU bitslice path is required = Phase 2).
# Results -> reports/cron/cliff_results.md
set -uo pipefail
cd "$HOME/Downloads/hawking" || exit 2
LOG=reports/cron/cliff_results.md
mkdir -p reports/cron
PY=python3.12
jget(){ $PY -c "import sys,json;print(json.load(sys.stdin)['ppl'])" 2>/dev/null; }
lppl(){ llama-perplexity -m "$1" -f /tmp/ppl20k.txt -c 512 2>&1 | grep -o 'PPL = [0-9.]*' | tail -1 | grep -o '[0-9.]*'; }

echo "# Cliff watchdog — waiting for 32B bake" > "$LOG"; date >> "$LOG"
while [ ! -f models/qwen32b-tq3.tq ] || pgrep -f 'release/tq_bake' >/dev/null; do sleep 60; done
sleep 10
{
  echo ""; echo "## 32B bake done"; date; ls -la models/qwen32b-tq3.tq | awk '{print "tq:",$5"B"}'
  # loader convention: <weights>.tq next to the GGUF
  cp -f models/qwen32b-tq3.tq models/qwen32b-q4km.tq 2>/dev/null

  echo ""; echo "## 1.5B iso-model 3-way (Hawking TQ3 vs llama Q4_K vs MLX 4-bit, same model)"
  cat docs/plans/condense_master_plan_2026_06_22.md docs/plans/native_tq_serving_impl.md README.md 2>/dev/null | head -c 20000 > /tmp/ppl20k.txt
  hf=$(PPL_TEXT=/tmp/ppl20k.txt $PY tools/condense/ppl_bench.py scratch/qwen-15b - f16 2>/dev/null | jget)
  ht=$(PPL_TEXT=/tmp/ppl20k.txt $PY tools/condense/ppl_bench.py scratch/qwen-15b scratch/qwen15b-tq3.safetensors tq3 2>/dev/null | jget)
  lf=$(lppl scratch/qwen15b-f16.gguf); lq=$(lppl scratch/qwen15b-q4km.gguf)
  [ -d scratch/qwen15b-mlx4 ]  || $PY -m mlx_lm.convert --hf-path scratch/qwen-15b -q --q-bits 4 --mlx-path scratch/qwen15b-mlx4 >/dev/null 2>&1
  [ -d scratch/qwen15b-mlxbf ] || $PY -m mlx_lm.convert --hf-path scratch/qwen-15b --dtype bfloat16 --mlx-path scratch/qwen15b-mlxbf >/dev/null 2>&1
  mf=$($PY tools/condense/mlx_ppl.py scratch/qwen15b-mlxbf /tmp/ppl20k.txt 2>/dev/null)
  m4=$($PY tools/condense/mlx_ppl.py scratch/qwen15b-mlx4 /tmp/ppl20k.txt 2>/dev/null)
  $PY - "${hf:-0}" "${ht:-0}" "${lf:-0}" "${lq:-0}" "${mf:-0}" "${m4:-0}" <<'PY'
import sys
hf,ht,lf,lq,mf,m4=(float(x) for x in sys.argv[1:7])
def d(q,f): return f"+{(q/f-1)*100:.1f}%" if f and q else "NA"
print(f"| engine | bpw | ppl-degradation vs own f16 |")
print(f"|---|--:|--:|")
print(f"| Hawking TQ3 (PTQ) | ~3.6 | {d(ht,hf)} (DENSEST; recovery closes it) |")
print(f"| llama Q4_K_M | ~4.9 | {d(lq,lf)} |")
print(f"| MLX 4-bit | ~4.5 | {d(m4,mf)} |")
PY

  echo ""; echo "## 32B contender: llama Q4_K decode tps (20GB on 19GB -> swaps)"
  perl -e 'alarm 300; exec @ARGV' llama-cli -m models/qwen32b-q4km.gguf -p "The science of operations is" -n 16 -no-cnv 2>&1 | grep -iE 'eval time|tokens per second|t/s' | tail -3 || echo "(swapped/timed out — the point of the cliff)"

  echo ""; echo "## 32B Hawking TQ serve ATTEMPT (CPU path Q12-inflates ~128GB -> expect OOM = needs GPU Phase 2)"
  perl -e 'alarm 300; exec @ARGV' env HAWKING_QWEN_TQ=1 ./target/release/hawking generate --weights models/qwen32b-q4km.gguf --prompt "The science of operations is" --max-new-tokens 8 --temperature 0 2>&1 | tail -5 || echo "(OOM/timeout — confirms GPU bitslice serving needed)"
  echo ""; echo "## DONE"; date
} >> "$LOG" 2>&1
