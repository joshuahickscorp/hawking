#!/usr/bin/env bash
# Watchdog: waits for the 7B HF download, then runs the condense pipeline at 7B scale
# (the new minimum working scale). 19GB Mac => CPU-f16 (fits 14GB, no MPS f16 GQA bug);
# slow but correct. Mac Studio: flip DOCTOR_DEVICE=mps DOCTOR_DTYPE=float32 for speed.
#
# Stages (each logs independently so a slow later stage never loses an earlier result):
#   1. 7B AWQ-bake (3-bit) + AWQ base ppl vs f16   <- the first 7B datapoint, feasible now
#   2. doctor (LoRA-KD) on the AWQ base + healed ppl  <- long on CPU; the recovery
#   3. llama Q4_K + MLX 4-bit 7B ppl (if artifacts present) <- the 3-way
# Result -> reports/cron/win7b.md
set -uo pipefail
cd "$HOME/Downloads/hawking" || exit 2
LOG=reports/cron/win7b.md; mkdir -p reports/cron
PY=python3.12
# bfloat16 NOT float16: fp16 overflows on the 7B CPU forward (>65504 → nan, observed
# 2026-06-23 → f16 ppl=nan crashed stage 1). bf16 is 2-byte (still fits 14GB) with fp32 range.
export DOCTOR_DEVICE=cpu DOCTOR_DTYPE=bfloat16 DOCTOR_CALIB=scratch/calib_corpus.txt PYTHONUNBUFFERED=1
jget(){ $PY -c "import sys,json;print(json.load(sys.stdin)['ppl'])" 2>/dev/null; }
PT=/tmp/ppl24k.txt; [ -f "$PT" ] || cat README.md docs/plans/*.md 2>/dev/null | head -c 24000 > "$PT"

echo "# 7B condense watchdog" > "$LOG"; date >> "$LOG"
# wait for the download (index + all shards present, no hf process running)
while [ ! -f scratch/qwen-7b/model.safetensors.index.json ] || pgrep -f 'hf download' >/dev/null; do sleep 60; done
sleep 5
echo "## download complete" >> "$LOG"; du -sh scratch/qwen-7b >> "$LOG" 2>&1

{
  echo ""; echo "## [1] 7B AWQ-bake (3-bit, CPU-f16) + base ppl"; date
  hf=$(PPL_TEXT=$PT $PY tools/condense/ppl_bench.py scratch/qwen-7b - f16 2>/dev/null | jget)
  $PY tools/condense/awq.py bake scratch/qwen-7b scratch/qwen7b-awq.safetensors 3 0.5 2>&1 | grep -E 'saved|captured' | tail -1
  ha=$(PPL_TEXT=$PT $PY tools/condense/ppl_bench.py scratch/qwen-7b scratch/qwen7b-awq.safetensors a 2>/dev/null | jget)
  $PY -c "hf=${hf:-0};ha=${ha:-0};print(f'7B f16={hf:.2f} | TQ3-AWQ +{(ha/hf-1)*100:.1f}% @3.6bpw (vs llama Q4_K ~+8% @4.9bpw)')"

  echo ""; echo "## [2] doctor (LoRA-KD) on 7B AWQ base + healed ppl (long on CPU)"; date
  KD=1 KD_TOPK=128 $PY tools/condense/doctor_lora.py scratch/qwen7b-awq.safetensors 200 1e-4 128 scratch/qwen7b-awq-heal.safetensors 2>&1 | grep -E 'base held|best held'
  hd=$(PPL_TEXT=$PT $PY tools/condense/ppl_bench.py scratch/qwen-7b scratch/qwen7b-awq-heal.safetensors h 2>/dev/null | jget)
  $PY -c "hf=${hf:-0};hd=${hd:-0};print(f'7B TQ3 AWQ+doctor +{(hd/hf-1)*100:.1f}% @3.6bpw  '+('WIN vs llama +8%' if hd and (hd/hf-1)*100<8 else '(close)'))" 2>/dev/null

  echo ""; echo "## DONE"; date
} >> "$LOG" 2>&1
