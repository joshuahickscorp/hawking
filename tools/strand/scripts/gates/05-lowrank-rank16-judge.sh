#!/usr/bin/env bash
# Real-PPL judge for the output-space low-rank residual at rank-16 on up_proj (on top of de-bias).
cd /Users/scammermike/Downloads/strand || exit 1
OUT=research/lowrank-ppl-up16.json
[ "$1" = "--pending" ] && { [ -f "$OUT" ] && exit 1 || exit 0; }
/usr/local/bin/python3 scripts/lowrank-residual-ppl.py \
  --base scratch/qwen-05b --recon research/mp-frontier/dp_d4_r2/recon \
  --actmean research/actmean-qwen05b.json --rank 16 --match up_proj \
  --ctx 2048 --chunks 64 --device cpu --tag up16 --out "$OUT"
/usr/local/bin/python3 scripts/promote.py "$OUT" --model qwen-05b --quiet >> scratch/gate-chain.log 2>&1
