#!/usr/bin/env bash
# KL-routed PV: protect up/v/gate (the real output-damage classes per rung-kl), NOT down.
cd /Users/scammermike/Downloads/strand || exit 1
OUT=research/pv-dp/pv-kl-routed.json
STATE=research/pv-dp/pv-kl-routed.pt
PRIOR=research/pv-dp/pv-dp.json   # fires only after the down-protect PV verdict lands
[ "$1" = "--pending" ] && { { [ -f "$PRIOR" ] && [ ! -f "$OUT" ]; } && exit 0 || exit 1; }
[ -f "$PRIOR" ] && /usr/local/bin/python3 scripts/promote.py "$PRIOR" --model qwen-05b --quiet >> scratch/gate-chain.log 2>&1
COMMON=(
  --model scratch/qwen-05b --quant strand --bits 2 --steps 200 --lr 0.0001 --kd --kd-cache research/pv-dp/kdcache --device mps --ctx 512
  --strand-flags "--bits 2 --l 12 --outlier-channel 1 --rung-config configs/mp-kl-routed.json --threads 12"
  --eval-chunks 64 --eval-ctx 2048 --eval-every 150 --save "$STATE"
  --lineage research/pv-lineage.jsonl --arm-name pv_kl_routed_q2
  --out "$OUT"
)
# wall-clock opt (2026-06-13): batch1/accum4 is numerically identical to batch2/accum2
# (same effective batch, same accumulated gradient) but ~half the peak activation memory
# -> less swap pressure on the 18GB box. Gate-preserving (does NOT change the science / the
# <26.77 cloud-gating comparison). eval-every 75->150 drops 2 mid-run evals (monitoring only).
/usr/local/bin/python3 scripts/strand-qat.py "${COMMON[@]}" \
  --batch 1 --grad-accum 4 --grad-checkpoint >> research/pv-dp/run-kl-routed.log 2>&1
rc=$?
if [ ! -f "$OUT" ]; then
  echo "[gate10 $(date '+%H:%M:%S')] batch2 rc=$rc produced no json; retry batch1/accum4" >> research/pv-dp/run-kl-routed.log
  EXTRA=()
  [ -f "$STATE" ] && EXTRA=(--init-state "$STATE")
  /usr/local/bin/python3 scripts/strand-qat.py "${COMMON[@]}" "${EXTRA[@]}" \
    --batch 1 --grad-accum 4 >> research/pv-dp/run-kl-routed.log 2>&1
fi
[ -f "$OUT" ] && /usr/local/bin/python3 scripts/promote.py "$OUT" --model qwen-05b --quiet >> scratch/gate-chain.log 2>&1
