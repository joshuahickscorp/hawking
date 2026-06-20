#!/usr/bin/env bash
# strand-act2-night2.sh — NIGHT 2: rung-3 PV (train through what you ship) + full-box utilization.
#
# Empirical inputs (will.md §4/§7): proxy transfer DEAD (3,013 vs 80.7) → the forward must be
# the REAL encoder recon (strand mode in strand-qat.py). Full 0.5B requant ≈ 15 min @12T → 8T
# in-loop (CPU lane holds 4). QAT+KD = 10.2–10.5 GB capped; batch-1 only. 7B eval (~15 GB)
# cannot coexist with QAT → quant overnight (--no-eval), eval when the GPU lane is done.
#
#   GPU lane:  A  rung-3 PV 0.5B (300 steps, requant-75, KD)        — the headline experiment
#              A2 [gate: A_after < 81] +600 steps from A's weights  — beat the PTQ floor? extend
#              B  [gate: ternary 800→1000 slope > 4%] ternary +1000 from the 1k export
#   CPU lane:  D-guard (mp-down3 if night-1 left it missing)
#              L  Llama-2-7B download attempt (network-bound, backgrounded, graceful skip)
#              S7 Qwen-7B q2_l12_out1 PTQ floor — quant only, threads 4, per-shard resume
#              S7-eval once the GPU lane exits (MPS free)
#
# Launch gate: waits for scratch/.polish-landed (created after the polish-phase2 merge + tests)
# AND for the night-1 orchestrator to exit. Relaunch-safe: every phase skips on its artifact.
#
#   nohup caffeinate -dimsu ./scripts/strand-act2-night2.sh \
#       > scratch/qwen-05b/night2.log 2>&1 & disown

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

PY=/usr/local/bin/python3
M=scratch/qwen-05b
M7=scratch/qwen-7b
SENTINEL=scratch/.polish-landed
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.92
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7

log()      { echo "[n2 $(date '+%H:%M:%S')] $*"; }
avail_gb() { df -g / | awk 'NR==2{print $4}'; }
disk_ok()  { [ "$(avail_gb)" -ge "${2:-4}" ] || { log "$1: SKIP — $(avail_gb)GB free < ${2:-4}GB"; return 1; }; }
ppl_of()   { "$PY" -c "import json;print(json.load(open('$1'))['$2'])" 2>/dev/null || echo ""; }

QAT=("$PY" scripts/strand-qat.py --model "$M" --ctx 512 --eval-chunks 64 --eval-ctx 2048
     --grad-accum 4 --batch 1 --device mps --grad-checkpoint --lr 1e-4 --kd)
PVFLAGS="--bits 2 --l 12 --outlier-channel 1 --threads 8"

log "night-2 armed: waiting for polish landing + night-1 exit ($(avail_gb)GB free)"
until [ -f "$SENTINEL" ]; do sleep 30; done
while pgrep -f 'strand-act2-overnight.sh' >/dev/null 2>&1; do sleep 60; done
while pgrep -f 'strand-qat.py' >/dev/null 2>&1; do sleep 60; done
log "gates open — lanes starting"

# ─────────────────────────────── GPU LANE ───────────────────────────────────
gpu_lane() {
    # A: rung-3 PV — forward IS the deployment recon, STE backward, requant every 75
    if [ ! -f "$M/qat-pv.json" ]; then
        disk_ok "arm A" 8 && {
        log "[gpu] ARM A: rung-3 PV, 300 steps, requant-75 (the headline)"
        "${QAT[@]}" --model "$M" --quant strand --steps 300 --train-chunks 1200 \
            --requant-every 75 --eval-every 75 --strand-flags "$PVFLAGS" \
            --save "$M/qat-pv.pt" --save-hf "$M/qat-pv-hf" --out "$M/qat-pv.json" \
            >> "$M/qat-pv.log" 2>&1 || log "[gpu] ARM A rc=$?"
        [ -f "$M/qat-pv.json" ] && [ -d "$M/qat-pv-hf" ] && rm -f "$M/qat-pv.pt"
        }
    else log "[gpu] ARM A: skip (artifact exists)"; fi

    local AP; AP="$(ppl_of "$M/qat-pv.json" ppl_after)"
    log "[gpu] ARM A verdict: PV recon ppl=${AP:-none} (PTQ floor 80.7, bf16 12.55)"

    # A2: if PV beat (or basically matched) the floor with only 300 steps, extend
    if [ -n "$AP" ] && "$PY" -c "exit(0 if float('$AP') < 81 else 1)" 2>/dev/null; then
        if [ ! -f "$M/qat-pv2.json" ] && [ -d "$M/qat-pv-hf" ]; then
            disk_ok "arm A2" 8 && {
            log "[gpu] ARM A2: GATE OPEN (${AP} < 81) — +600 steps, requant-100"
            "${QAT[@]}" --model "$M/qat-pv-hf" --quant strand --steps 600 --train-chunks 2400 \
                --requant-every 100 --eval-every 100 --strand-flags "$PVFLAGS" \
                --save "$M/qat-pv2.pt" --save-hf "$M/qat-pv2-hf" --out "$M/qat-pv2.json" \
                >> "$M/qat-pv2.log" 2>&1 || log "[gpu] ARM A2 rc=$?"
            [ -f "$M/qat-pv2.json" ] && [ -d "$M/qat-pv2-hf" ] && rm -f "$M/qat-pv2.pt"
            log "[gpu] ARM A2 verdict: ppl=$(ppl_of "$M/qat-pv2.json" ppl_after)"
            }
        fi
    else
        log "[gpu] ARM A2: gate closed — PV needs requant-cadence/LR iteration, not more steps"
    fi

    # B: ternary continuation if the 1k run was still learning at the end
    if [ -f "$M/qat-ternary-kd-1k.json" ] && [ -d "$M/qat-ternary-kd-1k-hf" ] \
       && [ ! -f "$M/qat-ternary-kd-2k.json" ]; then
        local E8 E10
        E8="$(grep -oE 'eval @step800: ppl=[0-9.]+' "$M/qat-ternary-kd-1k.log" | grep -oE '[0-9.]+$' | tail -1)"
        E10="$(ppl_of "$M/qat-ternary-kd-1k.json" ppl_after)"
        if [ -n "$E8" ] && [ -n "$E10" ] && \
           "$PY" -c "exit(0 if ($E8-$E10)/$E8 > 0.04 else 1)" 2>/dev/null; then
            disk_ok "arm B" 8 && {
            log "[gpu] ARM B: ternary slope $E8→$E10 still falling — +1000 steps"
            "${QAT[@]}" --model "$M/qat-ternary-kd-1k-hf" --quant ternary --steps 1000 \
                --train-chunks 4000 --eval-every 200 \
                --save "$M/qat-ternary-kd-2k.pt" --save-hf "$M/qat-ternary-kd-2k-hf" \
                --out "$M/qat-ternary-kd-2k.json" \
                >> "$M/qat-ternary-kd-2k.log" 2>&1 || log "[gpu] ARM B rc=$?"
            [ -f "$M/qat-ternary-kd-2k.json" ] && rm -f "$M/qat-ternary-kd-2k.pt"
            log "[gpu] ARM B verdict: ppl=$(ppl_of "$M/qat-ternary-kd-2k.json" ppl_after)"
            }
        else
            log "[gpu] ARM B: gate closed (slope ${E8:-?}→${E10:-?} ≤ 4%)"
        fi
    fi
    log "[gpu] lane done"
}

# ─────────────────────────────── CPU LANE ───────────────────────────────────
cpu_lane() {
    local GPID="$1"
    # D-guard: night-1's mp-down3, if it died
    local D_JSON="$M/reopen/q2_l12_out1_down3/ppl_q2_l12_out1_down3.json"
    if [ ! -f "$D_JSON" ]; then
        log "[cpu] D-guard: mp-down3 missing — rerunning (threads 4, eval cpu)"
        OMP_NUM_THREADS=8 nice -n 10 ./scripts/strand-7b-ppl.sh "$(cd "$M" && pwd)" \
            --mp-config configs/mp-2bit-down3.json --mp-fallback 2 --l 12 --outlier-channel 1 \
            --threads 4 --device cpu --label q2_l12_out1_down3 --limit-chunks 64 --resume \
            --out-dir "$M/reopen/q2_l12_out1_down3" >> "$M/mp-down3.log" 2>&1 \
            || log "[cpu] D-guard rc=$?"
        [ -f "$D_JSON" ] && { rm -rf "$M/reopen/q2_l12_out1_down3/recon"; \
            log "[cpu] D verdict: ppl=$(ppl_of "$D_JSON" ppl)"; }
    fi

    # L: Llama-2-7B download attempt (gated repo — skips fast without an accepted token)
    if [ ! -d scratch/llama2-7b ] && disk_ok "llama dl" 20; then
        log "[cpu] L: attempting Llama-2-7b-hf download (background, graceful skip)"
        nohup nice -n 15 ./scripts/download-model.sh meta-llama/Llama-2-7b-hf --out scratch/llama2-7b --include "*.safetensors" --include "*.json" --include "*.model" \
            > scratch/llama2-dl.log 2>&1 &
    fi

    # S7: the 7B 2-bit PTQ floor (canon 7B q2 is still l=6 = 213!) — quant only, all night
    local S7_JSON="$M7/reopen/q2_l12_out1/ppl_q2_l12_out1.json"
    if [ ! -f "$S7_JSON" ]; then
        if disk_ok "S7 quant" 25; then
            log "[cpu] S7: 7B q2_l12_out1 quant (threads 4, per-shard resume, NO eval)"
            OMP_NUM_THREADS=4 nice -n 12 ./scripts/strand-7b-ppl.sh "$(cd "$M7" && pwd)" \
                --bits 2 --l 12 --outlier-channel 1 --threads 4 --skip-calib --no-eval \
                --label q2_l12_out1 --limit-chunks 64 --resume \
                --out-dir "$M7/reopen/q2_l12_out1" >> "$M7/q2-l12-out1.log" 2>&1 \
                || log "[cpu] S7 quant rc=$? (resumable per shard)"
            log "[cpu] S7 quant pass done — waiting for GPU lane before eval"
            while kill -0 "$GPID" 2>/dev/null; do sleep 120; done
            log "[cpu] S7-eval: GPU free — evaluating recon on MPS"
            ./scripts/strand-7b-ppl.sh "$(cd "$M7" && pwd)" \
                --bits 2 --l 12 --outlier-channel 1 --threads 4 --skip-calib \
                --label q2_l12_out1 --limit-chunks 64 --resume --device mps \
                --out-dir "$M7/reopen/q2_l12_out1" >> "$M7/q2-l12-out1.log" 2>&1 \
                || log "[cpu] S7 eval rc=$?"
            [ -f "$S7_JSON" ] && log "[cpu] S7 VERDICT: 7B q2_l12_out1 ppl=$(ppl_of "$S7_JSON" ppl) (old l=6 canon: 213)"
        fi
    else log "[cpu] S7: skip (artifact exists)"; fi
    log "[cpu] lane done"
}

gpu_lane & GPID=$!
cpu_lane "$GPID" & CPID=$!
wait "$GPID" "$CPID"
log "NIGHT 2 COMPLETE — $(avail_gb)GB free"
