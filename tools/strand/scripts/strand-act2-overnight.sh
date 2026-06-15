#!/usr/bin/env bash
# strand-act2-overnight.sh — Act-2 (QAT) overnight ladder on the 0.5B, TWO PARALLEL LANES.
#
#   GPU lane (MPS):  B uniform+KD 300  →  F ternary+KD 1000  →  [gate C<70] E uniform+KD 1000
#   CPU lane:        C STRAND-quant+eval of B's weights  →  D mp-down3  →  E-quant after E
#
# The lanes use different compute (MPS training vs CPU Viterbi) and sync via artifact files.
# Memory rails (two machine freezes taught us — will.md §7): QAT capped via MPS watermark env,
# freezes non-proj params, --grad-checkpoint; CPU-lane quants run --threads 4 while the GPU lane
# is busy (12 when idle); CPU-lane PPL evals run on --device cpu while the GPU lane is busy
# (~5th-digit PPL delta vs MPS — irrelevant at our magnitudes; the json records the device).
# Disk rails (tonight's 1.4GB-free crash): ≥4GB free required before each save-heavy phase;
# .pt shadows deleted once json+hf-dir exist; recon dirs deleted once their ppl json lands.
#
# QAT speed: batch 2 × accum 2 (gradient-identical to batch 1 × accum 4, fewer/larger MPS
# kernels); automatic batch-1 retry if a run produces no result json.
#
# Launch:
#   nohup caffeinate -dimsu ./scripts/strand-act2-overnight.sh \
#       > scratch/qwen-05b/act2-overnight.log 2>&1 & disown
# Every phase skips itself if its terminal artifact exists — relaunch anytime.

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

PY=/usr/local/bin/python3
M=scratch/qwen-05b
STATE="$M/.gpu-lane-state"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.92
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7

log()      { echo "[orch $(date '+%H:%M:%S')] $*"; }
avail_gb() { df -g / | awk 'NR==2{print $4}'; }
disk_ok()  { [ "$(avail_gb)" -ge 4 ] || { log "$1: SKIP — only $(avail_gb)GB free (<4GB)"; return 1; }; }
gpu_busy() { [ "$(cat "$STATE" 2>/dev/null || echo busy)" = "busy" ] || pgrep -f 'strand-qat.py' >/dev/null 2>&1; }

QAT_COMMON=(--model "$M" --ctx 512 --eval-chunks 64 --eval-ctx 2048
            --device mps --grad-checkpoint --lr 1e-4)

# run_qat <json> <log> <args...>  — batch2 first, batch1 retry if no json appears
run_qat() {
    local json="$1" lg="$2"; shift 2
    disk_ok "QAT $json" || return 1
    "$PY" scripts/strand-qat.py "${QAT_COMMON[@]}" --batch 2 --grad-accum 2 "$@" \
        >> "$lg" 2>&1 || log "QAT (batch2) rc=$? for $json"
    if [ ! -f "$json" ]; then
        log "QAT batch2 produced no $json — retrying with batch 1 / accum 4"
        disk_ok "QAT retry $json" || return 1
        "$PY" scripts/strand-qat.py "${QAT_COMMON[@]}" --batch 1 --grad-accum 4 "$@" \
            >> "$lg" 2>&1 || log "QAT (batch1 retry) rc=$? for $json"
    fi
    [ -f "$json" ]
}

# quant_eval <model_dir> <label> <outdir> <log> [extra ppl.sh args...]
quant_eval() {
    local mdir="$1" label="$2" outdir="$3" lg="$4"; shift 4
    local t=12 dev=mps
    if gpu_busy; then t=4; dev=cpu; fi
    log "quant+eval $label: threads=$t eval-device=$dev"
    OMP_NUM_THREADS=8 nice -n 10 ./scripts/strand-7b-ppl.sh "$(cd "$mdir" && pwd)" \
        --threads "$t" --device "$dev" --label "$label" --limit-chunks 64 --resume \
        --out-dir "$outdir" "$@" >> "$lg" 2>&1 || log "quant+eval $label rc=$?"
    if [ -f "$outdir/ppl_${label}.json" ]; then
        rm -rf "$outdir/recon"           # re-derivable; the json is the science
        log "$label done: ppl=$("$PY" -c "import json;print(round(json.load(open('$outdir/ppl_${label}.json'))['ppl'],2))" 2>/dev/null)"
    fi
}

# ─────────────────────────────── GPU LANE ───────────────────────────────────
gpu_lane() {
    echo busy > "$STATE"
    while pgrep -f 'strand-qat.py' >/dev/null 2>&1; do sleep 60; done   # in-flight run

    # B: uniform-2bit + KD, 300 steps -> tuned HF dir (the transfer-probe input)
    if [ ! -f "$M/qat-u2-kd.json" ]; then
        log "[gpu] phase B: uniform-2bit+KD 300 steps"
        run_qat "$M/qat-u2-kd.json" "$M/qat-u2-kd.log" \
            --quant uniform --bits 2 --kd --steps 300 --train-chunks 1200 --eval-every 100 \
            --save "$M/qat-u2-kd.pt" --save-hf "$M/qat-u2-kd-hf" --out "$M/qat-u2-kd.json" \
            || log "[gpu] phase B FAILED"
        [ -f "$M/qat-u2-kd.json" ] && [ -d "$M/qat-u2-kd-hf" ] && rm -f "$M/qat-u2-kd.pt"
    else log "[gpu] phase B: skip (artifact exists)"; fi

    # F: ternary + KD, 1000 steps (BitNet track; 300-step run hit 260.0 PPL and was still falling)
    if [ ! -f "$M/qat-ternary-kd-1k.json" ]; then
        log "[gpu] phase F: ternary+KD 1000 steps"
        run_qat "$M/qat-ternary-kd-1k.json" "$M/qat-ternary-kd-1k.log" \
            --quant ternary --kd --steps 1000 --train-chunks 4000 --eval-every 200 \
            --save "$M/qat-ternary-kd-1k.pt" --save-hf "$M/qat-ternary-kd-1k-hf" \
            --out "$M/qat-ternary-kd-1k.json" \
            || log "[gpu] phase F FAILED"
        [ -f "$M/qat-ternary-kd-1k.json" ] && [ -d "$M/qat-ternary-kd-1k-hf" ] && rm -f "$M/qat-ternary-kd-1k.pt"
    else log "[gpu] phase F: skip (artifact exists)"; fi

    # E gate: C's verdict (CPU lane computes it during F). Wait ≤30 min if needed.
    local C_JSON="$M/reopen/qatU2KD300_strand/ppl_qatU2KD300_q2l12out1.json"
    for _ in $(seq 30); do [ -f "$C_JSON" ] && break; sleep 60; done
    local CP
    CP="$("$PY" -c "import json;print(json.load(open('$C_JSON'))['ppl'])" 2>/dev/null || echo "")"
    log "[gpu] E gate: STRAND-after-QAT ppl=${CP:-none} (floor 80.7, open if <70)"
    if [ -n "$CP" ] && "$PY" -c "exit(0 if float('$CP') < 70 else 1)" 2>/dev/null; then
        if [ ! -f "$M/qat-u2-kd-1k.json" ]; then
            log "[gpu] phase E: GATE OPEN — uniform-2bit+KD 1000 steps"
            run_qat "$M/qat-u2-kd-1k.json" "$M/qat-u2-kd-1k.log" \
                --quant uniform --bits 2 --kd --steps 1000 --train-chunks 4000 --eval-every 200 \
                --save "$M/qat-u2-kd-1k.pt" --save-hf "$M/qat-u2-kd-1k-hf" --out "$M/qat-u2-kd-1k.json" \
                || log "[gpu] phase E FAILED"
            [ -f "$M/qat-u2-kd-1k.json" ] && [ -d "$M/qat-u2-kd-1k-hf" ] && rm -f "$M/qat-u2-kd-1k.pt"
        fi
    else
        log "[gpu] phase E: skipped (gate closed — rung-3 in-loop machinery is the path)"
    fi
    echo idle > "$STATE"
    log "[gpu] lane done"
}

# ─────────────────────────────── CPU LANE ───────────────────────────────────
cpu_lane() {
    local GPID="$1"
    # C: STRAND quant + canon eval of B's tuned weights (+ tuned-bf16 baseline)
    local C_JSON="$M/reopen/qatU2KD300_strand/ppl_qatU2KD300_q2l12out1.json"
    if [ ! -f "$C_JSON" ]; then
        until [ -f "$M/qat-u2-kd.json" ]; do
            kill -0 "$GPID" 2>/dev/null || { log "[cpu] gpu lane died pre-B; skipping C"; break; }
            sleep 30
        done
        if [ -d "$M/qat-u2-kd-hf" ]; then
            log "[cpu] phase C: STRAND l12+out1 of tuned-300 weights"
            quant_eval "$M/qat-u2-kd-hf" qatU2KD300_q2l12out1 "$M/reopen/qatU2KD300_strand" \
                "$M/qatU2KD300-strand.log" \
                --bits 2 --l 12 --outlier-channel 1 --skip-calib --fp16-baseline
            log "[cpu] PHASE C VERDICT: $(cat "$C_JSON" 2>/dev/null || echo MISSING)"
        fi
    else log "[cpu] phase C: skip (artifact exists)"; fi

    # D: mp-down3 PTQ ceiling point (threads 4 always — wide down_proj tensors at l12)
    local D_JSON="$M/reopen/q2_l12_out1_down3/ppl_q2_l12_out1_down3.json"
    if [ ! -f "$D_JSON" ]; then
        log "[cpu] phase D: mp-down3 (down_proj@3 rest@2, l12+out1)"
        local dev=mps; gpu_busy && dev=cpu
        OMP_NUM_THREADS=8 nice -n 10 ./scripts/strand-7b-ppl.sh "$(cd "$M" && pwd)" \
            --mp-config configs/mp-2bit-down3.json --mp-fallback 2 --l 12 --outlier-channel 1 \
            --threads 4 --device "$dev" --label q2_l12_out1_down3 --limit-chunks 64 --resume \
            --out-dir "$M/reopen/q2_l12_out1_down3" >> "$M/mp-down3.log" 2>&1 \
            || log "[cpu] phase D rc=$?"
        [ -f "$D_JSON" ] && rm -rf "$M/reopen/q2_l12_out1_down3/recon"
        log "[cpu] phase D verdict: $(cat "$D_JSON" 2>/dev/null || echo MISSING) (vs q2_l12_out1=80.7)"
    else log "[cpu] phase D: skip (artifact exists)"; fi

    # E-quant: if/when E's tuned dir appears (gate may stay closed — bounded wait on gpu lane)
    local E_JSON="$M/reopen/qatU2KD1k_strand/ppl_qatU2KD1k_q2l12out1.json"
    while kill -0 "$GPID" 2>/dev/null && [ ! -f "$M/qat-u2-kd-1k.json" ]; do sleep 60; done
    if [ -d "$M/qat-u2-kd-1k-hf" ] && [ ! -f "$E_JSON" ]; then
        log "[cpu] phase E-quant: STRAND l12+out1 of tuned-1k weights"
        quant_eval "$M/qat-u2-kd-1k-hf" qatU2KD1k_q2l12out1 "$M/reopen/qatU2KD1k_strand" \
            "$M/qatU2KD1k-strand.log" \
            --bits 2 --l 12 --outlier-channel 1 --skip-calib --fp16-baseline
    fi
    log "[cpu] lane done"
}

# ─────────────────────────────── run both ───────────────────────────────────
log "Act-2 overnight v2: two lanes, $(avail_gb)GB free disk"
gpu_lane & GPID=$!
cpu_lane "$GPID" & CPID=$!
wait "$GPID" "$CPID"
log "ALL PHASES COMPLETE — $(avail_gb)GB free disk"
