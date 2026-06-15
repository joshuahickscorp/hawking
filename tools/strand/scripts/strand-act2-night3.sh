#!/usr/bin/env bash
# strand-act2-night3.sh — THE MAXIMAL RUN. One continuous operation appended to night-1:
# every QAT rung the box can hold, both PTQ floors at 7B, crash-resume on every arm.
#
#   GPU lane (gated chain, ~12–20 h):
#     A    PV-2bit 300 steps, requant-75            — the headline: beat the 80.7 PTQ floor
#     A2   [A < 81]            +600, requant-100    — extend a win
#     A3   [A2 < 0.85×A]       +1000, requant-150   — chase the bf16 gap (2nd-epoch data)
#     Aalt [81 ≤ A < 250]      300, requant-37      — cadence probe (mechanism works, knob wrong)
#     P3   PV-3bit 300, requant-75 (UNCONDITIONAL)  — the shipping rung: push 3-bit toward bf16
#     B    [ternary 800→1000 slope > 4%] +1000      — BitNet track depth
#     B2   [B 800→1000 slope > 4%]       +1000      — more depth
#     M    PV-1.5bit (vec d2) 300, requant-75       — MOONSHOT: PTQ is 16,213-dead here;
#                                                     re-learning is the only thing that can work
#   CPU lane:
#     D-guard, 0.5B q3_l12_out1 floor, 0.5B vec-d2 floor (M's anchor), Llama-2 download
#     (HF_TOKEN-gated), 7B q2_l12_out1 quant all night (--no-eval, per-shard resume),
#     7B eval when GPU frees (+146-window confirm if < 100), then 7B q3_l12_out1 stretch.
#
# Crash-resume: arms pass --init-state <arm>.pt when a checkpoint exists (clean-as-you-go
# deletes .pt only after the arm's json+hf land, so a leftover .pt = a crashed arm).
# A guardian process relaunches this script if it dies before scratch/.night3-done exists.
#
# Launch gate: scratch/.polish-landed (the polish merge + --no-eval flag) + night-1 exit.
#   nohup caffeinate -dimsu ./scripts/strand-act2-night3.sh >> scratch/qwen-05b/night3.log 2>&1 & disown

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

PY=/usr/local/bin/python3
M=scratch/qwen-05b
M7=scratch/qwen-7b
SENTINEL=scratch/.polish-landed
DONE=scratch/.night3-done
# STRICT 16.5GB total budget (owner directive — memory pressure crashes this box):
#   QAT python (MPS)        ≤ 10.9 GB  (watermark 0.82 × 13.32 recommendedMax; measured need 10.2–10.5)
#   S7/floor quants (4T)    ≈ 2–3 GB   (governor-paused at the warn line)
#   in-loop requants (8T)   ≈ 2.5 GB   (run only while QAT idles at ~6 GB)
#   system allowance        ≈ 3 GB
# scratch/governor.sh enforces the total: SIGSTOP the resumable CPU quants at warn,
# requants too at critical, SIGCONT when normal. Pause, never kill — zero work loss.
# Metal encode SIGKILLs on 7B-wide tensors (in_features 18944) and loses ~8x to
# 12-way CPU anyway — force the CPU encode path for every quant in this run.
export STRAND_NO_GPU=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85
# (strand-mode converged need ≈13.1GB: 12.41+0.35 allocated + 0.30 ask at the 12.92 cap.
#  1.0 = 13.32GB. With the 7B ladder on the pod this box has ONE consumer: 13.3+3 sys
#  ≈ 16.3 total, inside the 16.5 budget. If it OOMs past 13.32 the fix is footprint
#  surgery (KD-free PV A/B), not more cap.)
# (strand-mode true peak ≈12.4GB — wq_ext +1.4GB over ternary/uniform arms; 0.92=12.25
#  was 0.15 short. 0.97=12.92GB covers it; the governor enforces the box total.)
# (0.82 starved training: PV mode holds +1.4GB of wq_ext recon buffers — every arm OOM'd.
#  0.92=12.25GB is the two-nights-proven value; the governor owns the 16.5GB total.)

log()      { echo "[n3 $(date '+%d %H:%M:%S')] $*"; }
avail_gb() { df -g / | awk 'NR==2{print $4}'; }
disk_ok()  { [ "$(avail_gb)" -ge "${2:-8}" ] || { log "$1: SKIP — $(avail_gb)GB < ${2:-8}GB"; return 1; }; }
ppl_of()   { "$PY" -c "import json;print(json.load(open('$1'))['$2'])" 2>/dev/null || echo ""; }
lt()       { "$PY" -c "exit(0 if float('$1') < float('$2') else 1)" 2>/dev/null; }

QAT=("$PY" scripts/strand-qat.py --ctx 512 --eval-chunks 64 --eval-ctx 2048
     --grad-accum 4 --batch 1 --device mps --grad-checkpoint --lr 1e-4 --kd)
PV2="--bits 2 --l 12 --outlier-channel 1 --threads 12"
PV3="--bits 3 --l 12 --outlier-channel 1 --threads 12"
PVM="--bits 3 --vec-dim 2 --l 12 --outlier-channel 1 --threads 12"

# arm <name> <model_dir> -- <extra qat args...>   (artifact-skipped, crash-resumed)
arm() {
    local name="$1" mdl="$2"; shift 2; [ "$1" = "--" ] && shift
    local json="$M/qat-$name.json" pt="$M/qat-$name.pt" hf="$M/qat-$name-hf"
    if [ -f "$json" ]; then log "[gpu] $name: skip (artifact exists)"; return 0; fi
    disk_ok "arm $name" 8 || return 1
    local RES=(); [ -f "$pt" ] && { RES=(--init-state "$pt"); log "[gpu] $name: crash-resume from $pt"; }
    log "[gpu] ARM $name starting (model=$mdl)"
    "${QAT[@]}" --model "$mdl" ${RES[@]+"${RES[@]}"} "$@" \
        --save "$pt" --save-hf "$hf" --out "$json" >> "$M/qat-$name.log" 2>&1 \
        || log "[gpu] $name rc=$?"
    [ -f "$json" ] && [ -d "$hf" ] && rm -f "$pt"
    log "[gpu] $name verdict: ppl_after=$(ppl_of "$json" ppl_after)"
}


# pv_arm <name> <model_dir> <flags> <total_steps> <seg_steps> — SEGMENTED strand arm.
# MPS pool fragmentation is unfixable in-process (4 OOM sites mapped 2026-06-10): each
# segment is a FRESH python [load -> init-requant -> eval(clean pool) -> train -> save -> exit];
# the finisher (steps=0) does the last requant+eval+export on a pristine pool. Requant
# cadence == segment size; data walks forward via --chunk-offset (science identical).
pv_arm() {
    local name="$1" mdl="$2" flags="$3" total="$4" seg="$5"
    local json="$M/qat-$name.json" pt="$M/qat-$name.pt" hf="$M/qat-$name-hf"
    local prog="$M/.qat-$name.seg"
    if [ -f "$json" ]; then log "[gpu] $name: skip (artifact exists)"; return 0; fi
    disk_ok "arm $name" 8 || return 1
    local done_steps=0; [ -f "$prog" ] && done_steps="$(cat "$prog")"
    while [ "$done_steps" -lt "$total" ]; do
        local RES=(); [ -f "$pt" ] && RES=(--init-state "$pt")
        log "[gpu] $name segment @${done_steps}/${total} (+${seg})"
        "${QAT[@]}" --model "$mdl" ${RES[@]+"${RES[@]}"} --quant strand \
            --steps "$seg" --train-chunks $((seg*4)) --chunk-offset $((done_steps*4)) \
            --requant-every 0 --eval-every 0 --skip-after --strand-flags "$flags" \
            --save "$pt" >> "$M/qat-$name.log" 2>&1 \
            || { log "[gpu] $name segment rc=$? @${done_steps} (resumable)"; return 1; }
        done_steps=$((done_steps+seg)); echo "$done_steps" > "$prog"
    done
    log "[gpu] $name finisher (requant -> eval -> export, pristine pool)"
    "${QAT[@]}" --model "$mdl" --init-state "$pt" --quant strand --steps 0 \
        --train-chunks 4 --requant-every 0 --eval-every 0 --strand-flags "$flags" \
        --save-hf "$hf" --out "$json" >> "$M/qat-$name.log" 2>&1 \
        || log "[gpu] $name finisher rc=$?"
    [ -f "$json" ] && [ -d "$hf" ] && rm -f "$pt" "$prog"
    log "[gpu] $name verdict: ppl_after=$(ppl_of "$json" ppl_after)"
}

slope_gate() {  # slope_gate <log> <json> — 800→1000 improvement > 4%?
    local e8 e10
    e8="$(grep -oE 'eval @step800: ppl=[0-9.]+' "$1" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
    e10="$(ppl_of "$2" ppl_after)"
    [ -n "$e8" ] && [ -n "$e10" ] && "$PY" -c "exit(0 if ($e8-$e10)/$e8 > 0.04 else 1)" 2>/dev/null
}

log "night-3 armed ($(avail_gb)GB free) — waiting for polish landing + night-1 exit"
until [ -f "$SENTINEL" ]; do sleep 30; done
while pgrep -f 'strand-act2-overnight.sh' >/dev/null 2>&1; do sleep 60; done
while pgrep -f 'strand-qat.py' >/dev/null 2>&1; do sleep 60; done
log "gates open — lanes starting"

# ─────────────────────────────── GPU LANE ───────────────────────────────────
gpu_lane() {
    # A: the headline
    pv_arm pv "$M" "$PV2" 300 75
    local AP; AP="$(ppl_of "$M/qat-pv.json" ppl_after)"

    if [ -n "$AP" ] && lt "$AP" 81; then
        [ -d "$M/qat-pv-hf" ] && pv_arm pv2 "$M/qat-pv-hf" "$PV2" 600 100
        local A2P; A2P="$(ppl_of "$M/qat-pv2.json" ppl_after)"
        if [ -n "$A2P" ] && "$PY" -c "exit(0 if float('$A2P') < 0.85*float('$AP') else 1)" 2>/dev/null; then
            [ -d "$M/qat-pv2-hf" ] && pv_arm pv3run "$M/qat-pv2-hf" "$PV2" 1000 200
        else log "[gpu] A3 gate closed (A2=${A2P:-none} vs 0.85×A=${AP})"; fi
    elif [ -n "$AP" ] && lt "$AP" 250; then
        log "[gpu] A-alt: cadence probe (A=${AP} — mechanism live, knob wrong)"
        pv_arm pvfast "$M" "$PV2" 300 50
    else
        log "[gpu] A extensions skipped (A=${AP:-none} ≥ 250 — proxy track carries tomorrow)"
    fi

    # P3: the shipping rung (unconditional)
    pv_arm pv3bit "$M" "$PV3" 300 75

    # B / B2: ternary depth
    if [ -f "$M/qat-ternary-kd-1k.json" ] && [ -d "$M/qat-ternary-kd-1k-hf" ] \
       && slope_gate "$M/qat-ternary-kd-1k.log" "$M/qat-ternary-kd-1k.json"; then
        arm ternary-kd-2k "$M/qat-ternary-kd-1k-hf" -- --quant ternary --steps 1000 \
            --train-chunks 4000 --eval-every 200
        if [ -d "$M/qat-ternary-kd-2k-hf" ] \
           && slope_gate "$M/qat-ternary-kd-2k.log" "$M/qat-ternary-kd-2k.json"; then
            arm ternary-kd-3k "$M/qat-ternary-kd-2k-hf" -- --quant ternary --steps 1000 \
                --train-chunks 4000 --eval-every 200
        else log "[gpu] B2 gate closed"; fi
    else log "[gpu] B gate closed (1k slope ≤ 4% or artifacts missing)"; fi

    # M: the sub-2-bit moonshot (PTQ floor here = 16,213 — only re-learning can work)
    pv_arm pv15bit "$M" "$PVM" 300 75

    log "[gpu] lane done"
}

# ─────────────────────────────── CPU LANE ───────────────────────────────────
cpu_lane() {
    local GPID="$1"

    # D-guard: night-1's mp-down3
    local D_JSON="$M/reopen/q2_l12_out1_down3/ppl_q2_l12_out1_down3.json"
    if [ ! -f "$D_JSON" ]; then
        log "[cpu] D-guard: mp-down3 rerun"
        OMP_NUM_THREADS=8 nice -n 10 ./scripts/strand-7b-ppl.sh "$(cd "$M" && pwd)" \
            --mp-config configs/mp-2bit-down3.json --mp-fallback 2 --l 12 --outlier-channel 1 \
            --threads 4 --device cpu --label q2_l12_out1_down3 --limit-chunks 64 --resume \
            --out-dir "$M/reopen/q2_l12_out1_down3" >> "$M/mp-down3.log" 2>&1 || log "[cpu] D rc=$?"
        [ -f "$D_JSON" ] && { rm -rf "$M/reopen/q2_l12_out1_down3/recon";
            log "[cpu] D verdict: $(ppl_of "$D_JSON" ppl)"; }
    fi

    # Floors the new arms compare against (each ~25 min, threads 4, eval cpu)
    local lbl json
    # q4_l12: the FREE 4-bit lever — canon 4-bit ran at default L (256 states); l=12 costs
    # zero density (register width, not payload). 0.5B anchor q4@defaultL = 13.948 (bf16 12.55).
    for spec in "q3_l12_out1|--bits 3 --outlier-channel 1" "vecd2_l12_out1|--bits 3 --vec-dim 2 --outlier-channel 1" "q4_l12|--bits 4"; do
        lbl="${spec%%|*}"
        json="$M/reopen/$lbl/ppl_$lbl.json"
        if [ ! -f "$json" ]; then
            log "[cpu] floor $lbl: quant+eval"
            # shellcheck disable=SC2086
            OMP_NUM_THREADS=8 nice -n 10 ./scripts/strand-7b-ppl.sh "$(cd "$M" && pwd)" \
                ${spec##*|} --l 12 --threads 4 --skip-calib \
                --device cpu --label "$lbl" --limit-chunks 64 --resume \
                --out-dir "$M/reopen/$lbl" >> "$M/floors.log" 2>&1 || log "[cpu] $lbl rc=$?"
            [ -f "$json" ] && { rm -rf "$M/reopen/$lbl/recon";
                log "[cpu] FLOOR $lbl: ppl=$(ppl_of "$json" ppl)"; }
            [ "$lbl" = "q4_l12" ] && [ -f "$json" ] && \
                log "[cpu] 4-BIT LEVER VERDICT: q4_l12=$(ppl_of "$json" ppl) vs q4@defaultL=13.948 (free if lower; 7B confirm queues tomorrow if >0.5% better)"
        fi
    done

    # L: Llama-2 (gated repo — needs HF_TOKEN; skips fast otherwise)
    if [ ! -d scratch/llama2-7b ] && disk_ok "llama" 20; then
        log "[cpu] L: Llama-2-7b-hf download attempt (HF_TOKEN $([ -n "${HF_TOKEN:-}" ] && echo present || echo ABSENT))"
        nohup nice -n 15 ./scripts/download-model.sh meta-llama/Llama-2-7b-hf \
            --out scratch/llama2-7b --include "*.safetensors" --include "*.json" --include "*.model" \
            > scratch/llama2-dl.log 2>&1 &
    fi

    # 7B ladder (q2/q3/mp_light/q4 + Llama-2/Mistral anchors) MOVED TO THE RUNPOD
    # (27 vCPU ≈ 6x local; scratch/pod-paste.sh is the block the owner pastes).
    log "[cpu] 7B ladder: delegated to the pod — local lane done"

    log "[cpu] lane done"
}

gpu_lane & GPID=$!
cpu_lane "$GPID" & CPID=$!
wait "$GPID" "$CPID"
date > "$DONE"
log "NIGHT 3 COMPLETE — $(avail_gb)GB free"
