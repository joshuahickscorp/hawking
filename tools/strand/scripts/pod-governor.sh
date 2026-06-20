#!/usr/bin/env bash
# pod-governor.sh — STRICT pod-side storage + memory moderation (2026-06-10).
# Runs ON the pod: enforcement survives the laptop's network route. Pause, never
# kill (the local governor's law). Log: /workspace/pod-governor.log
#
# DISK (200GB network volume; du is the quota truth — df shows the whole cluster):
#   WARN  >=170GB: reclaim pre-authorized re-derivables, in order:
#     1) recon dirs whose ppl json is already banked (eval done => recon re-derivable)
#     2) orphaned partial .tmp shards (no quant running, >60min old, empty json sidecar)
#   CRIT  >=185GB: SIGSTOP producers (quantize-model + download-model) until <170.
#   NEVER deletes: complete model weights, ppl jsons, results, logs, or chain recons
#   still awaiting their eval (protected by the banked-json rule).
# MEMORY (cgroup cap 125GB; the 21:02 OOM killed mp_light's quant at 121GB):
#   >=120GB: SIGSTOP quantize-model. <=110GB: SIGCONT. Cache reclaim gets room.
LOG=/workspace/pod-governor.log
log(){ echo "[pgov $(date '+%d %H:%M:%S')] $*" >> "$LOG"; }
vol_gb(){ du -s /workspace 2>/dev/null | awk '{printf "%d", $1/1048576}'; }
mem_gb(){ awk '/^anon /{printf "%d", $2/1e9}' /sys/fs/cgroup/memory.stat 2>/dev/null; }
paused=""
log "pod-governor up (disk warn170/crit185 of 200GB; mem pause120/resume110 of 125GB)"
while :; do
    V=$(vol_gb); M=$(mem_gb)
    # ── memory law ──
    if [ -n "$M" ] && [ "$M" -ge 120 ] && [ -z "$paused" ]; then
        pkill -STOP quantize-model 2>/dev/null && { paused=mem; log "MEM ${M}GB>=120: quant PAUSED"; }
    elif [ "$paused" = "mem" ] && [ -n "$M" ] && [ "$M" -le 110 ]; then
        pkill -CONT quantize-model 2>/dev/null; paused=""; log "MEM ${M}GB<=110: quant RESUMED"
    fi
    # ── disk law ──
    if [ -n "$V" ] && [ "$V" -ge 170 ]; then
        log "DISK ${V}GB>=170: reclaiming"
        for d in /workspace/strand/scratch/*/reopen/*/; do
            if ls "$d"ppl_*.json >/dev/null 2>&1 && [ -d "${d}recon" ]; then
                sz=$(du -s "${d}recon" 2>/dev/null | awk '{printf "%d", $1/1048576}')
                rm -rf "${d}recon" && log "reclaimed recon ${d} (${sz}GB)"
            fi
            [ "$(vol_gb)" -lt 160 ] && break
        done
        if ! pgrep quantize-model >/dev/null 2>&1; then
            find /workspace/strand/scratch/*/reopen/ -maxdepth 2 -name '.tmp-*.safetensors' -mmin +60 2>/dev/null \
            | while read -r t; do
                [ -s "$t.json" ] || { rm -f "$t" "$t.json"; log "reclaimed orphan partial $t"; }
            done
        fi
        V=$(vol_gb)
        if [ "$V" -ge 185 ] && [ -z "$paused" ]; then
            pkill -STOP quantize-model 2>/dev/null
            pkill -STOP -f download-model 2>/dev/null
            paused=disk; log "DISK ${V}GB>=185 CRIT: producers PAUSED"
        fi
    fi
    if [ "$paused" = "disk" ] && [ -n "$V" ] && [ "$V" -lt 170 ]; then
        pkill -CONT quantize-model 2>/dev/null
        pkill -CONT -f download-model 2>/dev/null
        paused=""; log "DISK ${V}GB<170: producers RESUMED"
    fi
    sleep 120
done
