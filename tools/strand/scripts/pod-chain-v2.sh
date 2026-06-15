#!/usr/bin/env bash
# pod-chain.sh v3.6 — PER-MODEL LEGS with strict storage sequencing (2026-06-10).
# The bulk-download design would blow the 200GB volume (~260GB peak); each leg now:
#   download -> baseline eval (offload, on-pod) -> q2 quant -> recon eval -> bank
#   jsons -> DELETE model + recon -> next model.
# Peak ~= one model footprint. 70B leg: quant with ROLLING source-shard delete
# (peak ~155GB), recon KEPT, eval deferred (70B bf16 ~140GB > the 125GB cgroup —
# needs the 1h eval-pod cameo; owner decides Sunday). Sentinels: SKIP-70B, SKIP-MP.
# Gates on the LADDER draining first. Optional HF token: /workspace/HF_TOKEN.
cd /workspace/strand
. "$HOME/.cargo/env" 2>/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
LOG=/workspace/strand-chain.log
R=/workspace/strand-results
log(){ echo "[chain $(date "+%d %H:%M:%S")] $*" >> "$LOG"; }
export STRAND_NO_GPU=1 HF_HUB_DISABLE_XET=1
[ -f /workspace/HF_TOKEN ] && export HF_TOKEN="$(cat /workspace/HF_TOKEN)" && log "HF_TOKEN loaded"
PPL=/workspace/s7p-chain.sh
EV=/workspace/eval-ppl.py

has_weights(){ # complete iff all index-listed shards present (single-file models pass)
  local d="$1"
  [ -f "$d/model.safetensors" ] && return 0
  local want have
  want=$(python3 -c "import json;print(len(set(json.load(open(\"$d/model.safetensors.index.json\"))[\"weight_map\"].values())))" 2>/dev/null) || return 1
  have=$(ls "$d"/*.safetensors 2>/dev/null | wc -l | tr -d " ")
  [ -n "$want" ] && [ "$have" = "$want" ]
}
dl_until_complete(){ local repo="$1" out="$2" tries=0
  while ! has_weights "$out"; do
    tries=$((tries+1)); [ "$tries" -gt 14 ] && { log "dl $out: GAVE UP after 14 tries"; return 1; }
    if pgrep -f "download-model.sh.*$out" >/dev/null 2>&1; then log "dl $out: active, waiting"; sleep 600; continue; fi
    log "dl $out: attempt $tries ($repo)"
    nice -n 15 ./scripts/download-model.sh "$repo" --out "$out" \
      --include "*.safetensors" --include "*.json" --include "*.model" --include "*.txt" >> "$LOG" 2>&1
    has_weights "$out" || sleep 1200
  done
  log "dl $out: COMPLETE"; return 0
}
ev(){ # ev <load_dir> <tag> <out_json> — canon offload eval (validated: reproduces 6.6289 exactly)
  local dir="$1" tag="$2" oj="$3"
  [ -f "$oj" ] && { log "eval $tag: banked"; return 0; }
  log "eval $tag starting (offload)"
  OMP_NUM_THREADS=24 EVAL_GPU_GB=18 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 python3 "$EV" "$dir" 2048 64 offload bfloat16 "$tag" "$oj" >> "$LOG" 2>&1 \
    && { cp "$oj" "$R/" 2>/dev/null; log "eval $tag: $(tr -d " \n" < "$oj" | head -c 120)"; } \
    || log "eval $tag rc=$?"
}
# Cloud-waste guard: a NEW scale leg may run only if its 0.5B gate earned
# PROMOTE_CLOUD. The pod does not recompute — it trusts the promotion.state that
# promote.py already stamped locally and that was mirrored to /workspace/gates/.
# Baseline-campaign legs pass no gate (empty arg) and run as before.
# Usage:  require_promoted <gate-json-basename> || return 1
require_promoted(){
  local gate="$1"
  [ -z "$gate" ] && return 0
  local g="/workspace/gates/$gate"
  [ -f "$g" ] || { log "GATE $gate MISSING — refusing scale (no stamped 0.5B promotion on pod)"; return 1; }
  local state; state=$(grep -oE '"state"[[:space:]]*:[[:space:]]*"[A-Z_]+"' "$g" | head -1 | grep -oE '[A-Z_]+"$' | tr -d '"')
  [ "$state" = "PROMOTE_CLOUD" ] || { log "GATE $gate state=${state:-none} != PROMOTE_CLOUD — refusing scale"; return 1; }
  log "GATE $gate PROMOTE_CLOUD — scale authorized"; return 0
}
q(){ local model="$1" label="$2"; shift 2
  local out="scratch/$model/reopen/$label"
  has_weights "scratch/$model" || { log "$model/$label: SKIP (no weights)"; return 1; }
  [ -f "$out/.quant-done" ] && { log "$model/$label: already done"; return 0; }
  log "QUANT $model/$label starting"
  SHARD_JOBS="${SHARD_JOBS:-2}" bash "$PPL" "$(cd "scratch/$model" && pwd)" "$@" --threads 96 --no-eval --resume --skip-calib \
      --label "$label" --out-dir "$out" >> "$LOG" 2>&1 \
    && { date > "$out/.quant-done"; log "QUANT $model/$label DONE"; } \
    || log "QUANT $model/$label rc=$?"
}
rolling_src_delete(){ # 70B only: delete source shards whose recon .tmp sidecar is banked
  local model="$1" out="$2"
  while pgrep -f "s7p-chain.sh.*$model" >/dev/null 2>&1; do
    for tj in "$out"/.tmp-*.safetensors.json; do
      [ -s "$tj" ] || continue
      src="scratch/$model/$(basename "${tj%.json}" | sed "s/^\.tmp-//")"
      [ -f "$src" ] && { rm -f "$src"; log "rolling-delete source $(basename "$src")"; }
    done
    sleep 180
  done
}

log "=== CHAIN v3.4 ARMED (per-model legs, storage-sequenced) ==="
log "waiting for ladder to drain"
while pgrep -f "strand-ladder[.]sh" >/dev/null 2>&1; do sleep 300; done
log "ladder drained — post-ladder 7B model weight cleanup"
for m in qwen-7b llama2-7b; do
  if ls scratch/$m/reopen/*/ppl_*.json >/dev/null 2>&1; then
    rm -f scratch/$m/model-*.safetensors scratch/$m/model.safetensors 2>/dev/null
    log "cleanup: $m weights deleted (jsons banked; re-downloadable)"
  fi
done

# ── LEG 1: Qwen2.5-14B ──
if [ ! -f "$R/ppl_q2_l12_out1_14b.json" ]; then
  dl_until_complete Qwen/Qwen2.5-14B scratch/qwen-14b && {
    ev scratch/qwen-14b baseline_14b "$R/ppl_baseline_14b.json"
    q qwen-14b q2_l12_out1 --bits 2 --l 12 --outlier-channel 1
    ev scratch/qwen-14b/reopen/q2_l12_out1/recon q2_l12_out1_14b "$R/ppl_q2_l12_out1_14b.json"
    if [ ! -f /workspace/SKIP-MP ]; then
      q qwen-14b mp_light_l12_out1 --bits 3 --mp-config configs/mp-light.json --mp-fallback 3 --l 12 --outlier-channel 1
      ev scratch/qwen-14b/reopen/mp_light_l12_out1/recon mp_light_14b "$R/ppl_mp_light_14b.json"
    fi
    if [ -f "$R/ppl_q2_l12_out1_14b.json" ]; then rm -rf scratch/qwen-14b; log "LEG 14B complete — model deleted"
    else log "LEG 14B FAILED (q2 json missing) — model RETAINED for retry"; fi
  }
fi

# ── LEG 2: Qwen2.5-32B ──
if [ ! -f "$R/ppl_q2_l12_out1_32b.json" ]; then
  dl_until_complete Qwen/Qwen2.5-32B scratch/qwen-32b && {
    ev scratch/qwen-32b baseline_32b "$R/ppl_baseline_32b.json"
    q qwen-32b q2_l12_out1 --bits 2 --l 12 --outlier-channel 1
    ev scratch/qwen-32b/reopen/q2_l12_out1/recon q2_l12_out1_32b "$R/ppl_q2_l12_out1_32b.json"
    if [ ! -f /workspace/SKIP-MP ]; then
      q qwen-32b mp_light_l12_out1 --bits 3 --mp-config configs/mp-light.json --mp-fallback 3 --l 12 --outlier-channel 1
      ev scratch/qwen-32b/reopen/mp_light_l12_out1/recon mp_light_32b "$R/ppl_mp_light_32b.json"
    fi
    if [ -f "$R/ppl_q2_l12_out1_32b.json" ]; then rm -rf scratch/qwen-32b; log "LEG 32B complete — model deleted"
    else log "LEG 32B FAILED (q2 json missing) — model RETAINED for retry"; fi
  }
fi

# ── LEG 3: Llama-2-70B (the boss) — quant + recon only; eval needs the cameo pod ──
if [ -f /workspace/SKIP-70B ]; then log "70B leg SKIPPED (sentinel)"
else
  dl_until_complete NousResearch/Llama-2-70b-hf scratch/llama2-70b && {
    log "70B BOSS LEG: quant with rolling source delete (recon KEPT for the eval cameo)"
    out70=scratch/llama2-70b/reopen/q2_l12_out1
    rolling_src_delete llama2-70b "$out70" &
    q llama2-70b q2_l12_out1 --bits 2 --l 12 --outlier-channel 1
    log "70B quant done; recon retained at $out70/recon — eval via cameo pod (owner gate)"
  }
fi
log "=== CHAIN v3.4 COMPLETE ==="
