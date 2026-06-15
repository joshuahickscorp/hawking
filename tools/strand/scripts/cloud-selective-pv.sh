#!/usr/bin/env bash
# cloud-selective-pv.sh — the selective-PV-at-scale recipe (7B + 32B) that drives the
# 2-bit loss-tax from 0.324 toward <=0.15. This is the main cloud dollar-burn; it runs
# on the pod ONLY after the 0.5B selective-PV-KL-routed gate earns PROMOTE_CLOUD.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IT DOES (per model leg: 7B, then 32B):
#   1. require_promoted — refuse unless a stamped PROMOTE_CLOUD 0.5B gate sentinel is
#      present (mirrored to /workspace/gates/). NO unproven cloud burn.
#   2. calib-actmean.py — per-projection activation feature means (de-bias calibration),
#      written once per model (skipped if banked).
#   3. strand-qat.py --quant strand — selective PV (training-through the REAL Rust
#      encoder's recon) on the RED tensor set, q2 base, KD on, de-bias on, ctx 512.
#      Output: a drop-in tuned HF dir (--save-hf).
#   4. deploy_quant_recon + canon eval on the tuned HF dir — quantize-model RED@4 +
#      de-bias recon, then eval-ppl.py (canon harness) gives the CANON WikiText-2 PPL
#      of the QAT'd-then-STRAND-quantized weights (same harness as the bf16 anchor),
#      so the shipped number matches the deployment config PV trained through.
#   5. promote.py on the resulting ppl_*.json — billed loss-tax stamped into the json
#      (so the result self-describes; the mirror-down sweep picks it up).
#
# WHY CLASS-LEVEL RED (not the 0.5B per-layer regex):
#   rung-kl (research/rung-kl-dp_d4_r2.json) ranked up_proj / v_proj / gate_proj as the
#   high-output-KL (RED) tensor CLASSES; down_proj is NOT red. The decisive, TRANSFERABLE
#   signal is the class — the 0.5B's exact red list is layer-indexed (layers 0-23) and
#   does NOT map onto 7B (28 layers) or 32B (64 layers). So selective PV here protects by
#   CLASS regex, and the deployment quant simultaneously LIFTS those same classes to 4-bit
#   via configs/mp-kl-routed.json (v/up/gate=4, q/k/o/down=2) — the side-info-cheap RED
#   protection. PV re-learns the 2-bit classes through the recon; the RED classes ride at
#   4-bit with de-bias on top.
#
# THE STACK (all measured-good levers, 2026-06-11..13):
#   --quant strand              forward = the REAL encoder recon (proxy-transfer is DEAD)
#   q2 base, RED@4 via rung-cfg  the KL-routed mixed-precision deployment config
#                               (configs/mp-kl-routed.json is a flat class->bits map =
#                                quantize-model's --rung-config shape, NOT --mp-config)
#   --outlier-channel 1          pre-RHT top-|w| side-channel (+1%, the live PTQ lever)
#   --actmean <calib.json>       output de-bias (ADOPTED: -28.7% PPL on dp_d4_r2)
#   --kd                         KL-distill from the frozen FP teacher
#   --ctx 512                    the proven-fast PV setting (matches the 0.5B gate)
#   WSD warmup 0.05 / cooldown 0.2   Apple low-bit QAT schedule
#   (C2 stack — scale/outlier-position/init coding — composes on top once its own gate
#    clears; add its flags to STRAND_FLAGS_BASE below. C2 is NOT armed here.)
#
# WHY THIS SCRIPT OWNS THE DEPLOYMENT-QUANT SHARD LOOP (not strand-7b-ppl.sh):
#   strand-7b-ppl.sh forwards neither --rung-config NOR --actmean to quantize-model
#   (it only knows --mp-config / --outlier-channel). Both the RED@4 protection and the
#   de-bias correction MUST reach the deployment recon, so this script drives
#   quantize-model per-shard directly (deploy_quant_recon, mirroring s7p's proven
#   recast), then evals the recon dir through the SAME canon shim (eval-ppl.py ->
#   tools/strand_eval) so the PPL is harness-identical to the bf16 anchor.
#
# THIS SCRIPT DOES NOT RUN HERE (no cloud GPU on the dev box). It is the artifact to
# scp to the pod and launch detached. See the ARMING block at the bottom of this file.
#
# Machine-stamp target: RunPod CUDA pod, /workspace/strand checkout. NOT run on the
# Apple dev box (selective-PV training needs CUDA; the box's MPS GPU is owned by a
# separate local run). Numbers are produced ON THE POD only.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

cd /workspace/strand 2>/dev/null || { echo "FATAL: /workspace/strand not found (run ON THE POD)"; exit 2; }
. "$HOME/.cargo/env" 2>/dev/null || export PATH="$HOME/.cargo/bin:$PATH"

# ── paths / knobs (override via env) ──
R="${R:-/workspace/strand-results}"
GATES="${GATES:-/workspace/gates}"
LOG="${LOG:-/workspace/cloud-selective-pv.log}"
PY="${PY:-python3}"
QAT="${QAT:-scripts/strand-qat.py}"
EVAL="${EVAL:-scripts/eval-ppl.py}"               # canon eval shim (-> tools/strand_eval; harness_key)
QUANT_BIN="${QUANT_BIN:-target/release/quantize-model}"
CALIB="${CALIB:-scripts/calib-actmean.py}"
PROMOTE="${PROMOTE:-scripts/promote.py}"
RUNG_CFG="${RUNG_CFG:-configs/mp-kl-routed.json}"   # v/up/gate=4, q/k/o/down=2 (the RED protection)
# Eval budget for the canon offload PPL (matches pod-chain-v2.sh ev()).
EVAL_GPU_GB="${EVAL_GPU_GB:-18}"
OMP_NUM_THREADS_EVAL="${OMP_NUM_THREADS_EVAL:-24}"

# The 0.5B gate sentinel each leg requires (basename under $GATES). This is the json
# scripts/gates/10-pv-kl-routed.sh produces and promote.py stamps PROMOTE_CLOUD.
GATE_SENTINEL="${GATE_SENTINEL:-pv-kl-routed.json}"

# PV training hyperparams (match the 0.5B KL-routed gate; ctx 512 is the proven-fast lane).
STEPS="${STEPS:-300}"
LR="${LR:-1e-4}"
CTX="${CTX:-512}"
EVAL_CHUNKS="${EVAL_CHUNKS:-64}"
EVAL_CTX="${EVAL_CTX:-2048}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
COOLDOWN_FRAC="${COOLDOWN_FRAC:-0.2}"
SELPV_REQUANT="${SELPV_REQUANT:-75}"   # requant cadence inside PV

# CLASS-level RED protection set (transferable; see header). PV trains these classes
# (they re-learn through the 2-bit recon); the deployment config lifts them to 4-bit.
PV_TENSORS="${PV_TENSORS:-v_proj|up_proj|gate_proj}"

# Threads for the in-loop + deployment Rust encoder (pod has the cores; 96 on the chain box).
QUANT_THREADS="${QUANT_THREADS:-96}"

# Deployment quant config — used by BOTH the in-loop PV requant (via --strand-flags) AND
# the final deploy_quant_recon. RED@4 via rung-config + outlier channel. De-bias
# (--actmean) is appended per-model after calibration. To stack C2, append its flags here.
STRAND_FLAGS_BASE="${STRAND_FLAGS_BASE:---bits 2 --l 12 --outlier-channel 1 --rung-config $RUNG_CFG}"

# CUDA hygiene for the long PV run.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET=1
[ -f /workspace/HF_TOKEN ] && export HF_TOKEN="$(cat /workspace/HF_TOKEN)"

mkdir -p "$R" "$GATES"
log(){ echo "[cloud-pv $(date '+%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ── completeness check: all index-listed shards present (single-file models pass) ──
has_weights(){
  local d="$1"
  [ -f "$d/model.safetensors" ] && return 0
  local want have
  want=$($PY -c "import json;print(len(set(json.load(open('$d/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null) || return 1
  have=$(ls "$d"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  [ -n "$want" ] && [ "$have" = "$want" ]
}

# ── recast one quantizer f32 safetensors -> bf16 (verbatim from strand-7b-ppl.sh) ──
recast_one(){
  local src="$1" dst="$2"
  "$PY" - "$src" "$dst" <<'RECAST_PY'
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]
tensors = {}
with safe_open(src, framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if t.dtype == torch.float32:
            t = t.to(torch.bfloat16)
        tensors[k] = t.contiguous()
save_file(tensors, dst, metadata={"format": "pt"})
print(f"[recast] {dst}: {len(tensors)} tensors -> bf16", flush=True)
RECAST_PY
}

copy_aux(){  # config + tokenizer + index so the recon dir is a loadable HF model
  local src="$1" dst="$2"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
           vocab.json merges.txt special_tokens_map.json added_tokens.json \
           model.safetensors.index.json; do
    [ -f "$src/$f" ] && cp -f "$src/$f" "$dst/$f"
  done
}

# ── deployment recon: quantize the tuned HF dir per-shard with RED@4 (--rung-config) +
# de-bias (--actmean) + outlier channel, recast to bf16, assemble a loadable recon dir.
# strand-7b-ppl.sh forwards neither --rung-config nor --actmean, so we drive
# quantize-model directly (same per-shard contract it uses). The recon's
# effective_bpw is harvested from the first shard sidecar for density billing. ──
deploy_quant_recon(){
  local hfdir="$1" recon="$2" actmean="$3"
  mkdir -p "$recon"
  local qflags=( --bits 2 --l 12 --rung-config "$RUNG_CFG"
                 --outlier-channel 1 --actmean "$actmean" --threads "$QUANT_THREADS" )
  local shards=()
  if [ -f "$hfdir/model.safetensors" ] && ! ls "$hfdir"/model-*-of-*.safetensors >/dev/null 2>&1; then
    shards=( "$hfdir/model.safetensors" )
  else
    shards=( "$hfdir"/model-*-of-*.safetensors )
  fi
  [ -e "${shards[0]}" ] || { log "deploy-quant: no shards in $hfdir"; return 1; }
  local n=${#shards[@]} i=0
  for shard in "${shards[@]}"; do
    i=$((i+1)); local base dst tmp
    base="$(basename "$shard")"; dst="$recon/$base"; tmp="$recon/.tmp-$base"
    if [ -s "$dst" ] && [ -s "$dst.json" ]; then
      log "deploy-quant shard $i/$n $base: RESUME (recon present)"; continue
    fi
    log "deploy-quant shard $i/$n $base: quantize RED@4 + de-bias"
    "$QUANT_BIN" --in "$shard" --out "$tmp" "${qflags[@]}" >> "$LOG" 2>&1 \
      || { log "deploy-quant shard $base rc=$? — ABORT"; return 1; }
    cp -f "$tmp.json" "$dst.json" 2>/dev/null || true
    recast_one "$tmp" "$dst" >> "$LOG" 2>&1 || { log "recast $base rc=$? — ABORT"; return 1; }
    rm -f "$tmp"
  done
  copy_aux "$hfdir" "$recon"
  return 0
}

# harvest aggregate effective_bpw from a recon shard sidecar (for density billing)
recon_bpw(){
  local recon="$1" sc
  sc=$(ls "$recon"/*.safetensors.json 2>/dev/null | head -1)
  [ -n "$sc" ] || return 0
  "$PY" -c "import json,sys;print(json.load(open('$sc'))['aggregate']['effective_bpw'])" 2>/dev/null
}

# ── THE GATE: refuse to burn cloud unless the 0.5B selective-PV-KL-routed gate earned
# PROMOTE_CLOUD. Same contract as pod-chain-v2.sh require_promoted: read the stamped
# json mirrored to $GATES; require promotion.state == PROMOTE_CLOUD. ──
require_promoted(){
  local gate="$1"
  local g="$GATES/$gate"
  [ -f "$g" ] || { log "GATE $gate MISSING under $GATES — refusing selective-PV (no stamped 0.5B promotion)"; return 1; }
  # exact same extraction as pod-chain-v2.sh (works whether or not jq is present)
  local state
  state=$(grep -oE '"state"[[:space:]]*:[[:space:]]*"[A-Z_]+"' "$g" | head -1 | grep -oE '[A-Z_]+"$' | tr -d '"')
  [ "$state" = "PROMOTE_CLOUD" ] || { log "GATE $gate state=${state:-none} != PROMOTE_CLOUD — refusing selective-PV"; return 1; }
  log "GATE $gate PROMOTE_CLOUD — selective-PV-at-scale authorized"; return 0
}

# ── de-bias calibration (once per model; skipped if banked) ──
calibrate(){
  local model="$1" out="$2"
  [ -f "$out" ] && { log "actmean $out: banked"; return 0; }
  log "calib-actmean $model -> $out (de-bias feature means, GPU)"
  "$PY" "$CALIB" --model "$model" --out "$out" \
      --split train --ctx "$CTX" --chunks 8 --device cuda --dtype float32 >> "$LOG" 2>&1 \
    || { log "calib-actmean $model rc=$? — de-bias UNAVAILABLE, leg ABORT"; return 1; }
  [ -f "$out" ]
}

# ── one selective-PV leg: calib -> PV (save-hf) -> canon PPL on tuned dir -> promote ──
leg(){
  local model="$1" tag="$2" anchor="$3"   # anchor: substring promote.py maps to a bf16 PPL
  local mdir="scratch/$model"
  local sentinel="$R/cloudpv_${tag}_DONE"
  [ -f "$sentinel" ] && { log "LEG $tag: already complete"; return 0; }
  has_weights "$mdir" || { log "LEG $tag: SKIP (no weights at $mdir — download first)"; return 1; }

  local actmean="$mdir/actmean.json"
  calibrate "$mdir" "$actmean" || return 1
  local strand_flags="$STRAND_FLAGS_BASE --actmean $actmean --threads $QUANT_THREADS"

  # ---- selective PV (CUDA). Tuned shadow weights written as a drop-in HF dir. ----
  local hfdir="$mdir/selpv-hf"
  local pv_json="$R/pv_selpv_${tag}.json"
  if [ ! -d "$hfdir" ]; then
    log "PV $tag: selective-PV q2 RED@4 de-bias KD ctx$CTX steps$STEPS lr$LR (pv-tensors=$PV_TENSORS)"
    "$PY" "$QAT" \
      --model "$mdir" --quant strand --bits 2 --l 12 \
      --steps "$STEPS" --lr "$LR" --ctx "$CTX" \
      --requant-every "$SELPV_REQUANT" \
      --kd --grad-checkpoint \
      --warmup-frac "$WARMUP_FRAC" --cooldown-frac "$COOLDOWN_FRAC" \
      --device cuda \
      --pv-tensors "$PV_TENSORS" \
      --strand-flags "$strand_flags" \
      --strand-dir "$mdir/strand-pv" \
      --eval-chunks "$EVAL_CHUNKS" --eval-ctx "$EVAL_CTX" \
      --arm-name "cloud_selpv_${tag}" --lineage-label science \
      --save-hf "$hfdir" \
      --out "$pv_json" >> "$LOG" 2>&1 \
      || { log "PV $tag rc=$? — see $LOG (HF dir NOT written; leg ABORT)"; return 1; }
    "$PY" "$PROMOTE" "$pv_json" --model "$anchor" --quiet >> "$LOG" 2>&1 || true
  else
    log "PV $tag: tuned HF dir present ($hfdir) — skipping PV, going to canon eval"
  fi
  [ -d "$hfdir" ] || { log "PV $tag: no tuned HF dir — leg ABORT"; return 1; }

  # ---- CANON PPL of the tuned-then-STRAND-quantized weights (same harness as bf16) ----
  # Quantize the tuned HF dir with the SAME deployment config (RED@4 + de-bias) into a
  # recon dir, then eval on WikiText-2 via the canon shim (harness-identical to the
  # bf16 anchor). The recon's effective_bpw is merged in for density billing.
  local label="selpv_${tag}"
  local outdir="$mdir/reopen/$label"
  local recon="$outdir/recon"
  local final="$R/ppl_${label}.json"
  mkdir -p "$outdir"
  if [ ! -f "$final" ]; then
    log "DEPLOY-QUANT $tag: RED@4 + de-bias recon of tuned dir -> $recon"
    deploy_quant_recon "$hfdir" "$recon" "$actmean" || { log "DEPLOY-QUANT $tag FAILED"; return 1; }
    log "CANON-EVAL $tag: WikiText-2 ctx$EVAL_CTX chunks$EVAL_CHUNKS (offload)"
    OMP_NUM_THREADS="$OMP_NUM_THREADS_EVAL" EVAL_GPU_GB="$EVAL_GPU_GB" \
      "$PY" "$EVAL" "$recon" "$EVAL_CTX" "$EVAL_CHUNKS" offload bfloat16 \
        "selpv_${tag}" "$final" >> "$LOG" 2>&1 \
      || { log "CANON-EVAL $tag rc=$? — see $LOG"; return 1; }
    # stamp the deployment density (effective_bpw) into the eval json so promote.py
    # can bill loss-tax AGAINST a known bit-rate.
    local bpw; bpw=$(recon_bpw "$recon")
    if [ -n "$bpw" ]; then
      "$PY" - "$final" "$bpw" <<'BPW_PY'
import json, sys
p, bpw = sys.argv[1], float(sys.argv[2])
o = json.load(open(p)); o["eff_bpw"] = bpw
o["deploy"] = {"rung_config": "mp-kl-routed", "debias": True, "outlier_channel": 1, "base_bits": 2}
json.dump(o, open(p, "w"), indent=2)
print(f"[bpw] stamped eff_bpw={bpw} into {p}", flush=True)
BPW_PY
    fi
  fi

  # ---- bank + promote the canon result (loss-tax stamped into the json) ----
  if [ -f "$final" ]; then
    "$PY" "$PROMOTE" "$final" --model "$anchor" >> "$LOG" 2>&1 || true
    log "LEG $tag DONE: $(tr -d ' \n' < "$final" | head -c 240)"
    date > "$sentinel"
  else
    log "LEG $tag: canon ppl json missing — NOT marking done"; return 1
  fi
}

log "=== CLOUD SELECTIVE-PV ARMED (7B + 32B, q2 RED@4 + de-bias + KD, ctx$CTX) ==="
log "gate sentinel required: $GATES/$GATE_SENTINEL (state must be PROMOTE_CLOUD)"
log "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //') | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Single gate guards BOTH scale legs (the 0.5B selective-PV-KL-routed proof transfers
# to the class-level recipe used at 7B and 32B).
require_promoted "$GATE_SENTINEL" || { log "REFUSING all scale legs — gate not PROMOTE_CLOUD"; exit 1; }

# ── LEG 1: Qwen2.5-7B ──
[ -f /workspace/SKIP-7B ]  || leg qwen-7b  7b  qwen-7b

# ── LEG 2: Qwen2.5-32B ──
[ -f /workspace/SKIP-32B ] || leg qwen-32b 32b qwen-32b

log "=== CLOUD SELECTIVE-PV COMPLETE ==="
log "results banked in $R/ppl_selpv_*.json (promote.py-stamped); mirror-down sweep ingests them"

# ═════════════════════════════════════════════════════════════════════════════
# ARMING — how the operator deploys + launches this (runs FROM THE DEV BOX).
# (Pod identity matches scripts/conductor.sh: root@213.192.2.110 -P 40078,
#  key ~/.ssh/id_ed25519. Confirm with the owner before each run; the pod's
#  TCP host/port rotates.)
#
#   POD=213.192.2.110 ; PORT=40078 ; KEY=~/.ssh/id_ed25519
#   SCP="scp -o BatchMode=yes -o IdentitiesOnly=yes -i $KEY -P $PORT"
#   SSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $KEY -p $PORT root@$POD"
#
#   # 0) PREREQUISITE: the 0.5B selective-PV-KL-routed gate must have earned PROMOTE_CLOUD.
#   #    Run it locally (it self-gates on its prior), then confirm the stamped state:
#   #      bash scripts/gates/10-pv-kl-routed.sh
#   #      python3 scripts/promote.py research/pv-dp/pv-kl-routed.json --model qwen-05b --dry-run
#   #    -> the grammar line MUST read  state=PROMOTE_CLOUD  before arming.
#
#   # 1) deploy the script + the RED-protection config + the local gate sentinel.
#   $SSH 'mkdir -p /workspace/strand/scripts /workspace/strand/configs /workspace/gates'
#   $SCP scripts/cloud-selective-pv.sh root@$POD:/workspace/strand/scripts/
#   $SCP configs/mp-kl-routed.json     root@$POD:/workspace/strand/configs/
#   $SCP research/pv-dp/pv-kl-routed.json root@$POD:/workspace/gates/pv-kl-routed.json   # the GATE
#   $SSH 'chmod +x /workspace/strand/scripts/cloud-selective-pv.sh'
#
#   # 2) ensure deps on the pod (these are pod-resident in the live campaign; verify):
#   #    - target/release/quantize-model  (cargo build -p strand-quant --release --bin quantize-model)
#   #    - scripts/strand-qat.py (PV harness), scripts/calib-actmean.py (de-bias calib),
#   #      scripts/eval-ppl.py + tools/strand_eval/ (canon eval shim), scripts/promote.py
#   #    - python deps: torch(+cuda), transformers, datasets, safetensors
#   #    - 7B/32B weights at scratch/qwen-7b , scratch/qwen-32b (use scripts/download-model.sh
#   #      on the pod; do NOT bulk-download both if the volume is tight — run 7B, then 32B).
#
#   # 3) LAUNCH DETACHED (survives disconnect; logs to /workspace/cloud-selective-pv.log):
#   $SSH 'cd /workspace/strand && nohup bash scripts/cloud-selective-pv.sh \
#            > /workspace/cloud-selective-pv.nohup 2>&1 &'
#   #    (override knobs inline, e.g.  SKIP-32B for a 7B-only first pass:
#   #       $SSH "touch /workspace/SKIP-32B" )
#
#   # 4) WATCH:
#   $SSH 'tail -n 40 /workspace/cloud-selective-pv.log'
#
#   # 5) MIRROR RESULTS DOWN (conductor's pod tick already does this every 10th poll;
#   #    or pull manually). ppl_selpv_*.json carry the promote.py stamp:
#   $SCP "root@$POD:/workspace/strand-results/ppl_selpv_*.json" scratch/pod-results/
#   #    then bill locally:
#   #      for f in scratch/pod-results/ppl_selpv_*.json; do \
#   #        python3 scripts/promote.py "$f" --model qwen-7b --dry-run; done
#
# GATE CONDITION (one line): the script refuses to run any scale leg unless
#   /workspace/gates/pv-kl-routed.json has promotion.state == PROMOTE_CLOUD
# (stamped by promote.py on the 0.5B selective-PV-KL-routed result and mirrored up in
# step 1). No sentinel, or any other state, => exit 1, zero cloud burn.
#
# SUCCESS TARGET: ppl_selpv_7b.json loss_tax_nats <= 0.15 (from the q2 base 0.324).
#   promote.py computes loss_tax = ln(PPL_quant / 6.629) for 7B automatically.
# ═════════════════════════════════════════════════════════════════════════════
