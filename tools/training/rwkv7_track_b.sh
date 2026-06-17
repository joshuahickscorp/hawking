#!/bin/bash
# ============================================================================
# Track B orchestrator — DPO (SimPO) polish pass on top of the SFT checkpoint.
#   Stage 1: build preference pairs (gold chosen + SFT rejected at temp 0.7)
#   Stage 2: SimPO DPO training (last-16 layers, MPS)
#   Stage 3: export -> GGUF Q4_K_M
#   Stage 4: coherence + tps measure vs SFT
#
# Launch:
#   nohup bash tools/training/rwkv7_track_b.sh >> artifacts/rwkv7_posttrain/track_b.log 2>&1 &
# ============================================================================
set -uo pipefail
cd /Users/scammermike/Downloads/dismantle
source .venv-rwkv/bin/activate 2>/dev/null

ART=artifacts/rwkv7_posttrain
LOG=$ART/track_b.log
CHOSEN=$ART/dpo_chosen.jsonl
PAIRS=$ART/dpo.jsonl
SFT_STATE=$ART/sft_out/final/state_dict.pt
DPO_OUT=$ART/dpo_out
HFOUT=$ART/dpo_hf
DPO_GGUF=models/rwkv7-g1-04-dpo-Q4_K_M.gguf
SFT_GGUF=models/rwkv7-g1-04-sft-Q4_K_M.gguf
EVAL=$ART/eval_prompts.jsonl
BIN=target/release/dismantle
exec > >(tee -a "$LOG") 2>&1

stage(){ echo ""; echo "============ [$(date +%H:%M:%S)] $* ============"; }

stage "TRACK B START — DPO (SimPO) on SFT checkpoint"

# ---- Stage 1: build preference pairs ----
stage "1/4 build DPO pairs (SFT rejected samples at temp 0.7)"
if [ -f "$PAIRS" ] && [ "$(wc -l < "$PAIRS")" -ge 2000 ]; then
  echo "[skip] $PAIRS already has $(wc -l < "$PAIRS") rows"
else
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3.12 tools/training/rwkv7_dpo_build_pairs.py \
    --chosen "$CHOSEN" \
    --gguf "$SFT_GGUF" \
    --bin "$BIN" \
    --out "$PAIRS" \
    --max-new-tokens 200 \
    --temperature 0.7 \
    --resume \
    || { echo "!!! pair build failed — stopping"; exit 1; }
fi
N_PAIRS=$(wc -l < "$PAIRS")
echo "pair file: $PAIRS ($N_PAIRS rows)"
[ "$N_PAIRS" -lt 100 ] && { echo "!!! too few pairs ($N_PAIRS) — stopping"; exit 1; }

# ---- Stage 2: SimPO DPO training ----
stage "2/4 SimPO DPO training (MPS, last-16 layers)"
if [ -f "$DPO_OUT/final/state_dict.pt" ]; then
  echo "[skip] $DPO_OUT/final/state_dict.pt already exists"
else
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3.12 tools/training/rwkv7_dpo_torch.py \
    --sft-state "$SFT_STATE" \
    --pairs "$PAIRS" \
    --out "$DPO_OUT" \
    --device mps \
    --last-n-layers 16 \
    --max-length 448 \
    --grad-accum 16 \
    --lr 5e-6 \
    --beta 2.0 \
    --gamma 0.5 \
    --save-every 50 \
    --epochs 1 \
    || echo "DPO exited nonzero — will try latest checkpoint"
fi
CKPT="$DPO_OUT/final/state_dict.pt"
[ -f "$CKPT" ] || CKPT="$DPO_OUT/latest/state_dict.pt"
[ -f "$CKPT" ] || { echo "!!! no usable DPO checkpoint — stopping"; exit 1; }
echo "using checkpoint: $CKPT"

# ---- Stage 3: export DPO model -> GGUF Q4_K_M ----
stage "3/4 export DPO checkpoint -> GGUF Q4_K_M"
python3.12 tools/training/rwkv7_export_hf.py \
  --state-dict "$CKPT" --out-dir "$HFOUT" --gguf "$DPO_GGUF" \
  || { echo "!!! EXPORT FAILED — stopping"; exit 1; }
[ -f "$DPO_GGUF" ] || { echo "!!! no DPO GGUF — stopping"; exit 1; }

# ---- Stage 4: coherence + tps measure ----
stage "4/4 coherence measure — DPO vs SFT vs BASE"
BASE_GGUF=models/rwkv7-g1-04/rwkv7-0.4B-g1.Q4_K_M.gguf
python3.12 - "$BIN" "$DPO_GGUF" "$SFT_GGUF" "$BASE_GGUF" "$EVAL" "$ART/measure_dpo.jsonl" <<'PY'
import json, subprocess, sys, re
from collections import Counter
BIN, DPO, SFT, BASE, EVAL, OUTP = sys.argv[1:7]
def gen(gguf, user):
    p = f"<|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant:"
    r = subprocess.run([BIN,"generate","--weights",gguf,"--prompt",p,"--max-new-tokens","160"],
                       capture_output=True, text=True)
    out = r.stdout.strip()
    return out.split("\n\n",1)[0].strip()[:600]
def degenerate(t):
    toks = t.split()
    if len(toks) < 12: return False
    grams = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
    return max(Counter(grams).values(), default=0) >= 4 or bool(re.search(r"(.)\1{20,}", t))
rows=[json.loads(l) for l in open(EVAL)]
n_dpo=0; n_sft=0; recs=[]
for r in rows:
    u=r["user"]; d=gen(DPO,u); s=gen(SFT,u); b=gen(BASE,u)
    dd=degenerate(d); ds=degenerate(s)
    n_dpo+=dd; n_sft+=ds
    recs.append({"bucket":r["bucket"],"user":u[:80],"dpo":d,"sft":s,"base":b,
                 "dpo_degenerate":dd,"sft_degenerate":ds})
    print(f"  [{r['bucket']:>20}] dpo_deg={dd} sft_deg={ds}  Q={u[:48]!r}")
    print(f"      DPO : {d[:140]!r}")
json.dump(recs, open(OUTP,"w"), indent=1)
print(f"\n[coherence] DPO {n_dpo}/20 deg, SFT {n_sft}/20 deg ({'IMPROVED' if n_dpo<=n_sft else 'WORSE'})")
print(f"[coherence] full outputs -> {OUTP}")
PY

stage "TRACK B (DPO) COMPLETE"
echo "Review $ART/measure_dpo.jsonl for the DPO GO/NO-GO decision."
echo "  DPO GGUF: $DPO_GGUF"
echo "  GO criteria: DPO degenerate count <= SFT (2/20) + coherence qualitatively better."
