#!/bin/bash
# ============================================================================
# Track A orchestrator — drives the whole on-device RWKV-7 post-train pipeline
# as ONE managed process: SFT-on-gold -> export GGUF -> coherence+tps measure
# -> GO/NO-GO report. Single log, resumable per stage, GPU single-tenant.
#
# Launch once in the background and watch artifacts/rwkv7_posttrain/track_a.log:
#   nohup tools/training/rwkv7_track_a.sh > /dev/null 2>&1 &
# Stage 5 (DPO) is intentionally a separate follow-on, gated on the Stage-4
# verdict (the F2 plan: SFT alone may be enough to ship).
# ============================================================================
set -uo pipefail
cd /Users/scammermike/Downloads/dismantle
source .venv-rwkv/bin/activate 2>/dev/null

ART=artifacts/rwkv7_posttrain
LOG=$ART/track_a.log
OUT=$ART/sft_out
HFOUT=$ART/sft_hf
SFT_GGUF=models/rwkv7-g1-04-sft-Q4_K_M.gguf
BASE_GGUF=models/rwkv7-g1-04/rwkv7-0.4B-g1.Q4_K_M.gguf
EVAL=$ART/eval_prompts.jsonl
BIN=target/release/dismantle
exec > >(tee -a "$LOG") 2>&1

stage(){ echo ""; echo "============ [$(date +%H:%M:%S)] $* ============"; }

stage "TRACK A START — SFT -> export -> measure"

# ---- Stage 1: SFT on gold (MPS, exclusive GPU) ----
stage "1/4 SFT on gold corpus (MPS)"
if [ -f "$OUT/final/state_dict.pt" ]; then
  echo "[skip] $OUT/final/state_dict.pt already exists"
else
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3.12 tools/training/rwkv7_sft_torch.py \
    --device mps --out "$OUT" --epochs 1 --grad-accum 16 --lr 2e-5 --max-length 384 \
    --last-n-layers 16 --save-every 25 \
    || echo "SFT exited nonzero — will try the latest checkpoint (saved every 25 steps)"
fi
# Prefer the final checkpoint; fall back to the most recent (a near-complete run is still usable).
CKPT="$OUT/final/state_dict.pt"
[ -f "$CKPT" ] || CKPT="$OUT/latest/state_dict.pt"
[ -f "$CKPT" ] || { echo "!!! no usable checkpoint (final or latest) — stopping"; exit 1; }
echo "using checkpoint: $CKPT"

# ---- Stage 2: export -> GGUF f16 ----
stage "2/4 export trained checkpoint -> GGUF Q4_K_M"
python3.12 tools/training/rwkv7_export_hf.py \
  --state-dict "$CKPT" --out-dir "$HFOUT" --gguf "$SFT_GGUF" \
  || { echo "!!! EXPORT FAILED — stopping"; exit 1; }
[ -f "$SFT_GGUF" ] || { echo "!!! no SFT GGUF — stopping"; exit 1; }

# ---- Stage 3: coherence measure (SFT vs base on the eval battery) ----
stage "3/4 coherence measure — eval battery (incl. degeneration triggers)"
python3.12 - "$BIN" "$SFT_GGUF" "$BASE_GGUF" "$EVAL" "$ART/measure_sft.jsonl" <<'PY'
import json, subprocess, sys, re
BIN, SFT, BASE, EVAL, OUTP = sys.argv[1:6]
def gen(gguf, user):
    p = f"<|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant:"
    r = subprocess.run([BIN,"generate","--weights",gguf,"--prompt",p,"--max-new-tokens","160"],
                       capture_output=True, text=True)
    out = r.stdout.strip()
    # dismantle prints the completion; trim at the first blank-line turn break
    return out.split("\n\n",1)[0].strip()[:600]
def degenerate(t):
    toks = t.split()
    if len(toks) < 12: return False
    # any 3-gram repeated >=4x  => degenerate loop
    grams = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
    from collections import Counter
    return max(Counter(grams).values(), default=0) >= 4 or bool(re.search(r"(.)\1{20,}", t))
rows=[json.loads(l) for l in open(EVAL)]
n_deg=0; recs=[]
for r in rows:
    u=r["user"]; s=gen(SFT,u); b=gen(BASE,u)
    d=degenerate(s); n_deg+=d
    recs.append({"bucket":r["bucket"],"user":u[:80],"sft":s,"base":b,"sft_degenerate":d})
    print(f"  [{r['bucket']:>20}] deg={d}  Q={u[:48]!r}")
    print(f"      SFT : {s[:140]!r}")
json.dump(recs, open(OUTP,"w"), indent=1)
print(f"\n[coherence] {len(rows)} prompts, {n_deg} degenerate ({'PASS' if n_deg==0 else 'CHECK'})")
print(f"[coherence] full outputs -> {OUTP}")
PY

# ---- Stage 4: tps measure (flat decode — the SSM win must hold) ----
stage "4/4 tps measure (decode tok/s, flat-context)"
for tag in SFT BASE; do
  g=$SFT_GGUF; [ "$tag" = BASE ] && g=$BASE_GGUF
  P=$(printf '<|rwkv_tokenizer_end_of_text|>User: Write a short paragraph about the ocean.\n\nAssistant:')
  echo "--- $tag ($g) ---"
  $BIN generate --weights "$g" --prompt "$P" --max-new-tokens 128 --explain-performance 2>/dev/null \
    | grep -iE "tok/s|decode|tokens/sec|tps" | head -3 || echo "(no perf line — time it manually)"
done

stage "TRACK A (SFT) COMPLETE"
echo "Review $ART/measure_sft.jsonl for the GO/NO-GO decision on DPO."
echo "  GO criteria: SFT non-degenerate on the battery + coherent >= base + tps holds (~76 tok/s flat)."
echo "  If GO and you want the polish pass, run the DPO follow-on (gold-as-chosen)."
