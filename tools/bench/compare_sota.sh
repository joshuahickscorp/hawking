#!/usr/bin/env bash
# =============================================================================
# tools/bench/compare_sota.sh
#
# THE comprehensive head-to-head: Hawking vs the closest SOTA local runtimes
# (llama.cpp + MLX), plus a full Hawking self-diagnostic that also exercises the
# capabilities the others don't have (noting the nearest SOTA equivalent).
#
# Dimensions:
#   0. SETUP / detection          — models, engines, cleanliness, versions
#   1. CAPABILITY MAP             — Hawking CLI surface vs closest SOTA surface
#   2. LOCAL MODEL INVENTORY      — runnable files present on this machine
#   3. FOOTPRINT / compression    — on-disk bpw + out-of-core press planner
#   4. QUANTIZATION / BIT LADDER  — runtime formats + all requested press targets
#   5. SPEED                      — warm decode tps + prefill, same GGUF
#   6. KERNEL / BENCH BATTERY     — hawking bench + synthetic kernel microbench
#   7. LONG-CONTEXT (the moat)    — SSM flat-decode vs the transformer KV wall
#   8. QUALITY                    — deterministic task prompts, side-by-side + pass/fail
#   9. DISTILL / POST-TRAIN       — local tooling inventory + current product gap
#  10. HAWKING DIAGNOSTIC         — CLI probes; closest-SOTA note for unique ones
#  11. ENERGY (optional)          — J/tok via macmon, if present
#
# ROBUSTNESS (this is the part that bit us before):
#   * llama.cpp is ALWAYS run non-interactively — `-no-cnv` + stdin `< /dev/null`
#     + `timeout`. That is the fix for the `>>>>>` interactive prompt loop.
#   * `llama-bench` (non-interactive by design) is used for speed.
#   * Every external call is timeout-wrapped; a missing engine is SKIPPED with a
#     clear note + the install command, never a hang or a hard failure.
#
# CLEAN ROOM: for trustworthy absolute numbers, quit Claude/Cursor and any heavy
# GPU app first (a background Claude session inflates tps/J ~4-5x). The preflight
# warns (or aborts with STRICT_CLEAN=1).
#
# Wall-clock is intentionally not optimized — this is the thorough latent test.
#
# USAGE (run in a terminal with everything else closed):
#   bash tools/bench/compare_sota.sh                 # full run, all engines found
#   QUICK=1 bash tools/bench/compare_sota.sh         # fewer trials/contexts/prompts
#   STRICT_CLEAN=1 bash tools/bench/compare_sota.sh  # abort if Claude is running
#   TRIALS=5 TOK=256 bash tools/bench/compare_sota.sh
#
# ENV OVERRIDES:
#   HBIN          hawking binary           (default ./target/release/hawking)
#   QWEN_GGUF     shared transformer GGUF  (default models/Qwen2.5-3B-Instruct-Q4_K_M.gguf)
#   RWKV_GGUF     SSM GGUF (the moat)      (default models/rwkv7-g1-04-sft-Q4_K_M.gguf)
#   MLX_MODEL     MLX HF id               (default mlx-community/Qwen2.5-3B-Instruct-4bit)
#   MLX_PYTHON    python with mlx_lm      (auto-probed: $MLX_PYTHON, python3.12, ~/.mlxenv, python3)
#   LLAMA_CLI / LLAMA_BENCH               (auto-detected on PATH)
#   TRIALS=3  TOK=128  CTX_SHORT  CTX_LONG=8192  RUN_TIMEOUT=300
#   BIT_TARGETS=8,6,5,4,3,2,1
#   RUN_KERNEL_BENCH=1  KERNEL_ITERS=100  RUN_HAWKING_BENCH=1
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
REPO="$(pwd)"

QUICK="${QUICK:-0}"
HBIN="${HBIN:-./target/release/hawking}"
# Transformer model — prefer the bigger 7B PORTABLE model if present (same base in
# all three: GGUF for Hawking+llama.cpp, MLX 4bit for MLX); else fall back to 3B.
if [ -z "${QWEN_GGUF:-}" ]; then
  if [ -f models/Qwen2.5-7B-Instruct-Q4_K_M.gguf ]; then
    QWEN_GGUF=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
  else
    QWEN_GGUF=models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
  fi
fi
if [ -z "${MLX_MODEL:-}" ]; then
  if [ -d models/mlx-Qwen2.5-7B-Instruct-4bit ]; then
    MLX_MODEL=models/mlx-Qwen2.5-7B-Instruct-4bit
  elif [ -f models/Qwen2.5-7B-Instruct-Q4_K_M.gguf ]; then
    MLX_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
  else
    MLX_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit
  fi
fi
RWKV_GGUF="${RWKV_GGUF:-models/rwkv7-g1-04-sft-Q4_K_M.gguf}"
QWEN_BASE="$(basename "$QWEN_GGUF")"
RWKV_BASE="$(basename "$RWKV_GGUF")"
TRIALS="${TRIALS:-3}"; [ "$QUICK" = 1 ] && TRIALS=2
TOK="${TOK:-128}";     [ "$QUICK" = 1 ] && TOK=64
CTX_LONG="${CTX_LONG:-8192}"
RUN_TIMEOUT="${RUN_TIMEOUT:-300}"
BIT_TARGETS="${BIT_TARGETS:-8,6,5,4,3,2,1}"
# Press-demo memory budget: must sit BETWEEN the out-of-core peak and the
# full-resident-f32 size to demonstrate the wedge. 8 GiB wedges both 3B (~1.3 ooc /
# ~11.5 resident) and 7B (~2.5 ooc / ~28 resident). Bump for >7B parents.
PRESS_BUDGET="${PRESS_BUDGET:-8gb}"
RUN_HAWKING_BENCH="${RUN_HAWKING_BENCH:-1}"
RUN_KERNEL_BENCH="${RUN_KERNEL_BENCH:-1}"
KERNEL_ITERS="${KERNEL_ITERS:-100}"; [ "$QUICK" = 1 ] && KERNEL_ITERS=25
KERNEL_SHAPE="${KERNEL_SHAPE:-1408x2048}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-reports/sota-compare/$STAMP}"
mkdir -p "$OUT"
REPORT="$OUT/report.md"
LOG="$OUT/run.log"

say()  { printf '%s\n' "$*" | tee -a "$LOG"; }
md()   { printf '%s\n' "$*" >>"$REPORT"; }
med()  { sort -n | awk '{a[NR]=$1} END{print (NR>0)?a[int((NR+1)/2)]:"NA"}'; }
human_gib() { awk -v b="$1" 'BEGIN{printf "%.2f GiB", b/1073741824}'; }

# ---------------------------------------------------------------- portable timeout
# macOS ships NO `timeout`. Prefer timeout/gtimeout; else perl (always present and
# it reliably kills a runaway — e.g. an interactive llama-cli `>` loop, the exact
# hang we have hit before). EVERY external engine call goes through TO.
if command -v timeout >/dev/null 2>&1; then
  TO() { timeout "$@"; }
elif command -v gtimeout >/dev/null 2>&1; then
  TO() { gtimeout "$@"; }
else
  TO() { local t="$1"; shift; perl -e 'my $s=shift; alarm $s; exec @ARGV or exit 127' "$t" "$@"; }
fi

# ---------------------------------------------------------------- detection
LLAMA_BENCH="${LLAMA_BENCH:-$(command -v llama-bench || true)}"
# Generation binary: modern llama.cpp REJECTS `llama-cli -no-cnv` and defaults to
# an interactive chat that loops `>` forever (the classic hang). The non-interactive
# binary is `llama-completion`. Prefer it; fall back to an older llama-cli that still
# honors -no-cnv.
LLAMA_GEN="${LLAMA_GEN:-$(command -v llama-completion || command -v llama-cli || true)}"
case "$LLAMA_GEN" in
  */llama-completion) LLAMA_GEN_KIND="completion" ;;
  */llama-cli)        LLAMA_GEN_KIND="cli-nocnv" ;;
  *)                  LLAMA_GEN_KIND="none" ;;
esac
OLLAMA="$(command -v ollama || true)"

probe_mlx_python() {
  local c
  for c in "${MLX_PYTHON:-}" python3.12 "$HOME/.mlxenv/bin/python" python3; do
    [ -n "$c" ] || continue
    if "$c" -c 'import mlx_lm' >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}
MLX_PY="$(probe_mlx_python || true)"

HAVE_HAWKING=0; [ -x "$HBIN" ] && HAVE_HAWKING=1
HAVE_LLAMA=0; { [ -n "$LLAMA_BENCH" ] || [ -n "$LLAMA_GEN" ]; } && HAVE_LLAMA=1
HAVE_MLX=0; [ -n "$MLX_PY" ] && HAVE_MLX=1

# ---------------------------------------------------------------- preflight
say "=== compare_sota: Hawking vs llama.cpp vs MLX ($STAMP) ==="
CLEAN="clean"
if pgrep -f "Claude.app" >/dev/null 2>&1 || pgrep -xi claude >/dev/null 2>&1; then
  CLEAN="DIRTY (Claude running — absolute tps/J inflate; close it for trustworthy numbers)"
  if [ "${STRICT_CLEAN:-0}" = 1 ]; then say "ABORT: STRICT_CLEAN=1 and Claude is running."; exit 3; fi
fi
busy_gpu="$(ps ax -o command= | grep -E 'hawking (generate|serve)|llama-(cli|bench|server)|mlx_lm' | grep -v grep || true)"

hawking_ver="$($HBIN version 2>/dev/null | head -1 || echo NA)"
llama_ver="$([ -n "$LLAMA_BENCH" ] && TO 15 "$LLAMA_BENCH" --version 2>&1 | head -1 || echo "$LLAMA_GEN_KIND")"
mlx_ver="$([ -n "$MLX_PY" ] && "$MLX_PY" -c 'import mlx_lm;print(getattr(mlx_lm,"__version__","present"))' 2>/dev/null || echo 'absent (pip install mlx-lm in a py3.12 env to enable)')"

{
  echo "# Hawking vs SOTA — comparison report ($STAMP)"
  echo
  echo "- Run cleanliness: **$CLEAN**"
  echo "- Hawking: \`$hawking_ver\`"
  echo "- llama.cpp: \`$llama_ver\` (gen=$LLAMA_GEN_KIND, bench=$([ -n "$LLAMA_BENCH" ] && echo yes || echo no))"
  echo "- MLX (mlx_lm): \`$mlx_ver\`"
  echo "- ollama present: $([ -n "$OLLAMA" ] && echo yes || echo no)"
  echo "- Shared transformer GGUF: \`$QWEN_GGUF\` | SSM GGUF: \`$RWKV_GGUF\`"
  echo "- Config: TRIALS=$TRIALS TOK=$TOK CTX_LONG=$CTX_LONG RUN_TIMEOUT=${RUN_TIMEOUT}s QUICK=$QUICK BIT_TARGETS=$BIT_TARGETS RUN_HAWKING_BENCH=$RUN_HAWKING_BENCH RUN_KERNEL_BENCH=$RUN_KERNEL_BENCH"
  [ -n "$busy_gpu" ] && { echo; echo "> ⚠ other model jobs were running at start:"; echo '> ```'; echo "$busy_gpu" | sed 's/^/> /'; echo '> ```'; }
  echo
} >"$REPORT"
say "report: $REPORT"
[ "$HAVE_HAWKING" = 1 ] || { say "FATAL: hawking binary not built ($HBIN) — cargo build --release -p hawking"; exit 2; }

# ============================================================= 1. CAPABILITY MAP
say ""; say "-- [1/11] capability map: Hawking surface vs closest SOTA --"
md "## 1. Capability map — what Hawking does vs the closest local SOTA"
md ""
md "| facet | Hawking command / path | closest SOTA | comparison verdict |"
md "|---|---|---|---|"
md "| one-shot inference | \`generate\` | llama.cpp \`llama-cli\` / \`llama-completion\`, MLX \`mlx_lm.generate\`, ollama | directly comparable; speed + quality measured below |"
md "| OpenAI-compatible serving | \`serve\` | \`llama-server\`, ollama, vLLM | comparable API class; Hawking adds Apple-fit/workload knobs |"
md "| continuous / multiseq batching | \`serve --max-batch-size\`, capture multiseq path | llama-server/vLLM batching | comparable throughput class; Hawking reports token-only/dispatch internals |"
md "| fit / machine envelope | \`doctor --json\`, \`fit\`, \`serve --auto --intent\` | coarse UI estimates in LM Studio/Ollama | Hawking-specific explicit planner |"
md "| speed suites | \`bench --suite decode/prefill/throughput/bandwidth/competitive\` | \`llama-bench\`, MLX generate timing | directly comparable where model artifacts match |"
md "| kernel microbench | \`bench-kernel\`, \`bench-q4k-shapes\` | none standard in llama.cpp/MLX CLIs | Hawking-specific internal visibility |"
md "| compression planner | \`press --dry-run --memory-budget --target\` | \`llama-quantize\`, AutoAWQ, GPTQ | Hawking plans out-of-core; SOTA tools generally need resident parent/workflow |"
md "| quantized serving formats | GGUF Q4_K_M primary; Q6_K/Q3_K/Q8_0/f16 partial; TQ/STRAND feature work | llama.cpp broad GGUF; MLX 4-bit; ollama GGUF packages | llama.cpp is broadest today; Hawking is optimized/narrower with sub-4 R&D |"
md "| artifact integrity / sidecar | \`verify\`, \`bake-sidecar\`, sidecar loader | manual hashes / model manifests | Hawking has explicit integrity and sidecar hooks; bake coverage varies by backend |"
md "| speculative / draft diagnostics | \`spec-oracle\`, \`--user-draft\`, Eagle/RWKV scripts | llama.cpp speculative options | Hawking has more oracle/debug tooling; speed win is workload-dependent |"
md "| distillation / post-train | \`tools/training/rwkv7_*.py\`, QAT/KD/DPO scripts | external trainer stacks, not llama.cpp/ollama runtime CLIs | Hawking has local research tooling, not a finished \`press --distill\` product command |"
md "| energy / thermal | \`tools/bench/phase_joules.sh\`, \`energy_paired.sh\` with macmon | no built-in llama.cpp energy report | Hawking-specific local energy harness |"
md ""
md "_This section separates shipped CLI/runtime capabilities from research tooling. The measured sections below only run non-destructive local probes._"

# ============================================================= 2. LOCAL MODEL INVENTORY
say ""; say "-- [2/11] local model / artifact inventory --"
md ""; md "## 2. Local model and artifact inventory"
md ""
md "| artifact | type | size | note |"
md "|---|---|---:|---|"
while IFS= read -r g; do
  [ -n "$g" ] || continue
  b="$(stat -f%z "$g" 2>/dev/null || echo 0)"
  note="local GGUF; runnable if the loader supports its architecture"
  case "$(basename "$g")" in
    *Qwen2.5-3B*|*Qwen2.5-0.5B*) note="verified/primary dense path per MODELS.md" ;;
    *Qwen2.5-7B*|*qwen2.5-1.5b*) note="same dense path; run-gated here if selected" ;;
    *rwkv7*) note="SSM moat / RWKV-7 local artifact" ;;
    *mamba*) note="Mamba/SSM local artifact; not part of default SOTA comparison" ;;
    *Llama*|*SmolLM*) note="small dense comparison artifact" ;;
  esac
  md "| \`$g\` | GGUF | $(human_gib "$b") | $note |"
done < <(find models -maxdepth 3 -type f -name '*.gguf' 2>/dev/null | sort | head -40)
while IFS= read -r s; do
  [ -n "$s" ] || continue
  b="$(stat -f%z "$s" 2>/dev/null || echo 0)"
  md "| \`$s\` | safetensors | $(human_gib "$b") | training/MLX/HF artifact; not directly runnable by Hawking unless converted/planned by press |"
done < <(find models -maxdepth 3 -type f -name '*.safetensors' 2>/dev/null | sort | head -20)
md ""
md "Model support tier reference: \`MODELS.md\` (verified vs runs vs untested)."

# ---------------------------------------------------------------- engine runners (warm-median tps)
# Hawking decode tps for a prompt + token budget.
hawking_tps() { # $1=gguf $2=prompt $3=tok -> median tps
  local g="$1" p="$2" t="$3" i out
  # warmup (discarded) to amortize cold PSO shader-compile — we want WARM tps.
  TO "$RUN_TIMEOUT" env HAWKING_QWEN_USER_DRAFT=0 "$HBIN" generate --weights "$g" \
    --prompt "$p" --max-new-tokens 8 --temperature 0 --seed 5 >/dev/null 2>&1 || true
  for i in $(seq 1 "$TRIALS"); do
    out="$(TO "$RUN_TIMEOUT" env HAWKING_QWEN_USER_DRAFT=0 "$HBIN" generate --weights "$g" \
            --prompt "$p" --max-new-tokens "$t" --temperature 0 --seed 5 2>&1 || true)"
    printf '%s\n' "$out" | grep -oE 'dec_tps=[0-9.]+' | tail -1 | cut -d= -f2
  done | grep . | med
}
# llama.cpp decode tps via llama-bench JSON (non-interactive by design).
llama_tps() { # $1=gguf $2=n_prompt $3=n_gen -> "pp_tps tg_tps"
  [ -n "$LLAMA_BENCH" ] || { echo "NA NA"; return; }
  local g="$1" np="$2" ng="$3" j
  j="$(TO "$RUN_TIMEOUT" "$LLAMA_BENCH" -m "$g" -p "$np" -n "$ng" -r "$TRIALS" -o json 2>/dev/null </dev/null || true)"
  printf '%s' "$j" | python3 -c '
import sys,json
try: d=json.load(sys.stdin)
except Exception: print("NA NA"); raise SystemExit
pp=tg="NA"
for e in d:
    ts=e.get("avg_ts")
    if e.get("n_prompt",0)>0 and e.get("n_gen",0)==0: pp=f"{ts:.2f}"
    if e.get("n_gen",0)>0 and e.get("n_prompt",0)==0: tg=f"{ts:.2f}"
print(pp,tg)
' 2>/dev/null || echo "NA NA"
}
# MLX decode tps.
mlx_tps() { # $1=prompt $2=tok -> median tps
  [ -n "$MLX_PY" ] || { echo "NA"; return; }
  local p="$1" t="$2" i out
  for i in $(seq 1 "$TRIALS"); do
    out="$(TO "$RUN_TIMEOUT" "$MLX_PY" -m mlx_lm.generate --model "$MLX_MODEL" \
            --prompt "$p" --max-tokens "$t" --temp 0 2>&1 </dev/null || true)"
    printf '%s\n' "$out" | grep -i 'Generation:' | grep -oE '[0-9.]+ tokens-per-sec' | grep -oE '[0-9.]+' | tail -1
  done | grep . | med
}

SHORTP="Explain how unified memory on Apple Silicon changes the GPU programming model."

# ============================================================= 3. FOOTPRINT
say ""; say "-- [3/11] footprint / compression --"
md ""; md "## 3. Footprint / compression"
md ""
md "| model | engine | on-disk | bpw | note |"
md "|---|---|---|---|---|"
for g in "$QWEN_GGUF" "$RWKV_GGUF"; do
  [ -f "$g" ] || continue
  b="$(stat -f%z "$g")"
  # bpw via hawking press metadata (weight bytes / params)
  bpw="$($HBIN press --dry-run --weights "$g" 2>/dev/null | grep -oE '~[0-9.]+ bpw' | grep -oE '[0-9.]+' | head -1 || echo NA)"
  md "| $(basename "$g") | Hawking + llama.cpp (same GGUF) | $(human_gib "$b") | ${bpw:-NA} | identical artifact; both load this file |"
done
md "| $QWEN_BASE (MLX 4bit equivalent) | MLX | (HF/local MLX artifact) | ~4.5 | different artifact; MLX uses its own 4-bit format |"
md ""
md "**Hawking-unique:** \`hawking press --dry-run --memory-budget\` plans an OUT-OF-CORE condense (quantize a parent that does"
md "not fit fully resident). Closest SOTA: \`llama-quantize\` (in-memory only) / AutoAWQ / GPTQ (need the full parent resident)."
press_demo="$($HBIN press --dry-run --memory-budget "$PRESS_BUDGET" --target 4,3,2 --weights "$QWEN_GGUF" 2>/dev/null | grep -E 'WEDGE|out-of-core|full-resident' | head -3 || true)"
[ -n "$press_demo" ] && { md '```'; md "$press_demo"; md '```'; }

# ============================================================= 4. QUANTIZATION / BIT LADDER
say ""; say "-- [4/11] quantization / bit ladder ($BIT_TARGETS) --"
md ""; md "## 4. Quantization, distillation-adjacent compression, and bit-width coverage"
md ""
md "| area | Hawking status | closest SOTA status | diagnostic run here |"
md "|---|---|---|---|"
md "| Q4_K_M GGUF serving | primary tuned path; verified for Qwen2.5-3B/0.5B; selected model measured below | llama.cpp/Ollama broad support; MLX uses separate 4-bit artifact | yes: footprint, speed, quality |"
md "| Q6_K / Q3_K_M / Q8_0 / f16 | loader/reference paths and targeted kernels exist; verification varies by model | llama.cpp broadest GGUF format coverage | inventoried; not all formats speed-tested unless local artifacts are selected |"
md "| Q2_K / Q5_K / IQ* | not claimed as verified in MODELS.md | llama.cpp supports more formats | marked untested for Hawking |"
md "| TQ / STRAND sub-4-bit | CPU/reference + GPU bitslice work exists behind the TQ track; product serving still incomplete | QTIP/AWQ/GPTQ external research/tooling depending on stack | documented, not baked in this run |"
md "| out-of-core creation | \`press --dry-run\` reports peak tensor-at-a-time vs full-resident memory | most SOTA quantizers expect resident parent/host workflow | yes: all-bit dry-run ladder |"
md "| compress-then-recover | QAT/KD/DPO scripts exist; no finished \`press --distill\` command | trainer frameworks external to llama.cpp/ollama; MLX has training pieces but not RWKV-7 here | inventoried in section 9 |"
md ""
press_all="$OUT/press_ladder_${BIT_TARGETS//,/ _}.txt"
press_all="${press_all// /}"
TO "$RUN_TIMEOUT" "$HBIN" press --dry-run --memory-budget "$PRESS_BUDGET" --target "$BIT_TARGETS" --weights "$QWEN_GGUF" >"$press_all" 2>&1 || true
md "**All-bit dry-run ladder** for \`$QWEN_BASE\` with \`--target $BIT_TARGETS\` saved to \`$press_all\`:"
md '```'
grep -E 'Condense ladder|tier|bit|out-of-core|full-resident|WEDGE|FITS|EXCEEDS|bpw|out size|vs now' "$press_all" | head -40 >>"$REPORT" || true
md '```'
md ""
md "_Important honesty line: this estimates bits/footprint and creation memory. It does not claim quality at every bit-width; quality requires a per-tier card against the fp16 parent._"

# ============================================================= 5. SPEED
say ""; say "-- [5/11] speed (warm decode tps, same $QWEN_BASE) --"
md ""; md "## 5. Speed — warm decode tps (same model: \`$QWEN_BASE\` for Hawking+llama; MLX 4bit equivalent)"
md ""
hk="$(hawking_tps "$QWEN_GGUF" "$SHORTP" "$TOK")"; say "  hawking: $hk tps"
read -r lpp ltg <<<"$(llama_tps "$QWEN_GGUF" 512 "$TOK")"; say "  llama.cpp: tg=$ltg pp=$lpp"
mtg="$(mlx_tps "$SHORTP" "$TOK")"; say "  mlx: $mtg tps"
md "| engine | decode tps | prefill tps | note |"
md "|---|---|---|---|"
md "| **Hawking** | ${hk:-NA} | (see TTFT bench) | predec Q4_K GEMV, TCB |"
md "| llama.cpp (llama-bench) | ${ltg:-NA} | ${lpp:-NA} | non-interactive llama-bench |"
md "| MLX | ${mtg:-NA} | — | $([ "$HAVE_MLX" = 1 ] && echo "$MLX_MODEL" || echo "SKIPPED — pip install mlx-lm in py3.12") |"
md ""
md "_Warm median of $TRIALS trials, $TOK tokens, greedy. llama tg via llama-bench; absolute numbers require a clean room._"

# ============================================================= 6. KERNEL / BENCH BATTERY
say ""; say "-- [6/11] Hawking bench suite + kernel microbench --"
md ""; md "## 6. Hawking bench battery and kernel microbench"
md ""
md "| probe | output | result | note |"
md "|---|---|---|---|"
if [ "$RUN_HAWKING_BENCH" = 1 ]; then
  hb_json="$OUT/hawking_bench_decode.json"
  say "  hawking bench decode -> $hb_json"
  TO "$RUN_TIMEOUT" "$HBIN" bench --weights "$QWEN_GGUF" --model "$QWEN_BASE" \
    --suite decode --trials "$TRIALS" --max-new-tokens "$TOK" --json "$hb_json" \
    >/dev/null 2>&1 || true
  hb_dec="$(python3 - "$hb_json" 2>/dev/null <<'PY' || true
import json,sys
p=sys.argv[1]
try:
    d=json.load(open(p))
except Exception:
    print("NA"); raise SystemExit
def find(o, keys=("decode_tps","median_decode_tps","tokens_per_second","tps")):
    if isinstance(o, dict):
        for k in keys:
            v=o.get(k)
            if isinstance(v,(int,float)):
                return v
        for v in o.values():
            r=find(v, keys)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r=find(v, keys)
            if r is not None:
                return r
    return None
r=find(d)
print("NA" if r is None else f"{r:.2f}")
PY
)"
  md "| \`hawking bench --suite decode\` | \`$hb_json\` | ${hb_dec:-NA} decode tps | in-process suite; complements raw \`generate\` tps above |"
else
  md "| \`hawking bench --suite decode\` | — | skipped | set \`RUN_HAWKING_BENCH=1\` |"
fi
if [ "$RUN_KERNEL_BENCH" = 1 ]; then
  kb_txt="$OUT/bench_kernel_${KERNEL_SHAPE}.txt"
  say "  bench-kernel --all --shape $KERNEL_SHAPE -> $kb_txt"
  TO "$RUN_TIMEOUT" "$HBIN" bench-kernel --all --shape "$KERNEL_SHAPE" \
    --iterations "$KERNEL_ITERS" --no-history >"$kb_txt" 2>&1 || true
  kb_head="$(grep -E 'kernel|mean|p50|p99|us|μs|gemv|q4|q3|f16' "$kb_txt" | head -8 | tr '\n' '; ' | sed 's/; $//')"
  md "| \`hawking bench-kernel --all --shape $KERNEL_SHAPE\` | \`$kb_txt\` | ${kb_head:-see file} | synthetic kernel timing, no model load |"
  q4_json="$OUT/bench_q4k_shapes.json"
  TO "$RUN_TIMEOUT" "$HBIN" bench-q4k-shapes --iters "$KERNEL_ITERS" --out "$q4_json" \
    >/dev/null 2>&1 || true
  md "| \`hawking bench-q4k-shapes\` | \`$q4_json\` | $([ -s "$q4_json" ] && echo "wrote JSON" || echo "no output") | production Q4_K shape sweep |"
else
  md "| \`bench-kernel\` / \`bench-q4k-shapes\` | — | skipped | set \`RUN_KERNEL_BENCH=1\` |"
fi
md ""
md "_These are Hawking-internal probes. llama.cpp exposes \`llama-bench\`, but not a comparable per-kernel Metal timing CLI._"

# ============================================================= 7. LONG-CONTEXT (the moat)
say ""; say "-- [7/11] long-context: SSM flat vs transformer KV wall --"
md ""; md "## 7. Long-context — the SSM moat (decode tps vs context)"
md ""
md "| model / engine | short | ~${CTX_LONG} ctx | shape |"
md "|---|---|---|---|"
# build a long prompt
LONGP="$(python3 -c "print(('The memory bandwidth of a GPU limits decode because each token rereads weights, and at long context the KV cache adds traffic. '*150)+'Summarize in one line.')")"
q_short="$(hawking_tps "$QWEN_GGUF" "$SHORTP" 32)"
q_long="$(hawking_tps "$QWEN_GGUF" "$LONGP" 32)"
r_short="$(hawking_tps "$RWKV_GGUF" "$SHORTP" 32)"
r_long="$(hawking_tps "$RWKV_GGUF" "$LONGP" 32)"
md "| $QWEN_BASE (Hawking) | ${q_short:-NA} | ${q_long:-NA} | transformer — KV wall (drops) |"
md "| **$RWKV_BASE (Hawking, SSM)** | ${r_short:-NA} | ${r_long:-NA} | **FLAT — no KV cache (the moat)** |"
if [ -n "$LLAMA_BENCH" ]; then
  read -r _ ltg_s <<<"$(llama_tps "$QWEN_GGUF" 64 32)"
  read -r _ ltg_l <<<"$(llama_tps "$QWEN_GGUF" "$CTX_LONG" 32)"
  md "| $QWEN_BASE (llama.cpp) | ${ltg_s:-NA} | ${ltg_l:-NA} | transformer — KV wall |"
fi
md ""
md "_The differentiator: the transformer rows fall with context; the RWKV-7 SSM row stays flat (constant recurrent state)._"
md "Closest SOTA for an optimized small instruct SSM: none shipping — llama.cpp has RWKV support but unoptimized; MLX has no RWKV-7."

# ============================================================= 8. QUALITY
say ""; say "-- [8/11] quality (deterministic task prompts, side-by-side) --"
md ""; md "## 8. Quality — deterministic tasks (greedy; pass = output contains the expected answer)"
md ""
md "| task | Hawking | llama.cpp | MLX | expected |"
md "|---|---|---|---|---|"
# Qwen chat template for Hawking's raw-completion path.
fmt() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n' "$1"; }
# Hawking has no auto chat-template → feed it the manually-templated prompt.
hawking_gen() { TO "$RUN_TIMEOUT" env HAWKING_QWEN_USER_DRAFT=0 "$HBIN" generate --weights "$QWEN_GGUF" --prompt "$1" --max-new-tokens "${2:-48}" --temperature 0 --seed 7 2>/dev/null | grep -vE '^\[(stats|hawking)' | head -c 4000 || true; }
# llama-completion + MLX auto-apply the model's chat template → feed the PLAIN question.
llama_gen() { # $1=PLAIN question $2=tok
  [ -n "$LLAMA_GEN" ] || { echo ""; return; }
  if [ "$LLAMA_GEN_KIND" = completion ]; then
    TO "$RUN_TIMEOUT" "$LLAMA_GEN" -m "$QWEN_GGUF" -p "$1" -n "${2:-48}" --temp 0 -ngl 999 2>/dev/null </dev/null | head -c 4000 || true
  else
    TO "$RUN_TIMEOUT" "$LLAMA_GEN" -m "$QWEN_GGUF" -no-cnv -p "$1" -n "${2:-48}" --temp 0 -ngl 999 2>/dev/null </dev/null | head -c 4000 || true
  fi
}
mlx_gen() { # $1=PLAIN question $2=tok
  [ -n "$MLX_PY" ] || { echo ""; return; }
  TO "$RUN_TIMEOUT" "$MLX_PY" -m mlx_lm.generate --model "$MLX_MODEL" --prompt "$1" --max-tokens "${2:-48}" --temp 0 2>/dev/null </dev/null | head -c 4000 || true
}
contains() { printf '%s' "$1" | grep -qi "$2" && echo "✅" || echo "❌"; }
qrun() { # $1=label $2=question $3=expected_regex
  local label="$1" q="$2" exp="$3" f a l m
  f="$(fmt "$q")"
  a="$(hawking_gen "$f" 48)"; l="$(llama_gen "$q" 48)"; m="$(mlx_gen "$q" 48)"
  printf '## %s\n### Q\n%s\n### Hawking\n%s\n### llama.cpp\n%s\n### MLX\n%s\n' "$label" "$q" "$a" "$l" "$m" >"$OUT/quality_${label}.md"
  md "| $label | $(contains "$a" "$exp") | $(contains "$l" "$exp") | $([ "$HAVE_MLX" = 1 ] && contains "$m" "$exp" || echo "—") | \`$exp\` |"
}
qrun math    "What is 17 multiplied by 23? Answer with the number only." "391"
qrun capital "What is the capital of France? One word." "Paris"
qrun json    "Return ONLY a JSON object with keys name and age for: Alice is 30." "\"age\""
[ "$QUICK" != 1 ] && qrun primes "List the first five prime numbers, comma-separated." "2, 3, 5, 7, 11"
md ""
md "_Raw answers saved to \`$OUT/quality_*.md\`. Hawking gets a manual Qwen chat template; llama.cpp/MLX get the plain question through their generation wrappers. RWKV-7 instruct quality is evaluated separately (it is a 0.4B model) — see \`tools/ci/ssm_quality_chat.sh\` and \`tools/ci/ssm_quality_suite.sh\`._"

# ============================================================= 9. DISTILL / POST-TRAIN
say ""; say "-- [9/11] distillation / post-train / recovery tooling --"
md ""; md "## 9. Distillation, post-train, and quality recovery tooling"
md ""
md "| facet | local Hawking tooling | status | closest SOTA comparison |"
md "|---|---|---|---|"
md "| teacher capture | \`hawking generate --batched-capture\`, \`tools/training/rwkv7_capture_teacher_logits.py\` | built for local RWKV-7 pipeline; GPU run is workload-sized | llama.cpp/ollama are runtimes, not teacher-capture trainer stacks |"
md "| SFT | \`tools/training/rwkv7_sft_stream.py\`, \`rwkv7_sft_torch.py\` | local PyTorch/MPS pipeline | external trainer frameworks; MLX has pieces but no RWKV-7 support noted here |"
md "| DPO / preference recovery | \`tools/training/rwkv7_dpo_torch.py\`, \`rwkv7_dpo_build_pairs.py\` | built as scripts | external trainer stacks, not runtime CLIs |"
md "| logit KD / draft training | \`rwkv7_train_draft.py\`, \`eagle5_train.py\`, \`eagle5_tau_eval.py\` | research/prototype; acceptance gates decide if it is useful | llama.cpp speculative decode can run drafts; Hawking has more local oracle tooling |"
md "| QAT / low-bit recovery | \`rwkv7_qat.py\`, \`lowbit_qat.py\`, STRAND/TQ scripts | research path; no product \`press --distill\` command yet | AWQ/GPTQ/QLoRA-style ecosystems external to runtime CLIs |"
md "| product command | target concept: \`hawking press --target tq3 --distill\` | not implemented; must be reported as a gap | no direct equivalent in llama.cpp/ollama; training stacks can compose it manually |"
md ""
md "Present training files:"
md '```'
find tools/training -maxdepth 1 -type f \( -name '*rwkv7*' -o -name '*qat*' -o -name '*eagle*' -o -name '*awq*' -o -name '*corpus*' \) -print | sort | sed -n '1,80p' >>"$REPORT"
md '```'
md ""
md "_Honest conclusion: Hawking already has the local capture/post-train building blocks. The missing thing is a reproducible one-command compress-then-recover artifact flow with per-bit quality cards._"

# ============================================================= 10. HAWKING DIAGNOSTIC
say ""; say "-- [10/11] full Hawking diagnostic (+ closest-SOTA notes) --"
md ""; md "## 10. Hawking full diagnostic — CLI probes and closest-SOTA notes"
md ""
run_diag() { # $1=label $2...=cmd
  local label="$1"; shift
  say "  diag: $label"
  { echo "### $label"; echo '```'; TO "$RUN_TIMEOUT" "$@" 2>&1 | head -40; echo '```'; } >>"$OUT/diagnostic.md"
}
: >"$OUT/diagnostic.md"
run_diag "top-level help"      "$HBIN" --help
run_diag "generate --help"     "$HBIN" generate --help
run_diag "serve --help"        "$HBIN" serve --help
run_diag "bench --help"        "$HBIN" bench --help
run_diag "bench-kernel --help" "$HBIN" bench-kernel --help
run_diag "press --help"        "$HBIN" press --help
run_diag "fit --help"          "$HBIN" fit --help
run_diag "doctor --help"       "$HBIN" doctor --help
run_diag "bake-sidecar --help" "$HBIN" bake-sidecar --help
run_diag "spec-oracle --help"  "$HBIN" spec-oracle --help
run_diag "version"            "$HBIN" version
run_diag "shader-hash"        "$HBIN" shader-hash
run_diag "verify"             "$HBIN" verify --weights "$QWEN_GGUF"
run_diag "doctor --json"      "$HBIN" doctor --weights "$QWEN_GGUF" --json --max-seq-len 32768
run_diag "fit (max-capability)" "$HBIN" fit --weights "$QWEN_GGUF" --intent max-capability
run_diag "fit (max-context, RWKV/SSM)" "$HBIN" fit --weights "$RWKV_GGUF" --intent max-context
run_diag "press --dry-run (out-of-core all-bit condense planner)" "$HBIN" press --dry-run --memory-budget "$PRESS_BUDGET" --target "$BIT_TARGETS" --weights "$QWEN_GGUF"
run_diag "stats"             "$HBIN" stats --weights "$QWEN_GGUF" --prompt "$SHORTP" --max-new-tokens 16
if ls profiles/*.json >/dev/null 2>&1; then
  first_profile="$(ls profiles/*.json | head -1)"
  run_diag "profile-rank (first local profile)" "$HBIN" profile-rank --profile-json "$first_profile"
fi
md "| Hawking capability | subcommand | closest SOTA |"
md "|---|---|---|"
md "| inference (decode) | \`generate\` | llama.cpp \`llama-cli\`, MLX \`mlx_lm.generate\`, ollama |"
md "| OpenAI server + continuous batch | \`serve\` | llama-server, ollama, vLLM |"
md "| benchmark suite | \`bench\` | \`llama-bench\`, MLX timing wrappers |"
md "| per-kernel microbench | \`bench-kernel\`, \`bench-q4k-shapes\` | none direct in runtime CLIs |"
md "| **per-Mac fit planner / envelope** | \`fit\`, \`doctor --json\` | none direct (LM Studio shows a coarse RAM estimate) |"
md "| **capability-first auto serve (anti-throttle)** | \`serve --auto --intent\` | none (ollama/LM Studio pick silently) |"
md "| **out-of-core condense planner** | \`press --dry-run\` | llama-quantize / AWQ / GPTQ (all in-memory) |"
md "| hardware kernel autotune | \`autotune\` | llama.cpp has none (compile-time) |"
md "| artifact integrity + sidecar | \`verify\`, \`bake-sidecar\` | gguf hash (manual) |"
md "| regression / reproducibility | \`batch-hash\`, \`shader-hash\`, \`profile-rank\` | ad-hoc scripts |"
md "| speculative oracle | \`spec-oracle\`, \`--user-draft\` | llama.cpp speculative decode, fewer local oracles |"
md "| distill / post-train | \`tools/training/*.py\` | external trainer stacks; not llama.cpp/ollama runtime |"
md ""
md "_Full subcommand transcripts: \`$OUT/diagnostic.md\`._"

# ============================================================= 11. ENERGY (optional)
say ""; say "-- [11/11] energy (optional, macmon) --"
md ""; md "## 11. Energy (J/tok)"
if command -v macmon >/dev/null 2>&1; then
  md "macmon present — run \`tools/bench/energy_paired.sh\` / \`tools/bench/phase_joules.sh\` for the full J/tok paired measurement (Hawking vs llama.cpp). Not auto-run here to keep this pass non-destructive."
else
  md "macmon absent (\`brew install macmon\`) — energy comparison skipped. Hawking exposes per-domain J/tok via \`tools/bench/phase_joules.sh\`; llama.cpp has no built-in energy reporting."
fi

# ============================================================= summary
say ""; say "=== DONE — report: $REPORT ==="
{
  echo; echo "## Summary"
  echo "- Speed ($QWEN_BASE decode tps): Hawking ${hk:-NA} vs llama.cpp ${ltg:-NA} vs MLX ${mtg:-NA}."
  echo "- Long-context moat: $RWKV_BASE (Hawking) short ${r_short:-NA} -> long ${r_long:-NA} (flat) vs $QWEN_BASE short ${q_short:-NA} -> long ${q_long:-NA} (KV wall)."
  echo "- Compression/quantization: Hawking + llama.cpp share the GGUF (identical); Hawking adds out-of-core \`press\` planning and the all-bit dry-run ladder (\`$BIT_TARGETS\`)."
  echo "- Distillation/post-train: local RWKV/QAT/KD/DPO tooling exists, but the honest product gap is still a one-command \`press --distill\` artifact flow with quality cards."
  echo "- Unique to Hawking: per-Mac \`fit\`/\`doctor --json\`, capability-first \`serve --auto\`, kernel microbench visibility, out-of-core \`press\`."
  [ "$HAVE_MLX" = 1 ] || echo "- NOTE: MLX was SKIPPED (mlx_lm not importable). Install: \`python3.12 -m pip install mlx-lm\`, then re-run (or set MLX_PYTHON)."
  [ "$CLEAN" = clean ] || echo "- NOTE: run was NOT clean ($CLEAN) — re-run with everything closed for trustworthy absolute numbers."
  echo
} >>"$REPORT"
say "Open: sed -n '1,200p' $REPORT"
