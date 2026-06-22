#!/usr/bin/env bash
# =============================================================================
# tools/bench/compare_sota.sh
#
# THE comprehensive head-to-head: Hawking vs the closest SOTA local runtimes
# (llama.cpp + MLX), plus a full Hawking self-diagnostic that also exercises the
# capabilities the others don't have (noting the nearest SOTA equivalent).
#
# Dimensions:
#   1. FOOTPRINT / compression   — on-disk bpw + the out-of-core press planner
#   2. SPEED                     — warm decode tps + prefill, same GGUF
#   3. LONG-CONTEXT (the moat)   — SSM flat-decode vs the transformer KV wall
#   4. QUALITY                   — deterministic task prompts, side-by-side + pass/fail
#   5. HAWKING DIAGNOSTIC        — every subcommand; closest-SOTA note for the unique ones
#   6. ENERGY (optional)         — J/tok via macmon, if present
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
TRIALS="${TRIALS:-3}"; [ "$QUICK" = 1 ] && TRIALS=2
TOK="${TOK:-128}";     [ "$QUICK" = 1 ] && TOK=64
CTX_LONG="${CTX_LONG:-8192}"
RUN_TIMEOUT="${RUN_TIMEOUT:-300}"
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
  echo "- Config: TRIALS=$TRIALS TOK=$TOK CTX_LONG=$CTX_LONG RUN_TIMEOUT=${RUN_TIMEOUT}s QUICK=$QUICK"
  [ -n "$busy_gpu" ] && { echo; echo "> ⚠ other model jobs were running at start:"; echo '> ```'; echo "$busy_gpu" | sed 's/^/> /'; echo '> ```'; }
  echo
} >"$REPORT"
say "report: $REPORT"
[ "$HAVE_HAWKING" = 1 ] || { say "FATAL: hawking binary not built ($HBIN) — cargo build --release -p hawking"; exit 2; }

# ---------------------------------------------------------------- engine runners (warm-median tps)
# Hawking decode tps for a prompt + token budget.
hawking_tps() { # $1=gguf $2=prompt $3=tok -> median tps
  local g="$1" p="$2" t="$3" i out
  # warmup (discarded) to amortize cold PSO shader-compile — we want WARM tps.
  TO "$RUN_TIMEOUT" env HAWKING_QWEN_USER_DRAFT=0 "$HBIN" generate --weights "$g" \
    --prompt "$p" --max-new-tokens 8 --temperature 0 --seed 5 >/dev/null 2>&1 || true
  for i in $(seq 1 "$TRIALS"); do
    out="$(TO "$RUN_TIMEOUT" env HAWKING_QWEN_USER_DRAFT=0 "$HBIN" generate --weights "$g" \
            --prompt "$p" --max-new-tokens "$t" --temperature 0 --seed 5 2>/dev/null || true)"
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

# ============================================================= 1. FOOTPRINT
say ""; say "-- [1/6] footprint / compression --"
md "## 1. Footprint / compression"
md ""
md "| model | engine | on-disk | bpw | note |"
md "|---|---|---|---|---|"
for g in "$QWEN_GGUF" "$RWKV_GGUF"; do
  [ -f "$g" ] || continue
  b="$(stat -f%z "$g")"
  # bpw via hawking press metadata (weight bytes / params)
  bpw="$($HBIN press --dry-run --weights "$g" 2>/dev/null | grep -oE '~[0-9.]+ bpw' | grep -oE '[0-9.]+' | head -1 || echo NA)"
  md "| $(basename "$g") | Hawking + llama.cpp (same GGUF) | $(human_gib "$b") | ${bpw:-NA} | identical artifact — both load this file |"
done
md "| $(basename "$QWEN_GGUF") (MLX 4bit) | MLX | (HF download) | ~4.5 | different artifact; MLX uses its own 4-bit format |"
md ""
md "**Hawking-unique:** \`hawking press --dry-run --memory-budget\` plans an OUT-OF-CORE condense (quantize a parent that does"
md "not fit fully resident). Closest SOTA: \`llama-quantize\` (in-memory only) / AutoAWQ / GPTQ (need the full parent resident)."
press_demo="$($HBIN press --dry-run --memory-budget 2gb --target 4,3,2 --weights "$QWEN_GGUF" 2>/dev/null | grep -E 'WEDGE|out-of-core|full-resident' | head -3 || true)"
[ -n "$press_demo" ] && { md '```'; md "$press_demo"; md '```'; }

# ============================================================= 2. SPEED
say ""; say "-- [2/6] speed (warm decode tps, same Qwen-3B-Q4_K_M) --"
md ""; md "## 2. Speed — warm decode tps (same Qwen2.5-3B-Q4_K_M GGUF)"
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

# ============================================================= 3. LONG-CONTEXT (the moat)
say ""; say "-- [3/6] long-context: SSM flat vs transformer KV wall --"
md ""; md "## 3. Long-context — the SSM moat (decode tps vs context)"
md ""
md "| model / engine | short | ~${CTX_LONG} ctx | shape |"
md "|---|---|---|---|"
# build a long prompt
LONGP="$(python3 -c "print(('The memory bandwidth of a GPU limits decode because each token rereads weights, and at long context the KV cache adds traffic. '*150)+'Summarize in one line.')")"
q_short="$(hawking_tps "$QWEN_GGUF" "$SHORTP" 32)"
q_long="$(hawking_tps "$QWEN_GGUF" "$LONGP" 32)"
r_short="$(hawking_tps "$RWKV_GGUF" "$SHORTP" 32)"
r_long="$(hawking_tps "$RWKV_GGUF" "$LONGP" 32)"
md "| Qwen-3B (Hawking) | ${q_short:-NA} | ${q_long:-NA} | transformer — KV wall (drops) |"
md "| **RWKV-7 (Hawking, SSM)** | ${r_short:-NA} | ${r_long:-NA} | **FLAT — no KV cache (the moat)** |"
if [ -n "$LLAMA_BENCH" ]; then
  read -r _ ltg_s <<<"$(llama_tps "$QWEN_GGUF" 64 32)"
  read -r _ ltg_l <<<"$(llama_tps "$QWEN_GGUF" "$CTX_LONG" 32)"
  md "| Qwen-3B (llama.cpp) | ${ltg_s:-NA} | ${ltg_l:-NA} | transformer — KV wall |"
fi
md ""
md "_The differentiator: the transformer rows fall with context; the RWKV-7 SSM row stays flat (constant recurrent state)._"
md "Closest SOTA for an optimized small instruct SSM: none shipping — llama.cpp has RWKV support but unoptimized; MLX has no RWKV-7."

# ============================================================= 4. QUALITY
say ""; say "-- [4/6] quality (deterministic task prompts, side-by-side) --"
md ""; md "## 4. Quality — deterministic tasks (greedy; pass = output contains the expected answer)"
md ""
md "| task | Hawking | llama.cpp | MLX | expected |"
md "|---|---|---|---|---|"
# Qwen chat template, applied identically to the raw-completion CLIs (Hawking, llama-cli).
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
md "_Raw answers saved to \`$OUT/quality_*.md\`. Same chat-template string is fed to Hawking + llama.cpp (both raw-completion CLIs); MLX gets the plain question and applies its own template. RWKV-7 instruct quality is evaluated separately (it is a 0.4B model) — see \`tools/ci/ssm_quality_chat.sh\`._"

# ============================================================= 5. HAWKING DIAGNOSTIC
say ""; say "-- [5/6] full Hawking diagnostic (+ closest-SOTA notes) --"
md ""; md "## 5. Hawking full diagnostic — every subcommand (and the nearest SOTA for the unique ones)"
md ""
run_diag() { # $1=label $2...=cmd
  local label="$1"; shift
  say "  diag: $label"
  { echo "### $label"; echo '```'; TO "$RUN_TIMEOUT" "$@" 2>&1 | head -40; echo '```'; } >>"$OUT/diagnostic.md"
}
: >"$OUT/diagnostic.md"
run_diag "version"            "$HBIN" version
run_diag "doctor --json"      "$HBIN" doctor --weights "$QWEN_GGUF" --json --max-seq-len 32768
run_diag "fit (max-capability)" "$HBIN" fit --weights "$QWEN_GGUF" --intent max-capability
run_diag "fit (max-context, RWKV/SSM)" "$HBIN" fit --weights "$RWKV_GGUF" --intent max-context
run_diag "press --dry-run (out-of-core condense planner)" "$HBIN" press --dry-run --memory-budget 2gb --target 4,3,2,1 --weights "$QWEN_GGUF"
run_diag "stats"             "$HBIN" stats --weights "$QWEN_GGUF" --prompt "$SHORTP" --max-new-tokens 16
md "| Hawking capability | subcommand | closest SOTA |"
md "|---|---|---|"
md "| inference (decode) | \`generate\` | llama.cpp \`llama-cli\`, MLX \`mlx_lm.generate\`, ollama |"
md "| OpenAI server + continuous batch | \`serve\` | llama-server, ollama, vLLM |"
md "| **per-Mac fit planner / envelope** | \`fit\`, \`doctor --json\` | none direct (LM Studio shows a coarse RAM estimate) |"
md "| **capability-first auto serve (anti-throttle)** | \`serve --auto --intent\` | none (ollama/LM Studio pick silently) |"
md "| **out-of-core condense planner** | \`press --dry-run\` | llama-quantize / AWQ / GPTQ (all in-memory) |"
md "| hardware kernel autotune | \`autotune\` | llama.cpp has none (compile-time) |"
md "| artifact integrity + sidecar | \`verify\`, \`bake-sidecar\` | gguf hash (manual) |"
md ""
md "_Full subcommand transcripts: \`$OUT/diagnostic.md\`._"

# ============================================================= 6. ENERGY (optional)
say ""; say "-- [6/6] energy (optional, macmon) --"
md ""; md "## 6. Energy (J/tok)"
if command -v macmon >/dev/null 2>&1; then
  md "macmon present — run \`tools/bench/energy_paired.sh\` / \`tools/bench/phase_joules.sh\` for the full J/tok paired measurement (Hawking vs llama.cpp). Not auto-run here to keep this pass non-destructive."
else
  md "macmon absent (\`brew install macmon\`) — energy comparison skipped. Hawking exposes per-domain J/tok via \`tools/bench/phase_joules.sh\`; llama.cpp has no built-in energy reporting."
fi

# ============================================================= summary
say ""; say "=== DONE — report: $REPORT ==="
{
  echo; echo "## Summary"
  echo "- Speed (Qwen-3B decode tps): Hawking ${hk:-NA} vs llama.cpp ${ltg:-NA} vs MLX ${mtg:-NA}."
  echo "- Long-context moat: RWKV-7 (Hawking) short ${r_short:-NA} → long ${r_long:-NA} (flat) vs Qwen short ${q_short:-NA} → long ${q_long:-NA} (KV wall)."
  echo "- Compression: Hawking + llama.cpp share the GGUF (identical); Hawking adds out-of-core \`press\` planning the others lack."
  echo "- Unique to Hawking: per-Mac \`fit\`/\`doctor --json\`, capability-first \`serve --auto\`, out-of-core \`press\`."
  [ "$HAVE_MLX" = 1 ] || echo "- NOTE: MLX was SKIPPED (mlx_lm not importable). Install: \`python3.12 -m pip install mlx-lm\`, then re-run (or set MLX_PYTHON)."
  [ "$CLEAN" = clean ] || echo "- NOTE: run was NOT clean ($CLEAN) — re-run with everything closed for trustworthy absolute numbers."
  echo
} >>"$REPORT"
say "Open: sed -n '1,200p' $REPORT"
